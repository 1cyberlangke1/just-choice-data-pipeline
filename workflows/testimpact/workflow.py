"""测试影响选择 workflow：**特征注入式**（覆盖表与跑法档位表由代码写，改动由后端写）。

流程与日志行分诊（`workflows/logtriage`）、聊天回复门控（`workflows/chatgate`）同构
（`plan → sample → render → ask`），区别在「判什么」：

- **覆盖表、跑法档位表是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**一次改动**（一行提交标题 + 一份 unified diff）：真实 diff 是长尾的，模板拼出来的
  脱离现实分布；
- **改了哪些文件由代码按字面算**（`features.touched_in` 读 `diff --git` 那一行），所以标签是
  注入特征经纯函数算出来的（`features.plan_for` + `label_for`），模型碰不到；写出来的改动必须
  兑现这份计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` / `FrameJudge` 再各查一遍）。

**为什么要这一族**：社区里 `jev-test-filter`（★15）拿 Jev 给 git diff 里的每条测试打分、吐出
`vitest` / `node:test` / `Playwright` / `cargo test` / `go test` 的过滤参数；`jev-flash-router`
把它写成「文件选择、任务路径与**测试影响判断**」；`semdecide` 与 `jev-cli` 把同一种判断接到 CI
与 shell 的退出码上。共同点是一份**改动**进来、要一个**立刻能落地**的跑法，而不是让人通读整份
diff 再猜。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按触及组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`flavor=revert`（回滚）这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("testimpact")` 可用）。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Final

from jc.jsontypes import JsonValue
from jc.knobs import require_pool
from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import ImpactCase
from .spec import ImpactSpec

WORKFLOW_NAME = "testimpact"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `flavor=revert`：它是**只影响渲染、不影响标签**的那一维——标签由 `touch` 算
# （`flavor` 一个字符都不参与），改动性质只决定这份 diff 长成什么样。同分布侧一次都见不到它，
# 模型必须真的按覆盖表读，而不是靠「回滚这种改动大概该跑什么」的先验蒙。
#
# 反过来，留出某个 `touch` 会让 `count` / `which` / `route` 三族的档位在同分布侧缺一块，
# 那些档位就取不到了——均衡闸与审计都会当场报出来。所以留出的是 `flavor`。
OOD_FLAVORS: Final[frozenset[str]] = frozenset({features.OOD_FLAVOR})

# 同分布切分的哈希分桶边界（组合留出：桶按触及组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "touch": "这次改动触及哪几条路径",
    "flavor": "这是一次什么性质的改动",
}


@dataclass(frozen=True)
class ImpactConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 16×4 = 64 组，512 让枚举器稳稳走**全枚举**
      （成对覆盖在这里只会白扔组合）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 512
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasing_pool", self.phrasing_pool)
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


def _restricted(**overrides: tuple[JsonValue, ...]) -> dict[str, tuple[JsonValue, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    dims: dict[str, tuple[JsonValue, ...]] = dict(features.DIMS)
    dims.update(overrides)
    return dims


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        flavor=tuple(value for value in features.ALL_FLAVORS if value not in OOD_FLAVORS)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含 OOD 取值的维度表（`test_ood` 用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(
            flavor=tuple(value for value in features.ALL_FLAVORS if value in OOD_FLAVORS)
        )
    ]


def bucket_of(case: ImpactCase) -> int:
    """输入：一个 case。输出：它的哈希桶号（0~99）。

    预期行为：只按触及组合算，与切分名无关——同一个组合永远落在同一个桶里，所以「同一组合不跨
    切分」是算出来的，不是靠人排的。
    """
    return int(
        hashlib.sha256(f"{SPLIT_SEED}|{case.touch}".encode("utf-8")).hexdigest()[:8], 16
    ) % 100


def _slice(cases: list[ImpactCase], split: str) -> list[ImpactCase]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, 触及组合, 性质) 排序再切片，比例才是**精确**的。
    """
    ordered = sorted(
        cases, key=lambda case: (bucket_of(case), case.touch, case.flavor)
    )
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def _dims_of(case: ImpactCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多两个派生维度）。

    预期行为：`as_dict` 的两个维度之外再加 `need`（需要的套件集合）与 `level`（跑法档位）
    ——小批量里最容易被忽略的正是它们：只按 `touch` / `flavor` 铺开的话，10 条很可能一条
    「只跑冒烟」或「跑全套」都出不来，`count` / `which` / `route` 三族的某些档位当场取不到，
    均衡闸与审计都会报出来。
    """
    return {
        "touch": case.touch,
        "flavor": case.flavor,
        "need": features.touch_key(case.need()),
        "level": case.level(),
    }


def _cover_pick(cases: list[ImpactCase], limit: int) -> list[ImpactCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：触及组合 → 性质 → 套件集合 → 跑法档位），
    同分时取原顺序里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`touch` 有 16 个取值、`level` 有 4 个，纯截断的话 10 条的小批量很可能
    一条「跑全套」都出不来，`route` 的某些档位也取不到，均衡闸与审计都会当场报出来。覆盖优先
    把这种「小批量必然缺一档」的坑填掉，代价只是 case 顺序不再等于桶序（切分归属仍是桶算的）。
    """
    names = ("touch", "flavor", "need", "level")
    chosen: list[ImpactCase] = []
    seen: dict[str, set[str]] = {name: set() for name in names}
    remaining = list(cases)
    while remaining and len(chosen) < limit:
        keys = [
            (
                sum(1 for name in names if _dims_of(case)[name] not in seen[name]),
                -index,
            )
            for index, case in enumerate(remaining)
        ]
        best = max(range(len(remaining)), key=lambda index: keys[index])
        case = remaining.pop(best)
        chosen.append(case)
        for name in names:
            seen[name].add(_dims_of(case)[name])
    return chosen


def cases_for(config: ImpactConfig) -> list[ImpactCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], ImpactCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(features.build_cases(config.cover_limit, dims=id_dims()), config.split)
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class ImpactWorkflow(BackendHolder):
    """测试影响选择：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 改动由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> ImpactConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return ImpactConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: ImpactConfig, case: ImpactCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> ImpactSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, ImpactCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return ImpactSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: ImpactSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那一份改动，先按注入特征校验（不合格抛错，管线据此重渲染），再与代码
        渲染的覆盖表 / 档位表 / 抬头拼起来；后端没注入就报错——`needs` 非空时管线一定会注入，
        这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("testimpact workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        title = draft.data.get("title")
        diff = draft.data.get("diff")
        if not isinstance(title, str) or not isinstance(diff, str):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.ContentDraft(title=title, diff=diff)
        render.check_draft(parsed, spec.case, loc)
        return render.to_state(parsed, loc)

    def ask(
        self, spec: ImpactSpec, rng: random.Random, locale: str
    ) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「提交标题」）。

        预期行为：只报模型写的那一行提交标题；覆盖表与档位表是代码写的（按语言取规范文案），
        diff 是代码，一律不报——把 diff 当正文会让「汉字占比」这类判据把好数据当坏数据。
        """
        parsed = render.parse_state(sample.get("state"))
        title = parsed.title
        return {"提交标题": title} if title and title.strip() else {}



@register(WORKFLOW_NAME)
def _factory() -> ImpactWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return ImpactWorkflow()
