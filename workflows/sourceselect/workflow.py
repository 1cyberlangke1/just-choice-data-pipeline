"""来源选择（sourceselect）workflow：**特征注入式**（候选表与选源口径由代码写，查询由后端写）。

流程与检索结果判读（`workflows/search`）同构（`plan → sample → render → ask`），区别在
「判什么」：

- `search` 问的是「**这条结果**可不可信」（对单条结果判读，来源元数据只有一格）；
- 这里问的是「**该用哪个来源**」——在一张候选来源表里挑一个，挑不出来就弃权。
  与模型路由（`workflows/modelroute`）的分工：那个的候选是模型、门槛是数值档位；
  这里的候选是来源、属性是「类型 × 时效 × 覆盖度」三个枚举档，而且**要求只写在查询里**
  （modelroute 把门槛摊成状态行）。

**候选表与选源口径是结构化事实**，由代码渲染进 state；后端只写**一句查询**（真实请求的
语气是长尾的）。标签由注入特征经纯函数算出，模型碰不到；五个题族里 `need` 由那句查询承载，
交给 `judges.TraceJudge` 反解校验——答不回来就说明文本与标签脱节了。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 case 的哈希分桶，同 case 不跨切分；
- test_ood：**值留出**——表长 `size=4` 这一档，同分布侧一次都不出现（见 `OOD_SIZES`）。

导入本模块即自注册（`jc.workflow.load("sourceselect")` 可用）。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Final

from jc.judge import Judge
from jc.jsontypes import JsonValue
from jc.knobs import require_pool
from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import SourceCase
from .spec import SourceSpec

WORKFLOW_NAME = "sourceselect"
WORKFLOW_VERSION = "0.1.0"

#: 只在 test_ood 出现的取值（值留出）。
#:
#: 为什么是 `size=4`：它是唯一一个留出后**两边都不杀档**的取值。都拿枚举逐档验过
#: （见 `tests/workflows/sourceselect/test_pipeline.py` 的不变量测试）：
#: - 留出某个 `need` → `need` 那一族的选项表少一个正确答案（死选项）；
#: - 留出 `near_miss=1` → `near_miss` 那一族在同分布侧恒为 False（答案退化成常量）；
#: - 留出某个 `shape` → `enough` / `pick` / `cover` 至少一族在留出侧恒为同一个值
#:   （`tied` 那一侧永远是「有达标行」、`none` 那一侧永远弃权）；
#: - `size=4` 只改表的**长度**，选源口径一个字都不动，五个题族在两侧的每一档都还在。
OOD_SIZES: Final[frozenset[int]] = frozenset({4})

#: 同分布切分的哈希分桶边界（组合留出：桶按 case 算，同 case 永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "need": "首要要求",
    "size": "候选表条数",
    "shape": "表的形状",
    "near_miss": "有没有诱饵行",
}


@dataclass(frozen=True)
class SourceSelectConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 3×3×3×2 = 54 组、同分布空间 3×3×2×2 = 36 组，
      128 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 128
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasing_pool", self.phrasing_pool)
        if self.cover_limit < 1:
            raise ValueError(f"cover_limit 必须 >= 1：{self.cover_limit}")
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


def _restricted(**overrides: tuple[JsonValue, ...]) -> dict[str, tuple[JsonValue, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    dims = dict(features.DIMS)
    dims.update(overrides)
    return dims


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(size=tuple(v for v in features.SIZES if v not in OOD_SIZES))


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（表长留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(size=tuple(sorted(OOD_SIZES)))]


def bucket_of(case: SourceCase, seed: int = SPLIT_SEED) -> int:
    """输入：case、种子。输出：0~99 的桶号。

    预期行为：同一 case 必得同一桶，不同 case 近似均匀——切分单位是 **case**，不是样本。
    不用内置 `hash()`（它对字符串每进程随机化），否则切分会随进程漂移。
    """
    return int(
        hashlib.sha256(f"{seed}|{case.fingerprint()}".encode("utf-8")).hexdigest()[:8], 16
    ) % 100


def _slice(cases: list[SourceCase], split: str) -> list[SourceCase]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, case 指纹) 排序再切片，比例才是**精确**的。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), case.fingerprint()))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def cases_for(config: SourceSelectConfig) -> list[SourceCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], SourceCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(features.build_cases(config.cover_limit, dims=id_dims()), config.split)
    return cases[: config.case_limit] if config.case_limit else cases


class SourceSelectWorkflow(BackendHolder):
    """来源选择：候选表、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 那句查询由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> SourceSelectConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return SourceSelectConfig(split=split)

    def stats_dimensions(self, case: SourceCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return dict(case.as_dict())

    def plan_sample(
        self, rng: random.Random, config: SourceSelectConfig, case: SourceCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> SourceSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return SourceSpec(
            case=split.params["case"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: SourceSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一句查询，再与代码渲染的候选表 / 选源口径拼起来；
        后端没注入就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，
        绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("sourceselect workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.case, loc, spec.render_task_id))
        render.check_draft(draft.data, loc)
        return render.to_state(draft.data, spec.plan, loc)

    def ask(self, spec: SourceSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。预期行为：未知名字直接报错。"""
        out: list[Judge] = []
        for name in names:
            if name == "hygiene":
                out.append(judges_mod.HygieneJudge(locale))
            elif name == "frame":
                out.append(judges_mod.FrameJudge(locale))
            elif name == "trace":
                out.append(judges_mod.TraceJudge(self.backend, locale))
            else:
                raise KeyError(f"未知判别器：{name!r}（可用：hygiene / frame / trace）")
        return tuple(out)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（那句查询）。

        预期行为：只报模型写的那句查询；候选表与选源口径的取值词元由代码写，不报。
        """
        query = render.parse_state(sample.get("state")).query
        return {"查询": query} if query.strip() else {}



@register(WORKFLOW_NAME)
def _factory() -> SourceSelectWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return SourceSelectWorkflow()
