"""记忆写入门控 workflow：**特征注入式**（判据表与档位表由代码写，候选记忆由后端写）。

流程与提示注入（`workflows/promptinject`）、内容审核（`workflows/guardrail`）同构
（`plan → sample → render → ask`），区别在「判什么」：

- **入库判据表、处置档位表、记忆抬头是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**候选记忆**（一条偏好、一段配置、一句从文档里抄来的结论，读起来是长尾的，模板拼出来
  的脱离现实分布）；
- **命中哪几条由代码按字面算**（`features.PATTERNS` 的四条正则），所以标签是注入特征经纯函数
  算出来的（`features.plan_for` + `label_for`），模型碰不到；写出来的记忆必须兑现这份计划
  （`render.check_draft` 在渲染期查、`judges.HygieneJudge` / `FrameJudge` 再各查一遍）。

**与 `promptinject` 的区别**：promptinject 判的是「Agent **读进来**的外部内容有没有在试图操纵
它」；本 workflow 判的是「Agent **准备写出去**的一条记忆该不该落库」。一个管入口、一个管出口，
判据空间（覆盖指令 / 伪造角色 / 外送数据 / 诱导执行 vs 作用域膨胀 / 含凭据 / 冲突 / 来源）、
处置语义与素材形态都不一样，不重复。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`form=quote`（记忆写成引用原话）这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("memorygate")` 可用）。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from jc.backend.types import GenerationBackend
from jc.judge import Judge
from jc.jsontypes import JsonValue
from jc.knobs import require_pool
from jc.workflow import LabeledQuestion, SplitSpec, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import MemoryGateCase
from .spec import MemoryGateSpec

WORKFLOW_NAME = "memorygate"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `form=quote`：它是**只影响渲染、不影响标签**的那一维——标签由 `hits` 算
# （`form` 一个字符都不参与），写法只决定这条记忆长成什么样（一句话 / 分点 / 引用原话）。
# 同分布侧一次都见不到它，模型必须真的按字面判，而不是靠「这种写法里出现 X 就报警」的先验蒙。
#
# 反过来，留出 `hits` 会让 `count` / `combo` / `gate` 三族各少几档（16 组少一半），那些档位在
# 同分布侧就取不到了——均衡闸与审计都会当场报出来。所以留出的是 `form`。
OOD_FORMS: Final[frozenset[str]] = frozenset({features.FORM_QUOTE})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "hits": "该命中哪几条判据",
    "form": "记忆写成什么形态",
}


@dataclass(frozen=True)
class MemoryGateConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 16×3 = 48 组，512 让枚举器稳稳走**全枚举**
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
    return _restricted(form=tuple(value for value in features.FORMS if value not in OOD_FORMS))


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含 OOD 取值的维度表（`test_ood` 用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(form=tuple(value for value in features.FORMS if value in OOD_FORMS))]


def bucket_of(case: MemoryGateCase) -> int:
    """输入：一个 case。输出：它的哈希桶号（0~99）。

    预期行为：只按 tag 组合算，与切分名无关——同一个组合永远落在同一个桶里，所以「同一组合不跨
    切分」是算出来的，不是靠人排的。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{SPLIT_SEED}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[MemoryGateCase], split: str) -> list[MemoryGateCase]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def _dims_of(case: MemoryGateCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多一个「命中条数」）。

    预期行为：`as_dict` 的两个维度之外再加 `count`（命中条数）——小批量里最容易被忽略的正是它：
    只按 `hits` / `form` 铺开的话，10 条会落在「0 条命中 2 条、1 条命中 8 条」上，`count` 与
    `gate` 两族的答案分布当场偏到 80%，均衡闸标红。
    """
    return {"hits": case.hits, "form": case.form, "count": str(len(case.tags()))}


def _cover_pick(cases: list[MemoryGateCase], limit: int) -> list[MemoryGateCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：命中组合 → 写法 → 命中条数），同分时取原顺序里
    靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`hits` 有 16 个取值、`count` 有 5 个，纯截断的话 10 条的小批量很可能一条
    「4 条全中」都出不来，`combo` 与 `gate` 的某些档位也取不到，均衡闸与审计都会当场报出来。
    覆盖优先把这种「小批量必然缺一档」的坑填掉，代价只是 case 顺序不再等于桶序（切分归属仍是桶
    算的，没变）。
    """
    names = ("hits", "form", "count")
    chosen: list[MemoryGateCase] = []
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


def cases_for(config: MemoryGateConfig) -> list[MemoryGateCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], MemoryGateCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(features.build_cases(config.cover_limit, dims=id_dims()), config.split)
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class MemoryGateWorkflow:
    """记忆写入门控：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 候选记忆由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")

    def __init__(self) -> None:
        """输入：无。输出：实例。预期行为：后端要等 `bind_backend` 才注入。"""
        self.backend: GenerationBackend | None = None

    def bind_backend(self, backend: GenerationBackend) -> None:
        """输入：生成后端。输出：无。预期行为：存下后端，`render` 与 `trace` 判别器要用。"""
        self.backend = backend

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：四个切分名。"""
        return ("train", "val", "test_iid", "test_ood")

    def config_for(self, split: str) -> MemoryGateConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return MemoryGateConfig(split=split)

    def build_cases(self, config: MemoryGateConfig) -> list[MemoryGateCase]:
        """输入：采样参数。输出：tag 组合列表。"""
        return cases_for(config)

    def case_label(self, case: MemoryGateCase) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return case.describe()

    def stats_dimensions(self, case: MemoryGateCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：各维取值（均衡统计用）。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：各维的中文说明。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: MemoryGateConfig, case: MemoryGateCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法编号由 RNG 从池里抽（它不是 case 维度），入库计划在 `sample` 阶段按
        「case 指纹 + 问法编号」确定性地算出来。
        """
        return SplitSpec(
            name=config.split,
            params={
                "case": case,
                "phrasing": rng.choice(config.phrasing_pool),
                "seq": seq,
            },
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> MemoryGateSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, MemoryGateCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return MemoryGateSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: MemoryGateSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那条候选记忆，先按注入特征校验（不合格抛错，管线据此重渲染），再与代码
        渲染的判据表 / 档位表 / 抬头拼起来；后端没注入就报错——`needs` 非空时管线一定会注入，
        这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("memorygate workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        text = draft.data.get("text")
        if not isinstance(text, str):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.ContentDraft(text=text)
        render.check_draft(parsed, spec.case)
        return render.to_state(parsed, loc)

    def ask(self, spec: MemoryGateSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def reparse(self, state: object, *, case: object = None) -> render.ParsedContent:
        """输入：渲染后的 state。输出：纯代码能反解的部分（判据表 / 记忆）。"""
        del case
        return render.parse_state(state)

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。预期行为：未知名字直接报错。"""
        out: list[Judge] = []
        for name in names:
            if name == "hygiene":
                out.append(judges_mod.HygieneJudge(locale))
            elif name == "frame":
                out.append(judges_mod.FrameJudge(locale))
            elif name == "trace":
                if self.backend is None:
                    raise RuntimeError("trace 判别器需要后端，但还没被 bind_backend 注入")
                out.append(judges_mod.TraceJudge(self.backend, locale))
            else:
                raise KeyError(f"未知判别器：{name!r}（可用：hygiene / frame / trace）")
        return tuple(out)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「候选记忆」）。

        预期行为：只报模型写的那条候选记忆；`tags` 是判据的编号（由语言表写），不报。
        """
        parsed = render.parse_state(sample.get("state"))
        text = parsed.text
        return {"候选记忆": text} if text and text.strip() else {}

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分审计计数, 缺失项列表)。"""
        return audit_mod.audit_run(run_dir, splits, locale)

    def available_locales(self) -> list[str]:
        """输入：无。输出：该 workflow 支持的语言标签列表。"""
        return locales.available()


@register(WORKFLOW_NAME)
def _factory() -> MemoryGateWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return MemoryGateWorkflow()
