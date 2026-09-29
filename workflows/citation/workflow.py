"""引文核查 workflow：**特征注入式**（文档骨架、引文与声称出处由代码写，短文正文由后端写）。

流程与检索重排（`workflows/rerank`）同构（`plan → sample → render → ask`），区别在「判什么」：

- **文档抬头、节抬头、句子前缀、引文抬头、声称出处抬头是结构化事实**，由代码渲染进 state
  ——让模型去写只会写歪；
- 后端只写**标题与 6 句话**（一篇短文的正文是长尾的，模板拼的脱离现实）；
- **引文由代码从这篇短文里剪**（整句照抄 / 截前半句 / 换掉一个数字），所以「引文在不在文档
  里、出自第几节」是**注入特征经纯函数**算出来的（`features.plan_for` + `label_for`），
  模型碰不到；写出来的正文必须兑现这份计划（`render.check_draft` 在渲染期查、
  `judges.HygieneJudge` 再查一遍）。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「引文落在该节第几句 = 最后一句」这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("citation")` 可用）。
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
from .features import CitationCase
from .spec import CitationSpec

WORKFLOW_NAME = "citation"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是「引文落在该节第几句 = 最后一句」：它是**只影响渲染、不影响标签**的那一维——标签由
# `form` / `where` / `place` 三者算（引文出自哪一节只看 `place`），`target` 只决定同一节里
# 引用哪一句，换一句不会让任何一族的答案在训练侧退化成常量（逐档枚举验过，见
# `tests/workflows/citation/test_pipeline.py`）。
#
# 反过来，留出 `form` 会让 `fabricated` 一档消失、留出 `where` 会让 `mislocated` 一档消失
# （`verdict` 的三档就少一档），留出 `place` 会让 `section` 的「第 3 节」取不到。这三类都不能留。
OOD_TARGETS: Final[frozenset[str]] = frozenset({"last"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260929
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "form": "引文造法",
    "where": "声称出处对不对",
    "place": "引文落在哪一节",
    "target": "引文落在该节第几句",
}


@dataclass(frozen=True)
class CitationConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 3×2×3×1 = 18 组、全空间 3×2×3×2 = 36 组，
      512 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；
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
        target=tuple(value for value in features.TARGETS if value not in OOD_TARGETS)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（引文位置留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(
            target=tuple(value for value in features.TARGETS if value in OOD_TARGETS)
        )
    ]


def bucket_of(case: CitationCase) -> int:
    """输入：一个 case。输出：它的哈希桶号（0~99）。

    预期行为：只按 tag 组合算，与切分名无关——同一个组合永远落在同一个桶里，所以「同一组合
    不跨切分」是算出来的，不是靠人排的。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{SPLIT_SEED}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[CitationCase], split: str) -> list[CitationCase]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def _cover_pick(cases: list[CitationCase], limit: int) -> list[CitationCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：造法 → 声称出处 → 落点），同分时取原顺序里
    靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`form` 有三个取值、`verdict` 的三个档位一一对应它，`where` 有
    两个取值、`in_claimed` 的两个档位一一对应它——纯截断的话，10 条的小批量很可能一个
    `fabricated` 都出不来，均衡闸与审计都会当场报出来。覆盖优先把这种「小批量必然缺一档」
    的坑填掉，代价只是 case 顺序不再等于桶序（切分归属仍是桶算的，没变）。
    """
    names = ("form", "where", "place")
    chosen: list[CitationCase] = []
    seen: dict[str, set[JsonValue]] = {name: set() for name in names}
    remaining = list(cases)
    while remaining and len(chosen) < limit:
        keys = [
            (
                sum(1 for name in names if case.as_dict()[name] not in seen[name]),
                -index,
            )
            for index, case in enumerate(remaining)
        ]
        best = max(range(len(remaining)), key=lambda index: keys[index])
        case = remaining.pop(best)
        chosen.append(case)
        for name in names:
            seen[name].add(case.as_dict()[name])
    return chosen


def cases_for(config: CitationConfig) -> list[CitationCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], CitationCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(features.build_cases(config.cover_limit, dims=id_dims()), config.split)
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class CitationWorkflow:
    """引文核查：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 短文正文由后端写：本 workflow 不是纯构造式
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

    def config_for(self, split: str) -> CitationConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return CitationConfig(split=split)

    def build_cases(self, config: CitationConfig) -> list[CitationCase]:
        """输入：采样参数。输出：tag 组合列表。"""
        return cases_for(config)

    def case_label(self, case: CitationCase) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return case.describe()

    def stats_dimensions(self, case: CitationCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "form": case.form,
            "where": case.where,
            "place": case.place,
            "target": case.target,
        }

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: CitationConfig, case: CitationCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法编号由 RNG 从池里抽（它不是 case 维度），引文计划在 `sample` 阶段按
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

    def sample(self, rng: random.Random, split: SplitSpec) -> CitationSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, CitationCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return CitationSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: CitationSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要标题与 6 句话，先按引文计划校验（不合格抛错，管线据此重渲染），再与
        代码渲染的抬头 / 节号 / 引文 / 声称出处拼起来；后端没注入就报错——`needs` 非空时管线
        一定会注入，这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」
        这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("citation workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        title = draft.data.get("title")
        sentences = draft.data.get("sentences")
        if not isinstance(title, str) or not isinstance(sentences, list):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.CitationDraft(
            title=title, sentences=[str(item) for item in sentences]
        )
        render.check_draft(parsed, spec.plan, loc)
        return render.to_state(parsed, spec.plan, loc)

    def ask(self, spec: CitationSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def reparse(self, state: object, *, case: object = None) -> render.ParsedCitation:
        """输入：渲染后的 state。输出：纯代码能反解的部分（标题 / 分节句子 / 引文 / 声称出处）。"""
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

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分审计计数, 缺失项列表)。"""
        return audit_mod.audit_run(run_dir, splits, locale)

    def available_locales(self) -> list[str]:
        """输入：无。输出：该 workflow 支持的语言标签列表。"""
        return locales.available()


@register(WORKFLOW_NAME)
def _factory() -> CitationWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return CitationWorkflow()
