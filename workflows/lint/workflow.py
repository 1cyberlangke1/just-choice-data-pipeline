"""行级规则检查 workflow：**特征注入式**（规则表与处置办法由代码写，片段原文由后端写）。

流程与打分评分（`workflows/rubric`）同构（`plan → sample → render → ask`），区别在「判什么」：

- **片段抬头、规则名与判据、处置办法表是结构化事实**，由代码渲染进 state——让模型去写只会
  写歪；
- 后端只写**那 8 行原文**（真实代码 / 日志 / 文档里的行是长尾的，模板拼的脱离现实）；
- 哪几行命中由**注入特征经纯函数**算出（`features.plan_for`），模型碰不到；写出来的行必须
  兑现这份计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` 再查一遍）。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「体裁 = 文档」这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("lint")` 可用）。
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
from .features import LintCase
from .spec import LintSpec

WORKFLOW_NAME = "lint"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是 `genre=doc`：训练侧只见过源码与日志两种体裁，OOD 考的就是「同一套规则判据换到
# 文档体裁还认不认」。而三条体裁的**答案分布完全一样**（命中行数仍取 0~8 均匀、规则表仍是
# 那五条），所以留出它不会让任何一族的答案在训练侧退化成常量——逐档枚举验过
# （`tests/workflows/lint/test_pipeline.py`）。
#
# 反过来，留出某条**规则**（比如 `long`）会把「按模式判命中」与「按长度判命中」劈成两半，
# 训练侧再也见不到长度判据那一类，`line` 族就少了一整种考法；留出某个**命中行数**更不行，
# `count` 的 9 个取值会缺一格。这两类都不能留。
OOD_GENRES: Final[frozenset[str]] = frozenset({"doc"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260929
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "rule": "查哪条规则",
    "genre": "片段体裁",
    "hits": "命中行数",
}


@dataclass(frozen=True)
class LintConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 5×2×9 = 90 组、全空间 5×3×9 = 135 组，
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


def _restricted(**overrides: tuple[str, ...]) -> dict[str, tuple[JsonValue, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    dims: dict[str, tuple[JsonValue, ...]] = dict(features.DIMS)
    dims.update(overrides)
    return dims


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(genre=tuple(v for v in features.GENRES if v not in OOD_GENRES))


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（体裁留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(genre=tuple(v for v in features.GENRES if v in OOD_GENRES))]


def bucket_of(case: LintCase) -> int:
    """输入：一个 case。输出：它的哈希桶号（0~99）。

    预期行为：只按 tag 组合算，与切分名无关——同一个组合永远落在同一个桶里，所以「同一组合
    不跨切分」是算出来的，不是靠人排的。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{SPLIT_SEED}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[LintCase], split: str) -> list[LintCase]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def _cover_pick(cases: list[LintCase], limit: int) -> list[LintCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：命中行数 → 规则 → 体裁），同分时取原顺序
    里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`hits` 有九个取值，而 `spot` 只在「恰好一行命中」时才出——纯截断的
    话，10 条的小批量有约三成概率一条 `spot` 都出不来，审计会当场报「题族 spot 一条都没出」。
    覆盖优先把这种「小批量必然缺一族」的坑填掉，代价只是 case 顺序不再等于桶序（切分归属
    仍是桶算的，没变）。
    """
    names = ("hits", "rule", "genre")
    chosen: list[LintCase] = []
    seen: dict[str, set[JsonValue]] = {name: set() for name in names}
    remaining = list(cases)
    while remaining and len(chosen) < limit:
        keys = [
            (sum(1 for name in names if case.as_dict()[name] not in seen[name]), -index)
            for index, case in enumerate(remaining)
        ]
        best = max(range(len(remaining)), key=lambda index: keys[index])
        case = remaining.pop(best)
        chosen.append(case)
        for name in names:
            seen[name].add(case.as_dict()[name])
    return chosen


def cases_for(config: LintConfig) -> list[LintCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], LintCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(features.build_cases(config.cover_limit, dims=id_dims()), config.split)
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class LintWorkflow:
    """行级规则检查：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 片段原文由后端写：本 workflow 不是纯构造式
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

    def config_for(self, split: str) -> LintConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return LintConfig(split=split)

    def build_cases(self, config: LintConfig) -> list[LintCase]:
        """输入：采样参数。输出：tag 组合列表。"""
        return cases_for(config)

    def case_label(self, case: LintCase) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return case.describe()

    def stats_dimensions(self, case: LintCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "rule": case.rule,
            "genre": case.genre,
            "hits": case.hits,
        }

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: LintConfig, case: LintCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法编号由 RNG 从池里抽（它不是 case 维度），行计划在 `sample` 阶段按
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

    def sample(self, rng: random.Random, split: SplitSpec) -> LintSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, LintCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return LintSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: LintSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那 8 行原文，先按行计划校验（不合格抛错，管线据此重渲染），再与
        代码渲染的抬头 / 判据 / 处置办法表拼起来；后端没注入就报错——`needs` 非空时管线一定
        会注入，这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「片段由模型写」这个
        前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("lint workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        raw = draft.data.get("lines")
        if not isinstance(raw, list):
            raise ValueError(f"渲染草稿里没有 lines 字段：{raw!r}")
        lines = [str(item) for item in raw]
        render.check_draft(lines, spec.plan)
        return render.to_state(lines, spec.plan, loc)

    def ask(self, spec: LintSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def reparse(self, state: object, *, case: object = None) -> render.ParsedLint:
        """输入：渲染后的 state。输出：纯代码能反解的部分（片段 / 规则 / 体裁 / 处置办法表）。"""
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
                out.append(judges_mod.TraceJudge(self.backend, locale))
            else:
                raise KeyError(f"未知判别器：{name!r}（可用：hygiene / frame / trace）")
        return tuple(out)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：空表（这一族没有「模型写的自然语言正文」）。

        预期行为：state 是「一行中文抬头 + 一段源码」，**模型写的是那段代码**——代码与语言无关
        （真机 zh 侧汉字占比低到 0.099 全是正常产物）。抬头由代码按语言表写，天然一致。
        所以这里报空表：把代码喂给语言判据只会把好数据当坏数据。
        """
        del sample
        return {}

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分审计计数, 缺失项列表)。"""
        return audit_mod.audit_run(run_dir, splits, locale)

    def available_locales(self) -> list[str]:
        """输入：无。输出：该 workflow 支持的语言标签列表。"""
        return locales.available()


@register(WORKFLOW_NAME)
def _factory() -> LintWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return LintWorkflow()
