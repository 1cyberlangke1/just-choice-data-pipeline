"""改动引入性判定 workflow：**特征注入式**（改动记录与判定口径由代码写，发现与修法由后端写）。

流程与来源选择（`workflows/sourceselect`）同构（`plan → sample → render → ask`），区别在
「判什么」：

- **改动记录、判定口径是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**一条审查发现 + 一条建议修法**（真实审查意见的措辞是长尾的，模板拼出来的
  「第 N 行有问题」不像人话）；
- 标签由**注入特征**经纯函数算出，模型碰不到。六个题族里三件由那两行文本承载
  （指向哪一行、引用哪个符号、修法点没点名），交给 `judges.TraceJudge` 反解校验
  ——答不回来就说明文本与标签脱节了。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「记录 3 行」这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("findtriage")` 可用）。
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
from .features import RecordCase
from .spec import FindTriageSpec

WORKFLOW_NAME = "findtriage"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是 `size=3`：表长只决定记录有几行，**不决定任何一档标签**——五条口径里没有一条
# 提到行数，所以留出它不杀任何一档，逐档枚举验过（`tests/workflows/findtriage/test_pipeline.py`）。
#
# 反过来，留出 `cite=untouched` 会把处置档冻成「丢弃」、留出 `evidence=none` 会把证据档
# 冻成 `none`、留出 `fix=vague` 会把「落不落地」冻成 false——那三个都会让 OOD 侧整族退化，
# 不能留。`decoy` 只有两档且只喂一族题，留出它会让那一族变成常量，也不能留。
OOD_SIZES: Final[frozenset[int]] = frozenset({3})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "cite": "发现指向的那一行在本次改动里的处境",
    "evidence": "发现引用的证据符号落在哪里",
    "fix": "建议修法点没点名具体符号",
    "size": "改动记录的条数",
    "decoy": "记录里有没有近形符号的诱饵行",
}


@dataclass(frozen=True)
class FindTriageConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 3×3×2×3×2 = 108 组、剔除 1 对禁用对后 96 组，
      512 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 512
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasing_pool", self.phrasing_pool)
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


def _restricted(**overrides: tuple[JsonValue, ...]) -> dict[str, tuple[JsonValue, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    dims = dict(features.DIMS)
    dims.update(overrides)
    return dims


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(size=tuple(value for value in features.SIZES if value not in OOD_SIZES))


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（表长留出）。"""
    return [
        _restricted(size=tuple(value for value in features.SIZES if value in OOD_SIZES))
    ]


def bucket_of(case: RecordCase) -> int:
    """输入：一个 case。输出：它的哈希桶号（0~99）。

    预期行为：只按 tag 组合算，与切分名无关——同一个组合永远落在同一个桶里，
    所以「同一组合不跨切分」是算出来的，不是靠人排的。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{SPLIT_SEED}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[RecordCase], split: str) -> list[RecordCase]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def cases_for(config: FindTriageConfig) -> list[RecordCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, str], ...], RecordCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(features.build_cases(config.cover_limit, dims=id_dims()), config.split)
    return cases[: config.case_limit] if config.case_limit else cases


class FindTriageWorkflow:
    """改动引入性判定：正文、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 发现与修法由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")

    def __init__(self) -> None:
        """输入：无。输出：实例。预期行为：后端要等 `bind_backend` 才注入。"""
        self.backend: GenerationBackend | None = None

    def bind_backend(self, backend: GenerationBackend) -> None:
        """输入：生成后端。输出：无。预期行为：存下后端，`render` 与 `trace` 判别器要用。"""
        self.backend = backend

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：支持的切分名。"""
        return ("train", "val", "test_iid", "test_ood")

    def config_for(self, split: str) -> FindTriageConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return FindTriageConfig(split=split)

    def build_cases(self, config: FindTriageConfig) -> list[RecordCase]:
        """输入：采样参数。输出：tag 组合列表。"""
        return cases_for(config)

    def case_label(self, case: RecordCase) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: RecordCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "cite": case.cite,
            "evidence": case.evidence,
            "fix": case.fix,
            "size": case.size,
            "decoy": case.decoy,
        }

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: FindTriageConfig, case: RecordCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。"""
        return SplitSpec(
            name=config.split,
            params={
                "case": case,
                "phrasing": rng.choice(config.phrasing_pool),
                "seq": seq,
            },
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> FindTriageSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        del rng
        return FindTriageSpec(
            case=split.params["case"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: FindTriageSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一条发现与一条修法，再与代码渲染的记录表 / 口径表拼起来；
        后端没注入就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，
        绝不能悄悄降级成模板（那会让「发现由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("findtriage workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        plan = spec.plan
        draft = self.backend.generate(render.build_task(plan, loc, spec.render_task_id))
        render.check_draft(draft.data, plan, loc)
        return render.to_state(draft.data, plan, loc)

    def ask(self, spec: FindTriageSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def reparse(self, state: object, *, case: object = None) -> render.ParsedFinding:
        """输入：渲染后的 state。输出：纯代码能反解的部分（发现 / 修法 / 记录 / 口径）。"""
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
        """输入：契约样本。输出：必须与声明语言一致的正文（发现与修法两行）。

        预期行为：只报模型写的那两行；改动记录与判定口径的取值词元由代码写，不报。
        """
        frame = render.parse_state(sample.get("state"))
        out: dict[str, str] = {}
        if frame.finding.strip():
            out["发现"] = frame.finding
        if frame.fix.strip():
            out["修法"] = frame.fix
        return out

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分审计计数, 缺失项列表)。"""
        return audit_mod.audit_run(run_dir, splits, locale)

    def available_locales(self) -> list[str]:
        """输入：无。输出：该 workflow 支持的语言标签列表。"""
        return locales.available()


@register(WORKFLOW_NAME)
def _factory() -> FindTriageWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return FindTriageWorkflow()
