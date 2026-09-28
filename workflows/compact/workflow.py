"""会话上下文压缩 workflow：**特征注入式**（处置表与位置行由代码写，那一块正文由后端写）。

流程与工具调用门控（`workflows/toolgate`）同构（`plan → sample → render → ask`），区别在
「判什么」：

- **处置表与「本块位置」那一行是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**一块上下文正文**（真实会话里那一段是长尾的，模板拼的脱离现实）；
- 标签由**注入特征**经纯函数算出，模型碰不到。六个题族里两个由那段正文承载
  （这一块是什么 / 有没有承载持久事实），交给 `judges.TraceJudge` 反解校验——答不回来
  就说明文本与标签脱节了。

与检索结果判读（`workflows/search`）的分工：那个 workflow 判**一条搜索结果该不该信**，
文本是模型写的、来源元数据是代码写的；这里判**一块会话上下文该怎么处置**
（保留 / 截断 / 摘要 / 丢弃），文本是模型写的、位置元数据是代码写的。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「轮次位置 = 当前这一轮」这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("compact")` 可用）。
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
from .features import TagSet
from .spec import CompactSpec

WORKFLOW_NAME = "compact"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是 `age=current`：它是唯一一个留出后**既不动题族选项、也不动任何规则与档位**的取值。
# 都拿枚举逐档验过（见 `tests/workflows/compact/test_pipeline.py` 的不变量测试）：
# - 留出 `kind` 的某个取值 → 直接杀掉 `kind` 那一族的这一档（`kind` 是题族选项）；
# - 留出 `fact` 的某个取值 → 杀掉 `fact` 那一族的这一档；
# - 留出 `pins` 的某个取值 → 杀掉 `pin_overrode` 的答案空间（钉住那一格是它唯一为真的路）；
# - 留出 `age=old` → `critical_old` / `tight_old` / `old_truncate` 三条规则与 `truncate`
#   这一档一起消失（三条的条件里都有 `age=old`）；
# - 留出 `age=recent` → 优先级 2 档与兜底那条 `none` 一起消失；
# - 留出 `budget` 的任何一档都会杀规则：`critical` 杀 `critical_old`，
#   `tight` 杀 `tight_old`，`ample` 杀 `old_truncate`（留出它之后 `age=old` 只剩
#   `tight` / `critical` 两种预算，全被前面两条规则抢走）——**这一条是拿枚举验出来的**，
#   第一版就是错在留了 `ample`；
# - `age=current` 不出现在任何一条规则的条件里（只有 `old` 出现），留出它之后
#   `none` / `old_truncate` 仍然到得了，四档处置与四个优先级也都还在。
OOD_AGES: Final[frozenset[str]] = frozenset({"current"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260928
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "kind": "这一块是什么",
    "fact": "有没有承载持久事实",
    "age": "轮次位置",
    "pins": "是否钉住",
    "budget": "剩余预算",
}


@dataclass(frozen=True)
class CompactConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 5×2×3×2×2 = 120 组、剔除 2 对禁用对后 96 组，
      256 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；不带维度限制的全空间
      180 组剔除 2 对禁用对后 144 组，同样在 256 之下、也走全枚举
      （`tests/workflows/compact/test_features.py` 用它验覆盖器）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 256
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasing_pool", self.phrasing_pool)
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


def _restricted(**overrides: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    dims = dict(features.DIMS)
    dims.update(overrides)
    return dims


def id_dims() -> dict[str, tuple[str, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(age=tuple(v for v in features.AGES if v not in OOD_AGES))


def ood_dims() -> list[dict[str, tuple[str, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（轮次位置留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(age=tuple(sorted(OOD_AGES)))]


def bucket_of(case: TagSet, seed: int = SPLIT_SEED) -> int:
    """输入：tag 组合、种子。输出：0~99 的桶号。

    预期行为：同一组合必得同一桶，不同组合近似均匀——切分单位是**组合**，不是样本。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[TagSet], split: str) -> list[TagSet]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def cases_for(config: CompactConfig) -> list[TagSet]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, str], ...], TagSet] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(features.build_cases(config.cover_limit, dims=id_dims()), config.split)
    return cases[: config.case_limit] if config.case_limit else cases


class CompactWorkflow:
    """会话上下文压缩：正文、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 那一块正文由后端写：本 workflow 不是纯构造式
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

    def config_for(self, split: str) -> CompactConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return CompactConfig(split=split)

    def build_cases(self, config: CompactConfig) -> list[TagSet]:
        """输入：采样参数。输出：tag 组合列表。"""
        return cases_for(config)

    def case_label(self, case: TagSet) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: TagSet) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "kind": case.kind,
            "fact": case.fact,
            "age": case.age,
            "pins": case.pins,
            "budget": case.budget,
        }

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: CompactConfig, case: TagSet, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。"""
        return SplitSpec(
            name=config.split,
            params={
                "tags": case,
                "phrasing": rng.choice(config.phrasing_pool),
                "seq": seq,
            },
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> CompactSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return CompactSpec(
            tags=split.params["tags"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: CompactSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一块上下文正文，再与代码渲染的处置表 / 位置行拼起来；
        后端没注入就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，
        绝不能悄悄降级成模板（那会让「正文由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("compact workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.tags, loc, spec.render_task_id))
        render.check_draft(draft.data, loc)
        return render.to_state(draft.data, spec.tags, loc)

    def ask(self, spec: CompactSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.tags, locales.load(locale), spec.phrasing)

    def reparse(self, state: object, *, case: object = None) -> render.ParsedFrame:
        """输入：渲染后的 state。输出：纯代码能反解的部分（正文 / 位置行 / 处置表）。"""
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

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分审计计数, 缺失项列表)。"""
        return audit_mod.audit_run(run_dir, splits, locale)

    def available_locales(self) -> list[str]:
        """输入：无。输出：该 workflow 支持的语言标签列表。"""
        return locales.available()


@register(WORKFLOW_NAME)
def _factory() -> CompactWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return CompactWorkflow()
