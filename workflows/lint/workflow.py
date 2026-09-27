"""行级规则检查（lint）workflow：**构造式**（片段与全部答案都由规则算出来，`needs` 为空、零 token）。

覆盖的决策形状（社区用例地图缺口：语义 grep / 语义 lint 家族，20+ 个第三方项目）：
- **逐行判定**：一段片段 + 一条自然语言规则 → 每一行命中不命中（语义 grep 的本体）；
- **计数与门控**：命中几行、按处置办法走哪一档（放行 / 人工复核 / 拦下）；
- **定位**：恰好只有一行命中时，是哪一行（高基数 Choice）。

片段固定 8 行，命中行数取 0~8 均匀，所以四条题族的答案分布都由构造保证均衡（`line` 的
True 占比 0.5、`count` 九档各 1/9、`gate` 三档各 1/3、`spot` 在 8 行里均匀）。

切分按 PLAN 3.3：
- train / val / test_iid：**素材留出**——按素材（四维指纹）的哈希分桶切片，同一份片段不跨切分；
- test_ood：**值留出**——规则 `long`（按长度判的那条）、体裁 `doc`、第二套句面在训练侧
  一个都不出现。

导入本模块即自注册（`jc.workflow.load("lint")` 可用）。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

from jc.judge import Judge
from jc.workflow import LabeledQuestion, SplitSpec, register

from . import judges as judges_mod
from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import LintCase, LintConfig, LintSpec

WORKFLOW_NAME = "lint"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（素材留出：桶按**素材**算，同一份片段永远同切分）。
SPLIT_SEED: Final[int] = 20260929
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 训练侧取值：只留 spec 里标了「见过」的那几档。排序是为了顺序稳定
#: （frozenset 的迭代顺序随进程变化，不能直接拿来枚举）。
ID_CONFIG: Final[LintConfig] = LintConfig(
    split="train",
    rules=tuple(sorted(spec_mod.SEEN_RULES)),
    genres=tuple(sorted(spec_mod.SEEN_GENRES)),
    hits=spec_mod.HITS,
    phrasings=tuple(sorted(spec_mod.SEEN_PHRASINGS)),
)
#: 留出档：取值表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的组合。
OOD_CONFIG: Final[LintConfig] = LintConfig(split="test_ood", ood_only=True)

DIMENSION_TITLES: Final[dict[str, str]] = {
    "rule": "检查规则",
    "genre": "片段体裁",
    "hits": "命中行数",
    "phrasing": "问法",
}


def bucket_of(material: str, seed: int = SPLIT_SEED) -> int:
    """输入：素材指纹（`spec.material_of`）、种子。输出：0~99 的桶号。

    预期行为：同一素材必得同一桶，不同素材近似均匀；不用内置 `hash()`
    （它对字符串每进程随机化），否则切分会随进程漂移。
    """
    return int(hashlib.sha256(f"{seed}|{material}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[LintCase], split: str) -> list[LintCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：**先按素材分组，再按位置切**——同一份片段整组落在同一档。
    切片用**位置**而不是桶号过滤：只按桶号过滤时哈希方差会让 val 只剩个位数。
    """
    groups: dict[str, list[LintCase]] = {}
    for case in cases:
        groups.setdefault(spec_mod.material_of(case), []).append(case)
    ordered = sorted(groups, key=lambda material: (bucket_of(material), material))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return [
        case
        for material in ordered[low * total // 100 : high * total // 100]
        for case in groups[material]
    ]


def cases_for(config: LintConfig) -> list[LintCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合；
    其余切分按素材的哈希桶分组后切片，同一份片段不跨切分。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        return cases
    return _slice(cases, config.split)


@register(WORKFLOW_NAME)
class LintWorkflow:
    """行级规则检查：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    #: 纯构造式：`needs` 为空，管线不会初始化任何后端。
    needs: frozenset[str] = frozenset()
    judges: tuple[str, ...] = ("roundtrip", "label")

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：支持的切分名。"""
        return ("train", "val", "test_iid", "test_ood")

    def config_for(self, split: str) -> LintConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`train` / `val` / `test_iid` 三档共用同一份取值表、只换 `split` 字段——
        所以这里**必须**返回新对象（`replace`），不能返回同一个单例：三档共用一个单例会让
        `cases_for` 三次都按同一档切片（切出来三份一模一样的数据）。
        """
        if split == "test_ood":
            return OOD_CONFIG
        if split in SPLIT_BUCKETS:
            return replace(ID_CONFIG, split=split)
        raise KeyError(f"未知切分：{split!r}（可用：{self.split_names()}）")

    def build_cases(self, config: LintConfig) -> list[LintCase]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。"""
        return cases_for(config)

    def case_label(self, case: LintCase) -> str:
        """输入：一个 case。输出：一行可读描述（报告与文件名用）。"""
        return case.describe()

    def stats_dimensions(self, case: LintCase) -> dict[str, Any]:
        """输入：一个 case。输出：{维度名: 取值}（四个维度全摊平）。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: LintConfig, case: LintCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：句面（问法）**是 case 的一个维度**（它要被留出），所以这里不掷骰子——
        同一 case 的每条样本都得到同一份计划（`per_case > 1` 时会出重复样本，
        多样性交给 case 矩阵里的问法维度）。
        """
        del rng
        return SplitSpec(name=config.split, params={"case": case, "seq": seq})

    def sample(self, rng: random.Random, split: SplitSpec) -> LintSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        return spec_mod.sample_spec(split.params["case"], split.name, split.params["seq"])

    def render(self, spec: LintSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（多行文本）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件
        必得同一段文本，所以「两次生成逐字节一致」这条能直接查。
        """
        del rng
        return render.render_state(spec, locales.load(locale))

    def ask(self, spec: LintSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        loc = locales.load(locale)
        plan = spec.plan
        return questions.build(
            plan.rule, render.render_lines(plan, loc), plan.matched(), loc, plan.phrasing
        )

    def reparse(self, state: Any, *, case: Any = None) -> parse.LintReading:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用 `locales.merged()` 的合并表
        （各语言的规则短语取并集，互不重叠由测试锁住）。
        """
        del case
        if not isinstance(state, str):
            raise TypeError(f"state 必须是字符串，收到 {type(state).__name__}")
        return parse.read_state(state, locales.merged())

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。预期行为：未知名字直接报错。"""
        out: list[Judge] = []
        for name in names:
            if name == "roundtrip":
                out.append(judges_mod.RoundtripJudge(locale=locale))
            elif name == "label":
                out.append(judges_mod.LabelJudge(locale=locale))
            else:
                raise KeyError(f"未知判别器：{name!r}（可用：roundtrip / label）")
        return tuple(out)

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, Any], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。"""
        from . import audit as audit_mod  # 延迟导入：避免与 audit 互相 import 成环

        return audit_mod.audit_run(run_dir, splits, locale)

    def available_locales(self) -> list[str]:
        """输入：无。输出：支持的语言列表。"""
        return locales.available()

    def bind_backend(self, backend: Any) -> None:
        """空操作：行级规则检查是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("lint workflow 是纯构造式的，不该被绑定生成后端")
