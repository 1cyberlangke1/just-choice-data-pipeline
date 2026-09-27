"""结构恢复（autoformat）workflow：**构造式**（文档与全部判断都由规则算出来，`needs` 为空、零 token）。

它在 PLAN 3.2 的分类里属于「抽取式」的近亲——素材（一份丢掉标记的纯文本备忘）由构造器
造好，答案是从素材**按读法推**出来的，所以判别器能把文本反解回来独立重算。

覆盖的决策形状（官方 use-case map 的 structure recovery）：
- **结构恢复 / 结构化抽取**：`join`（相邻两行是不是同一块的折行）、`block_type`（逐块分类）；
- **分类**：`callout_kind`（提示框是备注 / 小技巧 / 警告）；
- **检测**：`list_ordered`（那个列表是不是步骤序列）。

官方 cookbook 用**两遍请求**：第一遍逐行问「是不是接着上一行」把硬换行拼回块，第二遍逐块
问「是哪一类内容」。本 workflow 把两遍都做成题：`join` 是第一遍，`block_type` 与伴随题是
第二遍。真实部署里第二遍的输入（块的行区间）是第一遍算出来的，本数据集把两者放进同一条
样本里并行问——依赖关系在部署时才出现。

切分按 PLAN 3.3：
- train / val / test_iid：**素材留出**——按素材（五维指纹）的哈希分桶切片，同一份文档不跨切分；
- test_ood：**值留出**——`incident` / `migration` 两个主题、行宽 `56`、提示框种类 `tip`、
  命令族 `container` 在训练侧一个都不出现。

**关于 `block_type` 的分布**：一份文档里段落天然比引用、提示框、代码块多，所以六类选项
不可能等概率。题族照样**打乱选项**——类型由选项描述承载、不靠位置承载，打乱之后
「答案位置」不再泄露标签（范式 §5.8）。均衡报告会如实标出这个偏斜，它是数据本身的形状。

**关于 `join` 的分布**：折行对（同一块）与块边界（不同块）的条数由文档骨架决定——块越多
边界越密，所以 `mixed` 形状的真值只占 0.35、`callout` 形状占 0.59，整体约 0.41。这是
「相邻行对」这个问法在文档上的必然形状（真实硬换行文本同样如此），所以**不为了凑 0.5 去丢
样本**；均衡报告照常如实标它。

导入本模块即自注册（`jc.workflow.load("autoformat")` 可用）。
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
from .spec import DocCase, DocConfig, DocSpec

WORKFLOW_NAME = "autoformat"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（素材留出：桶按**素材**算，同一份文档永远同切分）。
SPLIT_SEED: Final[int] = 20260927
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 训练侧取值：只留 spec 里标了「见过」的那几档。排序是为了顺序稳定
#: （frozenset 的迭代顺序随进程变化，不能直接拿来枚举）。
ID_CONFIG: Final[DocConfig] = DocConfig(
    split="train",
    topics=tuple(sorted(spec_mod.SEEN_TOPICS)),
    wraps=tuple(sorted(spec_mod.SEEN_WRAPS)),
    variants=tuple(sorted(spec_mod.SEEN_VARIANTS)),
    code_families=tuple(sorted(spec_mod.SEEN_CODE_FAMILIES)),
    phrasings=(0, 1),
)
#: 留出档：取值表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的组合。
OOD_CONFIG: Final[DocConfig] = DocConfig(split="test_ood", ood_only=True)

DIMENSION_TITLES: Final[dict[str, str]] = {
    "topic": "主题",
    "wrap": "行宽（字符）",
    "shape": "文档形状",
    "variant": "变体",
    "code_family": "命令族",
}


def bucket_of(material: str, seed: int = SPLIT_SEED) -> int:
    """输入：素材指纹（`spec.material_of`）、种子。输出：0~99 的桶号。

    预期行为：同一素材必得同一桶，不同素材近似均匀；不用内置 `hash()`
    （它对字符串每进程随机化），否则切分会随进程漂移。
    """
    return int(hashlib.sha256(f"{seed}|{material}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[DocCase], split: str) -> list[DocCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：**先按素材分组，再按位置切**——同一份文档整组落在同一档。
    切片用**位置**而不是桶号过滤：只按桶号过滤时哈希方差会让 val 只剩个位数。
    """
    groups: dict[str, list[DocCase]] = {}
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


def cases_for(config: DocConfig) -> list[DocCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合；
    其余切分按素材的哈希桶分组后切片，同一份文档不跨切分。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        return cases
    return _slice(cases, config.split)


@register(WORKFLOW_NAME)
class AutoformatWorkflow:
    """结构恢复：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs: frozenset[str] = frozenset()  # 构造式：文档与答案都由规则算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：支持的切分名。"""
        return tuple(SPLIT_BUCKETS) + ("test_ood",)

    def config_for(self, split: str) -> DocConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用全取值表 + `ood_only`；其余切分用训练侧取值表，
        **并且把 `split` 换成调用方要的那一档**——`ID_CONFIG` 是个单例，直接返回它会让
        train/val/test_iid 三次都按同一档切片（切出来三份一模一样的数据）。
        """
        if split == "test_ood":
            return OOD_CONFIG
        if split in SPLIT_BUCKETS:
            return replace(ID_CONFIG, split=split)
        raise KeyError(f"未知切分：{split!r}（可用：{self.split_names()}）")

    def build_cases(self, config: DocConfig) -> list[DocCase]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。"""
        return cases_for(config)

    def case_label(self, case: DocCase) -> str:
        """输入：一个 case。输出：一行可读描述（报告与文件名用）。"""
        return case.describe()

    def stats_dimensions(self, case: DocCase) -> dict[str, Any]:
        """输入：一个 case。输出：{维度名: 取值}（五个维度全摊平）。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: DocConfig, case: DocCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法在这里抽（同一 case 的多条样本会用不同问法）；
        文档内容由 case 指纹决定，`sample` 阶段不再做任何随机决策。
        """
        return SplitSpec(
            name=config.split,
            params={"case": case, "phrasing": rng.choice(config.phrasings), "seq": seq},
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> DocSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        return spec_mod.sample_spec(
            split.params["case"], split.name, split.params["seq"], split.params["phrasing"]
        )

    def render(self, spec: DocSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件
        必得同一段文本，所以「两次生成逐字节一致」这条能直接查。
        """
        del rng
        return render.render_state(spec, locales.load(locale))

    def ask(self, spec: DocSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: Any, *, case: Any = None) -> parse.ParsedDoc:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用 `locales.merged()` 的合并表
        （各语言的关键词 / 命令 / 时序词取并集，互不重叠由测试锁住）。
        """
        del case
        if not isinstance(state, str):
            raise TypeError(f"state 必须是字符串，收到 {type(state).__name__}")
        return parse.parse_state(state, locales.merged())

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
        """空操作：结构恢复是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("结构恢复 workflow 是纯构造式的，不该被绑定生成后端")
