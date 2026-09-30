"""界面元素定位 workflow：**构造式**（一屏元素清单与全部判断都由规则算出来，`needs` 为空、零 token）。

它在 PLAN 3.2 的分类里属于「规则引擎式」——素材（屏号 + 动作标签 + 六行元素的四格）由
构造器造好，答案是从素材**按完全一致的匹配、可见且可点、近似文本三条规则推**出来的，
所以判别器能把文本反解回来独立重算。

覆盖的决策形状（社区用例地图：浏览器与计算机操作，83 个第三方项目，含 `Jev Ultrafast`
★9291「在一次请求中选择操作和 DOM 元素」/ `agent-desktop` ★1441 / `surf-cli` ★624 /
`Jev-cu` ★584 / `typesafe-computer-use` ★537 / `mobile-jev` ★240）：
- **有没有能落点的元素**：`found`（noul，总是出）；
- **有没有名字只是近似的元素**：`shadow`（noul，总是出）；
- **落在第几个元素上**：`target`（choice 6，只在有时）；
- **落在哪一类元素上**：`role`（choice 3，只在有时）；
- **有几个同名元素被挡住了**：`decoy`（score 3 档，只在有时）。

与 `browser` 的区别：`browser` 问「下一步做什么动作」（要读一串历史、判动作序列与完成度），
这家问「**这个动作落在哪个元素上**」——纯查表，没有历史、没有动作序列、没有完成度。

切分按 PLAN 3.3：
- train / val / test_iid：**素材留出**——按素材（六维指纹）的哈希分桶切片，
  同一屏整组不跨切分；
- test_ood：**值留出**——排布 `l5` 与第二套句面在训练侧一个都不出现。

导入本模块即自注册（`jc.workflow.load("elementpick")` 可用）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import hashlib
import random
from dataclasses import replace
from pathlib import Path
from typing import Final

from jc.backend.types import GenerationBackend
from jc.judge import Judge
from jc.workflow import LabeledQuestion, SplitSpec, register

from . import audit as audit_mod
from . import judges as judges_mod
from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import ElementCase, ElementConfig, ElementSpec

WORKFLOW_NAME = "elementpick"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（素材留出：桶按**素材**算，同一屏永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 训练侧取值：只留 spec 里标了「见过」的那几档。
#: 排序是为了顺序稳定（frozenset 的迭代顺序随进程变化，不能直接拿来枚举）。
ID_CONFIG: Final[ElementConfig] = ElementConfig(
    split="train",
    layouts=tuple(sorted(spec_mod.SEEN_LAYOUTS)),
    phrasings=tuple(sorted(spec_mod.SEEN_PHRASINGS)),
)
#: 留出档：取值表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的 case。
OOD_CONFIG: Final[ElementConfig] = ElementConfig(split="test_ood", ood_only=True)

DIMENSION_TITLES: Final[dict[str, str]] = {
    "label": "动作标签",
    "layout": "排布",
    "shape": "命中形状",
    "extras": "多出来的同名行",
    "shadow": "近似元素",
    "phrasing": "句面",
}


def bucket_of(material: str, seed: int = SPLIT_SEED) -> int:
    """输入：素材指纹（`spec.material_of`）、种子。输出：0~99 的桶号。

    预期行为：同一素材必得同一桶，不同素材近似均匀；不用内置 `hash()`
    （它对字符串每进程随机化），否则切分会随进程漂移。
    """
    return int(hashlib.sha256(f"{seed}|{material}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[ElementCase], split: str) -> list[ElementCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：**先按素材分组，再按位置切**——同一屏整组落在同一档。
    切片用**位置**而不是桶号过滤：只按桶号过滤时哈希方差会让 val 只剩个位数。
    """
    groups: dict[str, list[ElementCase]] = {}
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


def cases_for(config: ElementConfig) -> list[ElementCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合；
    其余切分按素材的哈希桶分组后切片，同一屏不跨切分。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        return cases
    return _slice(cases, config.split)


@register(WORKFLOW_NAME)
class ElementPickWorkflow:
    """界面元素定位：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 构造式：完全不碰后端
    judges: tuple[str, ...] = ("roundtrip", "label")

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：支持的切分名。"""
        return ("train", "val", "test_iid", "test_ood")

    def config_for(self, split: str) -> ElementConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用全开的留出配置；其余三个切分共用训练侧取值表，只换 `split`
        （切分靠 `_slice` 按素材哈希分，不靠取值表区分）。
        """
        if split == "test_ood":
            return OOD_CONFIG
        if split in SPLIT_BUCKETS:
            return replace(ID_CONFIG, split=split)
        raise KeyError(f"未知切分：{split!r}（可用：{self.split_names()}）")

    def build_cases(self, config: ElementConfig) -> list[ElementCase]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。"""
        return cases_for(config)

    def case_label(self, case: ElementCase) -> str:
        """输入：一个 case。输出：一行可读描述（报告与文件名用）。"""
        return case.describe()

    def stats_dimensions(self, case: ElementCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：{维度名: 取值}（六个维度全摊平）。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: ElementConfig, case: ElementCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法在这里抽（同一 case 的多条样本会用不同问法）；case 与序号一起放进
        params，`sample` 阶段不再做任何随机决策。
        """
        return SplitSpec(
            name=config.split,
            params={"case": case, "phrasing": rng.choice(config.phrasings), "seq": seq},
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> ElementSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（屏号、六行元素与四条判据都在这里定死）。"""
        return spec_mod.sample_spec(
            split.params["case"],
            split.name,
            split.params["seq"],
            split.params["phrasing"],
            rng,
        )

    def render(self, spec: ElementSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件必得同一段
        文本，所以「两次生成逐字节一致」这条能直接查。
        """
        del rng
        return render.build(spec, locales.load(locale))

    def ask(self, spec: ElementSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.rows, spec.case.label, locales.load(locale), spec.phrasing)

    def reparse(self, state: object, *, case: object = None) -> parse.ParsedScreen:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的
        结构反解（各语言的字段名与四张词表互不重叠，由 `tests/workflows/elementpick/
        test_locales.py` 锁住）。
        """
        del case
        return parse.parse_screen(state, parse.merged())

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

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：空表（纯构造族，没有「模型写的正文」）。

        预期行为：`needs` 为空，state 全由代码按语言模板拼，天然一致。
        """
        del sample
        return {}

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

        预期行为：只看产物本身——题族是否齐全、结构化答案的下标是否越界、以及
        **答案能不能由清单重算出来**（把判别器的判据搬到产物级，老产物也能事后查）。
        """
        return audit_mod.audit_run(run_dir, splits, locale)

    def available_locales(self) -> list[str]:
        """输入：无。输出：支持的语言列表。"""
        return locales.available()

    def bind_backend(self, backend: GenerationBackend) -> None:
        """空操作：界面元素定位是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("界面元素定位 workflow 是纯构造式的，不该被绑定生成后端")
