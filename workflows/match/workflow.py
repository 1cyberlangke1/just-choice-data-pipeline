"""记录匹配（match）workflow：**构造式**（两条记录、八个答案都由字段关系表算出来，零 token）。

覆盖的决策形状（官方 use-case map）：**实体对齐**——官方那篇 `entity_alignment` 拿两个
目录的 450 组候选对，每对问「1 个 Score + 若干伴随 Noul」。这里把「哪几栏对得上」做成
**可判定的文本位**：两条记录的每一栏要么逐字相同（`same`）、要么只是写法不同（`variant`，
大小写 / 空格 / 单位换算 / 同义词）、要么真的不同（`diff`），于是「像不像同一个商品」
退化成按题面给的对照表逐栏归一化再比。

与 `rerank` 的区别：rerank 是**一条列表内部的次序**，这里是**两条记录之间的字段对齐**。

切分按 PLAN 3.3：
- train / val / test_iid：**素材留出**——按素材（五维指纹）的哈希分桶切片，同一条请求整组不跨切分；
- test_ood：**值留出**——商品表最后一个（`juice`）与写法对最后一对在训练侧一个都不出现，
  而且 `diff` 那一栏的取值池也锁在同分布侧（`spec.diff_pool`），所以留出商品的名字在训练侧
  **文本**里同样一条都没有。字段关系表刻意不动：八个题族的答案分布是**手推**出来的
  （见 `spec.PATTERNS` 的注释），改一张表就得重推八个分布，不值。

导入本模块即自注册（`jc.workflow.load("match")` 可用）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import hashlib
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Final

from jc import jsonio
from jc.backend.types import GenerationBackend
from jc.judge import Judge
from jc.workflow import LabeledQuestion, SplitSpec, StandardMethods, family_of, plan_with_phrasing, register

from . import judges as judges_mod
from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import MatchCase, MatchConfig, MatchSpec

WORKFLOW_NAME = "match"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（素材留出：桶按**素材**算，同一条请求永远同切分）。
SPLIT_SEED: Final[int] = 20260926
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}


#: 训练侧取值：只留 spec 里标了「见过」的那几档（与 `diff` 取值池共用同一份过滤结果）。
ID_CONFIG: Final[MatchConfig] = MatchConfig(
    split="train",
    products=spec_mod.ID_PRODUCTS,
    spellings=spec_mod.ID_SPELLINGS,
)
#: 留出档：取值表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的 case。
OOD_CONFIG: Final[MatchConfig] = MatchConfig(split="test_ood", ood_only=True)

DIMENSION_TITLES: Final[dict[str, str]] = {
    "product": "商品",
    "pattern": "字段关系",
    "spelling": "写法对",
    "layout": "显示顺序",
    "extra": "额外栏",
}


def bucket_of(material: str, seed: int = SPLIT_SEED) -> int:
    """输入：素材指纹（`spec.material_of`）、种子。输出：0~99 的桶号。

    预期行为：同一素材必得同一桶，不同素材近似均匀；不用内置 `hash()`
    （它对字符串每进程随机化），否则切分会随进程漂移。
    """
    return int(hashlib.sha256(f"{seed}|{material}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[MatchCase], split: str) -> list[MatchCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：**先按素材分组，再按位置切**——同一条请求整组落在同一档。
    切片用**位置**而不是桶号过滤：只按桶号过滤时哈希方差会让 val 只剩个位数。
    """
    groups: dict[str, list[MatchCase]] = {}
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


def cases_for(config: MatchConfig) -> list[MatchCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合（留出档全要，
    不切片）；其余切分先按素材哈希分桶再切片。判据看 `split` 而不是 `ood_only`——
    调用方可以只给一个 `split="test_ood"` 的配置（管线测试就这么干），
    那时按桶号切会直接 `KeyError`。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        return cases
    return _slice(cases, config.split)


@register(WORKFLOW_NAME)
class MatchWorkflow(StandardMethods):
    """记录匹配：两条目录记录 → 处置结论 / 逐栏一致 / 档位。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 构造式：两条记录与答案都由字段关系表算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")
    split_names_tuple = tuple(SPLIT_BUCKETS) + ("test_ood",)
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_locales = locales

    def config_for(self, split: str) -> MatchConfig:
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

    def plan_sample(
        self, rng: random.Random, config: MatchConfig, case: MatchCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasings)

    def sample(self, rng: random.Random, split: SplitSpec) -> MatchSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（请求号与八个答案都在这里定死）。"""
        return spec_mod.sample_spec(
            rng, split.params["case"], split.name, split.params["seq"], split.params["phrasing"]
        )

    def render(self, spec: MatchSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件
        必得同一段文本，所以「两次生成逐字节一致」这条能直接查。
        """
        return render.build(spec, locales.load(locale))

    def ask(self, spec: MatchSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: object, *, case: object = None) -> parse.ParsedMatch:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的
        结构反解（各语言的标签互不重叠，由 `tests/workflows/match/test_locales.py` 锁住）。
        「原文 → 归一化后的值」留给判别器，它手上有语言文件。
        """
        return parse.parse_state(state, parse.merged_tables())

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

        预期行为：只看产物本身——题族是否齐全、choice 的答案编号是否越界、
        以及**答案能不能由 state 重算出来**（复用 `judges.check_sample`，与判别器同一把尺，
        这样「老产物」也能事后查）。
        """
        locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段（判定按「任一语言」走）
        # 用**合并表**反解：产物按 `locales[seq % len]` 轮转渲染，调用方给的语言未必是
        # 产物那一种；拿单一语言的标签去认，另一种语言的请求会整条「反解不出来」。
        labels = parse.merged_tables()
        known = set(questions.FAMILIES)
        report: dict[str, JsonValue] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, JsonValue] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "verdict": Counter(),
                "band": Counter(),
                "variant_count": Counter(),
                "index_bad": 0,
                "label_mismatch": 0,
                "unknown_families": Counter(),
            }
            if not path.exists():
                missing.append(f"{split}: 产物不存在（{path.name}）")
                report[split] = counts
                continue
            for _, row in jsonio.iter_jsonl(path):
                counts["samples"] += 1
                try:
                    parsed = parse.parse_state(row["state"], labels)
                    # 归一化也用合并同义词表（`locales.merged`），理由同上一行。
                    facts = judges_mod.recompute(parsed, locales.merged())
                except (ValueError, KeyError) as exc:
                    counts["label_mismatch"] += 1
                    missing.append(f"{split}: 有一条请求反解不出来（{exc}）")
                    continue
                counts["verdict"][facts.verdict] += 1
                counts["band"][facts.band] += 1
                counts["variant_count"][facts.variant_count] += 1
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`same_name_2` -> `same_name`）
                    counts["families"][family] += 1
                    counts["types"][question["type"]] += 1
                    if family not in known:
                        counts["unknown_families"][family] += 1
                    if question["type"] == "choice":
                        options = list(question["criteria"])
                        index = row["answers"][qid]["choice"]
                        if not isinstance(index, int) or not 0 <= index < len(options):
                            counts["index_bad"] += 1
                problems = judges_mod.check_sample_any_locale(row, facts)
                counts["label_mismatch"] += len(problems)
                if problems:
                    missing.append(f"{split}: {problems[0]}")
            for family in questions.FAMILIES:
                if counts["families"][family] == 0:
                    missing.append(f"{split}: 题族 {family} 一条都没出")
            if counts["index_bad"]:
                missing.append(f"{split}: {counts['index_bad']} 处编号异常")
            if counts["label_mismatch"]:
                missing.append(f"{split}: {counts['label_mismatch']} 条答案与 state 重算不一致")
            if counts["unknown_families"]:
                missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
            report[split] = counts
        return report, missing

    def bind_backend(self, backend: GenerationBackend) -> None:
        """空操作：记录匹配是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("记录匹配 workflow 是纯构造式的，不该被绑定生成后端")
