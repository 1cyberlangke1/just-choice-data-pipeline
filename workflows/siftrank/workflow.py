"""多批次排名聚合（siftrank）workflow：**构造式**（批次与四个答案都由代码算出来，零 token）。

覆盖的决策形状（社区清单）：`siftrank`（★210, Go）把同一批条目随机采样成小批次、反复让模型
排序，再把各批次的排名**聚合**成一条稳定的相关性顺序——它要解决的正是一次排序的不确定、
上下文装不下、模型漏项这三个毛病。这里把「聚合」做成**可判定的算术**：每个批次给条目按名次
计分，总分决定聚合顺序，总分相同就是并列；相邻名次之间落差最大的那一处是肘部，肘部以上算
「相关」。四个题族全部由这四个批次算出来。

与 `rerank` / `citation` 的区别：那两家是「一次排序 / 一次判读」，读的是**单张**列表；
这一族读的是**多张互相矛盾的**列表，考的是「几份排名怎么合并」。`cachereuse` 也是多条轨迹，
但它比的是**同一序列的逐步对齐**（有序、成对），这里比的是**跨批次的名次累加**（无序、全局）。

切分按 PLAN 3.3：
- train / val / test_iid：**素材留出**——按素材（批次形状 + 条目映射）的哈希分桶切片，
  同一份请求整组不跨切分；
- test_ood：**值留出**——排序提示 `audit` 在训练侧一个都不出现（它只改抬头下面那一行文字，
  不动任何规则分支，所以留出它是安全的）。

导入本模块即自注册（`jc.workflow.load("siftrank")` 可用）。
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
from .spec import RankCase, RankConfig, RankSpec

WORKFLOW_NAME = "siftrank"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（素材留出：桶按**素材**算，同一份请求永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 训练侧取值：排序提示表只留标了「见过」的那几档。
ID_CONFIG: Final[RankConfig] = RankConfig(split="train", flavors=spec_mod.ID_FLAVORS)
#: 留出档：提示表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的 case。
OOD_CONFIG: Final[RankConfig] = RankConfig(split="test_ood", ood_only=True)

DIMENSION_TITLES: Final[dict[str, str]] = {
    "shape": "批次形状",
    "perm": "条目映射",
    "flavor": "排序提示",
}


def bucket_of(material: str, seed: int = SPLIT_SEED) -> int:
    """输入：素材指纹（`spec.RankCase.material`）、种子。输出：0~99 的桶号。

    预期行为：同一素材必得同一桶，不同素材近似均匀；不用内置 `hash()`
    （它对字符串每进程随机化），否则切分会随进程漂移。
    """
    return int(hashlib.sha256(f"{seed}|{material}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[RankCase], split: str) -> list[RankCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：**先按素材分组，再按位置切**——同一份请求整组落在同一档。
    切片用**位置**而不是桶号过滤：只按桶号过滤时哈希方差会让 val 只剩个位数。
    """
    groups: dict[str, list[RankCase]] = {}
    for case in cases:
        groups.setdefault(case.material(), []).append(case)
    ordered = sorted(groups, key=lambda material: (bucket_of(material), material))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return [
        case
        for material in ordered[low * total // 100 : high * total // 100]
        for case in groups[material]
    ]


def cases_for(config: RankConfig) -> list[RankCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合（留出档全要，不切片）；
    其余切分先按素材哈希分桶再切片。判据看 `split` 而不是 `ood_only`——调用方可以只给一个
    `split="test_ood"` 的配置（管线测试就这么干），那时按桶号切会直接 `KeyError`。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        return cases
    return _slice(cases, config.split)


@register(WORKFLOW_NAME)
class SiftRankWorkflow(StandardMethods):
    """多批次排名聚合：四个批次的排名 → 头名 / 一对条目的比较 / 相关条目数 / 批次是否一致。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 构造式：批次与答案都由代码算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")
    split_names_tuple = tuple(SPLIT_BUCKETS) + ("test_ood",)
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_locales = locales

    def config_for(self, split: str) -> RankConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用全提示表 + `ood_only`；其余切分用训练侧提示表，**并且把 `split`
        换成调用方要的那一档**——`ID_CONFIG` 是个单例，直接返回它会让 train/val/test_iid 三次
        都按同一档切片（切出来三份一模一样的数据）。
        """
        if split == "test_ood":
            return OOD_CONFIG
        if split in SPLIT_BUCKETS:
            return replace(ID_CONFIG, split=split)
        raise KeyError(f"未知切分：{split!r}（可用：{self.split_names()}）")

    def plan_sample(
        self, rng: random.Random, config: RankConfig, case: RankCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasings)

    def sample(self, rng: random.Random, split: SplitSpec) -> RankSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（请求号与四个答案都在这里定死）。"""
        return spec_mod.sample_spec(
            rng, split.params["case"], split.name, split.params["seq"], split.params["phrasing"]
        )

    def render(self, spec: RankSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件必得同一段
        文本，所以「两次生成逐字节一致」这条能直接查。
        """
        return render.build(spec, locales.load(locale))

    def ask(self, spec: RankSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: object, *, case: object = None) -> parse.ParsedRank:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的结构反解
        （各语言的标签互不重叠，由 `tests/workflows/siftrank/test_locales.py` 锁住）。
        「条目名 → 条目键」留给判别器，它手上有语言文件。
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

        预期行为：只看产物本身——题族是否齐全、choice 的答案编号是否越界、以及**答案能不能由
        state 重算出来**（复用 `judges.check_sample_any_locale`，与判别器同一把尺，这样
        「老产物」也能事后查）。
        """
        locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段（判定按「任一语言」走）
        # 用**合并表**反解：产物按 `locales[seq % len]` 轮转渲染，调用方给的语言未必是产物那一种。
        tables = parse.merged_tables()
        words = locales.words()
        known = set(questions.FAMILIES)
        report: dict[str, JsonValue] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, JsonValue] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "cut": Counter(),
                "stable": Counter(),
                "relation": Counter(),
                "top": Counter(),
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
                    parsed = parse.parse_state(row["state"], tables)
                    pair = judges_mod.pair_of(row, words)
                    facts = judges_mod.facts_of(parsed, words, pair)
                except (ValueError, KeyError) as exc:
                    counts["label_mismatch"] += 1
                    missing.append(f"{split}: 有一条请求反解不出来（{exc}）")
                    continue
                counts["cut"][facts["cut"]] += 1
                counts["stable"][facts["stable"]] += 1
                counts["relation"][facts["relation"]] += 1
                counts["top"][facts["top"]] += 1
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`cut_size_2` -> `cut_size`）
                    counts["families"][family] += 1
                    counts["types"][question["type"]] += 1
                    if family not in known:
                        counts["unknown_families"][family] += 1
                    if question["type"] == "choice":
                        options = list(question["criteria"])
                        index = row["answers"][qid]["choice"]
                        if not isinstance(index, int) or not 0 <= index < len(options):
                            counts["index_bad"] += 1
                problems = judges_mod.check_sample_any_locale(row, facts, pair)
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
        """空操作：多批次排名聚合是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("多批次排名聚合 workflow 是纯构造式的，不该被绑定生成后端")
