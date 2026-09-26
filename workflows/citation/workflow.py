"""引文核查 workflow：**构造式**（文档与全部判断都由表格算出来，`needs` 为空、零 token）。

它在 PLAN 3.2 的分类里属于「校验式」：素材（一份来源文档 + 一条引文）由构造器造好，
答案是按规则从素材推出来的，所以判别器能把文本反解回来独立重算。

覆盖的决策形状（官方 use-case map）：
- **Verification**：`verdict`（四档结论）、`quote_source`（引文实际出处）、
  `relation`（声称小节的态度）、`found` / `located`（整句匹配与定位两步）；
- **Scoring**：`evidence_strength`（证据强度 0~3 级）；
- **Routing**：`action`（自动采信 / 交人复核——官方 cookbook 里那道阈值闸）。

切分按 PLAN 3.3：
- train / val / test_iid：**组合留出**——按 8 维组合的哈希分桶，同组合不跨切分。
  这里 8 维**全部**决定页面文本（不像 search 的 `probe` 只改题面），所以按组合分桶
  就是按素材分桶；这条在 `tmp/split_overlap.py` 的产物检查里复验过（跨切分重复 0）；
- test_ood：**值留出**——`rollout` 主题、`mislocated` 一档结论、`reworded` 一种引文写法、
  `off_topic` 一种干扰、第一种主张措辞，在训练侧一个都不出现。

导入本模块即自注册（`jc.workflow.load("citation")` 可用）。
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

from jc import jsonio
from jc.judge import Judge
from jc.workflow import LabeledQuestion, SplitSpec, family_of, register

from . import judges as judges_mod
from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import CitationCase, CitationConfig, CitationSpec

WORKFLOW_NAME = "citation"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（组合留出：桶按 case 的 8 维取值算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260926
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 训练侧取值：各维度只留 spec 里标了「见过」的那几档。
#: 排序是为了顺序稳定（frozenset 的迭代顺序随进程变化，不能直接拿来枚举）。
ID_CONFIG: Final[CitationConfig] = CitationConfig(
    split="train",
    topics=tuple(sorted(spec_mod.SEEN_TOPICS)),
    statuses=tuple(sorted(spec_mod.SEEN_STATUSES)),
    quote_forms=tuple(sorted(spec_mod.SEEN_QUOTE_FORMS)),
    section_counts=tuple(sorted(spec_mod.SEEN_SECTION_COUNTS)),
    noises=tuple(sorted(spec_mod.SEEN_NOISES)),
    claim_styles=tuple(sorted(spec_mod.SEEN_CLAIM_STYLES)),
    wheres=tuple(sorted(spec_mod.SEEN_WHERES)),
    doc_styles=tuple(sorted(spec_mod.SEEN_DOC_STYLES)),
)
#: 留出档：取值表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的 case。
OOD_CONFIG: Final[CitationConfig] = CitationConfig(split="test_ood", ood_only=True)

DIMENSION_TITLES: Final[dict[str, str]] = {
    "topic": "文档主题",
    "status": "核查结论档",
    "quote_form": "引文写法",
    "sections": "小节数",
    "noise": "干扰小节",
    "claim_style": "主张措辞",
    "where": "声称出处的位置",
    "doc_style": "小节编号写法",
}


def bucket_of(case: CitationCase, seed: int = SPLIT_SEED) -> int:
    """输入：case、种子。输出：0~99 的桶号。

    预期行为：同一组合必得同一桶，不同组合近似均匀——切分单位是**组合**，不是样本；
    不用内置 `hash()`（它对字符串每进程随机化），否则切分会随进程漂移。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[CitationCase], split: str) -> list[CitationCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的——只按桶号过滤时，
    哈希方差会让 val 只剩个位数；同一组合永远落在同一段。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def cases_for(config: CitationConfig) -> list[CitationCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合；
    其余切分按 8 维组合的哈希桶切片，同组合不跨切分。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        return cases
    return _slice(cases, config.split)


@register(WORKFLOW_NAME)
class CitationWorkflow:
    """引文核查：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 构造式：文档与答案都由表格算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：支持的切分名。预期行为：三个同分布档 + 一个值留出档。"""
        return tuple(SPLIT_BUCKETS) + ("test_ood",)

    def config_for(self, split: str) -> CitationConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：同分布档**必须**用 `replace` 造一份新的——`ID_CONFIG` 是个单例，
        直接返回它会让调用方改到全局，三个档切出同一份数据（shell 上实测踩过）。
        """
        if split == "test_ood":
            return OOD_CONFIG
        if split in SPLIT_BUCKETS:
            return replace(ID_CONFIG, split=split)
        raise KeyError(f"未知切分：{split!r}（可用：{list(SPLIT_BUCKETS) + ['test_ood']}）")

    def build_cases(self, config: CitationConfig) -> list[CitationCase]:
        """输入：采样参数。输出：该切分的 case 列表。"""
        return cases_for(config)

    def expected_values(self, split: str) -> dict[str, set[Any]]:
        """输入：切分名。输出：这一档**本该产出**的取值集合（结论 / 主题 / 引文出处 / 强度）。

        预期行为：由这一档自己的 case 列表算出来，**不写死全量词表**——值留出本来就让训练侧
        少几档（`rollout` 主题、`mislocated` 那一档），写死会天天误报「一条都没出」；何况
        `mislocated` 是 **case 维度**、它的结论是「证据不足」，压根不会出现在结论计数里。
        每一档都走真的规则引擎（`sample_spec` + `answers_of`），不另起一套判断。
        """
        config = self.config_for(split)
        out: dict[str, set[Any]] = {
            "verdicts": set(),
            "topics": set(),
            "sources": set(),
            "strengths": set(),
        }
        for index, case in enumerate(self.build_cases(config)):
            spec = spec_mod.sample_spec(random.Random(0), case, split, index, 0)
            answers = spec_mod.answers_of(*spec.truth())
            out["verdicts"].add(answers["verdict"])
            out["sources"].add(answers["quote_source"])
            out["strengths"].add(answers["evidence_strength"])
            out["topics"].add(case.topic)
        return out

    def case_label(self, case: CitationCase) -> str:
        """输入：case。输出：一行可读描述。"""
        return case.describe()

    def stats_dimensions(self, case: CitationCase) -> dict[str, Any]:
        """输入：case。输出：{维度名: 取值}。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: CitationConfig, case: CitationCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法在这里抽（同一 case 的多条样本会用不同问法）；
        case 与序号一起放进 params，`sample` 阶段不再做任何随机决策。
        """
        return SplitSpec(
            name=config.split,
            params={"case": case, "phrasing": rng.choice(config.phrasings), "seq": seq},
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> CitationSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（文档、主张、引文来历都在这里定死）。"""
        return spec_mod.sample_spec(
            rng, split.params["case"], split.name, split.params["seq"], split.params["phrasing"]
        )

    def render(self, spec: CitationSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件
        必得同一段文本，所以「两次生成逐字节一致」这条能直接查。
        """
        return render.build(spec, locales.load(locale))

    def ask(
        self, spec: CitationSpec, rng: random.Random, locale: str
    ) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: Any, *, case: Any = None) -> parse.ParsedCitation:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的
        结构反解（各语言的标签互不重叠，由 `tests/workflows/citation/test_locales.py` 锁住）。
        「原文 → 键」的翻译留给判别器，它手上有语言文件。
        """
        return parse.parse_citation(state, parse.merged_labels())

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
        """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

        预期行为：只看产物本身——题族是否齐全、**这一档本该出现的**结论 / 主题 / 引文出处 /
        证据强度是否都出现（期望值由 `expected_values` 从这一档自己的 case 列表算出来，
        不写死全量词表）、choice 的答案编号是否越界，以及**答案能不能由 state 重算出来**
        （把 `LabelJudge` 的判据搬到产物级，这样「老产物」也能事后查）。
        反解用合并标签表；主题靠语言文件的文档标题认。
        """
        labels = parse.merged_labels()
        loc = locales.load(locale)
        topic_labels = {name: loc["topic"][name]["label"] for name in spec_mod.TOPIC_NAMES}
        known = set(questions.FAMILIES)
        report: dict[str, Any] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, Any] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "verdicts": Counter(),
                "sources": Counter(),
                "actions": Counter(),
                "strengths": Counter(),
                "topics": Counter(),
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
                    parsed = parse.parse_citation(row["state"], labels)
                    expected = judges_mod.recompute(parsed)
                except ValueError as exc:
                    counts["label_mismatch"] += 1
                    missing.append(f"{split}: 有一条引文反解不出来（{exc}）")
                    continue
                counts["verdicts"][expected["verdict"]] += 1
                counts["sources"][expected["quote_source"]] += 1
                counts["actions"][expected["action"]] += 1
                counts["strengths"][expected["evidence_strength"]] += 1
                for name, label in topic_labels.items():
                    if parsed.doc.startswith(label):
                        counts["topics"][name] += 1
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`verdict_2` -> `verdict`）
                    counts["families"][family] += 1
                    counts["types"][question["type"]] += 1
                    if family not in known:
                        counts["unknown_families"][family] += 1
                    answer = row["answers"][qid]
                    if question["type"] == "choice":
                        options = list(question["criteria"])
                        index = answer["choice"]
                        if not 0 <= index < len(options):
                            counts["index_bad"] += 1
                            continue
                        got: Any = options[index]
                    elif question["type"] == "noul":
                        got = answer["noul"]
                    else:
                        got = answer["score"]
                    want = expected.get(family)
                    if want is not None and got != want:
                        counts["label_mismatch"] += 1
            for family in questions.FAMILIES:
                if counts["families"][family] == 0:
                    missing.append(f"{split}: 题族 {family} 一条都没出")
            want_values = self.expected_values(split)
            for verdict in sorted(want_values["verdicts"]):
                if counts["verdicts"][verdict] == 0:
                    missing.append(f"{split}: 结论 {verdict} 一条都没出")
            for topic in sorted(want_values["topics"]):
                if counts["topics"][topic] == 0:
                    missing.append(f"{split}: 主题 {topic} 一条都没出")
            for source in sorted(want_values["sources"]):
                if counts["sources"][source] == 0:
                    missing.append(f"{split}: 引文出处 {source} 一条都没出")
            for level in sorted(want_values["strengths"]):
                if counts["strengths"][level] == 0:
                    missing.append(f"{split}: 证据强度 {level} 级一条都没出")
            if counts["index_bad"]:
                missing.append(f"{split}: {counts['index_bad']} 条 choice 答案编号越界")
            if counts["label_mismatch"]:
                missing.append(f"{split}: {counts['label_mismatch']} 条答案与文本重算不一致")
            if counts["unknown_families"]:
                missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
            report[split] = counts
        return report, missing

    def available_locales(self) -> list[str]:
        """输入：无。输出：支持的语言列表。"""
        return locales.available()

    def bind_backend(self, backend: Any) -> None:
        """空操作：引文核查是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。预期行为：真被绑定说明接线错了，直接报错。
        """
        raise AssertionError("引文核查是纯构造式，不该被绑定后端")
