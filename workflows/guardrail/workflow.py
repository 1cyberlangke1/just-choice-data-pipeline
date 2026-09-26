"""内容审核（护栏）workflow：**构造式**（内容与全部判断都由规则表算出来，零 token）。

覆盖的决策形状（官方 use-case map）：
- **Detection**：六条互相独立的政策条款各一族 noul——这正是官方 `llm_guardrails` 的形态；
- **Scoring**：`severity`（严重度 0~4 档，九格矩阵写在题面里）；
- **Routing**：`action`（放行 / 复核 / 拦截 / 转专项，六条有序规则写在题面里）；
- **分类**：`top_clause`（危害级别最高的那一条，并列规则写在题面里）。

与 routing 的区别：routing 那几个检测题族共用**一张规则表**，这里的六条条款互相独立，
「同一条内容里哪几条同时成立」才是难点；另外这里多了两维**只改文本、不改答案**的
干扰项（`mentions` 擦边句、`wording` 措辞），专门压「见关键词就报警」和「按语气下结论」。

切分按 PLAN 3.3：
- train / val / test_iid：**素材留出**——按素材（七维指纹）的哈希分桶切片，同一屏内容整组不跨切分；
- test_ood：**值留出**——三条都落在表面形式上：`mentions` 里「辱骂 + 诈骗 + 作弊一起擦边」
  那一档、`history = warned`（没见过这个取值）、`wording = softened`（委婉写法）在训练侧
  一个都不出现。规则分支刻意不动：留出 `reach` 会把严重度挤塌（见 spec 末尾的注释）。

导入本模块即自注册（`jc.workflow.load("guardrail")` 可用）。
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
from .spec import GuardCase, GuardConfig, GuardSpec

WORKFLOW_NAME = "guardrail"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（素材留出：桶按**素材**算，同一屏内容永远同切分）。
SPLIT_SEED: Final[int] = 20260926
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 训练侧取值：只留 spec 里标了「见过」的那几档。
#: 排序是为了顺序稳定（frozenset 的迭代顺序随进程变化，不能直接拿来枚举）。
ID_CONFIG: Final[GuardConfig] = GuardConfig(
    split="train",
    mentions=tuple(sorted(spec_mod.SEEN_MENTIONS)),
    histories=tuple(sorted(spec_mod.SEEN_HISTORIES)),
    wordings=tuple(sorted(spec_mod.SEEN_WORDINGS)),
)
#: 留出档：取值表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的 case。
OOD_CONFIG: Final[GuardConfig] = GuardConfig(split="test_ood", ood_only=True)

DIMENSION_TITLES: Final[dict[str, str]] = {
    "reach": "传播面",
    "direction": "方向",
    "violations": "违规条款",
    "mentions": "擦边句",
    "audience": "受众",
    "history": "历史",
    "wording": "措辞",
}


def bucket_of(material: str, seed: int = SPLIT_SEED) -> int:
    """输入：素材指纹（`spec.material_of`）、种子。输出：0~99 的桶号。

    预期行为：同一素材必得同一桶，不同素材近似均匀；不用内置 `hash()`
    （它对字符串每进程随机化），否则切分会随进程漂移。
    """
    return int(hashlib.sha256(f"{seed}|{material}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[GuardCase], split: str) -> list[GuardCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：**先按素材分组，再按位置切**——同一屏内容整组落在同一档。
    切片用**位置**而不是桶号过滤：只按桶号过滤时哈希方差会让 val 只剩个位数。
    """
    groups: dict[str, list[GuardCase]] = {}
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


def cases_for(config: GuardConfig) -> list[GuardCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合；
    其余切分按素材的哈希桶分组后切片，同一屏内容不跨切分。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        return cases
    return _slice(cases, config.split)


@register(WORKFLOW_NAME)
class GuardrailWorkflow:
    """内容审核：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 构造式：内容与答案都由规则表算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：支持的切分名。"""
        return tuple(SPLIT_BUCKETS) + ("test_ood",)

    def config_for(self, split: str) -> GuardConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用全取值表 + `ood_only`；其余切分用训练侧取值表，
        **并且把 `split` 换成调用方要的那一档**——`ID_CONFIG` 是个单例，直接返回它会让
        train/val/test_iid 三次都按同一档切片（切出来三份一模一样的数据，shell 上实测踩过）。
        """
        if split == "test_ood":
            return OOD_CONFIG
        if split in SPLIT_BUCKETS:
            return replace(ID_CONFIG, split=split)
        raise KeyError(f"未知切分：{split!r}（可用：{self.split_names()}）")

    def build_cases(self, config: GuardConfig) -> list[GuardCase]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。"""
        return cases_for(config)

    def case_label(self, case: GuardCase) -> str:
        """输入：一个 case。输出：一行可读描述（报告与文件名用）。"""
        return case.describe()

    def stats_dimensions(self, case: GuardCase) -> dict[str, Any]:
        """输入：一个 case。输出：{维度名: 取值}（七个维度全摊平）。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: GuardConfig, case: GuardCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法在这里抽（同一 case 的多条样本会用不同问法）；
        case 与序号一起放进 params，`sample` 阶段不再做任何随机决策。
        """
        return SplitSpec(
            name=config.split,
            params={"case": case, "phrasing": rng.choice(config.phrasings), "seq": seq},
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> GuardSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（内容号与全部判断都在这里定死）。"""
        return spec_mod.sample_spec(
            rng, split.params["case"], split.name, split.params["seq"], split.params["phrasing"]
        )

    def render(self, spec: GuardSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件
        必得同一段文本，所以「两次生成逐字节一致」这条能直接查。
        """
        return render.build(spec, locales.load(locale))

    def ask(self, spec: GuardSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: Any, *, case: Any = None) -> parse.ParsedMessage:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的
        结构反解（各语言的标签互不重叠，由 `tests/workflows/guardrail/test_locales.py` 锁住）。
        「原文 → 键」的翻译留给判别器，它手上有语言文件。
        """
        return parse.parse_message(state, parse.merged_labels())

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

        预期行为：只看产物本身——题族是否齐全、choice 的答案编号是否越界、
        以及**答案能不能由内容重算出来**（把 `LabelJudge` 的判据搬到产物级，
        这样「老产物」也能事后查）。重算要语言文件（句模板按语言走），所以这里用调用方给的 `locale`。
        """
        loc = locales.load(locale)
        labels = parse.label_index(loc)
        known = set(questions.FAMILIES)
        report: dict[str, Any] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, Any] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "actions": Counter(),
                "severities": Counter(),
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
                    message = parse.parse_message(row["state"], labels)
                    expected = judges_mod.recompute(message, loc)
                except (ValueError, KeyError) as exc:
                    counts["label_mismatch"] += 1
                    missing.append(f"{split}: 有一条内容反解不出来（{exc}）")
                    continue
                counts["actions"][expected["action"]] += 1
                counts["severities"][expected["severity"]] += 1
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`severity_2` -> `severity`）
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
            if counts["index_bad"]:
                missing.append(f"{split}: {counts['index_bad']} 处编号异常")
            if counts["label_mismatch"]:
                missing.append(f"{split}: {counts['label_mismatch']} 条答案与内容重算不一致")
            if counts["unknown_families"]:
                missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
            report[split] = counts
        return report, missing

    def available_locales(self) -> list[str]:
        """输入：无。输出：支持的语言列表。"""
        return locales.available()

    def bind_backend(self, backend: Any) -> None:
        """空操作：内容审核是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("内容审核 workflow 是纯构造式的，不该被绑定生成后端")
