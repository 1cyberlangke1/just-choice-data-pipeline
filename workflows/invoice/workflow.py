"""单据字段抽取 workflow：**抽取式**（答案从素材里抽出来，程序可校验）。

与另外两个 workflow 的关系（PLAN 3.2 的分类表）：
- `maze` 纯构造式：素材与答案都由构造算法算出来，连文本都是算出来的；
- `mail` 特征注入式：标签先行，模型只写文本外壳；
- `invoice`（本 workflow）抽取式：**素材先由构造器造好**（一张单据），
  答案就是素材里的字段。所以它同样是纯构造式的（`needs` 为空、零 token），
  与 maze 的区别在「答案来自素材的抽取」——素材是**文本**，
  必须靠反解（`parse.py`）把「文本里到底写了什么」与真值对上。

切分按 PLAN 3.3：
- train / val / test_iid：**组合留出**——按 8 维组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——版式 `paragraph`、标签别名、货币 `eur`、日期 `dotted`
  在训练侧一个都不出现，`test_ood` 只收「至少含一个没见过的取值」的 case。

导入本模块即自注册（`jc.workflow.load("invoice")` 可用）。
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
from jc.workflow import LabeledQuestion, SplitSpec, family_of, register

from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import InvoiceCase, InvoiceConfig

WORKFLOW_NAME = "invoice"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（组合留出：桶按 case 的 8 维取值算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260925
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 训练侧取值：各维度只留 spec 里标了「见过」的那几档。
#: 排序是为了顺序稳定（frozenset 的迭代顺序随进程变化，不能直接拿来枚举）。
ID_CONFIG: Final[InvoiceConfig] = InvoiceConfig(
    split="train",
    layouts=tuple(sorted(spec_mod.SEEN_LAYOUTS)),
    label_styles=tuple(sorted(spec_mod.SEEN_LABEL_STYLES)),
    currencies=tuple(sorted(spec_mod.SEEN_CURRENCIES)),
    date_formats=tuple(sorted(spec_mod.SEEN_DATE_FORMATS)),
)
#: 留出档：取值表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的 case。
OOD_CONFIG: Final[InvoiceConfig] = InvoiceConfig(split="test_ood", ood_only=True)

DIMENSION_TITLES: Final[dict[str, str]] = {
    "layout": "版式",
    "labels": "标签风格",
    "currency": "货币",
    "date_format": "日期格式",
    "lines": "明细行数",
    "defects": "注入的错误行数",
    "subtotal_ok": "小计是否正确",
    "tax_state": "税额行状态",
}


def bucket_of(case: InvoiceCase, seed: int = SPLIT_SEED) -> int:
    """输入：case、种子。输出：0~99 的桶号。

    预期行为：同一组合必得同一桶，不同组合近似均匀——切分单位是**组合**，不是样本；
    不用内置 `hash()`（它对字符串每进程随机化），否则切分会随进程漂移。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[InvoiceCase], split: str) -> list[InvoiceCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的——只按桶号过滤时，
    哈希方差会让 val 只剩个位数；同一组合永远落在同一段。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def cases_for(config: InvoiceConfig) -> list[InvoiceCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合；
    其余切分按 8 维组合的哈希桶切片，同组合不跨切分。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        return cases
    return _slice(cases, config.split)


@register(WORKFLOW_NAME)
class InvoiceWorkflow:
    """单据字段抽取：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 抽取式：素材与答案都由构造器算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：四个切分（同分布三档 + 一个值留出的 OOD 档）。"""
        return ("train", "val", "test_iid", "test_ood")

    def config_for(self, split: str) -> InvoiceConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用全取值表 + `ood_only`；其余切分用训练侧取值表。
        未知切分直接报错（不猜、不回落）。
        """
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        if split == "test_ood":
            return OOD_CONFIG
        return replace(ID_CONFIG, split=split)

    def build_cases(self, config: InvoiceConfig) -> list[InvoiceCase]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。"""
        return cases_for(config)

    def case_label(self, case: InvoiceCase) -> str:
        """输入：一个 case。输出：一行可读描述（报告与文件名用）。"""
        return case.describe()

    def stats_dimensions(self, case: InvoiceCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：{维度名: 取值}（8 个维度全摊平）。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: InvoiceConfig, case: InvoiceCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法在这里抽（同一 case 的多条样本会用不同问法）；
        case 与序号一起放进 params，`sample` 阶段不再做任何随机决策。
        """
        return SplitSpec(
            name=config.split,
            params={"case": case, "phrasing": rng.choice(config.phrasings), "seq": seq},
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> spec_mod.InvoiceSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（金额、日期、明细全在这里定死）。"""
        return spec_mod.sample_spec(
            rng,
            split.params["case"],
            split.name,
            split.params["seq"],
            split.params["phrasing"],
        )

    def render(self, spec: spec_mod.InvoiceSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件
        必得同一段文本，所以「两次生成逐字节一致」这条能直接查。
        """
        return render.build(spec, locales.load(locale))

    def ask(
        self, spec: spec_mod.InvoiceSpec, rng: random.Random, locale: str
    ) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: object, *, case: object = None) -> parse.ParsedInvoice:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的反解
        （各语言的标签互不重叠，由 `tests/workflows/invoice/test_locales.py` 锁住）。
        """
        return parse.parse_document(state, parse.merged_labels())

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。预期行为：未知名字直接报错。"""
        from . import judges as judges_mod  # 延迟导入：避免与 judges 互相 import 成环

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
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

        预期行为：只看产物本身——题族是否齐全、题型分布、choice 的答案编号是否越界，
        以及「有税额行的单据才该出 `tax_consistent`」这条自洽性（从 state 反解税额字段来判断）。
        反解用合并标签表，所以不需要具体语言文件。
        """
        labels = parse.merged_labels()
        known = set(questions.FAMILIES)
        report: dict[str, JsonValue] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, JsonValue] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "index_bad": 0,
                "tax_mismatch": 0,
                "unknown_families": Counter(),
            }
            if not path.exists():
                missing.append(f"{split}: 产物不存在（{path.name}）")
                report[split] = counts
                continue
            for _, row in jsonio.iter_jsonl(path):
                counts["samples"] += 1
                has_tax = "tax" in parse.parse_document(row["state"], labels).fields
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`bad_lines_2` -> `bad_lines`）
                    counts["families"][family] += 1
                    counts["types"][question["type"]] += 1
                    if family not in known:
                        counts["unknown_families"][family] += 1
                    if question["type"] == "choice":
                        answer = row["answers"][qid]["choice"]
                        if not 0 <= answer < len(question["criteria"]):
                            counts["index_bad"] += 1
                    elif family == "tax_consistent" and not has_tax:
                        counts["tax_mismatch"] += 1
            for family in questions.FAMILIES:
                if family == "tax_consistent":
                    continue  # 条件题族：只在有税额行的单据上出，不要求每份都有
                if counts["families"][family] == 0:
                    missing.append(f"{split}: 题族 {family} 一条都没出")
            if counts["index_bad"]:
                missing.append(f"{split}: {counts['index_bad']} 条 choice 答案编号越界")
            if counts["tax_mismatch"]:
                missing.append(
                    f"{split}: {counts['tax_mismatch']} 条 tax_consistent 出现在没有税额行的单据上"
                )
            if counts["unknown_families"]:
                missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
            report[split] = counts
        _ = locale  # 反解用合并标签表（语言无关），不需要具体语言文件
        return report, missing

    def available_locales(self) -> list[str]:
        """输入：无。输出：支持的语言列表。"""
        return locales.available()

    def bind_backend(self, backend: GenerationBackend) -> None:
        """空操作：单据是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("单据 workflow 是纯构造式的，不该被绑定生成后端")
