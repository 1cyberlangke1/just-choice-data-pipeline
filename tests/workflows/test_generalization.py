"""泛化留出的设计不变量：哪个维度在训练侧必须变、哪个必须不变。

为什么要有：PLAN 3.3 的三种留出里，「维度留出」在我们这里只能落成「同分布侧该维**恒为
一个取值**」——这是**设计决定**，靠人记容易在改取值表时悄悄破掉：

- 有人往 invoice 的训练侧加一个 `aliased` 标签风格 → 维度留出没了；
- 有人把 mail 的某一维收成单值 → 那一族的答案在同分布侧退化成常量（均衡闸会标它）。

`tests/workflows/mail/test_pipeline.py` 已经锁了 mail 的**值留出**（curiosity / 短链接），
这里补的是**维度跨度**这一层。
"""

from __future__ import annotations

from collections import defaultdict


import pytest

from workflows.hybrid.workflow import HybridWorkflow
from workflows.invoice.workflow import InvoiceWorkflow
from workflows.mail.workflow import MailWorkflow
from workflows.match.workflow import MatchWorkflow
from workflows.functions.workflow import FunctionsWorkflow
from workflows.skillpick.workflow import SkillPickWorkflow
from workflows.sim.workflow import SimWorkflow
from workflows.bench.workflow import BenchWorkflow
from workflows.shell.workflow import ShellWorkflow
from workflows.modelroute.workflow import ModelRouteWorkflow
from workflows.verify.workflow import VerifyWorkflow
from workflows.toolgate.workflow import ToolGateWorkflow
from workflows.search.workflow import SearchWorkflow
from workflows.compact.workflow import CompactWorkflow
from workflows.browser.workflow import BrowserWorkflow
from workflows.routing.workflow import RoutingWorkflow
from workflows.review.workflow import ReviewWorkflow
from workflows.rubric.workflow import RubricWorkflow
from workflows.lint.workflow import LintWorkflow
from workflows.rerank.workflow import RerankWorkflow
from workflows.citation.workflow import CitationWorkflow
from workflows.guardrail.workflow import GuardrailWorkflow
from workflows.autoformat.workflow import AutoformatWorkflow
from workflows.promptinject.workflow import PromptInjectWorkflow
from workflows.memorygate.workflow import MemoryGateWorkflow
from workflows.logtriage.workflow import LogTriageWorkflow
from workflows.turntake.workflow import TurnTakeWorkflow
from workflows.adblock.workflow import AdBlockWorkflow
from workflows.alertrisk.workflow import AlertRiskWorkflow
from workflows.depcheck.workflow import DepCheckWorkflow
from workflows.debugstep.workflow import DebugStepWorkflow
from workflows.hirecheck.workflow import HireCheckWorkflow
from workflows.subqa.workflow import SubQaWorkflow
from workflows.sqlreview.workflow import SqlReviewWorkflow
from workflows.cachereuse.workflow import CacheReuseWorkflow
from workflows.clipcut.workflow import ClipCutWorkflow
from workflows.quoteread.workflow import QuoteReadWorkflow
from workflows.posttriage.workflow import PostTriageWorkflow
from workflows.receiptcat.workflow import ReceiptCatWorkflow
from workflows.siftrank.workflow import SiftRankWorkflow
from workflows.sourceselect.workflow import SourceSelectWorkflow
from workflows.findtriage.workflow import FindTriageWorkflow
from workflows.elementpick.workflow import ElementPickWorkflow


def spans_of(workflow: object, splits: tuple[str, ...]) -> dict[str, set[str]]:
    """输入：workflow、要统计的切分。输出：{维度名: 这些切分里出现过的取值}。

    预期行为：只走公开接口（`config_for` / `build_cases` / `stats_dimensions`），零 token。
    """
    spans: dict[str, set[str]] = defaultdict(set)
    for split in splits:
        for case in workflow.build_cases(workflow.config_for(split)):
            for dim, value in workflow.stats_dimensions(case).items():
                spans[dim].add(str(value))
    return spans


def id_splits_of(workflow: object) -> tuple[str, ...]:
    """输入：workflow。输出：除 `test_ood` 以外的切分名（同分布侧）。"""
    return tuple(split for split in workflow.split_names() if split != "test_ood")


@pytest.mark.parametrize(
    ("factory", "dim"),
    [
        (InvoiceWorkflow, "labels"),
        (HybridWorkflow, "layout"),
        (QuoteReadWorkflow, "names"),
   ],
)
def test_dimension_holdout_is_realized(factory: object, dim: str) -> None:
    """输入：workflow 与它被留出的维度。输出：无。

    预期行为：同分布侧**恰好一个取值**、且 `test_ood` 侧有同分布侧没见过的取值——
    这就是「整个维度没见过」在本格式里的落地形态（字段不可能缺席，只能恒为一个值）。
    """
    workflow = factory()
    id_values = spans_of(workflow, id_splits_of(workflow))[dim]
    ood_values = spans_of(workflow, ("test_ood",))[dim]
    assert len(id_values) == 1, f"{dim} 在同分布侧有多个取值：{sorted(id_values)}"
    assert ood_values - id_values, f"{dim} 的 test_ood 侧没有新取值：{sorted(ood_values)}"


def test_mail_has_no_dimension_holdout() -> None:
    """输入：无。输出：无。

    预期行为：mail 的每一维在同分布侧都**至少两个取值**。它靠值留出（curiosity / 短链接）
    而不是维度留出：mail 的每一维都至少被一族题问着，把某一维冻成单值会让那一族的答案在
    同分布侧退化成常量——均衡闸会当场标它，等于用「坏数据」换「更难 OOD」。
    """
    workflow = MailWorkflow()
    spans = spans_of(workflow, id_splits_of(workflow))
    frozen = {dim: sorted(values) for dim, values in spans.items() if len(values) < 2}
    assert not frozen, f"mail 有维度被冻成单值：{frozen}"


def value_spans(workflow: object, splits: tuple[str, ...]) -> dict[str, set[str]]:
    """输入：workflow、切分。输出：{维度名: 这些切分里出现过的取值}（同 `spans_of`，按取值看）。"""
    return {dim: values for dim, values in spans_of(workflow, splits).items()}


@pytest.mark.parametrize(
    ("factory", "dims"),
    [
        (MatchWorkflow, ("product", "spelling")),
       (FunctionsWorkflow, ("tool", "window", "alias", "phrasing")),
       (SkillPickWorkflow, ("layout", "phrasing")),
       (SimWorkflow, ("layout", "phrasing")),
        (BenchWorkflow, ("layout", "phrasing")),
        (ShellWorkflow, ("goal", "target")),
        (ModelRouteWorkflow, ("task_kind",)),
        (VerifyWorkflow, ("claim_kind", "tone")),
        (ToolGateWorkflow, ("tool", "policy")),
        (SearchWorkflow, ("freshness",)),
        (CompactWorkflow, ("age",)),
        (BrowserWorkflow, ("screen",)),
        (RoutingWorkflow, ("area",)),
        (ReviewWorkflow, ("change_kind",)),
        (RubricWorkflow, ("length",)),
        (LintWorkflow, ("genre",)),
        (RerankWorkflow, ("terms",)),
        (CitationWorkflow, ("target",)),
        (GuardrailWorkflow, ("decoy",)),
        (AutoformatWorkflow, ("topic", "wrap", "variant", "code_family")),
        (PromptInjectWorkflow, ("source",)),
        (MemoryGateWorkflow, ("form",)),
        (LogTriageWorkflow, ("source",)),
        (TurnTakeWorkflow, ("channel",)),
        (AdBlockWorkflow, ("surface",)),
        (AlertRiskWorkflow, ("source",)),
        (DepCheckWorkflow, ("ecosystem",)),
        (HireCheckWorkflow, ("channel",)),
        (DebugStepWorkflow, ("runtime",)),
        (SubQaWorkflow, ("track",)),
        (SqlReviewWorkflow, ("workload",)),
        (CacheReuseWorkflow, ("flavor",)),
        (ClipCutWorkflow, ("topics",)),
        (QuoteReadWorkflow, ("names",)),
        (PostTriageWorkflow, ("decoy",)),
        (ReceiptCatWorkflow, ("decoy",)),
        (SiftRankWorkflow, ("flavor",)),
        (SourceSelectWorkflow, ("size",)),
        (FindTriageWorkflow, ("size",)),
        (ElementPickWorkflow, ("layout", "phrasing")),
    ],
)
def test_value_holdout_is_realized(factory: object, dims: tuple[str, ...]) -> None:
    """输入：纯构造式 workflow 与它们被留出的维度。输出：无。

    预期行为：被点名的每个维度在 `test_ood` 侧都**真有**同分布侧没见过的取值——
    留出清单是设计意图，这条把它钉在数据上；只写注释不写断言的话，
    改一次取值表就能把 OOD 悄悄变回同分布。

    只看被点名的维度：留出某一维会**连带**带出别的维度的新取值
    ——那是留出的后果，不是漏留。
    """
    workflow = factory()
    id_spans = value_spans(workflow, id_splits_of(workflow))
    ood_spans = value_spans(workflow, ("test_ood",))
    for dim in dims:
        assert ood_spans[dim] - id_spans[dim], f"{dim} 的 test_ood 侧没有新取值"
