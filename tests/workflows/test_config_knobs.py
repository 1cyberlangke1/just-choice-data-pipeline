"""每个 workflow 的 config 都接上了旋钮校验：坏旋钮在构造期报人话。

为什么单独一个文件：`jc.knobs` 自己的单元测试在 `tests/core/test_knobs.py`，
这里只验「每个 config 真的调了它、报错里点名了字段」——是**接线检查**。
下面列出的字段表同时也是各 workflow 的全部采样旋钮清单。
"""

from __future__ import annotations

import pytest

from workflows.hybrid.spec import HybridConfig
from workflows.invoice.spec import InvoiceConfig
from workflows.citation.spec import CitationConfig
from workflows.mail.workflow import MailConfig
from workflows.maze.spec import MazeConfig
from workflows.routing.spec import RoutingConfig
from workflows.guardrail.spec import GuardConfig
from workflows.rerank.spec import RerankConfig
from workflows.match.spec import MatchConfig
from workflows.shell.spec import ShellConfig
from workflows.search.spec import SearchConfig
from workflows.autoformat.spec import DocConfig
from workflows.functions.spec import CallConfig
from workflows.lint.spec import LintConfig
from workflows.modelroute.spec import RouteConfig
from workflows.context.spec import SessionConfig
from workflows.browser.spec import ScreenConfig
from workflows.review.spec import ReviewConfig
from workflows.skillpick.spec import SkillConfig
from workflows.rubric.spec import RubricConfig
from workflows.sim.spec import SimConfig

#: (config 类, 旋钮名)：每个取值表清空后都必须当场报错。
EMPTY_POOLS: tuple[tuple[type, str], ...] = (
    (MazeConfig, "algos"),
    (MazeConfig, "formats"),
    (MazeConfig, "symbols"),
    (MazeConfig, "history_lens"),
    (MazeConfig, "mark_trails"),
    (MazeConfig, "option_details"),
    (MazeConfig, "deviations"),
    (MazeConfig, "phrasing_pool"),
    (InvoiceConfig, "layouts"),
    (InvoiceConfig, "label_styles"),
    (InvoiceConfig, "currencies"),
    (InvoiceConfig, "date_formats"),
    (InvoiceConfig, "line_counts"),
    (InvoiceConfig, "defect_counts"),
    (InvoiceConfig, "subtotal_flags"),
    (InvoiceConfig, "tax_states"),
    (InvoiceConfig, "phrasings"),
    (HybridConfig, "meeting_counts"),
    (HybridConfig, "openings"),
    (HybridConfig, "clashes"),
    (HybridConfig, "layouts"),
    (HybridConfig, "weekdays"),
    (HybridConfig, "duration_mixes"),
    (HybridConfig, "phrasings"),
    (HybridConfig, "themes"),
    (MailConfig, "phrasing_pool"),
    (ShellConfig, "dialects"),
    (ShellConfig, "ops"),
    (ShellConfig, "wrappers"),
    (ShellConfig, "goal_styles"),
    (ShellConfig, "read_counts"),
    (ShellConfig, "phrasings"),
    (SearchConfig, "topics"),
    (SearchConfig, "best_flavors"),
    (SearchConfig, "mixes"),
    (SearchConfig, "layouts"),
    (SearchConfig, "probes"),
    (SearchConfig, "pair_kinds"),
    (SearchConfig, "query_styles"),
    (SearchConfig, "phrasings"),
    (CitationConfig, "topics"),
    (CitationConfig, "statuses"),
    (CitationConfig, "quote_forms"),
    (CitationConfig, "section_counts"),
    (CitationConfig, "noises"),
    (CitationConfig, "claim_styles"),
    (CitationConfig, "wheres"),
    (CitationConfig, "doc_styles"),
    (CitationConfig, "phrasings"),
    (RoutingConfig, "areas"),
    (RoutingConfig, "issues"),
    (RoutingConfig, "tiers"),
    (RoutingConfig, "impacts"),
    (RoutingConfig, "urgencies"),
    (RoutingConfig, "compliances"),
    (RoutingConfig, "waits"),
    (RoutingConfig, "phrasings"),
    (GuardConfig, "reaches"),
    (GuardConfig, "directions"),
    (GuardConfig, "violations"),
    (GuardConfig, "mentions"),
    (GuardConfig, "audiences"),
    (GuardConfig, "histories"),
    (GuardConfig, "wordings"),
    (GuardConfig, "phrasings"),
    (RerankConfig, "topics"),
    (RerankConfig, "aspects"),
    (RerankConfig, "shapes"),
    (RerankConfig, "traps"),
    (RerankConfig, "wordings"),
    (RerankConfig, "phrasings"),
    (MatchConfig, "products"),
    (MatchConfig, "patterns"),
    (MatchConfig, "spellings"),
    (MatchConfig, "layouts"),
    (MatchConfig, "extras"),
    (MatchConfig, "phrasings"),
    (DocConfig, "topics"),
    (DocConfig, "wraps"),
    (DocConfig, "shapes"),
    (DocConfig, "variants"),
    (DocConfig, "code_families"),
    (DocConfig, "phrasings"),
    (CallConfig, "tools"),
    (CallConfig, "voices"),
    (CallConfig, "windows"),
    (CallConfig, "aliases"),
    (CallConfig, "phrasings"),
    (LintConfig, "rules"),
    (LintConfig, "genres"),
    (LintConfig, "hits"),
    (LintConfig, "phrasings"),
    (RouteConfig, "tasks"),
    (RouteConfig, "complexities"),
    (RouteConfig, "latencies"),
    (RouteConfig, "costs"),
    (RouteConfig, "retries"),
    (RouteConfig, "datas"),
    (RouteConfig, "phrasings"),
    (SessionConfig, "keep_forms"),
    (SessionConfig, "drop_forms"),
    (SessionConfig, "kind_sets"),
    (SessionConfig, "extras"),
    (SessionConfig, "orders"),
    (SessionConfig, "phrasings"),
    (ScreenConfig, "tasks"),
    (ScreenConfig, "forms"),
    (ScreenConfig, "dones"),
    (ScreenConfig, "decoy_states"),
    (ScreenConfig, "orders"),
    (ScreenConfig, "phrasings"),
    (ReviewConfig, "triples"),
    (ReviewConfig, "noises"),
    (ReviewConfig, "changes"),
    (ReviewConfig, "orders"),
    (ReviewConfig, "phrasings"),
    (SkillConfig, "tags"),
    (SkillConfig, "layouts"),
    (SkillConfig, "grants"),
    (SkillConfig, "inuse_modes"),
    (SkillConfig, "phrasings"),
    (RubricConfig, "sheets"),
    (RubricConfig, "layouts"),
    (RubricConfig, "slacks"),
    (RubricConfig, "phrasings"),
    (SimConfig, "boards"),
    (SimConfig, "layouts"),
    (SimConfig, "phrasings"),
)

#: (config 类, 旋钮名)：区间类旋钮，元素个数不对时必须当场报错。
BOUNDS: tuple[tuple[type, str], ...] = (
    (MazeConfig, "size"),
    (MazeConfig, "marks_range"),
    (MazeConfig, "questions_range"),
)


@pytest.mark.parametrize(("config_cls", "field"), EMPTY_POOLS)
def test_empty_pool_reports_field(config_cls: type, field: str) -> None:
    """输入：把某个取值表清空。输出：无。预期行为：构造期 ValueError，且报错点名字段。"""
    with pytest.raises(ValueError, match=field):
        config_cls(**{field: ()})


@pytest.mark.parametrize(("config_cls", "field"), BOUNDS)
def test_bad_bounds_report_field(config_cls: type, field: str) -> None:
    """输入：把某个区间旋钮清空。输出：无。预期行为：构造期 ValueError，且报错点名字段。"""
    with pytest.raises(ValueError, match=field):
        config_cls(**{field: ()})


@pytest.mark.parametrize(
    "config_cls",
    [
        MazeConfig,
        InvoiceConfig,
        HybridConfig,
        MailConfig,
        ShellConfig,
        SearchConfig,
        CitationConfig,
        RoutingConfig,
        GuardConfig,
        RerankConfig,
        MatchConfig,
        DocConfig,
        CallConfig,
        LintConfig,
        RouteConfig,
        SessionConfig,
        ScreenConfig,
        ReviewConfig,
        SkillConfig,
        RubricConfig,
        SimConfig,
    ],
)
def test_defaults_still_construct(config_cls: type) -> None:
    """输入：全默认参数。输出：无。预期行为：默认值都合法，校验不会误伤正常配置。"""
    config_cls()


def test_mail_case_limit_rejects_negative() -> None:
    """输入：负数 `case_limit`。输出：无。

    预期行为：报人话——以前负数会被 `cases[:case_limit]` 悄悄解释成「去掉尾部几条」。
    """
    with pytest.raises(ValueError, match="case_limit"):
        MailConfig(case_limit=-1)
