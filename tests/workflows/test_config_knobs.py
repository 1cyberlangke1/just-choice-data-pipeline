"""每个 workflow 的 config 都接上了旋钮校验：坏旋钮在构造期报人话。

为什么单独一个文件：`jc.knobs` 自己的单元测试在 `tests/core/test_knobs.py`，
这里只验「每个 config 真的调了它、报错里点名了字段」——是**接线检查**。
下面列出的字段表同时也是各 workflow 的全部采样旋钮清单。
"""

from __future__ import annotations

import pytest

from workflows.hybrid.spec import HybridConfig
from workflows.invoice.spec import InvoiceConfig
from workflows.mail.workflow import MailConfig
from workflows.maze.spec import MazeConfig
from workflows.match.spec import MatchConfig
from workflows.functions.spec import CallConfig
from workflows.skillpick.spec import SkillConfig
from workflows.sim.spec import SimConfig
from workflows.bench.spec import BenchConfig
from workflows.shell.workflow import ShellConfig
from workflows.modelroute.workflow import RouteConfig
from workflows.verify.workflow import VerifyConfig
from workflows.toolgate.workflow import ToolGateConfig
from workflows.search.workflow import SearchConfig
from workflows.compact.workflow import CompactConfig
from workflows.browser.workflow import BrowserConfig
from workflows.routing.workflow import RoutingConfig
from workflows.review.workflow import ReviewConfig

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
    (MatchConfig, "products"),
    (MatchConfig, "patterns"),
    (MatchConfig, "spellings"),
    (MatchConfig, "layouts"),
    (MatchConfig, "extras"),
    (MatchConfig, "phrasings"),
    (CallConfig, "tools"),
    (CallConfig, "voices"),
    (CallConfig, "windows"),
    (CallConfig, "aliases"),
    (CallConfig, "phrasings"),
    (SkillConfig, "tags"),
    (SkillConfig, "layouts"),
    (SkillConfig, "grants"),
    (SkillConfig, "inuse_modes"),
    (SkillConfig, "phrasings"),
    (SimConfig, "boards"),
    (SimConfig, "layouts"),
    (SimConfig, "phrasings"),
    (BenchConfig, "tables"),
    (BenchConfig, "layouts"),
    (BenchConfig, "phrasings"),
    (ShellConfig, "phrasing_pool"),
    (RouteConfig, "phrasing_pool"),
    (VerifyConfig, "phrasing_pool"),
    (ToolGateConfig, "phrasing_pool"),
    (SearchConfig, "phrasing_pool"),
    (CompactConfig, "phrasing_pool"),
    (BrowserConfig, "phrasing_pool"),
    (RoutingConfig, "phrasing_pool"),
    (ReviewConfig, "phrasing_pool"),
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
        MatchConfig,
        CallConfig,
        SkillConfig,
        SimConfig,
        BenchConfig,
        ShellConfig,
        RouteConfig,
        VerifyConfig,
        ToolGateConfig,
        SearchConfig,
        CompactConfig,
        BrowserConfig,
        RoutingConfig,
        ReviewConfig,
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
