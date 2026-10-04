"""每个 workflow 的 config 都接上了旋钮校验：坏旋钮在构造期报人话。

为什么单独一个文件：`jc.knobs` 自己的单元测试在 `tests/core/test_knobs.py`，
这里只验「每个 config 真的调了它、报错里点名了字段」——是**接线检查**。
下面列出的字段表同时也是各 workflow 的全部采样旋钮清单。

为什么不是一格一条参数化用例：这三条查的是**同一段接线**（构造期校验），101 + 51 条用例
只是把同一件事报 101 + 51 遍。收成循环之后失败信息里照样点名是哪个类、哪个字段，
用例数从 157 降到 4。
"""

from __future__ import annotations

import dataclasses

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
from workflows.rubric.workflow import RubricConfig
from workflows.lint.workflow import LintConfig
from workflows.rerank.workflow import RerankConfig
from workflows.citation.workflow import CitationConfig
from workflows.guardrail.workflow import GuardrailConfig
from workflows.autoformat.workflow import AutoformatConfig
from workflows.alertrisk.workflow import AlertRiskConfig
from workflows.depcheck.workflow import DepCheckConfig
from workflows.hirecheck.workflow import HireCheckConfig
from workflows.sqlreview.workflow import SqlReviewConfig
from workflows.subqa.workflow import SubQaConfig
from workflows.promptinject.workflow import PromptInjectConfig
from workflows.memorygate.workflow import MemoryGateConfig
from workflows.logtriage.workflow import LogTriageConfig
from workflows.turntake.workflow import TurnTakeConfig
from workflows.adblock.workflow import AdBlockConfig
from workflows.debugstep.workflow import DebugStepConfig
from workflows.receiptcat.workflow import ReceiptCatConfig
from workflows.clipcut.spec import ClipConfig
from workflows.quoteread.spec import QuoteConfig
from workflows.sourceselect.workflow import SourceSelectConfig
from workflows.findtriage.workflow import FindTriageConfig
from workflows.elementpick.workflow import ElementConfig
from workflows.planstep.spec import PlanConfig
from workflows.sentiment.spec import SentimentConfig
from workflows.cachereuse.workflow import ReuseConfig
from workflows.causechain.workflow import CauseConfig
from workflows.chatgate.workflow import ChatGateConfig
from workflows.posttriage.workflow import PostTriageConfig
from workflows.segmentation.workflow import SegmentationConfig
from workflows.siftrank.workflow import RankConfig
from workflows.testimpact.workflow import ImpactConfig
from workflows.tracescore.workflow import TraceScoreConfig

#: (config 类, 旋钮名)：每个取值表清空后都必须当场报错。
#: 类那一格写 `type[object]` 是**如实**的：这些 config 各是独立的 dataclass，没有共同基类，
#: 这里要的只是「类对象」这一点（构造期故意喂坏取值，所以调用点不能写具体参数类型）。
EMPTY_POOLS: tuple[tuple[type[object], str], ...] = (
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
    (RubricConfig, "phrasing_pool"),
    (LintConfig, "phrasing_pool"),
    (RerankConfig, "phrasing_pool"),
    (CitationConfig, "phrasing_pool"),
    (CitationConfig, "topics"),
    (GuardrailConfig, "phrasing_pool"),
    (AutoformatConfig, "topics"),
    (AutoformatConfig, "wraps"),
    (AutoformatConfig, "shapes"),
    (AutoformatConfig, "variants"),
    (AutoformatConfig, "code_families"),
    (AutoformatConfig, "phrasing_pool"),
    (SqlReviewConfig, "phrasing_pool"),
    (AlertRiskConfig, "phrasing_pool"),
    (AlertRiskConfig, "directions"),
    (DepCheckConfig, "phrasing_pool"),
    (DepCheckConfig, "categories"),
    (DepCheckConfig, "versions"),
    (DepCheckConfig, "licenses"),
    (HireCheckConfig, "phrasing_pool"),
    (HireCheckConfig, "roles"),
    (HireCheckConfig, "years"),
    (HireCheckConfig, "names"),
    (SubQaConfig, "phrasing_pool"),
    (SubQaConfig, "timecodes"),
    (PromptInjectConfig, "phrasing_pool"),
    (MemoryGateConfig, "phrasing_pool"),
    (LogTriageConfig, "phrasing_pool"),
    (TurnTakeConfig, "phrasing_pool"),
    (AdBlockConfig, "phrasing_pool"),
    (DebugStepConfig, "phrasing_pool"),
    (ReceiptCatConfig, "phrasing_pool"),
    (ClipConfig, "phrasings"),
    (QuoteConfig, "phrasings"),
    (SourceSelectConfig, "phrasing_pool"),
    (FindTriageConfig, "phrasing_pool"),
    (ElementConfig, "labels"),
    (ElementConfig, "layouts"),
    (ElementConfig, "shapes"),
    (ElementConfig, "extras"),
    (ElementConfig, "shadows"),
    (ElementConfig, "phrasings"),
    (PlanConfig, "phrasings"),
    (SentimentConfig, "phrasings"),
)

#: (config 类, 旋钮名)：区间类旋钮，元素个数不对时必须当场报错。
BOUNDS: tuple[tuple[type[object], str], ...] = (
    (MazeConfig, "size"),
    (MazeConfig, "marks_range"),
    (MazeConfig, "questions_range"),
)


def test_bad_knobs_report_their_field() -> None:
    """输入：把某个取值表 / 区间旋钮清空。输出：无。

    预期行为：构造期 ValueError，且报错**点名字段**——不点名的话，用户看到「取值表不能为空」
    还得自己去翻是哪个表。失败信息攒在 `bad` 里一起报，不因为第一格炸掉就看不见其余。
    """
    bad: list[str] = []
    for config_cls, field in EMPTY_POOLS + BOUNDS:
        try:
            config_cls(**{field: ()})
        except ValueError as error:
            if field not in str(error):
                bad.append(f"{config_cls.__name__}.{field} 的报错没点名该字段：{error}")
        else:
            bad.append(f"{config_cls.__name__}.{field} 清空后照样构造出来了——校验没接上")
    assert not bad, bad


#: 全库的采样配置类（构造期校验的接线检查都用它）。
CONFIGS: tuple[type[object], ...] = (
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
    RubricConfig,
    LintConfig,
    RerankConfig,
    CitationConfig,
    GuardrailConfig,
    AutoformatConfig,
    SqlReviewConfig,
    AlertRiskConfig,
    DepCheckConfig,
    HireCheckConfig,
    SubQaConfig,
    PromptInjectConfig,
    MemoryGateConfig,
    LogTriageConfig,
    TurnTakeConfig,
    AdBlockConfig,
    DebugStepConfig,
    ReceiptCatConfig,
    ClipConfig,
    QuoteConfig,
    SourceSelectConfig,
    FindTriageConfig,
    ElementConfig,
    PlanConfig,
    SentimentConfig,
    ReuseConfig,
    CauseConfig,
    ChatGateConfig,
    PostTriageConfig,
    SegmentationConfig,
    RankConfig,
    ImpactConfig,
    TraceScoreConfig,
)


def test_defaults_still_construct() -> None:
    """输入：全默认参数（`split` 没默认值的那几个补一个 `train`）。输出：无。

    预期行为：默认值都合法，校验不会误伤正常配置。
    """
    bad: list[str] = []
    for config_cls in CONFIGS:
        fields = {field.name: field for field in dataclasses.fields(config_cls)}
        split_field = fields.get("split")
        extra = (
            {"split": "train"}
            if split_field is not None and split_field.default is dataclasses.MISSING
            else {}
        )
        try:
            config_cls(**extra)
        except ValueError as error:
            bad.append(f"{config_cls.__name__} 的默认值过不了自己的校验：{error}")
    assert not bad, bad


def test_case_limit_rejects_negative() -> None:
    """输入：每个带 `case_limit` 的 config × 负数。输出：无。

    预期行为：构造期 ValueError 且点名 `case_limit`——以前负数会被 `cases[:case_limit]` 悄悄
    解释成「去掉尾部几条」。各家测试包不再各抄一份。
    """
    for config_cls in CONFIGS:
        if "case_limit" not in {field.name for field in dataclasses.fields(config_cls)}:
            continue
        with pytest.raises(ValueError, match="case_limit"):
            config_cls(case_limit=-1)


def test_shape_knobs_report_their_field() -> None:
    """输入：几条形状不对的旋钮。输出：无。

    预期行为：构造期 ValueError 且点名是哪个字段。这几条原本散在各自的测试包里，现在只留这一份：

    - `cover_limit=0`：全空间枚举当场变成空集，采样侧一个 case 都拿不到（只有**真校验了**
      这条的包列在这里——没校验的包是生产侧还没接线，不是这条尺子的事）；
    - `versions=(3,)`：区间旋钮写成单个元素，`[3:3]` 会静默取空；
    - `licenses=("MIT",)`：只给一侧的许可证池，另一半的 case 一采样就没得抽（范式 §9.61）；
    - `timecodes=("12:00",)`：`S3` 的正则按 `HH:MM:SS,mmm` 认，写歪了那条判据就哑了。
    """
    with pytest.raises(ValueError, match="cover_limit"):
        AutoformatConfig(cover_limit=0)
    with pytest.raises(ValueError, match="cover_limit"):
        SegmentationConfig(cover_limit=0)
    with pytest.raises(ValueError, match="versions"):
        DepCheckConfig(split="train", versions=(3,))
    with pytest.raises(ValueError, match="licenses"):
        DepCheckConfig(split="train", licenses=("MIT",))
    with pytest.raises(ValueError, match="timecodes"):
        SubQaConfig(split="train", timecodes=("12:00",))
