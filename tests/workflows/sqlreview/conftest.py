"""sqlreview 的假后端与草稿构造器：脚本化出一屏评审记录，不联网。

假后端按 `task.spec` 现造评审记录——**默认完全兑现计划**（记录里的形状由注入特征推出来），另外
几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。

**每条判据各占一整行**（`Q1` 只看 `statement:` 行、`Q2` 只看 `plan:` 行、`Q3` 只看 `ddl:` 行、
`Q4` 只看 `note:` 行），所以「踩中某一条」就是把它那一行换成对应的形状，四条之间不会互相遮蔽
——这是这一族从设计上避开「两条判据共用一段文字」的做法（范式 §9.51）。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft as _draft
from workflows.sqlreview import features, locales, render
from workflows.sqlreview.features import SqlReviewCase
from workflows.sqlreview.spec import SqlReviewSpec

#: 四行骨架：一屏平平无奇、可以直接执行的评审记录（四条判据一条都不踩），长度也落在
#: `render.MIN_CONTENT_CHARS` 以上。语义上它就是「直接执行」那一档要的东西（0 条命中的记录）。
BASE_LINES: Final[dict[str, str]] = {
    "statement": "statement: UPDATE orders SET status = 'paid' WHERE id = 42;",
    "plan": "plan: Index Scan using orders_pkey on orders  (cost=0.43..8.45 rows=1 width=12)",
    "ddl": "ddl: ALTER TABLE orders ADD COLUMN IF NOT EXISTS paid_at timestamptz;",
    "note": "note: 这次只改一行、走主键索引，评审过一遍，直接执行没问题喵",
}

#: 每条判据的「踩中形状」——它**真的落在自己那一行上**，所以四条各占一行、互不遮蔽。
HIT_LINES: Final[dict[str, str]] = {
    "Q1": "statement: UPDATE orders SET status = 'paid';",
    "Q2": "plan: Seq Scan on orders  (cost=0.00..18334.00 rows=1000000 width=12)",
    "Q3": "ddl: ALTER TABLE orders ADD COLUMN paid_at timestamptz;",
    "Q4": "note: 这个字段的回填脚本待确认，先记一笔",
}

#: 判据编号 → 它钉在哪一个结构键那一行上。
LINE_KEYS: Final[dict[str, str]] = {
    "Q1": "statement",
    "Q2": "plan",
    "Q3": "ddl",
    "Q4": "note",
}

#: 每条判据在评审记录里**原样出现**的那一段（反解引文用它，见 `quote_for`）。引文要**连键名一起**
#: 抄全（反解提示词里的口径），所以它自己单拿出来也满足那条判据。
QUOTES: Final[dict[str, str]] = {
    "Q1": "statement: UPDATE orders SET status = 'paid';",
    "Q2": "plan: Seq Scan on orders  (cost=0.00..18334.00 rows=1000000 width=12)",
    "Q3": "ddl: ALTER TABLE orders ADD COLUMN paid_at timestamptz;",
    "Q4": "note: 这个字段的回填脚本待确认",
}


def lines_of(tags: tuple[str, ...]) -> dict[str, str]:
    """输入：该命中的判据。输出：四行评审记录的取值表（键 → 那一行的全文）。"""
    lines = dict(BASE_LINES)
    for tag in features.TAG_IDS:
        if tag in tags:
            lines[LINE_KEYS[tag]] = HIT_LINES[tag]
    return lines


def render_lines(lines: dict[str, str]) -> str:
    """输入：四行取值表。输出：按 `features.KEY_LINES` 顺序拼起来的评审记录。"""
    return "\n".join(lines[key] for key in features.KEY_LINES if key in lines)


def packet_of(tags: tuple[str, ...]) -> str:
    """输入：该命中的判据。输出：一屏完全兑现它们的评审记录。

    预期行为：按 `TAG_IDS` 顺序把该命中的那几行换成「踩中形状」——每条判据各占一整行，换上去
    也不会互相遮蔽，所以按字面算出来的命中组合就是 `tags`。
    """
    return render_lines(lines_of(tags))


def quote_for(tag: str) -> str:
    """输入：判据编号。输出：该判据在记录里**原样出现**的那一段（引文）。

    预期行为：直接取 `QUOTES` 里那一段——它确实在 `packet_of` 的产物里，所以反解带上它就站得住
    （`judges.quote_holds` 会原样比对，且它自己也满足那条判据）。
    """
    return QUOTES[tag]


def trace_report(case: SqlReviewCase, *, lie: bool = False) -> list[dict[str, str]]:
    """输入：注入特征、要不要说反话。输出：反解草稿的 tags（每条带编号与原样引文）。

    预期行为：`lie=True` 时**漏报**最后一条（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    """
    tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
    if lie and tags:
        tags = tags[:-1]
    return [{"tag": tag, "quote": quote_for(tag)} for tag in tags]


def spec_for(
    case: SqlReviewCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> SqlReviewSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return SqlReviewSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(packet: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：一屏评审记录、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.ContentDraft(packet=packet), locales.load(lang))}


def faithful(case: SqlReviewCase, lang: str = "zh") -> tuple[SqlReviewSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：评审记录完全兑现计划。"""
    spec = spec_for(case)
    packet = packet_of(case.tags())
    render.check_draft(render.ContentDraft(packet=packet), case, locales.load(lang))
    return spec, sample_of(packet, lang)


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时把命中组合说反（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_line` 少写一条该命中的形状、`extra_line` 多写一条不该命中的形状；
    - `missing_key` 把 `note:` 那一行整行抹掉（验「四个结构键必须齐全」那道闸）；
    - `too_short` 把记录缩到一个字符、`fence` 给记录套上围栏。
    """

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
        drop_line: bool = False,
        extra_line: bool = False,
        missing_key: bool = False,
        too_short: bool = False,
        fence: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.drop_line = drop_line
        self.extra_line = extra_line
        self.missing_key = missing_key
        self.too_short = too_short
        self.fence = fence
        self.tasks: list[str] = []

    def packet_for(self, case: SqlReviewCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的一屏评审记录。"""
        tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
        if self.drop_line and tags:
            tags = tags[:-1]
        if self.extra_line:
            spare = [tag for tag in features.TAG_IDS if tag not in case.tags()]
            if spare:
                tags = [*tags, spare[0]]
        lines = lines_of(tuple(tags))
        if self.missing_key:
            lines.pop("note")
        if self.too_short:
            return "短"
        packet = render_lines(lines)
        return "```" + packet if self.fence else packet

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = SqlReviewCase(hits=str(spec["case"]["hits"]), workload=str(spec["case"]["workload"]))
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            report = trace_report(case, lie=self.trace_lies)
            return _draft({"tags": report}, "tags=" + "+".join(item["tag"] for item in report))
        packet = self.packet_for(case)
        return _draft({"packet": packet}, packet)


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
