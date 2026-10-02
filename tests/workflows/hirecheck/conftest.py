"""hirecheck 的假后端与草稿构造器：脚本化出一屏候选人材料，不联网。

假后端按 `task.spec` 现造候选人材料——**默认完全兑现计划**（材料里的形状由注入特征推出来），另外
几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。

**每条判据各占一整行**（`H1` 只看 `years:` 行、`H2` 只看 `skills:` 行、`H3` 只看 `permit:` 行、
`H4` 只看 `degree:` 行），所以「取到某一条」就是把它那一行换成对应的形状，四条之间不会互相遮蔽
——这是这一族从设计上避开「两条判据共用一段文字」的做法（范式 §9.51）。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft as _draft
from workflows.hirecheck import features, render
from workflows.hirecheck.features import HireCheckCase

#: 六行骨架：一屏平平无奇、看一眼就能排面试的材料（四条判据一条都不取到），长度也落在
#: `render.MIN_CONTENT_CHARS` 以上。语义上它就是「1 级」那一档要的东西（0 条命中的材料）。
BASE_LINES: Final[dict[str, str]] = {
    "candidate": "candidate: 林素",
    "years": "years: 6",
    "skills": "skills: python, postgres, docker, kubernetes, grpc",
    "permit": "permit: citizen",
    "degree": "degree: ms",
    "note": "note: 现在在一家支付公司做后端，主要负责清结算链路的服务化改造，是同事李工推荐过来的喵",
}

#: 英文版骨架（假后端按语言现造时用它）：同样是「一条都不取到」的材料。
BASE_LINES_EN: Final[dict[str, str]] = {
    "candidate": "candidate: Robin Hale",
    "years": "years: 6",
    "skills": "skills: python, postgres, docker, kubernetes, grpc",
    "permit": "permit: citizen",
    "degree": "degree: ms",
    "note": "note: backend engineer at a payments company right now, mostly on the clearing path, referred by a teammate",
}

#: 每条判据的「取到形状」——它**真的落在自己那一行上**，所以四条各占一行、互不遮蔽。
HIT_LINES: Final[dict[str, str]] = {
    "H1": "years: 1",
    "H2": "skills: python, go, postgres, docker",
    "H3": "permit: none",
    "H4": "degree: associate",
}

#: 判据编号 → 它钉在哪一个结构键那一行上（直接取 features 里那一份，两边不会漂移）。
LINE_KEYS: Final[dict[str, str]] = features.LINE_KEYS

#: 每条判据在材料里**原样出现**的那一段（反解引文用它，见 `quote_for`）。引文要**连键名一起**抄全
#: （反解提示词里的口径），所以它自己单拿出来也满足那条判据。
QUOTES: Final[dict[str, str]] = dict(HIT_LINES)


def base_lines(lang: str = "zh") -> dict[str, str]:
    """输入：语言。输出：那一种语言的六行骨架。预期行为：只影响 `note:` 那一行的语言。"""
    return dict(BASE_LINES_EN) if lang == "en" else dict(BASE_LINES)


def lines_of(tags: tuple[str, ...], lang: str = "zh") -> dict[str, str]:
    """输入：该取到的判据、语言。输出：六行材料的取值表（键 → 那一行的全文）。"""
    lines = base_lines(lang)
    for tag in features.TAG_IDS:
        if tag in tags:
            lines[LINE_KEYS[tag]] = HIT_LINES[tag]
    return lines


def render_lines(lines: dict[str, str]) -> str:
    """输入：六行取值表。输出：按 `features.KEY_LINES` 顺序拼起来的候选人材料。"""
    return "\n".join(lines[key] for key in features.KEY_LINES if key in lines)


def packet_of(tags: tuple[str, ...], lang: str = "zh") -> str:
    """输入：该取到的判据、语言。输出：一屏完全兑现它们的候选人材料。

    预期行为：按 `TAG_IDS` 顺序把该取到的那几行换成「取到形状」——每条判据各占一整行，换上去也不会
    互相遮蔽，所以按字面算出来的命中组合就是 `tags`。
    """
    return render_lines(lines_of(tags, lang))


def quote_for(tag: str) -> str:
    """输入：判据编号。输出：该判据在材料里**原样出现**的那一段（引文）。

    预期行为：直接取 `QUOTES` 里那一段——它确实在 `packet_of` 的产物里，所以反解带上它就站得住
    （`judges.quote_holds` 会原样比对，且它自己也满足那条判据）。
    """
    return QUOTES[tag]


def trace_report(
    case: HireCheckCase, *, lie: bool = False, level: int | None = None
) -> dict[str, JsonValue]:
    """输入：注入特征、漏报开关、可选等级覆盖。输出：反解草稿的数据。

    预期行为：默认完全读对（命中组合与等级都与计划一致）；`lie` 漏报最后一条命中；`level` 覆盖等级。
    引文一律用 `quote_for`——它满足「引文自己要合判据」那一条。
    """
    hits = list(case.tags())
    if lie and hits:
        hits = hits[:-1]
    truth = features.level_of(case.tags())
    return {
        "tags": [{"tag": tag, "quote": quote_for(tag)} for tag in hits],
        "level": truth if level is None else level,
    }


def sample_of(packet: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：一屏候选人材料、语言。输出：契约样本（只有 state，判别器只需要它）。"""
    return {"state": render.to_state(render.ContentDraft(packet=packet), render.locales.load(lang))}


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（`locale` 决定 `note:` 那一行写哪种语言；其余都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时漏报最后一条命中（验 `trace` 判别器）；
    - `trace_level_lies` 反解时把等级说成另一档（验「权重没算对」那一条路径）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_line` 少写一条该取到的形状、`extra_line` 多写一条不该取到的形状；
    - `missing_key` 把 `candidate:` 那一行整行抹掉（验「六个结构键必须齐全」那道闸）；
    - `empty_note` 把 `note:` 那一行的取值清空（验「自然语言正文不能空」那道闸）；
    - `verdict_note` 在 `note:` 那一行里下初筛结论（验范式 §9.59 那道闸）；
    - `leak_skill` 在 `note:` 那一行里提到岗位的硬性技能（验「备注不许重复结构键」那道闸）；
    - `too_short` 把材料缩到一个字符、`fence` 给材料套上围栏。
    """

    def __init__(
        self,
        *,
        locale: str = "zh",
        trace_lies: bool = False,
        trace_level_lies: bool = False,
        trace_raises: Exception | None = None,
        drop_line: bool = False,
        extra_line: bool = False,
        missing_key: bool = False,
        empty_note: bool = False,
        verdict_note: bool = False,
        leak_skill: bool = False,
        too_short: bool = False,
        fence: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.locale = locale
        self.trace_lies = trace_lies
        self.trace_level_lies = trace_level_lies
        self.trace_raises = trace_raises
        self.drop_line = drop_line
        self.extra_line = extra_line
        self.missing_key = missing_key
        self.empty_note = empty_note
        self.verdict_note = verdict_note
        self.leak_skill = leak_skill
        self.too_short = too_short
        self.fence = fence
        self.tasks: list[str] = []

    def packet_for(self, case: HireCheckCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的一屏候选人材料。"""
        tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
        if self.drop_line and tags:
            tags = tags[:-1]
        if self.extra_line:
            spare = [tag for tag in features.TAG_IDS if tag not in case.tags()]
            if spare:
                tags = [*tags, spare[0]]
        lines = lines_of(tuple(tags), self.locale)
        if self.missing_key:
            lines.pop("candidate")
        if self.empty_note:
            lines["note"] = "note:"
        if self.verdict_note:
            lines["note"] = "note: 这个人年限不足，先放着吧"
        if self.leak_skill:
            lines["note"] = "note: 他在上一家公司搭过 kubernetes 平台，动手能力不错"
        if self.too_short:
            return "短"
        packet = render_lines(lines)
        # 围栏单独成行：这样六个结构键仍然各占一行（模型把整段包进代码块就是这个形状），
        # 「格式残留」那道闸才是在考它自己，而不是先被「缺这几行」拦下来。
        return f"```\n{packet}\n```" if self.fence else packet

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = HireCheckCase(
            flags=str(spec["case"]["flags"]), channel=str(spec["case"]["channel"])
        )
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            # 说反话时也要给一个**合法**的等级（1~5）：给出界值会被 `TraceJudge` 当「格式没对齐」
            # 记成没判成，那验的就不是「权重没算对」这条路径了。
            level = None
            if self.trace_level_lies:
                truth = int(spec["level"])
                level = 1 if truth == len(features.LEVELS) else truth + 1
            report = trace_report(case, lie=self.trace_lies, level=level)
            return _draft(report, "tags=" + "+".join(str(item["tag"]) for item in report["tags"]))
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
