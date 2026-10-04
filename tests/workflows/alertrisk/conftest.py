"""alertrisk 的假后端与草稿构造器：脚本化出一屏警报记录，不联网。

假后端按 `task.spec` 现造警报记录——**默认完全兑现计划**（记录里的形状由注入特征推出来），另外
几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。

**每条判据各占一整行**（`R1` 只看 `exposure:` 行、`R2` 只看 `exploit:` 行、`R3` 只看 `impact:`
行），所以「取到某一条」就是把它那一行换成对应的形状，三条之间不会互相遮蔽——这是这一族从设计上
避开「两条判据共用一段文字」的做法（范式 §9.51）。
"""

from __future__ import annotations

from typing import Final

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft as _draft
from workflows.alertrisk import features, locales, render
from workflows.alertrisk.features import AlertRiskCase
from workflows.alertrisk.spec import AlertRiskSpec

#: 五行骨架：一屏平平无奇、看一眼就能关掉的警报（三条判据一条都不取到高危档），长度也落在
#: `render.MIN_CONTENT_CHARS` 以上。语义上它就是「1 级」那一档要的东西（0 条命中的记录）。
BASE_LINES: Final[dict[str, str]] = {
    "rule": "rule: EDR-T1059-SuspiciousChildProcess",
    "impact": "impact: hosts=3 -- 三台研发用的笔记本",
    "exposure": "exposure: reachable=internal -- 只有内网能访问",
    "exploit": "exploit: auth=required -- 需要域账号才能利用",
    "detail": "detail: 研发笔记本上的 node.exe 拉起了一次 powershell，父进程是构建脚本，看着像日常构建喵",
}

#: 英文版骨架（假后端按语言现造时用它）：同样是「一条都不取到」的记录。
BASE_LINES_EN: Final[dict[str, str]] = {
    "rule": "rule: EDR-T1059-SuspiciousChildProcess",
    "impact": "impact: hosts=3 -- three developer laptops",
    "exposure": "exposure: reachable=internal -- only reachable from the intranet",
    "exploit": "exploit: auth=required -- needs a domain account",
    "detail": "detail: node.exe on a developer laptop spawned powershell once, the parent is the build script, looks like a normal build",
}

#: 每条判据的「取到形状」——它**真的落在自己那一行上**，所以三条各占一行、互不遮蔽。
HIT_LINES: Final[dict[str, str]] = {
    "R1": "exposure: reachable=public -- 这台机器从公网就能访问到",
    "R2": "exploit: auth=none -- 不需要任何凭据就能打",
    "R3": "impact: hosts=24 -- 二十四台生产机器",
}

#: 判据编号 → 它钉在哪一个结构键那一行上。
LINE_KEYS: Final[dict[str, str]] = {
    "R1": "exposure",
    "R2": "exploit",
    "R3": "impact",
}

#: 每条判据在警报记录里**原样出现**的那一段（反解引文用它，见 `quote_for`）。引文要**连键名一起**
#: 抄全（反解提示词里的口径），所以它自己单拿出来也满足那条判据。
QUOTES: Final[dict[str, str]] = {
    "R1": "exposure: reachable=public -- 这台机器从公网就能访问到",
    "R2": "exploit: auth=none -- 不需要任何凭据就能打",
    "R3": "impact: hosts=24 -- 二十四台生产机器",
}


def base_lines(lang: str = "zh") -> dict[str, str]:
    """输入：语言。输出：那一种语言的五行骨架。预期行为：只影响 `detail:` 那一行的语言。"""
    return dict(BASE_LINES_EN) if lang == "en" else dict(BASE_LINES)


def lines_of(tags: tuple[str, ...], lang: str = "zh") -> dict[str, str]:
    """输入：该取到高危档的因子、语言。输出：五行警报记录的取值表（键 → 那一行的全文）。"""
    lines = base_lines(lang)
    for tag in features.TAG_IDS:
        if tag in tags:
            lines[LINE_KEYS[tag]] = HIT_LINES[tag]
    return lines


def render_lines(lines: dict[str, str]) -> str:
    """输入：五行取值表。输出：按 `features.KEY_LINES` 顺序拼起来的警报记录。"""
    return "\n".join(lines[key] for key in features.KEY_LINES if key in lines)


def packet_of(tags: tuple[str, ...], lang: str = "zh") -> str:
    """输入：该取到高危档的因子、语言。输出：一屏完全兑现它们的警报记录。

    预期行为：按 `TAG_IDS` 顺序把该取到的那几行换成「取到形状」——每条判据各占一整行，换上去也
    不会互相遮蔽，所以按字面算出来的命中组合就是 `tags`。
    """
    return render_lines(lines_of(tags, lang))


def quote_for(tag: str) -> str:
    """输入：因子编号。输出：该判据在记录里**原样出现**的那一段（引文）。

    预期行为：直接取 `QUOTES` 里那一段——它确实在 `packet_of` 的产物里，所以反解带上它就站得住
    （`judges.quote_holds` 会原样比对，且它自己也满足那条判据）。
    """
    return QUOTES[tag]


def trace_report(
    case: AlertRiskCase, *, lie: bool = False, level: int | None = None
) -> dict[str, JsonValue]:
    """输入：注入特征、要不要说反话、要不要换一个等级。输出：反解草稿（因子 + 引文 + 等级）。

    预期行为：`lie=True` 时**漏报**最后一条（读者没读出来），这是 `TraceJudge` 该拦的方向；
    多报一条是另一条路径——引文站不住就会被 `judges.quote_holds` 当误报丢掉（见 test_judges）。
    `level` 给了就用它，否则用矩阵算出来的那个。
    """
    tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
    if lie and tags:
        tags = tags[:-1]
    return {
        "tags": [{"tag": tag, "quote": quote_for(tag)} for tag in tags],
        "level": features.level_of(case.tags()) if level is None else level,
    }


def spec_for(
    case: AlertRiskCase,
    phrasing: int = 0,
    split: str = "train",
    seq: int = 1,
    direction: int = 0,
) -> AlertRiskSpec:
    """输入：case、问法编号、切分、序号、手法方向下标。输出：真值规格。"""
    return AlertRiskSpec(
        case=case,
        phrasing=phrasing,
        direction=direction,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(packet: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：一屏警报记录、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.ContentDraft(packet=packet), locales.load(lang))}


def faithful(
    case: AlertRiskCase, lang: str = "zh"
) -> tuple[AlertRiskSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：警报记录完全兑现计划。"""
    spec = spec_for(case)
    packet = packet_of(case.tags(), lang)
    render.check_draft(render.ContentDraft(packet=packet), case, locales.load(lang))
    return spec, sample_of(packet, lang)


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（`locale` 决定 `detail:` 那一行写哪种语言；其余都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时漏报最后一条因子（验 `trace` 判别器）；
    - `trace_level_lies` 反解时把等级说成另一档（验「矩阵没读对」那一条路径）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_line` 少写一条该取到的形状、`extra_line` 多写一条不该取到的形状；
    - `missing_key` 把 `rule:` 那一行整行抹掉（验「五个结构键必须齐全」那道闸）；
    - `empty_detail` 把 `detail:` 那一行的取值清空（验「自然语言正文不能空」那道闸）；
    - `too_short` 把记录缩到一个字符、`fence` 给记录套上围栏。
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
        empty_detail: bool = False,
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
        self.empty_detail = empty_detail
        self.too_short = too_short
        self.fence = fence
        self.tasks: list[str] = []

    def packet_for(self, case: AlertRiskCase) -> str:
        """输入：注入特征。输出：按开关写歪之后的一屏警报记录。"""
        tags = [tag for tag in features.TAG_IDS if tag in case.tags()]
        if self.drop_line and tags:
            tags = tags[:-1]
        if self.extra_line:
            spare = [tag for tag in features.TAG_IDS if tag not in case.tags()]
            if spare:
                tags = [*tags, spare[0]]
        lines = lines_of(tuple(tags), self.locale)
        if self.missing_key:
            lines.pop("rule")
        if self.empty_detail:
            lines["detail"] = "detail:"
        if self.too_short:
            return "短"
        packet = render_lines(lines)
        # 围栏单独成行：这样五个结构键仍然各占一行（模型把整段包进代码块就是这个形状），
        # 「格式残留」那道闸才是在考它自己，而不是先被「缺这几行」拦下来。
        return f"```\n{packet}\n```" if self.fence else packet

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = AlertRiskCase(
            highs=str(spec["case"]["highs"]), source=str(spec["case"]["source"])
        )
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            # 说反话时也要给一个**合法**的等级（1~5）：给出界值会被 `TraceJudge` 当「格式没对齐」
            # 记成没判成，那验的就不是「矩阵读错了」这条路径了。
            level = None
            if self.trace_level_lies:
                truth = int(spec["level"])
                level = 1 if truth == len(features.LEVELS) else truth + 1
            report = trace_report(case, lie=self.trace_lies, level=level)
            return _draft(report, "tags=" + "+".join(str(item["tag"]) for item in report["tags"]))
        packet = self.packet_for(case)
        return _draft({"packet": packet}, packet)


backend = _fakes.instance_fixture(FakeBackend)


fake_backend = _fakes.factory_fixture(FakeBackend)
