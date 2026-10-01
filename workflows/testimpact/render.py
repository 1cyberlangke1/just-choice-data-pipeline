"""测试影响选择的文本外壳：覆盖表与跑法档位表由代码写，改动（提交标题 + diff）由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **覆盖表、跑法档位表由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一次改动**（一行提交标题 + 一份 unified diff）：真实 diff 是长尾的，模板拼出来的
  脱离现实分布；
- **改了哪些文件由代码按字面算**（`features.touched_in` 读 `diff --git` 那一行），模型碰不到：
  提示词里逐条写明「必须改到哪些文件、一条不多一条不少」，`check_draft` 在渲染期按同一份正则
  复核一遍（不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍。

**为什么覆盖表要印进 state**：题面上的答案就是从这张表 + 改动读出来的，考生只读 state 与题面
就得能自己答（范式 §9.8）。表里带**类型列**（代码 / 文档）——盲区与文档路径的覆盖列都是空的，
分得开它们靠的是类型列，这一条不给出来题就没法答。

**为什么档位表要印进 state**：`route` 那一族问的是「按档位表该走哪一档」，档位表不印出来就
只能靠猜。档位表的每一行都取 `[fam.route.opt]` 里同一份文案（`threshold_line`），所以 state
里那张表和题面不可能各说各的。

**为什么把 diff 的写法钉死**：`diff --git a/路径 b/路径` 是「这次改到了哪个文件」的**唯一**
字面依据。不钉死的话，模型会写成「改了 handlers.py 里的一个边界判断」这种散文，机械口径读不出
落点，标签就没法从文本重算（审计也就失去了独立路径）。所以 `draft_problems` 里有一条硬形状：
块头之外的行只允许是 git 自己会写的那几种元信息，块内只允许 `+` / `-` / 空格开头。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.state_parse import recognize_locale
from jc.workflow import DraftError

from . import features, locales
from .features import ImpactCase
from .spec import ImpactSpec

#: 覆盖表与档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 提交标题与改动的长度上下限（与 `features` 同一份常量，测试两边都钉）。
MIN_TITLE_CHARS: Final[int] = features.MIN_TITLE_CHARS
MAX_TITLE_CHARS: Final[int] = features.MAX_TITLE_CHARS
MIN_DIFF_CHARS: Final[int] = features.MIN_DIFF_CHARS
MAX_DIFF_CHARS: Final[int] = features.MAX_DIFF_CHARS

#: 格式残留：单独成行的围栏。改动是代码，套上围栏就没法当 diff 读。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```diff", "```patch", "```text", "```json", "```markdown", "```md", "```toml"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")

#: diff 里允许出现的元信息行前缀（`git diff` 自己会写的那几类）。
META_PREFIXES: Final[tuple[str, ...]] = (
    "diff --git ",
    "index ",
    "new file mode ",
    "deleted file mode ",
    "similarity index ",
    "rename from ",
    "rename to ",
    "old mode ",
    "new mode ",
    "Binary files ",
    "GIT binary patch",
    "--- ",
    "+++ ",
)


class ContentDraft(BaseModel):
    """后端要填的草稿：一行提交标题 + 一份 unified diff。

    结构化事实（覆盖表、跑法档位表、两个抬头）全由组装器渲染，模型碰不到。
    """

    title: str = Field(
        description="提交标题：单独一行，说清这次改了什么；不要编号、不要围栏"
    )
    diff: str = Field(
        description="unified diff：每个文件一行 `diff --git a/路径 b/路径`，"
        "后面跟 `---` / `+++` / 至少一个 `@@` 块头与 `+` / `-` 开头的改动行"
    )


@dataclass(frozen=True)
class ParsedContent:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`paths` 覆盖表里列出的路径编号；`levels` 档位表里列出的档位（按表里的顺序）；
    `title` 提交标题；`diff` 改动正文（认不出来给 None）。
    """

    paths: tuple[str, ...]
    levels: tuple[str, ...]
    title: str | None
    diff: str | None

    @property
    def found(self) -> bool:
        """输入：无。输出：改动那一块认出来了没有。"""
        return self.diff is not None

    @property
    def touched(self) -> tuple[str, ...]:
        """输入：无。输出：改动按字面读出来的路径编号（审计侧那条独立路径）。"""
        return features.touched_in(self.diff or "")


def kind_label(loc: dict[str, JsonValue], pid: str) -> str:
    """输入：语言字典、路径编号。输出：类型列的显示文案（`（代码）` / ` (code)`）。

    预期行为：缺这一档就当场报错——覆盖表与提示词都靠它，缺了模型就会拿到半句话。
    """
    kind = features.PATH_KINDS.get(pid)
    table = loc.get("state")
    if kind is None or not isinstance(table, dict):
        raise ValueError(f"路径 {pid!r} 没有类型，或语言文件缺 [state]")
    key = f"kind_{kind}"
    if key not in table:
        raise ValueError(f"语言文件的 [state] 里缺 {key}（路径类型 {kind!r} 的显示文案）")
    return str(table[key])


def suite_name(loc: dict[str, JsonValue], suite: str) -> str:
    """输入：语言字典、套件名。输出：套件的显示名。

    预期行为：缺这一档就当场报错（新加一套测试却忘了写文案时，渲染会拿不到名字）。
    """
    table = loc.get("suite")
    entry = table.get(suite) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "name" not in entry:
        raise ValueError(f"语言文件里缺套件 {suite} 的文案")
    return str(entry["name"])


def flavor_copy(loc: dict[str, JsonValue], flavor: str) -> str:
    """输入：语言字典、改动性质。输出：这一档的说明。预期行为：缺这一档当场报错。"""
    table = loc.get("flavor")
    entry = table.get(flavor) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "desc" not in entry:
        raise ValueError(f"语言文件里缺改动性质 {flavor} 的说明")
    return str(entry["desc"])


def coverage_line(loc: dict[str, JsonValue], pid: str) -> str:
    """输入：语言字典、路径编号。输出：覆盖表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 文件路径 + 类型 + 规则分隔符 + 覆盖的套件`。分隔符按语言取
    （`state.path_sep` / `state.rule_sep`），解析时按同一份分隔符切回编号——**编号是结构键**，
    所以它排在最前面，与语言的书写顺序无关。覆盖为空时写 `state.none_cover`。
    """
    marks = loc["state"]
    sep = str(marks["path_sep"])
    rule_sep = str(marks["rule_sep"])
    joiner = str(marks["suite_joiner"])
    suites = features.COVERAGE[pid]
    cover = joiner.join(suite_name(loc, suite) for suite in suites) or str(marks["none_cover"])
    return f"{pid}{sep}{features.PATH_FILES[pid]}{kind_label(loc, pid)}{rule_sep}{cover}"


def ladder_line(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、跑法档位。输出：档位表里那一行（不含行前缀）。

    预期行为：`阈值描述 + 分隔符 + 跑法动作`；跑法动作与 `route` 那一族的选项取的是**同一份
    文案**（`fam.route.opt`），所以 state 里那张表和题面不可能各说各的。
    """
    marks = loc["state"]
    sep = str(marks["path_sep"])
    threshold = str(marks["threshold"][level])
    action = str(loc["fam"]["route"]["opt"][level])
    return f"{threshold}{sep}{action}"


def build_prompt(loc: dict[str, JsonValue], case: ImpactCase) -> str:
    """组装提示词。

    输入：语言字典、注入特征。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——这次改动是什么性质、必须改到哪些文件（逐条印上
    文件路径与类型）、diff 的写法（`diff --git` / `---` / `+++` / `@@` / `+` `-`）、标题与
    正文的长度区间、输出前自查。这些就是「受控合成」的规格，全在 `draft_problems` 里机械复核，
    模型写歪了当场重渲染。

    为什么**一个成品 diff 都不给**（范式 §9.45 / §9.64）：给一个具体例子，模型会把它**整段照抄**
    进产物。所以形状只写占位形态（`diff --git a/路径 b/路径`），可抄的成品一行都不给；
    `pasted_rules` 再兜住「把覆盖表或档位表整段贴进改动」这条残余捷径。
    """
    prompt = loc["prompt"]
    lines = [
        str(prompt["intro"]).format(min=MIN_DIFF_CHARS, max=MAX_DIFF_CHARS),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["flavor_line"]).format(flavor=flavor_copy(loc, case.flavor)),
        str(prompt["touch_header"]),
    ]
    for pid in case.touched():
        lines.append(
            str(prompt["touch_line"]).format(
                file=features.PATH_FILES[pid], kind=kind_label(loc, pid)
            )
        )
    lines.extend(
        [
            str(prompt["shape_rule"]),
            str(prompt["title_rule"]).format(min=MIN_TITLE_CHARS, max=MAX_TITLE_CHARS),
            str(prompt["length_rule"]).format(min=MIN_DIFF_CHARS, max=MAX_DIFF_CHARS),
            str(prompt["self_check"]),
            str(prompt["output"]),
        ]
    )
    return "\n".join(lines)


def build_task(spec: ImpactSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：一次生成请求。

    预期行为：`spec` 里带上注入特征与必须触及的文件——假后端与真后端都按它写改动；答案不在
    里面（答案由 `questions.build` 纯代码推）。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="testimpact",
        prompt=build_prompt(loc, spec.case),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "touch": list(spec.case.touched()),
            "files": [features.PATH_FILES[pid] for pid in spec.case.touched()],
            "flavor": spec.case.flavor,
        },
    )


def diff_shape_problems(diff: str) -> list[str]:
    """输入：改动正文。输出：不像 diff 的行（最多报前几条）。

    预期行为：认三种行——块头（`@@ -a,b +c,d @@`，之后进入块内）、元信息行（`diff --git` /
    `index` / `---` / `+++` 等 git 自己写的）、块内的改动行（`+` / `-` / 空格开头）。
    **块外的散文一行都不许有**：这条是「模型是不是真的写了 diff」的机械判据。
    """
    problems: list[str] = []
    in_hunk = False
    for raw in diff.splitlines():
        if not raw.strip():
            continue
        if features.HUNK_RE.match(raw):
            in_hunk = True
            continue
        if raw.startswith(("diff --git ", "index ", "--- ", "+++ ")):
            in_hunk = False
            continue
        if raw.startswith(META_PREFIXES) or raw.startswith("\\"):
            continue
        if in_hunk and raw.startswith(("+", "-", " ")):
            continue
        problems.append(raw)
    return problems[:3]


def pasted_rules(diff: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：改动正文、语言字典。输出：被**逐字贴进**改动的覆盖表 / 档位表行。

    预期行为：只认**完全相同的整行**（`coverage_line` / `ladder_line` 的产物）——改一个词就
    不再算。提示词里给模型看的唯一一份具体文字就是这两张表，所以模型能「抄」的只剩它；把表整段
    贴进改动，读起来就不是一次代码改动了。真实 diff 不会包含整行表格，所以这条**不会误伤**。
    """
    table = [
        *(coverage_line(loc, pid) for pid in features.PATH_IDS),
        *(ladder_line(loc, level) for level in features.LEVELS),
    ]
    return [entry for entry in table if entry in diff]


def draft_problems(
    draft: ContentDraft, case: ImpactCase, loc: dict[str, JsonValue]
) -> list[str]:
    """输入：草稿、注入特征、语言字典。输出：这份草稿兑现不了计划的地方（空列表 = 合格）。

    预期行为：逐条查标题形态、改动长度、围栏与 HTML 残留、行形状、**碰到的文件正好是要的那些**
    （一条不多一条不少）、块头与改动行真的存在、没有把表整段贴进来。

    为什么先查「抄表」再查形状：两条都是硬闸，顺序只影响报出来的是哪一条；先报「抄表」是因为
    那是模型最容易犯的偷懒，报出来更好定位（范式 §9.67 那条「闸门先后」的教训：顺序错了会让
    某一条闸门永远撞不到，这里两条都在长度检查之前）。
    """
    problems: list[str] = []
    title = draft.title.strip()
    diff = draft.diff.strip()
    if not title or "\n" in title:
        problems.append("提交标题必须是单独一行、非空")
    elif not MIN_TITLE_CHARS <= len(title) <= MAX_TITLE_CHARS:
        problems.append(
            f"提交标题长度 {len(title)} 不在 {MIN_TITLE_CHARS}~{MAX_TITLE_CHARS} 之间：{title!r}"
        )
    if not diff:
        problems.append("改动那一块是空的")
        return problems
    pasted = pasted_rules(diff, loc)
    if pasted:
        problems.append(f"改动把覆盖表或档位表的原文整段贴了进去：{pasted[:2]}")
    if not MIN_DIFF_CHARS <= len(diff) <= MAX_DIFF_CHARS:
        problems.append(f"改动长度 {len(diff)} 不在 {MIN_DIFF_CHARS}~{MAX_DIFF_CHARS} 之间")
    fences = [line for line in (raw.strip() for raw in diff.splitlines()) if line in LEFTOVER_LINES]
    if fences:
        problems.append(f"改动里混进了围栏：{sorted(set(fences))}")
    if any(tag in diff for tag in LEFTOVER_TAGS):
        problems.append("改动里混进了 HTML 换行标签")
    weird = diff_shape_problems(diff)
    if weird:
        problems.append(f"改动里有不像 diff 的行：{weird}")
    headers = [
        match.group(2)
        for raw in diff.splitlines()
        if (match := features.HEADER_RE.match(raw.strip()))
    ]
    wanted = [features.PATH_FILES[pid] for pid in case.touched()]
    if sorted(headers) != sorted(wanted):
        problems.append(f"改动碰的文件是 {sorted(headers)}，要的正好是 {sorted(wanted)}")
    if not any(features.HUNK_RE.match(raw) for raw in diff.splitlines()):
        problems.append("改动里没有任何 `@@` 块头")
    if not any(
        raw.startswith(("+", "-")) and not raw.startswith(("+++", "---"))
        for raw in diff.splitlines()
    ):
        problems.append("改动里没有任何 `+` / `-` 开头的改动行")
    return problems


def _hunk_span(start: str, count: int) -> str:
    """输入：这一侧的起点行号、块内行数。输出：块头里的 `起点[,行数]` 片段。

    预期行为：行数为 1 时省略计数（`git diff` 就是这么写的），为 0 时照写 `,0`。
    """
    return start if count == 1 else f"{start},{count}"


def normalize_hunks(diff: str) -> str:
    """输入：改动正文。输出：块头行数按块内实际行数重算后的改动正文。

    预期行为：只重算 `@@ -a,b +c,d @@` 里的 `b` / `d`——数一数块内「上下文或删除」有几行
    （旧侧）、「上下文或新增」有几行（新侧）。**起点行号、尾部上下文与正文一行都不动**；
    块外的行（`diff --git` / `index` / `---` / `+++` 等元信息）原样留着。输出按行重排，
    行尾换行不算内容。

    光秃秃的 `@@`（没给行号，模型常这么写）另算一档：**起点按 `1` 补**——真实 diff 里新文件的
    第一块就是这样，而这份改动本来就是编的，模型自己写的起点也不比它更「真」；题面只问哪些
    测试受影响，一行行号都不参与。补完形态与 git 写出来的一致，`diff_shape_problems` 那边靠
    `features.HUNK_RE` 放行。

    为什么这步归代码：真实 diff 是 git 生成的，行数**永远自洽**；而模型写块头基本是瞎猜
    （2026-09-30 小批量实测：18 条里 15 条对不上，通常差一两行）。题面不问行数，所以重算不丢
    任何信息，只是把「模型算不准的算术」收回代码手里——与「覆盖表由代码渲染」同一条理由。

    为什么只数带前缀的行：块内真正的 diff 行必然以 ` ` / `+` / `-` 开头；模型偶尔在块尾多留
    一个空行当分隔符，那不是 diff 行，不该记进行数。
    """
    out: list[str] = []
    lines = diff.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        match = features.HUNK_SPAN_RE.match(raw)
        if match is not None:
            old_start, new_start, tail = match.group(1), match.group(2), match.group(3)
        elif features.BARE_HUNK_RE.match(raw):
            # 光写 `@@`：起点行号模型没给（题面用不到），按第 1 行补，行数照下面重算。
            old_start = new_start = "1"
            tail = ""
        else:
            out.append(raw)
            index += 1
            continue
        body: list[str] = []
        index += 1
        while index < len(lines):
            probe = lines[index]
            if (
                features.HUNK_SPAN_RE.match(probe)
                or features.BARE_HUNK_RE.match(probe)
                or probe.startswith("diff --git ")
            ):
                break
            body.append(probe)
            index += 1
        old_count = sum(1 for line in body if line.startswith((" ", "-")))
        new_count = sum(1 for line in body if line.startswith((" ", "+")))
        head = f"@@ -{_hunk_span(old_start, old_count)} +{_hunk_span(new_start, new_count)} @@"
        out.append(f"{head}{tail}")
        out.extend(body)
    return "\n".join(out)


def check_draft(draft: ContentDraft, case: ImpactCase, loc: dict[str, JsonValue]) -> None:
    """输入：草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`）。

    预期行为：把 `draft_problems` 的结论翻成人话异常——管线看到 `DraftError.retryable_for_case`
    就知道「这一条是模型没照契约写」，重渲染一次（采样抖动，换个说法往往就对了）。
    """
    problems = draft_problems(draft, case, loc)
    if problems:
        raise DraftError("；".join(problems))


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：覆盖表 + 档位表 + 提交标题 + 改动。**两张表印在 state 里**，所以「这次改动该跑
    哪些测试」是自明的——考生只读 state 与题面就能答，不必知道外部规则（范式 §9.8）。
    改动过一道 `normalize_hunks`：块头里的行数是算术，模型写歪了由代码重算（见该函数）。
    """
    marks = loc["state"]
    parts = [str(marks["coverage_mark"])]
    parts.extend(f"{LINE_PREFIX}{coverage_line(loc, pid)}" for pid in features.PATH_IDS)
    parts.append(str(marks["ladder_mark"]))
    parts.extend(f"{LINE_PREFIX}{ladder_line(loc, level)}" for level in features.LEVELS)
    parts.append(str(marks["note_mark"]))
    parts.append(draft.title.strip())
    parts.append(str(marks["diff_mark"]))
    parts.append(normalize_hunks(draft.diff.strip()))
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认覆盖表抬头、档位表抬头、提交标题抬头、改动抬头；覆盖表里带行前缀的那些行取
    编号（按 `state.path_sep` 切），档位表里带行前缀的那些行取档位（按阈值文案反查），
    **改动抬头之后的所有行原样拼回改动**——抬头之外的内容一律不当结构，否则改动正文里出现
    一行以 `- ` 开头的删除行就会把覆盖表数歪。
    """
    marks = loc["state"]
    coverage_mark = str(marks["coverage_mark"])
    ladder_mark = str(marks["ladder_mark"])
    note_mark = str(marks["note_mark"])
    diff_mark = str(marks["diff_mark"])
    sep = str(marks["path_sep"])
    thresholds = {str(marks["threshold"][level]): level for level in features.LEVELS}
    paths: list[str] = []
    levels: list[str] = []
    title: str | None = None
    body: list[str] = []
    section = ""
    for raw in state.splitlines():
        text = raw.strip()
        if section == "diff":
            body.append(raw)
            continue
        if text.startswith(diff_mark):
            section = "diff"
            body.append(text[len(diff_mark) :].strip())
            continue
        if text.startswith(note_mark):
            section = "note"
            continue
        if text.startswith(coverage_mark):
            section = "coverage"
            continue
        if text.startswith(ladder_mark):
            section = "ladder"
            continue
        if section == "coverage" and text.startswith(LINE_PREFIX):
            paths.append(text[len(LINE_PREFIX) :].partition(sep)[0].strip())
        elif section == "ladder" and text.startswith(LINE_PREFIX):
            head = text[len(LINE_PREFIX) :].partition(sep)[0].strip()
            level = thresholds.get(head)
            if level is not None:
                levels.append(level)
        elif section == "note" and text and title is None:
            title = text
    content = "\n".join(body).strip() if body else None
    return ParsedContent(paths=tuple(paths), levels=tuple(levels), title=title, diff=content)


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出改动抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**改动抬头**而不是路径编号：编号（P1…P5）是结构键，与语言无关，拿它当判据会把
    别的语言的 state 也抢过来（这条坑在 lint 上踩过一次，见范式 §9.22）；改动抬头是渲染时按
    语言写死的、整段最后那一块的开头，才分得开。
    """
    return recognize_locale(state, locales, parse_with)


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（覆盖表 / 档位表 / 标题 / 改动）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出改动抬头的那一份就用它；一份都认不出来
    时返回空结果（`diff` 为 None）。
    """
    if not isinstance(state, str):
        return ParsedContent(paths=(), levels=(), title=None, diff=None)
    lang = detect_locale(state)
    if lang is None:
        return ParsedContent(paths=(), levels=(), title=None, diff=None)
    return parse_with(state, locales.load(lang))
