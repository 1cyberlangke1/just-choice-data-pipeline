"""安全警报风险定级的文本外壳：因子表、定级矩阵、处置档位表由代码写，一屏警报记录由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **因子表、定级矩阵、处置档位表与记录抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一屏警报记录**（真实的规则名、主机名、进程与请求长尾得很，模板拼出来的脱离现实分布）；
- **哪几个因子取到高危档由代码按字面算**（`features.PATTERNS` 那三条正则），模型碰不到：提示词里
  逐条写明「哪一行必须出现什么形状、别的地方不许出现」，`check_draft` 在渲染期按同一份判据复核
  一遍（不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍。

**因子原文与矩阵都印进 state**（范式 §9.44）：反解模型要判「这条警报取到了哪几个因子、定几级」，
而因子名是**压缩过的标签**——「暴露面对公网可达」推不出「`exposure:` 那一行要写
`reachable=public`」，矩阵更是只印在 state 里。材料里没有那份口径，题就不可自解（范式 §9.8）。

**提示词里一个示例记录都不给**（范式 §9.45）：给了成品句，模型会整段照抄，写一句「那只是例子」
也拦不住。所以这里只印**因子原文**（它本来就是完整规格）与一份**格式规格**（键名 + 占位符，
不是可抄的成品），`pasted_rules` 再兜住「把因子原文整段贴进去」。

**警报只写不执行**：这一族要的是**文本**——规则名、主机数、可达性、凭据要求都只写在记录里，
提示词里明说不要真的去连任何机器、跑任何东西（判据看的也只是字面，跑不跑都不改变标签）。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from . import features, locales
from .features import AlertRiskCase
from .spec import AlertRiskSpec

#: 因子表 / 矩阵 / 档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 一屏警报记录的字符数上下限。太短装不下五行，太长就不是「一屏」了。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS

#: 自然语言正文那一个字段的结构键（`language_texts` 只报它）。
PROSE_KEY: Final[str] = "detail"


class ContentDraft(BaseModel):
    """后端要填的草稿：那一屏警报记录。

    `packet` 是**一屏安全警报记录**（真实的规则名、主机名与描述长尾得很、由模型写）；结构化事实
    （因子表、定级矩阵、处置档位表、抬头、格式规格）全由组装器渲染，模型碰不到。
    """

    packet: str = Field(
        description="五行警报记录（rule / impact / exposure / exploit / detail）；不要围栏、不要解释"
    )


@dataclass(frozen=True)
class ParsedContent:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`facts` 因子表里列出的编号；`matrix` 矩阵表的行；`actions` 档位表的行；
    `text` 一屏警报记录（认不出来给 None）。
    """

    facts: tuple[str, ...]
    matrix: tuple[str, ...]
    actions: tuple[str, ...]
    text: str | None

    @property
    def found(self) -> bool:
        """输入：无。输出：警报记录那一块认出来了没有。"""
        return self.text is not None


def fact_copy(loc: dict[str, JsonValue], tag: str) -> dict[str, JsonValue]:
    """输入：语言字典、因子编号。输出：该因子的文案（name / rule / avoid）。

    预期行为：缺字段就当场报错——因子表与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    table = loc.get("fact")
    entry = table.get(tag) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺因子 {tag} 的文案")
    for key in ("name", "rule", "avoid"):
        if key not in entry:
            raise ValueError(f"因子 {tag} 的文案缺字段 {key}")
    return entry


def level_text(loc: dict[str, JsonValue], level: int) -> str:
    """输入：语言字典、风险等级（1~5）。输出：等级名。

    预期行为：`[level.L<级>].name`；缺这一档就当场报错（新加一档等级却忘了写文案时，渲染会拿不到
    句子）。`grade` 那一族的选项就是按 `features.LEVELS` 依次取它拼出来的，所以题目里的等级表与
    state 里矩阵右边的等级名是**同一份文案**，两边不可能各说各的。
    """
    table = loc.get("level")
    entry = table.get(f"L{level}") if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "name" not in entry:
        raise ValueError(f"语言文件里缺等级 {level} 的文案")
    return str(entry["name"])


def act_text(loc: dict[str, JsonValue], act: str) -> str:
    """输入：语言字典、处置档位名。输出：该档的动作文案。

    预期行为：`[act.<档位>].name`；缺这一档就当场报错。档位表的右半与 `act` 那一族的选项取的是
    **同一份文案**，所以 state 里那张表和题面不可能各说各的。
    """
    table = loc.get("act")
    entry = table.get(act) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "name" not in entry:
        raise ValueError(f"语言文件里缺处置档位 {act} 的文案")
    return str(entry["name"])


def source_copy(loc: dict[str, JsonValue], source: str) -> str:
    """输入：语言字典、报警通道。输出：该通道的形态描述。

    预期行为：缺这一档就当场报错（新加一档通道却忘了写文案时，渲染会拿不到句子）。
    """
    table = loc.get("source")
    entry = table.get(source) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "desc" not in entry:
        raise ValueError(f"语言文件里缺报警通道 {source} 的描述")
    return str(entry["desc"])


def fact_line(loc: dict[str, JsonValue], tag: str) -> str:
    """输入：语言字典、因子编号。输出：因子表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 因子名 + 规则分隔符 + 判据原文`。分隔符按语言取
    （`state.tag_sep`），解析时按同一份分隔符切回编号——**编号是结构键**，所以它排在分隔符前面，
    与语言的书写顺序无关。
    """
    sep = str(loc["state"]["tag_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    entry = fact_copy(loc, tag)
    return f"{tag}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def matrix_line(loc: dict[str, JsonValue], highs: tuple[str, ...]) -> str:
    """输入：语言字典、命中组合。输出：矩阵表里那一行（不含行前缀）。

    预期行为：`因子取值键 + 箭头 + 等级名`。左边的键是格式（`features.combo_text_key` 生成），
    右边的等级名按语言取——考生要能把记录里的取值映射到矩阵左边那一格，所以两边都得印出来。
    """
    arrow = str(loc["state"]["matrix_arrow"])
    return f"{features.combo_text_key(highs)}{arrow}{level_text(loc, features.level_of(highs))}"


def action_line(loc: dict[str, JsonValue], level: int) -> str:
    """输入：语言字典、风险等级。输出：档位表里那一行（不含行前缀）。

    预期行为：`等级名 + 分隔符 + 处置动作`；等级名与 `grade` 那一族的选项是同一份文案。
    """
    sep = str(loc["state"]["tag_sep"])
    return f"{level_text(loc, level)}{sep}{act_text(loc, features.act_for(level))}"


def matrix_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：矩阵表的全部行（不含行前缀，按 `features.HIT_SETS` 的顺序）。

    预期行为：八格取遍——矩阵不完整就等于「有些组合没定义」，那几格的题没法答。`FrameJudge`
    拿它逐行比对 state，少一行或多一行都会当场报出来。
    """
    return [matrix_line(loc, features.parse_hits(value)) for value in features.HIT_SETS]


def action_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：档位表的全部行（不含行前缀，按 `features.LEVELS` 的顺序）。"""
    return [action_line(loc, level) for level in features.LEVELS]


def format_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：格式规格那几行（键名 + 占位符）。

    预期行为：每个键一行，行首带该键自己的说明（`prompt.format` 里按 `features.KEY_LINES` 取）。
    键名与冒号是 state 格式的一部分，所以它们**不进语言文件**，只有说明文字按语言取；每一行的取值
    都写成占位符——规格里不许有能整段照抄的成品（范式 §9.52）。
    """
    spec = loc["prompt"]["format"]
    return [f"{LINE_PREFIX}{spec[key]}" for key in features.KEY_LINES]


def build_prompt(loc: dict[str, JsonValue], case: AlertRiskCase, direction: int) -> str:
    """组装提示词。

    输入：语言字典、注入特征、手法方向下标。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些形状（逐条印上因子原文）、出现在哪一行、
    不许出现哪些（逐条给反面写法）、这条警报来自哪条通道、长度区间、五行格式规格、输出前自查。
    这些就是「受控合成」的规格，判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么还要给一个**手法方向**（范式 §9.7）：只给通道时，提示词在同通道内逐字相同，模型把
    `detail:` 迅速塌成一句套话（实测 9 条中文里 5 条 EDR 全是 PowerShell、4 条 WAF 全是 SQL 注入）。
    方向由 `spec.direction` 抽出来、只影响那一行怎么写，与判据无关。

    为什么**一个示例记录都不给**（范式 §9.45）：只给一份具体记录当例子，模型会把它**整段照抄**
    进样本。因子原文本身就是完整规格，所以提示词只印因子原文 + 一份带占位符的格式规格。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状（写了个
    `hosts=12` 的 `impact:` 行，又顺手在 `exposure:` 行写个 `public`）——那一条的标签就废了。
    """
    prompt = loc["prompt"]
    highs = case.tags()
    missing = [tag for tag in features.TAG_IDS if tag not in highs]
    lines = [
        str(prompt["intro"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["text_only"]),
        str(prompt["source_line"]).format(source=source_copy(loc, case.source)),
        str(prompt["direction_line"]).format(
            direction=locales.pick(loc["prompt"]["directions"], direction)
        ),
        str(prompt["format_header"]),
        *format_lines(loc),
    ]
    if highs:
        lines.append(str(prompt["hits_header"]))
        for tag in highs:
            entry = fact_copy(loc, tag)
            lines.append(str(prompt["hits_line"]).format(name=entry["name"], rule=entry["rule"]))
        lines.append(str(prompt["vary_rule"]))
    if missing:
        lines.append(str(prompt["miss_header"]) if highs else str(prompt["no_hits"]))
        for tag in missing:
            entry = fact_copy(loc, tag)
            lines.append(
                str(prompt["miss_line"]).format(
                    name=entry["name"], avoid=entry["avoid"], rule=entry["rule"]
                )
            )
    lines.append(str(prompt["length_rule"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS))
    lines.append(str(prompt["self_check"]).format(language=loc["language_name"]))
    lines.append(str(prompt["output"]))
    return "\n".join(lines)


def build_task(spec: AlertRiskSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="alertrisk",
        prompt=build_prompt(loc, spec.case, spec.direction),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "highs": list(spec.case.tags()),
            "source": spec.case.source,
            "level": spec.plan.level,
        },
    )


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一屏警报记录、语言字典。输出：被**逐字贴进**样本的因子原文（按 `TAG_IDS` 顺序）。

    预期行为：只认**完全相同的整段**因子原文（`fact_copy(...)["rule"]`）——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是因子原文（示例记录一律不给，见
    `build_prompt`），所以模型能「抄」的只剩它——把因子的词表整段贴进记录，标签照样算命中（正则
    只看子串），但那一屏记录读起来是「写手在照着规格交作业」，不是一条真实告警。
    """
    body = " ".join(text.split())
    out: list[str] = []
    for tag in features.TAG_IDS:
        rule = " ".join(str(fact_copy(loc, tag)["rule"]).split())
        if rule and rule in body:
            out.append(str(fact_copy(loc, tag)["rule"]))
    return out


def check_draft(draft: ContentDraft, case: AlertRiskCase, loc: dict[str, JsonValue]) -> None:
    """输入：后端草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`）。

    预期行为：六道闸依次过——非空、不带围栏、长度落在区间内、五个结构键齐全、`detail:` 那一行
    不是空的（它是这一族唯一的自然语言正文，空了语言尺子就空转）、按字面算出来的命中组合与计划
    逐条相同、没有把因子原文整段贴进去。抛 `DraftError`（`retryable_for_case = True`）让管线
    重渲染一次。
    """
    text = draft.packet.strip()
    if not text:
        raise DraftError("警报记录是空的")
    if "```" in text:
        raise DraftError("警报记录里带了围栏")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"警报记录太短（至少 {MIN_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"警报记录太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    missing = features.missing_keys(text)
    if missing:
        raise DraftError(f"警报记录缺这几行（每个键要单独占一行、后面跟冒号）：{list(missing)}")
    if not features.line_value(text, PROSE_KEY):
        raise DraftError(f"警报记录的 `{PROSE_KEY}:` 那一行是空的——它是唯一一段自然语言正文")
    found = features.hits_of(text)
    if found != case.highs:
        raise DraftError(
            f"警报记录按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.highs or '无'}"
        )
    pasted = pasted_rules(text, loc)
    if pasted:
        raise DraftError(f"警报记录把因子原文原样贴了进去：{pasted}")


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：因子表 + 定级矩阵 + 处置档位表 + 记录抬头 + 一屏警报记录。**三张表都印在 state 里**，
    所以「这条警报取到了哪几档、定几级、该走哪一档」是自明的——考生只读 state 与题面就能答，
    不必知道外部规则（范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["fact_mark"])]
    parts.extend(f"{LINE_PREFIX}{fact_line(loc, tag)}" for tag in features.TAG_IDS)
    parts.append(f"{marks['matrix_mark']}{marks['matrix_note']}")
    parts.extend(f"{LINE_PREFIX}{line}" for line in matrix_lines(loc))
    parts.append(str(marks["rule_mark"]))
    parts.extend(f"{LINE_PREFIX}{line}" for line in action_lines(loc))
    parts.append(str(marks["content_mark"]))
    parts.append(draft.packet.strip())
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认四个抬头，按抬头把行分进四段；因子表里带行前缀的那些行取编号（按 `state.tag_sep`
    切），矩阵与档位表里带行前缀的那些行取**整行原文**（`FrameJudge` 拿它与规范行逐字比），
    **记录抬头之后的所有行原样拼回正文**——抬头之外的内容一律不当结构。
    """
    marks = loc["state"]
    fact_mark = str(marks["fact_mark"])
    matrix_mark = str(marks["matrix_mark"])
    rule_mark = str(marks["rule_mark"])
    content_mark = str(marks["content_mark"])
    sep = str(marks["tag_sep"])
    facts: list[str] = []
    matrix: list[str] = []
    actions: list[str] = []
    body: list[str] = []
    section = ""
    for raw in state.splitlines():
        text = raw.strip()
        if section == "content":
            body.append(raw)
            continue
        if text.startswith(content_mark):
            section = "content"
            body.append(text[len(content_mark) :].strip())
            continue
        if text.startswith(fact_mark):
            section = "fact"
            continue
        if text.startswith(matrix_mark):
            section = "matrix"
            continue
        if text.startswith(rule_mark):
            section = "rule"
            continue
        if not text.startswith(LINE_PREFIX):
            continue
        row = text[len(LINE_PREFIX) :].strip()
        if section == "fact":
            facts.append(row.partition(sep)[0].strip())
        elif section == "matrix":
            matrix.append(row)
        elif section == "rule":
            actions.append(row)
    content = "\n".join(body).strip() if body else None
    return ParsedContent(
        facts=tuple(facts), matrix=tuple(matrix), actions=tuple(actions), text=content
    )


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出记录抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**记录抬头**而不是因子编号：编号（R1…R3）是结构键，与语言无关，拿它当判据会把
    别的语言的 state 也抢过来（范式 §9.22）；记录抬头是渲染时按语言写死的、整段最后那一块的开头，
    才分得开。
    """
    if not isinstance(state, str):
        return None
    for lang in locales.available():
        if parse_with(state, locales.load(lang)).found:
            return lang
    return None


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（因子表 / 矩阵 / 档位表 / 警报记录）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出记录抬头的那一份就用它；一份都认不出来
    时返回空结果（`text` 为 None）。
    """
    if not isinstance(state, str):
        return ParsedContent(facts=(), matrix=(), actions=(), text=None)
    lang = detect_locale(state)
    if lang is None:
        return ParsedContent(facts=(), matrix=(), actions=(), text=None)
    return parse_with(state, locales.load(lang))


def detail_text(packet: str) -> str:
    """输入：一屏警报记录。输出：`detail:` 那一行的描述（没有就给空串）。

    预期行为：这是这一族里**唯一一段自然语言正文**（规则名、主机数、可达性与凭据要求都是格式），
    `Workflow.language_texts` 会把它报给语言尺子（范式 §9.50：只喂认得出语言的字段）。
    """
    return features.line_value(packet, PROSE_KEY)
