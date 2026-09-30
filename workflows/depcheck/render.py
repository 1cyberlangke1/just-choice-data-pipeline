"""依赖与许可证风险门控的文本外壳：标记表、定级规则、处置档位表由代码写，一屏依赖记录由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **标记表、定级规则、处置档位表与记录抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一屏依赖记录**（真实的包名、版本号、许可证标识与备注长尾得很，模板拼出来的脱离现实
  分布）；
- **哪几个标记命中由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里逐条
  写明「哪一行必须出现什么形状、别的地方不许出现」，`check_draft` 在渲染期按同一份判据复核一遍
  （不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍。

**标记原文与定级规则都印进 state**（范式 §9.44）：反解模型要判「这个依赖命中了哪几个标记、定几级」，
而标记名是**压缩过的标签**——「仓库已归档」推不出「`status:` 那一行要写 `archived`」，权重与公式
更是只印在 state 里。材料里没有那份口径，题就不可自解（范式 §9.8）。

**提示词里一个示例记录都不给**（范式 §9.45）：给了成品句，模型会整段照抄，写一句「那只是例子」
也拦不住。所以这里只印**标记原文**（它本来就是完整规格）与一份**格式规格**（键名 + 占位符，不是
可抄的成品），`pasted_rules` 再兜住「把标记原文整段贴进去」。

**这一族只写不查**：要的是**文本**——包名、版本号、许可证与漏洞计数都只写在记录里，提示词里明说
不要真的去装包、连注册表、查漏洞库（判据看的也只是字面，查不查都不改变标签）。

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
from .features import DepCheckCase
from .spec import DepCheckSpec

#: 标记表 / 定级规则 / 档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 一屏依赖记录的字符数上下限。太短装不下六行，太长就不是「清单里的那一条」了。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS

#: 自然语言正文那一个字段的结构键（`language_texts` 只报它）。
PROSE_KEY: Final[str] = "note"


class ContentDraft(BaseModel):
    """后端要填的草稿：那一屏依赖记录。

    `packet` 是**一屏依赖记录**（真实的包名、版本号与备注长尾得很、由模型写）；结构化事实
    （标记表、定级规则、处置档位表、抬头、格式规格）全由组装器渲染，模型碰不到。
    """

    packet: str = Field(
        description=(
            "六行依赖记录（package / version / license / status / advisories / note）；"
            "不要围栏、不要解释"
        )
    )


@dataclass(frozen=True)
class ParsedContent:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`facts` 标记表里列出的编号；`rule` 定级规则那几行；`actions` 档位表的行；
    `text` 一屏依赖记录（认不出来给 None）。
    """

    facts: tuple[str, ...]
    rule: tuple[str, ...]
    actions: tuple[str, ...]
    text: str | None

    @property
    def found(self) -> bool:
        """输入：无。输出：依赖记录那一块认出来了没有。"""
        return self.text is not None


def fact_copy(loc: dict[str, JsonValue], tag: str) -> dict[str, JsonValue]:
    """输入：语言字典、标记编号。输出：该标记的文案（name / rule / avoid）。

    预期行为：缺字段就当场报错——标记表与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    table = loc.get("fact")
    entry = table.get(tag) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺标记 {tag} 的文案")
    for key in ("name", "rule", "avoid"):
        if key not in entry:
            raise ValueError(f"标记 {tag} 的文案缺字段 {key}")
    return entry


def level_text(loc: dict[str, JsonValue], level: int) -> str:
    """输入：语言字典、风险等级（1~5）。输出：等级名。

    预期行为：`[level.L<级>].name`；缺这一档就当场报错（新加一档等级却忘了写文案时，渲染会拿不
    到句子）。
    """
    table = loc.get("level")
    entry = table.get(f"L{level}") if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "name" not in entry:
        raise ValueError(f"语言文件里缺等级 L{level} 的文案")
    return str(entry["name"])


def act_text(loc: dict[str, JsonValue], act: str) -> str:
    """输入：语言字典、处置档位名。输出：该档位的文案。预期行为：缺这一档就当场报错。"""
    table = loc.get("act")
    entry = table.get(act) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "name" not in entry:
        raise ValueError(f"语言文件里缺处置档位 {act} 的文案")
    return str(entry["name"])


def eco_copy(loc: dict[str, JsonValue], ecosystem: str) -> str:
    """输入：语言字典、生态。输出：该生态的形态描述。

    预期行为：缺这一档就当场报错（新加一个生态却忘了写文案时，渲染会拿不到句子）。
    """
    table = loc.get("eco")
    entry = table.get(ecosystem) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "desc" not in entry:
        raise ValueError(f"语言文件里缺生态 {ecosystem} 的描述")
    return str(entry["desc"])


def fact_line(loc: dict[str, JsonValue], tag: str) -> str:
    """输入：语言字典、标记编号。输出：标记表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 标记名 + 规则分隔符 + 判据原文`。分隔符按语言取
    （`state.tag_sep`），解析时按同一份分隔符切回编号——**编号是结构键**，所以它排在分隔符前面，
    与语言的书写顺序无关。
    """
    sep = str(loc["state"]["tag_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    entry = fact_copy(loc, tag)
    return f"{tag}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def weight_line() -> str:
    """输入：无。输出：权重那一行（`D1=2 D2=1 D3=2 D4=1`）。

    预期行为：**这是 state 格式的一部分**（定级规则表的第一行），不进语言文件——权重是数字，
    与语言无关；`FrameJudge` 拿它与代码渲染的规范行逐字比。
    """
    return " ".join(f"{tag}={features.LEVEL_WEIGHTS[tag]}" for tag in features.TAG_IDS)


def rule_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：定级规则那两行（公式 + 权重，不含行前缀）。

    预期行为：公式按语言取（`state.rule_formula`，带一个上限占位符），权重行由 `weight_line`
    生成。**两行都要印进 state**——考生要能自己算等级（范式 §9.8）。
    """
    formula = str(loc["state"]["rule_formula"]).format(max=features.MAX_LEVEL)
    return [formula, weight_line()]


def action_line(loc: dict[str, JsonValue], level: int) -> str:
    """输入：语言字典、风险等级。输出：档位表里那一行（不含行前缀）。

    预期行为：`等级名 + 分隔符 + 处置动作`；等级名与 `grade` 那一族的选项是同一份文案。
    """
    sep = str(loc["state"]["tag_sep"])
    return f"{level_text(loc, level)}{sep}{act_text(loc, features.act_for(level))}"


def action_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：档位表的全部行（不含行前缀，按 `features.LEVELS` 的顺序）。

    预期行为：五档取遍——档位表不完整就等于「有些等级没有处置」，那几档的题没法答。
    """
    return [action_line(loc, level) for level in features.LEVELS]


def format_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：格式规格那几行（键名 + 占位符）。

    预期行为：每个键一行，行首带该键自己的说明（`prompt.format` 里按 `features.KEY_LINES` 取）。
    键名与冒号是 state 格式的一部分，所以它们**不进语言文件**，只有说明文字按语言取；每一行的取值
    都写成占位符——规格里不许有能整段照抄的成品（范式 §9.52）。
    """
    spec = loc["prompt"]["format"]
    return [f"{LINE_PREFIX}{spec[key]}" for key in features.KEY_LINES]


def verdict_words(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：不许出现在 `note:` 那一行里的**风险结论词**。

    预期行为：缺这一张表就当场报错——`check_draft` 与 `HygieneJudge` 都靠它挡「把结论写进自由
    文字」。词表按语言给（每种语言里「结论」的写法不同），所以它进语言文件。

    为什么要有这道闸（范式 §9.59）：`note:` 是这一族**唯一**的自由文字，它一旦写出「上游已归档」
    「无已知漏洞」这种结论，就把结构键上的标签搬进了一句自然语言里——考生可以绕过那几行去读
    备注，题型要考的「按字面判」就落空了。真机产物里 zh 侧 10 条有 3 条踩了这条。
    """
    table = loc.get("hygiene")
    words = table.get("verdict_words") if isinstance(table, dict) else None
    if not isinstance(words, list) or not words:
        raise ValueError("语言文件里缺 hygiene.verdict_words（不许写进备注的风险结论词）")
    return tuple(str(word) for word in words)


def note_verdicts(note: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：`note:` 那一行的备注、语言字典。输出：命中了的结论词（一个都没有就给空表）。

    预期行为：大小写不敏感的子串匹配——结论词是**短词**（「归档」「漏洞」），不是整句；宁可让
    写手换一种说法，也不要把结论留在材料里。
    """
    lowered = note.lower()
    return [word for word in verdict_words(loc) if word.lower() in lowered]


def version_shape(hint: int) -> str:
    """输入：版本号尾段提示值。输出：提示词里那一句要填的**形状**（`x.y.<尾段>`）。

    预期行为：只给形状、不给成品——`x.y.47` 里的第一段与第二段仍是占位符，照抄不下去（范式 §9.52）。
    """
    return f"x.y.{hint}"


def build_prompt(
    loc: dict[str, JsonValue],
    case: DepCheckCase,
    category: int,
    version: int,
    license: str,
) -> str:
    """组装提示词。

    输入：语言字典、注入特征、依赖类别下标、版本号尾段提示值、许可证标识符。输出：完整提示词
    （不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些形状（逐条印上标记原文）、出现在哪一行、
    不许出现哪些（逐条给反面写法）、这个包来自哪个生态、长度区间、六行格式规格、输出前自查。
    这些就是「受控合成」的规格，判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么要给**样本级变量**（范式 §9.57 / §9.58 / §9.60）：只给生态时，提示词在同一生态内逐字
    相同，模型把自由文字迅速塌成同一句套话（`alertrisk` 与 `citation` 都真机踩过）。这里有三个，
    都只影响怎么写、一个字符都不参与判据：
    - `category`（`spec.category`）管包名与 `note:` 那一行的**用途方向**；
    - `version`（`spec.version`）管 `version:` 那一行的**尾段数字**——它是最后一个没有样本级变量
      的自由字段，真机上塌得最狠（28 条里 `1.4.2` 出现 7 次）。
    - `license`（`spec.license`）管 `license:` 那一行的**标识符**——它同样是「最后一个」没有样本级
      变量的自由字段，`dep-10b` 的 40 条里 `MIT` 占了七成（范式 §9.61）；池子按 `D1` 分半，所以
      抽到的值一定与计划相容。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状（写了个
    `count=3` 的 `advisories:` 行，又顺手在 `status:` 行写个 `archived`）——那一条的标签就废了。
    """
    prompt = loc["prompt"]
    hits = case.tags()
    missing = [tag for tag in features.TAG_IDS if tag not in hits]
    lines = [
        str(prompt["intro"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["text_only"]),
        str(prompt["eco_line"]).format(ecosystem=eco_copy(loc, case.ecosystem)),
        str(prompt["category_line"]).format(
            category=locales.pick(loc["prompt"]["categories"], category)
        ),
        str(prompt["version_line"]).format(version=version_shape(version)),
        str(prompt["license_line"]).format(license=license),
        str(prompt["format_header"]),
        *format_lines(loc),
    ]
    if hits:
        lines.append(str(prompt["hits_header"]))
        for tag in hits:
            entry = fact_copy(loc, tag)
            lines.append(str(prompt["hits_line"]).format(name=entry["name"], rule=entry["rule"]))
        lines.append(str(prompt["vary_rule"]))
    if missing:
        lines.append(str(prompt["miss_header"]) if hits else str(prompt["no_hits"]))
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


def build_task(spec: DepCheckSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="depcheck",
        prompt=build_prompt(loc, spec.case, spec.category, spec.version, spec.license),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "hits": list(spec.case.tags()),
            "ecosystem": spec.case.ecosystem,
            "level": spec.plan.level,
        },
    )


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一屏依赖记录、语言字典。输出：被**逐字贴进**样本的标记原文（按 `TAG_IDS` 顺序）。

    预期行为：只认**完全相同的整段**标记原文（`fact_copy(...)["rule"]`）——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是标记原文（示例记录一律不给，见
    `build_prompt`），所以模型能「抄」的只剩它——把标记的词表整段贴进记录，标签照样算命中（正则
    只看子串），但那一屏记录读起来是「写手在照着规格交作业」，不是一条真实的依赖记录。
    """
    body = " ".join(text.split())
    out: list[str] = []
    for tag in features.TAG_IDS:
        rule = " ".join(str(fact_copy(loc, tag)["rule"]).split())
        if rule and rule in body:
            out.append(str(fact_copy(loc, tag)["rule"]))
    return out


def check_draft(draft: ContentDraft, case: DepCheckCase, loc: dict[str, JsonValue]) -> None:
    """输入：后端草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`）。

    预期行为：九道闸依次过——非空、不带围栏、长度落在区间内、六个结构键齐全、`note:` 那一行不是
    空的（它是这一族唯一的自然语言正文，空了语言尺子就空转）、`note:` 那一行没有写进**风险结论**
    （范式 §9.59：写了就把结构键上的标签搬进了自由文字）、按字面算出来的命中组合与计划逐条相同、
    没有把标记原文整段贴进去。抛 `DraftError`（`retryable_for_case = True`）让管线重渲染一次。
    """
    text = draft.packet.strip()
    if not text:
        raise DraftError("依赖记录是空的")
    if "```" in text:
        raise DraftError("依赖记录里带了围栏")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"依赖记录太短（至少 {MIN_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"依赖记录太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    missing = features.missing_keys(text)
    if missing:
        raise DraftError(f"依赖记录缺这几行（每个键要单独占一行、后面跟冒号）：{list(missing)}")
    note = features.line_value(text, PROSE_KEY)
    if not note:
        raise DraftError(f"依赖记录的 `{PROSE_KEY}:` 那一行是空的——它是唯一一段自然语言正文")
    verdicts = note_verdicts(note, loc)
    if verdicts:
        raise DraftError(
            f"依赖记录的 `{PROSE_KEY}:` 那一行写进了风险结论（{verdicts}）——它只讲这个包是干什么"
            "用的、会用在什么场景、谁在维护、为什么现在要引入，不许评价它的风险状况"
        )
    found = features.hits_of(text)
    if found != case.flags:
        raise DraftError(
            f"依赖记录按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.flags or '无'}"
        )
    pasted = pasted_rules(text, loc)
    if pasted:
        raise DraftError(f"依赖记录把标记原文原样贴了进去：{pasted}")


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：标记表 + 定级规则 + 处置档位表 + 记录抬头 + 一屏依赖记录。**三张表都印在 state 里**，
    所以「这个依赖命中了哪几个标记、定几级、该走哪一档」是自明的——考生只读 state 与题面就能答，
    不必知道外部规则（范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["fact_mark"])]
    parts.extend(f"{LINE_PREFIX}{fact_line(loc, tag)}" for tag in features.TAG_IDS)
    parts.append(str(marks["rule_mark"]))
    parts.extend(f"{LINE_PREFIX}{line}" for line in rule_lines(loc))
    parts.append(str(marks["act_mark"]))
    parts.extend(f"{LINE_PREFIX}{line}" for line in action_lines(loc))
    parts.append(str(marks["content_mark"]))
    parts.append(draft.packet.strip())
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认四个抬头，按抬头把行分进四段；标记表里带行前缀的那些行取编号（按 `state.tag_sep`
    切），定级规则与档位表里带行前缀的那些行取**整行原文**（`FrameJudge` 拿它与规范行逐字比），
    **记录抬头之后的所有行原样拼回正文**——抬头之外的内容一律不当结构。
    """
    marks = loc["state"]
    fact_mark = str(marks["fact_mark"])
    rule_mark = str(marks["rule_mark"])
    act_mark = str(marks["act_mark"])
    content_mark = str(marks["content_mark"])
    sep = str(marks["tag_sep"])
    facts: list[str] = []
    rule: list[str] = []
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
        if text.startswith(rule_mark):
            section = "rule"
            continue
        if text.startswith(act_mark):
            section = "act"
            continue
        if not text.startswith(LINE_PREFIX):
            continue
        row = text[len(LINE_PREFIX) :].strip()
        if section == "fact":
            facts.append(row.partition(sep)[0].strip())
        elif section == "rule":
            rule.append(row)
        elif section == "act":
            actions.append(row)
    content = "\n".join(body).strip() if body else None
    return ParsedContent(
        facts=tuple(facts), rule=tuple(rule), actions=tuple(actions), text=content
    )


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出记录抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**记录抬头**而不是标记编号：编号（D1…D4）是结构键，与语言无关，拿它当判据会把
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
    """输入：state。输出：纯代码能反解的部分（标记表 / 定级规则 / 档位表 / 依赖记录）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出记录抬头的那一份就用它；一份都认不出来
    时返回空结果（`text` 为 None）。
    """
    if not isinstance(state, str):
        return ParsedContent(facts=(), rule=(), actions=(), text=None)
    lang = detect_locale(state)
    if lang is None:
        return ParsedContent(facts=(), rule=(), actions=(), text=None)
    return parse_with(state, locales.load(lang))


def note_text(packet: str) -> str:
    """输入：一屏依赖记录。输出：`note:` 那一行的备注（没有就给空串）。

    预期行为：这是这一族里**唯一一段自然语言正文**（包名、版本号、许可证标识、状态与计数都是
    格式），`Workflow.language_texts` 会把它报给语言尺子（范式 §9.50：只喂认得出语言的字段）。
    """
    return features.line_value(packet, PROSE_KEY)
