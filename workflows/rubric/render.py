"""打分与评分的文本外壳：材料清单与三张表摊平，待评材料由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **材料清单、评分单、档位表、处置表由代码渲染**——它们是结构化事实（这份材料多长、
  分不分条、有没有数字，以及评分单那张表），让模型去写只会写歪；
- 后端只写**待评材料**（真实提交里那几行是长尾的，模板拼的脱离现实）；
- 那段材料必须体现注入的 `placeholders` 与 `citations` 两维，交给 `judges.TraceJudge`
  反解一次再比对。

评分单的「观测」那一格**有两行不印数字**，印的是 `meta.count_mark`（「按材料数」）：
那两条判据（完整 / 有据）的观测只能从**材料正文**里数出来——留空占位一处都没有就是 4、
有一处就是 0；出处标记有就是 4、没有就是 1。这样材料就不是装饰：读者不读材料，
那两行就填不出来（计数口径写在段落头的括号里，见 locale 的 `score_columns`）。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
**本模块不执行任何代码**——它只把文本当字符串读写。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from . import features, locales
from .features import TagSet

#: 提示词里特征的固定顺序：五维全给（见模块文档）。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = tuple(features.DIMS)

#: Markdown 围栏残留：模型偶尔把整段包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = (
    "body_label",
    "manifest_label",
    "score_label",
    "band_label",
    "gate_label",
)

#: 「材料清单」那一行的格数（篇幅 | 形式 | 数据）。
MANIFEST_CELLS: Final[int] = 3

#: 评分单的格数（判据 | 权重 | 要求 | 观测）。
SCORE_CELLS: Final[int] = 4

#: 档位表的格数（档位 | 下界 | 上界）。
BAND_CELLS: Final[int] = 3

#: 处置表的格数（档位 | 不踩红线 | 踩红线）。
GATE_CELLS: Final[int] = 3

#: 「材料清单」三格的顺序（与 `features.DIMS` 里那三维的声明顺序一致）。
MANIFEST_ORDER: Final[tuple[str, ...]] = ("length", "form", "data")

#: 处置表的列（红线两态）。
GATE_COLUMNS: Final[tuple[str, ...]] = features.REDLINE_STATES


class ScoreDraft(BaseModel):
    """后端要填的草稿：**只有待评材料**。

    刻意只留一个字段——材料清单与三张表全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    body: str = Field(
        description="待评的那几行材料：整段行文，或者用 - 开头的行分条列出",
    )


def build_prompt(tags: TagSet, loc: dict[str, JsonValue], phrasing: int) -> str:
    """组装提示词。

    输入：一组注入特征、语言字典、问法编号。
    输出：完整提示词（五维都描述，不含答案）。
    预期行为：特征按 `DIM_ORDER` 固定顺序逐行列出；规则整段附在后面，每条前面加短横线
    并把 `{language}` 填成语言名——留着占位符会把 `{language}` 原样发给模型。
    """
    values = tags.as_dict()
    lines = [str(loc["prompt"]["intro"]), ""]
    lines.extend(locales.feature_line(loc, dim, values[dim]) for dim in DIM_ORDER)
    lines.append("")
    rules = loc["prompt"]["rules"]
    offset = phrasing % len(rules)
    rotated = list(rules[offset:]) + list(rules[:offset])
    lines.extend(f"- {rule.format(language=loc['language_name'])}" for rule in rotated)
    return "\n".join(lines)


def build_task(tags: TagSet, loc: dict[str, JsonValue], task_id: str) -> GenerationTask:
    """输入：注入特征、语言字典、任务 id。输出：`GenerationTask`。

    预期行为：`response_model` 是 `ScoreDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="grading",
        prompt=build_prompt(tags, loc, 0),
        response_model=ScoreDraft,
        seeds=(),
        spec={"tags": tags.as_dict()},
    )


def label_leak(text: str, loc: dict[str, JsonValue]) -> str:
    """输入：一段文本、语言字典。输出：文本开头抄了的段落名（没有就空串）。

    预期行为：只认「段落名 + 冒号」开头这一种写法——材料里正常提到这些词不算
    （模型很爱在开头再抄一遍「待评材料：」，抄进去就是一行重复标记）。
    """
    meta = loc["meta"]
    stripped = text.strip()
    if not stripped:
        return ""
    head = stripped.splitlines()[0].strip()
    for key in SECTION_KEYS:
        title = str(meta[key])
        for sep in (str(meta["colon"]), ":", "："):
            if head.startswith(f"{title}{sep}"):
                return title
    return ""


def banned_echo(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一段文本、语言字典。输出：文本里出现的表名（没有就空列表）。

    预期行为：只认 locale 里点名的那几条**高精度**话术——宁可漏检也不误判，
    误判会在渲染期白拒一条本来能用的数据。这份清单**刻意短**：材料是自然语言，
    「篇幅」「形式」这类词在正常叙述里会出现，收进来会误伤。大小写不敏感。
    """
    lowered = text.lower()
    return [
        str(phrase) for phrase in loc["banned_intent_phrases"] if str(phrase).lower() in lowered
    ]


def split_body_lines(raw: str) -> list[str]:
    """输入：草稿正文。输出：非空行（只去尾部空白，**保留行首的 `- ` 标记**）。

    预期行为：行首那个 `- ` 就是「分条」的凭据，去掉它就分不出行文与分条了，
    所以这里只 `rstrip`；判「空行」时再看 `strip()`。
    """
    out: list[str] = []
    for line in raw.splitlines():
        text = line.rstrip()
        if text.strip():
            out.append(text)
    return out


def check_draft(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、注入特征、语言字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：材料会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：

    - 非空、不含 Markdown 围栏；
    - 开头不许再写一遍段落名；不许出现表名（这两条先查：它们的诊断比「句数不对」具体）；
    - 句数落在 `features.length_bounds(篇幅)` 里（上界是宽容过的，见 `features`）；
    - 形式对得上：分条至少有一行 `- ` 开头，行文一行都不许有；
    - 数据对得上：有数据至少得有一个阿拉伯数字，无数据一个都不许有；
    - 两维正文口径也对得上：留空占位、出处标记**有就是有、没有就是没有**。

    口径**全部机械可判**（范式 §9.19 第一条教训），所以这一关能把「材料与标签脱节」
    在渲染期就挡掉并重试；`judges.TraceJudge` 那一关是**读者视角的复核**（人读材料
    得出的结论必须与机械口径一致），不是同一件事的重复。
    """
    body = str(data.get("body", ""))
    if not body.strip():
        raise DraftError("草稿的「待评材料」是空的")
    if FENCE_PATTERN.search(body):
        raise DraftError("草稿的「待评材料」里混进了 Markdown 围栏")
    leaked = label_leak(body, loc)
    if leaked:
        raise DraftError(f"草稿的「待评材料」把段落名「{leaked}」也写进去了")
    echoed = banned_echo(body, loc)
    if echoed:
        raise DraftError(f"草稿的「待评材料」把表名写进去了：{'、'.join(echoed)}")
    sentences = features.count_sentences(body)
    low, high = features.length_bounds(tags.length)
    if not low <= sentences <= high:
        raise DraftError(
            f"草稿的「待评材料」有 {sentences} 句，与篇幅 {tags.length!r} 要求的 {low}~{high} 句对不上"
        )
    bullets = features.bullet_lines(split_body_lines(body))
    if tags.form == "bulleted" and not bullets:
        raise DraftError("草稿的「待评材料」是「分条」，却一行 `- ` 开头的都没有")
    if tags.form == "prose" and bullets:
        raise DraftError("草稿的「待评材料」是「行文」，却出现了 `- ` 开头的分条行")
    numbered = features.has_number(body)
    if tags.data == "numbers" and not numbered:
        raise DraftError("草稿的「待评材料」是「有数据」，却一个阿拉伯数字都没有")
    if tags.data == "none" and numbered:
        raise DraftError("草稿的「待评材料」是「无数据」，却出现了阿拉伯数字")
    if features.has_placeholder(body) != (tags.placeholders == "yes"):
        raise DraftError("草稿的「待评材料」与「有没有留空占位」对不上")
    if features.has_citation(body) != (tags.citations == "yes"):
        raise DraftError("草稿的「待评材料」与「有没有出处标记」对不上")


def manifest_row(tags: TagSet) -> tuple[str, str, str]:
    """输入：注入特征。输出：「材料清单」那一行的三格（篇幅 | 形式 | 数据）。"""
    return (tags.length, tags.form, tags.data)


def manifest_lines(tags: TagSet, loc: dict[str, JsonValue]) -> list[str]:
    """输入：注入特征、语言字典。输出：清单行（只有一行，三格都是 ASCII 词元）。"""
    return [locales.row_line(loc, manifest_row(tags))]


def score_rows(tags: TagSet, loc: dict[str, JsonValue]) -> tuple[tuple[str, str, str, str], ...]:
    """输入：注入特征、语言字典。输出：评分单的规范形态（五行，判据 | 权重 | 要求 | 观测）。

    预期行为：权重与要求取自 `features`（固定表），观测取自 `features.OBS_TABLE`；
    **正文承载的那两条不印数字**，印 `meta.count_mark`——它们只能从材料里数出来。
    行序严格按 `features.CRITERIA`。
    """
    mark = str(loc["meta"]["count_mark"])
    out: list[tuple[str, str, str, str]] = []
    for name in features.CRITERIA:
        cell = mark if features.OBS_DIM[name] in features.BODY_DIMS else str(features.obs_for(name, tags))
        out.append((name, str(features.WEIGHT_OF[name]), str(features.NEED_OF[name]), cell))
    return tuple(out)


def score_lines(tags: TagSet, loc: dict[str, JsonValue]) -> list[str]:
    """输入：注入特征、语言字典。输出：评分单的每一行（判据 | 权重 | 要求 | 观测）。"""
    return [locales.row_line(loc, cells) for cells in score_rows(tags, loc)]


def count_marks() -> frozenset[str]:
    """输入：无。输出：各语言的「按材料数」标记（`meta.count_mark`）的集合。

    预期行为：一次 run 里语言按样本轮转（`locales[seq % len]`），所以审计拿到的 `locale`
    未必是产物那一种；观测那一格的判定必须写「等于**任一**语言的标记」，拿调用方给的那一种
    去比会把另一种语言的产物误判成「评分单与固定表对不上」。标记本身是各语言自己的词元，
    这里只是把它们收齐，不做翻译。
    """
    return frozenset(str(locales.load(lang)["meta"]["count_mark"]) for lang in locales.available())


def band_rows() -> tuple[tuple[str, str, str], ...]:
    """输入：无。输出：档位表的规范形态（三行，档位 | 下界 | 上界）。

    预期行为：顺序按 `features.BANDS`（低 → 中 → 高）；区间取自 `features.BAND_LINES`，
    表里缺格会在这里当场抛 KeyError。
    """
    return tuple(
        (band, str(features.BAND_LINES[band][0]), str(features.BAND_LINES[band][1]))
        for band in features.BANDS
    )


def band_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：档位表的每一行（档位 | 下界 | 上界）。"""
    return [locales.row_line(loc, cells) for cells in band_rows()]


def gate_rows() -> tuple[tuple[str, str, str], ...]:
    """输入：无。输出：处置表的规范形态（三行，档位 | 不踩红线 | 踩红线）。

    预期行为：行按 `features.BANDS` 的顺序，列按 `GATE_COLUMNS` 的顺序；动作取自
    `features.GATE_TABLE`，表里缺格会在这里当场抛 KeyError。
    """
    return tuple(
        (band, *(features.GATE_TABLE[(band, state)] for state in GATE_COLUMNS))
        for band in features.BANDS
    )


def gate_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：处置表的每一行（档位 | 不踩红线 | 踩红线）。"""
    return [locales.row_line(loc, cells) for cells in gate_rows()]


def to_state(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`body`）、注入特征、语言字典。
    输出：五段文本——待评材料 / 材料清单 / 评分单 / 档位表 / 处置表。
    预期行为：材料独占段落头下面那几行（一行一段文字，拼在段头后面会读成一整行）；
    后四段**由代码写**（它们是注入特征与固定表，不是模型的发挥），所以从 `tags` 推出来。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [header(loc, meta["body_label"])]
    lines.extend(split_body_lines(str(data["body"])))
    lines.append(
        f"{header(loc, meta['manifest_label'])}{meta['paren'].format(text=meta['manifest_columns'])}"
    )
    lines.extend(manifest_lines(tags, loc))
    lines.append(
        f"{header(loc, meta['score_label'])}{meta['paren'].format(text=meta['score_columns'])}"
    )
    lines.extend(score_lines(tags, loc))
    lines.append(
        f"{header(loc, meta['band_label'])}{meta['paren'].format(text=meta['band_columns'])}"
    )
    lines.extend(band_lines(loc))
    lines.append(
        f"{header(loc, meta['gate_label'])}{meta['paren'].format(text=meta['gate_columns'])}"
    )
    lines.extend(gate_lines(loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class ManifestState:
    """从「材料清单」那一行里反解出来的三格。"""

    length: str
    form: str
    data: str


@dataclass(frozen=True)
class ParsedRubric:
    """从 state 里纯代码反解出来的事实。"""

    body_lines: tuple[str, ...]
    manifest: ManifestState | None
    score: tuple[tuple[str, str, str, str], ...]
    band: tuple[tuple[str, str, str], ...]
    gate: tuple[tuple[str, str, str], ...]

    @property
    def body(self) -> str:
        """输入：无。输出：待评材料那一段的原文（行之间用换行拼起来）。"""
        return "\n".join(self.body_lines)

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：待评材料那一段是不是空的。"""
        return not self.body.strip()


def _sections(state: str, loc: dict[str, JsonValue]) -> dict[str, list[str]]:
    """输入：state、语言字典。输出：`{段落名: 该段的行列表}`。

    预期行为：段落头是**前缀匹配**（后面可能跟着括号里的说明），段头那一行的括号说明
    单独收进该段、不当数据行；其余行只去尾部空白——材料里的 `- ` 是「分条」的凭据，
    去掉就解不出来了。
    """
    meta = loc["meta"]
    titles = tuple(str(meta[key]) for key in SECTION_KEYS)
    out: dict[str, list[str]] = {}
    current: str | None = None
    for raw in state.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        matched = next(
            (title for title in titles if stripped.startswith(locales.header(loc, title))), None
        )
        if matched is not None:
            current = matched
            out.setdefault(current, [])
            rest = stripped[len(locales.header(loc, matched)) :].strip()
            if rest:
                out[current].append(rest)
            continue
        if current is not None and stripped:
            out[current].append(line)
    return out


def _cells(line: str, loc: dict[str, JsonValue]) -> tuple[str, ...] | None:
    """输入：一行、语言字典。输出：拆好的格子；不是数据行就 None。

    预期行为：只认 `row_prefix` 开头的行，按 `CELL_SEP` 拆并把每格去空白。
    """
    prefix = str(loc["meta"]["row_prefix"])
    text = line.strip()
    if not text.startswith(prefix):
        return None
    rest = text[len(prefix) :].strip()
    return tuple(cell.strip() for cell in rest.split(locales.CELL_SEP))


def body_lines(state: object, loc: dict[str, JsonValue]) -> list[str]:
    """输入：state、语言字典。输出：待评材料那一段的每一行（非文本 state 给空列表）。"""
    if not isinstance(state, str):
        return []
    return list(_sections(state, loc).get(str(loc["meta"]["body_label"]), []))


def body_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：待评材料那一段的原文（没有就空串）。"""
    return "\n".join(body_lines(state, loc))


def manifest_of(state: object, loc: dict[str, JsonValue]) -> ManifestState | None:
    """输入：state、语言字典。输出：解析出来的清单行（解不出就 None）。

    预期行为：只认形如 `- medium | prose | none` 的三格行；词元不在取值表里就当解不出来
    ——判别器与审计会拿它跟注入特征逐格比对，串味会当场暴露。
    """
    if not isinstance(state, str):
        return None
    for line in _sections(state, loc).get(str(loc["meta"]["manifest_label"]), []):
        cells = _cells(line, loc)
        if cells is None:
            continue  # 段落头那一行（列名）不是数据行，跳过
        if len(cells) != MANIFEST_CELLS:
            return None
        length, form, data = cells
        if (
            length in features.LENGTHS
            and form in features.FORMS
            and data in features.DATAS
        ):
            return ManifestState(length=length, form=form, data=data)
        return None
    return None


def score_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, str, str, str], ...]:
    """输入：state、语言字典。输出：解析出来的评分单（解不出就空元组）。

    预期行为：只认四格行、首格是判据名；一行对不上就跳过（观测那一格是数字或
    `meta.count_mark`，两者都收——前者是印出来的，后者要读的人自己去材料里数）。
    """
    if not isinstance(state, str):
        return ()
    mark = str(loc["meta"]["count_mark"])
    out: list[tuple[str, str, str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["score_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != SCORE_CELLS:
            continue
        if cells[0] not in features.CRITERIA:
            continue
        if not all(cell.isdigit() and len(cell) == 1 for cell in cells[1:3]):
            continue
        if cells[3] != mark and not (cells[3].isdigit() and len(cells[3]) == 1):
            continue
        out.append(tuple(cells))
    return tuple(out)


def band_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, str, str], ...]:
    """输入：state、语言字典。输出：解析出来的档位表（解不出就空元组）。

    预期行为：只认三格行且首格是档位名、后两格都是 0~99 的整数；一行对不上就跳过。
    """
    if not isinstance(state, str):
        return ()
    out: list[tuple[str, str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["band_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != BAND_CELLS:
            continue
        if cells[0] not in features.BANDS:
            continue
        if not all(cell.isdigit() and len(cell) <= 2 for cell in cells[1:]):
            continue
        out.append(tuple(cells))
    return tuple(out)


def gate_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, str, str], ...]:
    """输入：state、语言字典。输出：解析出来的处置表（解不出就空元组）。

    预期行为：只认三格行且首格是档位名、后两格都是处置词元；一行对不上就跳过
    （否则审计会把假表当好表）。
    """
    if not isinstance(state, str):
        return ()
    out: list[tuple[str, str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["gate_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != GATE_CELLS:
            continue
        if cells[0] not in features.BANDS:
            continue
        if not all(cell in features.GATES for cell in cells[1:]):
            continue
        out.append(tuple(cells))
    return tuple(out)


def obs_from_score(
    rows: tuple[tuple[str, str, str, str], ...], body: str, marks: frozenset[str]
) -> dict[str, int] | None:
    """输入：解析出来的评分单行、材料正文、「按材料数」标记的集合（`count_marks()`）。
    输出：`{判据名: 观测}`。

    预期行为：观测那一格是数字就直接用；是**任一语言**的「按材料数」标记就按机械口径从
    材料里数出来（留空占位 / 出处标记）——一次 run 里语言按样本轮转，拿调用方给的那一种
    语言去认标记，会把另一种语言的产物判成「重算不出观测」。判据名不认识、五行凑不齐、
    标记与维度对不上，一律给 None——审计拿这个结果与产物里记下的答案比，给个「猜出来的
    观测」会让审计变成橡皮图章。
    """
    out: dict[str, int] = {}
    for name, _weight, _need, cell in rows:
        dim = features.OBS_DIM.get(name)
        if dim is None:
            return None
        if cell in marks:
            out[name] = count_obs(dim, body)
        elif cell.isdigit() and len(cell) == 1:
            out[name] = int(cell)
        else:
            return None
    if set(out) != set(features.CRITERIA):
        return None
    return out


def count_obs(dim: str, body: str) -> int:
    """输入：维度名、材料正文。输出：这一维从材料里数出来的观测。

    预期行为：只对**正文承载**的两维有意义；其余三维的观测是印在评分单上的，数不出来
    就直接抛 KeyError——静默返回一个数字会让「谁负责这一维」这件事糊掉。
    """
    if dim not in features.BODY_DIMS:
        raise KeyError(f"维度 {dim!r} 的观测印在评分单上，不该从材料里数")
    if dim == "placeholders":
        seen = "yes" if features.has_placeholder(body) else "no"
    else:
        seen = "yes" if features.has_citation(body) else "no"
    return features.OBS_TABLE[dim][seen]


def parse_state(state: object) -> ParsedRubric:
    """输入：state。输出：纯代码能反解的部分（材料 + 清单 + 三张表）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    empty = ParsedRubric(body_lines=(), manifest=None, score=(), band=(), gate=())
    if not isinstance(state, str):
        return empty
    for lang in locales.available():
        loc = locales.load(lang)
        body = body_lines(state, loc)
        manifest = manifest_of(state, loc)
        if body or manifest is not None:
            return ParsedRubric(
                body_lines=tuple(body),
                manifest=manifest,
                score=score_of(state, loc),
                band=band_of(state, loc),
                gate=gate_of(state, loc),
            )
    return empty
