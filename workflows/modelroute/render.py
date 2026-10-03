"""模型路由与降级的文本外壳：把候选目录与门槛摊平成可复核的 state，请求由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- **候选目录与筛选门槛由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写一句**用户请求**（这活儿只有模型干得出来：真实请求的语气与材料是长尾的）；
- 那句请求必须体现注入的「任务类型」，交给 `judges.TraceJudge` 反解一次再比对。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
**本模块不执行任何东西**——它只把文本当字符串读写。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc import locale_files
from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.state_parse import split_sections
from jc.workflow import DraftError

from . import locales
from .features import CATALOG, Candidate, Needs, TagSet

#: 提示词里特征的固定顺序：顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("task_kind", "size", "latency", "budget", "health")

#: Markdown 围栏残留：模型偶尔把整句话包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 表格行与门槛行的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"


class RouteDraft(BaseModel):
    """后端要填的草稿：**只有一句用户请求**。

    刻意只留一个字段——候选目录、门槛、档位全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    request: str = Field(
        description="用户想做的事：一句话，写清楚这是什么任务、涉及什么材料与大致规模",
    )


def build_prompt(tags: TagSet, loc: dict[str, JsonValue], phrasing: int) -> str:
    """组装提示词。

    输入：一组注入特征、语言字典、问法编号。
    输出：完整提示词（只描述特征，不含答案）。
    预期行为：特征按 `DIM_ORDER` 固定顺序逐行列出；规则整段附在后面，每条前面加短横线
    并把 `{language}` 填成语言名——留着占位符会把 `{language}` 原样发给模型。
    """
    return locales.tagged_prompt(loc, tags.as_dict(), DIM_ORDER, phrasing)


def build_task(tags: TagSet, loc: dict[str, JsonValue], task_id: str) -> GenerationTask:
    """输入：注入特征、语言字典、任务 id。输出：`GenerationTask`。

    预期行为：`response_model` 是 `RouteDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="model-routing",
        prompt=build_prompt(tags, loc, 0),
        response_model=RouteDraft,
        seeds=(),
        spec={"tags": tags.as_dict()},
    )


#: 本家 `label_leak` 查的键表：三段的行首标记都由代码写，模型再抄一遍就是重复标记。
LEAK_KEYS: Final[tuple[str, ...]] = ("request_label", "catalog_label", "needs_label")


def label_leak(text: str, loc: dict[str, JsonValue]) -> str:
    """输入：一段文本、语言字典。输出：文本开头误写的段落名（没有就空串）。

    预期行为：只认「段落名 + 冒号」开头这一种写法——正文里正常提到这些词不算。查段落名的
    机制只有一份，住在 `jc.locale_files.label_leak`（大小写不敏感的口径也在那里）。
    """
    return locale_files.label_leak(text, loc, LEAK_KEYS)


def check_draft(data: dict[str, JsonValue], loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、语言字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：请求那一行会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 非空、单行、不含 Markdown 围栏；
    - 开头不许再写一遍段落名（模型很爱这么干，写进去就是一行重复标记）；
    - 不许出现候选目录里的任何模型名——请求里出现模型名，等于把答案泄进题面。
    """
    request = str(data.get("request", "")).strip()
    if not request:
        raise DraftError("草稿的「请求」是空的")
    if FENCE_PATTERN.search(request):
        raise DraftError("草稿的「请求」里混进了 Markdown 围栏")
    if "\n" in request:
        raise DraftError("草稿的「请求」是多行——它必须是一句话")
    leaked = label_leak(request, loc)
    if leaked:
        raise DraftError(f"草稿的「请求」把段落名「{leaked}」也写进去了")
    lowered = request.lower()
    named = [cand.name for cand in CATALOG if cand.name in lowered]
    if named:
        raise DraftError(f"草稿的「请求」里出现了候选模型名：{'、'.join(named)}")


def catalog_line(cand: Candidate, loc: dict[str, JsonValue]) -> str:
    """输入：一行候选、语言字典。输出：表格里的一行（纯数据，中英一致）。"""
    meta = loc["meta"]
    cells = [
        cand.name,
        cand.provider,
        f"{cand.window}k",
        str(cand.ability),
        str(cand.price),
        str(cand.latency),
    ]
    return f"{meta['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def needs_line(needs: Needs, loc: dict[str, JsonValue]) -> str:
    """输入：门槛、语言字典。输出：门槛那一行（纯数据，中英一致）。"""
    meta = loc["meta"]
    down = ",".join(needs.down) if needs.down else str(meta["none_mark"])
    cells = [f"{needs.window}k", str(needs.ability), str(needs.latency), str(needs.price), down]
    return f"{meta['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def to_state(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`request`）、注入特征、语言字典。
    输出：三段文本——请求 / 候选模型表 / 筛选门槛。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    表与门槛**由代码写**（它们是注入特征，不是模型的发挥），所以从 `tags` 推出来。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [f"{header(loc, meta['request_label'])}{str(data['request']).strip()}"]
    lines.append(f"{header(loc, meta['catalog_label'])}{meta['paren'].format(text=meta['catalog_columns'])}")
    lines.extend(catalog_line(cand, loc) for cand in CATALOG)
    lines.append(f"{header(loc, meta['needs_label'])}{meta['paren'].format(text=meta['needs_columns'])}")
    lines.append(needs_line(tags.needs, loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class ParsedFrame:
    """从 state 里纯代码反解出来的事实。"""

    request: str
    catalog: tuple[Candidate, ...]
    needs: Needs | None

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：请求那一行是不是空的。"""
        return not self.request.strip()


#: 转出去：切段机制只有一份，住在 `jc.state_parse.split_sections`（11 家逐字同构，只留那一份）。
def _sections(state: str, loc: dict[str, JsonValue]) -> dict[str, list[str]]:
    """输入：state、语言字典。输出：`{段落名: 该段的行列表}`（行已去首尾空白）。"""
    return split_sections(state, loc, ("request_label", "catalog_label", "needs_label"))


def request_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：请求那一行（没有就空串）。"""
    lines = request_lines(state, loc)
    return lines[0] if lines else ""


def request_lines(state: object, loc: dict[str, JsonValue]) -> list[str]:
    """输入：state、语言字典。输出：请求那一段的行列表。

    预期行为：正常只有一行；模型把请求写成多行时这里会有多行——卫生判别器就靠这个
    认出「请求是多行」。
    """
    if not isinstance(state, str):
        return []
    return _sections(state, loc).get(str(loc["meta"]["request_label"]), [])


def catalog_of(state: str, loc: dict[str, JsonValue]) -> tuple[Candidate, ...]:
    """输入：state、语言字典。输出：解析出来的候选目录（解不出就空元组）。

    预期行为：只认形如 `- a | b | 200k | 3 | 3 | 3` 的行；解不出来的行直接跳过——
    判别器与审计会拿它跟目录表逐行比对，缺行会当场暴露。
    """
    meta = loc["meta"]
    out: list[Candidate] = []
    for line in _sections(state, loc).get(str(meta["catalog_label"]), []):
        if not line.startswith(str(meta["row_prefix"])):
            continue
        cells = [cell.strip() for cell in line[len(str(meta["row_prefix"])) :].split(CELL_SEP)]
        if len(cells) != 6:
            continue
        try:
            out.append(
                Candidate(
                    name=cells[0],
                    provider=cells[1],
                    window=int(cells[2].rstrip("kK")),
                    ability=int(cells[3]),
                    price=int(cells[4]),
                    latency=int(cells[5]),
                )
            )
        except ValueError:
            continue
    return tuple(out)


def needs_of(state: str, loc: dict[str, JsonValue]) -> Needs | None:
    """输入：state、语言字典。输出：解析出来的门槛（解不出就 None）。"""
    meta = loc["meta"]
    for line in _sections(state, loc).get(str(meta["needs_label"]), []):
        if not line.startswith(str(meta["row_prefix"])):
            continue
        cells = [cell.strip() for cell in line[len(str(meta["row_prefix"])) :].split(CELL_SEP)]
        if len(cells) != 5:
            continue
        mark = str(meta["none_mark"])
        try:
            return Needs(
                window=int(cells[0].rstrip("kK")),
                ability=int(cells[1]),
                latency=int(cells[2]),
                price=int(cells[3]),
                down=() if cells[4] in {mark, ""} else tuple(cells[4].split(",")),
            )
        except ValueError:
            return None
    return None


def parse_state(state: object) -> ParsedFrame:
    """输入：state。输出：纯代码能反解的部分（请求 + 候选目录 + 门槛）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    if not isinstance(state, str):
        return ParsedFrame(request="", catalog=(), needs=None)
    for lang in locales.available():
        loc = locales.load(lang)
        request = request_of(state, loc)
        catalog = catalog_of(state, loc)
        if request or catalog:
            return ParsedFrame(request=request, catalog=catalog, needs=needs_of(state, loc))
    return ParsedFrame(request="", catalog=(), needs=None)
