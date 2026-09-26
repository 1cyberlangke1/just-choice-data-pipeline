"""检索 workflow 的渲染：把真值规格写成一页「检索结果」。

输入：`SearchSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「请求头 → 结果条数行 → 每条两行（标题行 + 摘要行）」：
标题行以 `[编号]` 开头、摘要行以摘要标签开头，两种语言共用一套反解。
"""

from __future__ import annotations

from typing import Any

from .spec import SearchSpec


def topic_fields(topic_name: str, loc: dict[str, Any]) -> dict[str, str]:
    """输入：主题键、语言字典。输出：标题 / 摘要模板要用的占位符取值。

    预期行为：同时给出 `label/p1/p2/metric/v1/v2` 与别名 `other/op1/op2`——
    「跑题」的那几条用的模板写的是 `{other}` / `{op1}`，别名一铺，
    渲染器就不必为跑题项单开一条分支（别名就是它自己的主题名与要点）。
    """
    entry = loc["topic"][topic_name]
    points = entry["points"]
    values = entry["values"]
    return {
        "label": entry["label"],
        "p1": points[0],
        "p2": points[1],
        "metric": entry["metric"],
        "v1": values[0],
        "v2": values[1],
        "other": entry["label"],
        "op1": points[0],
        "op2": points[1],
    }


def query_text(spec: SearchSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：用户问题那句话（问法由 case 决定）。

    预期行为：模板里**一定**嵌了主题名与两个要点词——判别器靠「问题里出现了哪个主题名」
    认出问题问的是什么，靠「摘要里出现了几个要点词」算覆盖度。
    """
    styles = loc["query"]["tpl"]
    return styles[spec.case.query_style % len(styles)].format(**topic_fields(spec.topic.name, loc))


def build(spec: SearchSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整页文本。

    预期行为：标题行的五个字段用语言文件里的分隔符连起来（反解按它切）；
    更新日期那一格带标签，反解按标签剥掉前缀。
    """
    meta = loc["meta"]
    colon = meta["colon"]
    sep = meta["field_sep"]
    fields = topic_fields(spec.topic.name, loc)
    points = f"{fields['p1']}{meta['list_sep']}{fields['p2']}"
    lines = [
        meta["title"],
        f"{meta['query_label']}{colon}{query_text(spec, loc)}",
        f"{meta['kind_label']}{colon}{loc['doc_kind'][spec.want_kind]}",
        f"{meta['since_label']}{colon}{spec.since}",
        f"{meta['points_label']}{colon}{points}",
        f"{meta['results_label']}{colon}",
    ]
    for row in spec.results:
        row_fields = topic_fields(row.topic, loc)
        title = loc["title"][row.title]["text"].format(**row_fields)
        snippet = loc["snippet"][row.snippet]["text"].format(**row_fields)
        head = sep.join(
            (
                f"[{row.index}] {row.doc_id}",
                title,
                loc["doc_kind"][row.kind],
                loc["authority"][row.authority],
                f"{meta['updated_label']} {row.updated}",
            )
        )
        lines.append(f"  {head}")
        lines.append(f"      {meta['snippet_label']}{colon}{snippet}")
    return "\n".join(lines)
