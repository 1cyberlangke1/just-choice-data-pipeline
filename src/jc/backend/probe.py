"""端点能力探测（PLAN 3.1.1）。

开机对每个 provider 探四项：`plain` / `json_object` / `json_schema` / `tools`。
结果**只决定从哪一档起步**，不替代本地校验——探测返 200 不等于服务端真强制了 schema。

实现上直接用 `openai` 客户端发裸请求（不走 instructor），因为要看的恰恰是
「服务端吃不吃这些参数」；`max_retries=0`，探测要快要干脆，不重试。

输入/输出：`probe_capabilities()` 收端点信息，出 `BackendCapabilities`。
预期行为：任何一项失败都不抛异常，只记一条 notes；**不回显 key**。
"""

from __future__ import annotations

from typing import Any

import openai

from .types import BackendCapabilities

# 探测用的最小 schema：一个字符串字段。越简单越能测出「吃不吃这个参数」。
PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

# 探测用的最小工具定义。
PROBE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "emit_answer",
        "description": "Emit the final answer.",
        "parameters": PROBE_SCHEMA,
    },
}


def _attempt(client: openai.OpenAI, model: str, label: str, notes: list[str], **extra: Any) -> bool:
    """输入：客户端、模型、探针名、notes 列表、额外请求参数。输出：这一项是否可用。

    预期行为：成功返回 True；失败返回 False 并把原因（**不含 key**）追加到 notes。
    """
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the required format."}],
            **extra,
        )
    except Exception as exc:  # noqa: BLE001 —— 探测就是要看它抛什么
        notes.append(f"{label}: 不可用（{type(exc).__name__}）")
        return False
    notes.append(f"{label}: 可用")
    return True


def probe_capabilities(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float = 60.0,
) -> BackendCapabilities:
    """输入：端点地址、key、模型名、超时。输出：四项能力探测结果。

    预期行为：四项各自独立，互不连坐；**永不发 `max_tokens`**（PLAN 3.1.2）。
    """
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
    notes: list[str] = []
    plain = _attempt(client, model, "plain", notes)
    json_object = _attempt(
        client, model, "json_object", notes, response_format={"type": "json_object"}
    )
    json_schema = _attempt(
        client,
        model,
        "json_schema",
        notes,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": PROBE_SCHEMA},
        },
    )
    tools = _attempt(client, model, "tools", notes, tools=[PROBE_TOOL], tool_choice="auto")
    return BackendCapabilities(
        plain=plain,
        json_object=json_object,
        json_schema=json_schema,
        tools=tools,
        notes=tuple(notes),
    )
