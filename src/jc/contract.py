"""契约层：常量、允许值类型表、错误码。

职责：JSONL 契约的唯一真源，供校验器、格式 case 枚举器、组装器共用。
输入/输出：只有常量与纯函数，无副作用、不含任何领域词表。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import json
import re
from typing import Final

# ---------- 顶层字段（只允许这四个，多出即拒绝） ----------
TOP_FIELDS: Final[tuple[str, ...]] = ("id", "state", "questions", "answers")

# ---------- 题型 ----------
TYPE_CHOICE: Final[str] = "choice"
TYPE_NOUL: Final[str] = "noul"
TYPE_SCORE: Final[str] = "score"
QUESTION_TYPES: Final[tuple[str, ...]] = (TYPE_CHOICE, TYPE_NOUL, TYPE_SCORE)

# 每道题只允许这三个字段
QUESTION_FIELDS: Final[tuple[str, ...]] = ("type", "instructions", "criteria")

# 每种答案只允许 type + 一个值字段
ANSWER_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    TYPE_CHOICE: ("type", "choice"),
    TYPE_NOUL: ("type", "noul"),
    TYPE_SCORE: ("type", "score"),
}

# ---------- 数量约束 ----------
CHOICE_MIN_OPTIONS: Final[int] = 2
CHOICE_MAX_OPTIONS: Final[int] = 255
SCORE_MIN_LEVELS: Final[int] = 2
SCORE_MAX_LEVELS: Final[int] = 10

# ---------- 描述值允许的形态（官方 EntryType 的本地化版本） ----------
DESC_KINDS: Final[tuple[str, ...]] = ("string", "object", "array", "null")

# ---------- 切分 ----------
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test_iid", "test_ood")

# id 规则：{任务}-{切分}-{序号:07d}，如 foo-train-0000042
ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9_]*-(?:" + "|".join(SPLITS) + r")-\d{7}$"
)

# ---------- 错误码 ----------
E_BOM: Final[str] = "E_BOM"
E_JSON_PARSE: Final[str] = "E_JSON_PARSE"
E_JSON_DUP_KEY: Final[str] = "E_JSON_DUP_KEY"
E_JSON_NAN: Final[str] = "E_JSON_NAN"
E_ROW_NOT_OBJECT: Final[str] = "E_ROW_NOT_OBJECT"

E_FIELD_MISSING: Final[str] = "E_FIELD_MISSING"
E_FIELD_UNKNOWN: Final[str] = "E_FIELD_UNKNOWN"
E_ID_INVALID: Final[str] = "E_ID_INVALID"
E_ID_DUP: Final[str] = "E_ID_DUP"
E_STATE_INVALID: Final[str] = "E_STATE_INVALID"
E_QUESTIONS_EMPTY: Final[str] = "E_QUESTIONS_EMPTY"

E_Q_TYPE: Final[str] = "E_Q_TYPE"
E_Q_FIELD_UNKNOWN: Final[str] = "E_Q_FIELD_UNKNOWN"
E_INSTRUCTIONS_INVALID: Final[str] = "E_INSTRUCTIONS_INVALID"
E_DESC_TYPE: Final[str] = "E_DESC_TYPE"

E_CHOICE_CRITERIA_TYPE: Final[str] = "E_CHOICE_CRITERIA_TYPE"
E_CHOICE_OPTION_COUNT: Final[str] = "E_CHOICE_OPTION_COUNT"
E_CHOICE_OPTION_NAME: Final[str] = "E_CHOICE_OPTION_NAME"
E_CHOICE_OPTION_DUP: Final[str] = "E_CHOICE_OPTION_DUP"

E_NOUL_CRITERIA_KEY: Final[str] = "E_NOUL_CRITERIA_KEY"

E_SCORE_CRITERIA_TYPE: Final[str] = "E_SCORE_CRITERIA_TYPE"
E_SCORE_LEVEL_COUNT: Final[str] = "E_SCORE_LEVEL_COUNT"
E_SCORE_LEVEL_NULL: Final[str] = "E_SCORE_LEVEL_NULL"

E_ANSWERS_KEYSET: Final[str] = "E_ANSWERS_KEYSET"
E_ANSWER_SHAPE: Final[str] = "E_ANSWER_SHAPE"
E_ANSWER_TYPE_MISMATCH: Final[str] = "E_ANSWER_TYPE_MISMATCH"
E_INDEX_NOT_INT: Final[str] = "E_INDEX_NOT_INT"
E_INDEX_RANGE: Final[str] = "E_INDEX_RANGE"
E_NOUL_NOT_BOOL: Final[str] = "E_NOUL_NOT_BOOL"

ERROR_CODES: Final[tuple[str, ...]] = (
    E_BOM,
    E_JSON_PARSE,
    E_JSON_DUP_KEY,
    E_JSON_NAN,
    E_ROW_NOT_OBJECT,
    E_FIELD_MISSING,
    E_FIELD_UNKNOWN,
    E_ID_INVALID,
    E_ID_DUP,
    E_STATE_INVALID,
    E_QUESTIONS_EMPTY,
    E_Q_TYPE,
    E_Q_FIELD_UNKNOWN,
    E_INSTRUCTIONS_INVALID,
    E_DESC_TYPE,
    E_CHOICE_CRITERIA_TYPE,
    E_CHOICE_OPTION_COUNT,
    E_CHOICE_OPTION_NAME,
    E_CHOICE_OPTION_DUP,
    E_NOUL_CRITERIA_KEY,
    E_SCORE_CRITERIA_TYPE,
    E_SCORE_LEVEL_COUNT,
    E_SCORE_LEVEL_NULL,
    E_ANSWERS_KEYSET,
    E_ANSWER_SHAPE,
    E_ANSWER_TYPE_MISMATCH,
    E_INDEX_NOT_INT,
    E_INDEX_RANGE,
    E_NOUL_NOT_BOOL,
)


def json_kind(value: object) -> str:
    """返回 JSON 层面的类型名。

    输入：任意 Python 值。
    输出：'null' / 'boolean' / 'integer' / 'number' / 'string' / 'array' / 'object'。
    预期行为：bool 优先于 int 判定（isinstance(True, int) 为真，必须先排除）。
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def is_index(value: object) -> bool:
    """判断是否为合法编号：必须是 int 且不是 bool（True/False 不是编号）。

    输入：任意 Python 值。
    输出：bool。预期行为：1.0、True、"1" 一律返回 False。
    """
    return isinstance(value, int) and not isinstance(value, bool)


def desc_kind_ok(value: object) -> bool:
    """判断描述值形态是否在允许集合内（string / object / array / null）。

    输入：任意 Python 值。
    输出：bool。预期行为：数字、布尔一律 False。
    """
    return json_kind(value) in DESC_KINDS


def gold_text(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> str:
    """把金答案还原成**文字**（人看、模型看、比对都用它）。

    输入：一道题、它的答案。输出：选项 / 等级的文字，或 noul 的文案。
    预期行为：
    - 编号越界给 `?`——越界是校验器该报的错，这里只保证调用方不炸；
    - noul 的 criteria 是可选的，没有文案就退回布尔字面量；
    - 认不出的形态原样给 JSON，总比给个空字符串强。
    """
    criteria = question.get("criteria")
    kind = answer.get("type")
    if kind == TYPE_CHOICE and isinstance(criteria, dict):
        values = list(criteria.values())
        index = answer.get("choice")
        return str(values[index]) if is_index(index) and 0 <= index < len(values) else "?"
    if kind == TYPE_SCORE and isinstance(criteria, list):
        index = answer.get("score")
        return str(criteria[index]) if is_index(index) and 0 <= index < len(criteria) else "?"
    if kind == TYPE_NOUL:
        value = bool(answer.get("noul"))
        if isinstance(criteria, dict):
            return str(criteria.get("true" if value else "false", value))
        return str(value)
    return json.dumps(answer, ensure_ascii=False)
