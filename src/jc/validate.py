"""契约校验：把一行样本映射为 0 或多个 Issue。

输入：解析后的 JSON 对象（dict）、可选行号、已见 id 集合。
输出：Issue 列表（空表示通过）；`validate_file` 汇总成报告。
预期行为：只报问题、不抛异常、不静默跳过；同一行可能同时报多个错误码。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from . import contract, jsonio

_PURE_INT_NAME: Final[re.Pattern[str]] = re.compile(r"^[+-]?[0-9]+$")


@dataclass(frozen=True)
class Issue:
    """一条校验问题。

    字段：line 行号（无则 None）、sample_id 样本 id（无则 None）、code 错误码、
    path JSON 路径、msg 人类可读说明。
    """

    code: str
    path: str
    msg: str
    line: int | None = None
    sample_id: str | None = None

    def __str__(self) -> str:
        where = f"第 {self.line} 行" if self.line is not None else "（无行号）"
        who = f" id={self.sample_id}" if self.sample_id else ""
        return f"{where}{who} [{self.code}] {self.path}: {self.msg}"


@dataclass
class ValidationReport:
    """文件级校验结果：总行数、通过数、问题列表、按错误码计数。"""

    total: int = 0
    passed: int = 0
    issues: list[Issue] = field(default_factory=list)
    parse_issues: list[Issue] = field(default_factory=list)

    @property
    def failed(self) -> int:
        """校验未通过的行数（含解析失败的行）。"""
        return self.total - self.passed

    def code_counts(self) -> dict[str, int]:
        """按错误码统计条数（含解析类错误）。"""
        counter: Counter[str] = Counter(i.code for i in self.issues)
        counter.update(i.code for i in self.parse_issues)
        return dict(sorted(counter.items()))

    def all_issues(self) -> list[Issue]:
        """解析问题在前、契约问题在后，便于按顺序阅读。"""
        return [*self.parse_issues, *self.issues]


def _is_pure_int_name(name: str) -> bool:
    """选项名是否是纯整数字符串（"0"、"12"、"-3" 一律算）。

    输入：选项名。
    输出：bool。预期行为：与编号混排时会歧义的名字返回 True。
    """
    return bool(_PURE_INT_NAME.match(name))


def _instructions_ok(value: object) -> bool:
    """instructions 是否合法：非空 string（去空白后）/ 非空 object / 非空 array。"""
    kind = contract.json_kind(value)
    if kind == "string":
        return bool(str(value).strip())
    if kind == "object":
        return bool(value)
    if kind == "array":
        return bool(value)
    return False


def _state_ok(value: object) -> bool:
    """state 是否合法：非空 string（去空白后）/ 非空 object / 非空 array。"""
    return _instructions_ok(value)


def validate_sample(
    obj: object,
    *,
    line: int | None = None,
    seen_ids: set[str] | None = None,
) -> list[Issue]:
    """校验一行样本。

    输入：解析后的对象、可选行号、已见 id 集合（用于 E_ID_DUP，调用方需先查后加）。
    输出：Issue 列表，空表示通过。
    预期行为：顶层、题目、答案三层逐层检查；题目本身非法时跳过其答案检查，避免噪声。
    """
    issues: list[Issue] = []
    sample_id: str | None = None

    def add(code: str, path: str, msg: str) -> None:
        issues.append(Issue(code=code, path=path, msg=msg, line=line, sample_id=sample_id))

    if contract.json_kind(obj) != "object":
        add(contract.E_ROW_NOT_OBJECT, "$", f"每行必须是 JSON 对象，实际是 {contract.json_kind(obj)}")
        return issues

    row: dict[str, JsonValue] = obj  # type: ignore[assignment]

    # ---------- 顶层字段 ----------
    if isinstance(row.get("id"), str):
        sample_id = row["id"]
    for name in contract.TOP_FIELDS:
        if name not in row:
            add(contract.E_FIELD_MISSING, "$", f"缺少顶层字段 {name}")
    for name in row:
        if name not in contract.TOP_FIELDS:
            add(contract.E_FIELD_UNKNOWN, f"$.{name}", f"顶层不允许字段 {name}")

    raw_id = row.get("id")
    if isinstance(raw_id, str):
        if not contract.ID_PATTERN.match(raw_id):
            add(contract.E_ID_INVALID, "$.id", f"id 不符合 {{任务}}-{{切分}}-{{序号:07d}}：{raw_id!r}")
        elif seen_ids is not None and raw_id in seen_ids:
            add(contract.E_ID_DUP, "$.id", f"id 重复：{raw_id}")
    elif raw_id is not None:
        add(contract.E_ID_INVALID, "$.id", f"id 必须是字符串，实际是 {contract.json_kind(raw_id)}")

    if "state" in row and not _state_ok(row["state"]):
        add(contract.E_STATE_INVALID, "$.state", f"state 必须是非空 string/object/array，实际是 {contract.json_kind(row['state'])}")

    # ---------- 题目 ----------
    qtypes: dict[str, str] = {}
    questions = row.get("questions")
    if questions is None:
        pass  # 已由 E_FIELD_MISSING 报过
    elif contract.json_kind(questions) != "object":
        add(contract.E_QUESTIONS_EMPTY, "$.questions", f"questions 必须是对象，实际是 {contract.json_kind(questions)}")
    elif not questions:
        add(contract.E_QUESTIONS_EMPTY, "$.questions", "questions 至少要有 1 题")
    else:
        for qid, question in questions.items():
            qpath = f"$.questions.{qid}"
            if not isinstance(qid, str) or not qid.strip():
                add(contract.E_QUESTIONS_EMPTY, qpath, "题号必须是非空字符串")
                continue
            if contract.json_kind(question) != "object":
                add(contract.E_Q_TYPE, qpath, f"题目必须是对象，实际是 {contract.json_kind(question)}")
                continue
            qtype = question.get("type")
            if qtype not in contract.QUESTION_TYPES:
                add(contract.E_Q_TYPE, f"{qpath}.type", f"type 必须是 choice/noul/score，实际是 {qtype!r}")
                continue
            qtypes[qid] = qtype
            for name in question:
                if name not in contract.QUESTION_FIELDS:
                    add(contract.E_Q_FIELD_UNKNOWN, f"{qpath}.{name}", f"题目不允许字段 {name}")
            if "instructions" not in question:
                add(contract.E_FIELD_MISSING, qpath, "题目缺少 instructions")
            elif not _instructions_ok(question["instructions"]):
                add(contract.E_INSTRUCTIONS_INVALID, f"{qpath}.instructions", "instructions 必须是非空 string/object/array")
            _validate_criteria(qtype, question, qpath, add)

    # ---------- 答案 ----------
    answers = row.get("answers")
    if answers is None:
        pass
    elif contract.json_kind(answers) != "object":
        add(contract.E_ANSWERS_KEYSET, "$.answers", f"answers 必须是对象，实际是 {contract.json_kind(answers)}")
    else:
        if isinstance(questions, dict):
            qkeys, akeys = set(questions), set(answers)
            for missing in sorted(qkeys - akeys):
                add(contract.E_ANSWERS_KEYSET, f"$.answers.{missing}", "answers 缺少该题号")
            for extra in sorted(akeys - qkeys):
                add(contract.E_ANSWERS_KEYSET, f"$.answers.{extra}", "answers 多出该题号")
        for qid, answer in answers.items():
            apath = f"$.answers.{qid}"
            qtype = qtypes.get(qid)
            if qtype is None:
                continue  # 题目非法时不再叠加答案噪声
            if contract.json_kind(answer) != "object":
                add(contract.E_ANSWER_SHAPE, apath, f"答案必须是对象，实际是 {contract.json_kind(answer)}")
                continue
            allowed = contract.ANSWER_FIELDS[qtype]
            for name in answer:
                if name not in allowed:
                    add(contract.E_ANSWER_SHAPE, f"{apath}.{name}", f"{qtype} 答案不允许字段 {name}")
            if answer.get("type") != qtype:
                add(contract.E_ANSWER_TYPE_MISMATCH, f"{apath}.type", f"答案 type 必须与题目一致（{qtype}）")
            _validate_answer_value(qtype, answer, apath, qid, questions, add)

    return issues


def _validate_criteria(qtype: str, question: dict[str, JsonValue], qpath: str, add: object) -> None:
    """按题型校验 criteria。

    输入：题型、题目对象、题目路径、Issue 追加函数。
    输出：无（通过 add 追加问题）。
    预期行为：choice 要求 2~255 个选项且选项名合法；score 要求 2~10 个非空等级；noul 只允许 true/false。
    """
    criteria = question.get("criteria")
    if qtype == contract.TYPE_CHOICE:
        if criteria is None:
            add(contract.E_FIELD_MISSING, qpath, "choice 缺少 criteria")
            return
        if contract.json_kind(criteria) != "object":
            add(contract.E_CHOICE_CRITERIA_TYPE, f"{qpath}.criteria", f"choice 的 criteria 必须是对象，实际是 {contract.json_kind(criteria)}")
            return
        count = len(criteria)
        if not (contract.CHOICE_MIN_OPTIONS <= count <= contract.CHOICE_MAX_OPTIONS):
            add(contract.E_CHOICE_OPTION_COUNT, f"{qpath}.criteria", f"选项数必须在 {contract.CHOICE_MIN_OPTIONS}~{contract.CHOICE_MAX_OPTIONS}，实际 {count}")
        lowered: set[str] = set()
        for name, desc in criteria.items():
            opath = f"{qpath}.criteria.{name}"
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                add(contract.E_CHOICE_OPTION_NAME, opath, f"选项名必须是非空且首尾无空白的字符串：{name!r}")
                continue
            if _is_pure_int_name(name):
                add(contract.E_CHOICE_OPTION_NAME, opath, f"选项名不能是纯整数字符串（会与编号混淆）：{name!r}")
            key = name.strip().lower()
            if key in lowered:
                add(contract.E_CHOICE_OPTION_DUP, opath, f"选项名去空白转小写后重复：{name!r}")
            lowered.add(key)
            if not contract.desc_kind_ok(desc):
                add(contract.E_DESC_TYPE, opath, f"描述值必须是 string/object/array/null，实际是 {contract.json_kind(desc)}")
    elif qtype == contract.TYPE_SCORE:
        if criteria is None:
            add(contract.E_FIELD_MISSING, qpath, "score 缺少 criteria")
            return
        if contract.json_kind(criteria) != "array":
            add(contract.E_SCORE_CRITERIA_TYPE, f"{qpath}.criteria", f"score 的 criteria 必须是数组，实际是 {contract.json_kind(criteria)}")
            return
        count = len(criteria)
        if not (contract.SCORE_MIN_LEVELS <= count <= contract.SCORE_MAX_LEVELS):
            add(contract.E_SCORE_LEVEL_COUNT, f"{qpath}.criteria", f"等级数必须在 {contract.SCORE_MIN_LEVELS}~{contract.SCORE_MAX_LEVELS}，实际 {count}")
        for idx, level in enumerate(criteria):
            lpath = f"{qpath}.criteria[{idx}]"
            if level is None:
                add(contract.E_SCORE_LEVEL_NULL, lpath, "score 等级不允许 null（等级描述就是等级定义）")
            elif not contract.desc_kind_ok(level):
                add(contract.E_DESC_TYPE, lpath, f"等级描述必须是 string/object/array，实际是 {contract.json_kind(level)}")
    else:  # noul
        if criteria is None:
            return
        if contract.json_kind(criteria) != "object":
            add(contract.E_DESC_TYPE, f"{qpath}.criteria", f"noul 的 criteria 必须是对象，实际是 {contract.json_kind(criteria)}")
            return
        for key, desc in criteria.items():
            if key not in ("true", "false"):
                add(contract.E_NOUL_CRITERIA_KEY, f"{qpath}.criteria.{key}", f"noul 的 criteria 只允许 true/false，实际 {key!r}")
            if not contract.desc_kind_ok(desc):
                add(contract.E_DESC_TYPE, f"{qpath}.criteria.{key}", f"描述值必须是 string/object/array/null，实际是 {contract.json_kind(desc)}")


def _validate_answer_value(
    qtype: str,
    answer: dict[str, JsonValue],
    apath: str,
    qid: str,
    questions: object,
    add: object,
) -> None:
    """按题型校验答案取值。

    输入：题型、答案对象、答案路径、题号、questions 对象、Issue 追加函数。
    输出：无（通过 add 追加问题）。
    预期行为：choice/score 的编号必须是 int 且在 [0, len(criteria))；noul 必须是 JSON 布尔。
    """
    if qtype == contract.TYPE_NOUL:
        if contract.json_kind(answer.get("noul")) != "boolean":
            add(contract.E_NOUL_NOT_BOOL, f"{apath}.noul", f"noul 答案必须是 JSON 布尔，实际是 {contract.json_kind(answer.get('noul'))}")
        return

    field = "choice" if qtype == contract.TYPE_CHOICE else "score"
    value = answer.get(field)
    if not contract.is_index(value):
        add(contract.E_INDEX_NOT_INT, f"{apath}.{field}", f"编号必须是整数（不接受布尔/浮点），实际是 {contract.json_kind(value)}")
        return
    if not isinstance(questions, dict):
        return
    question = questions.get(qid)
    if not isinstance(question, dict):
        return
    criteria = question.get("criteria")
    if contract.json_kind(criteria) in ("object", "array"):
        size = len(criteria)
        if not (0 <= int(value) < size):
            add(contract.E_INDEX_RANGE, f"{apath}.{field}", f"编号必须在 [0, {size})，实际 {value}")


def validate_file(path: str | Path, *, max_issues: int = 0) -> ValidationReport:
    """校验一个 JSONL 文件。

    输入：文件路径；max_issues>0 时只收集前 N 条问题（用于快速预览）。
    输出：ValidationReport（总行数、通过数、问题列表、错误码计数）。
    预期行为：解析失败的行计入 total 与 failed，并带行号记录；不中断整个文件的校验。
    """
    report = ValidationReport()
    seen_ids: set[str] = set()
    rows: Iterator[tuple[int, object]] = _safe_iter(path, report)
    for lineno, obj in rows:
        report.total += 1
        issues = validate_sample(obj, line=lineno, seen_ids=seen_ids)
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            seen_ids.add(obj["id"])
        if issues:
            report.issues.extend(issues)
            if max_issues and len(report.issues) >= max_issues:
                break
        else:
            report.passed += 1
    return report


def _safe_iter(path: str | Path, report: ValidationReport) -> Iterator[tuple[int, object]]:
    """迭代 JSONL，把解析错误记进报告并继续（坏行用 (行号, None) 占位）。

    输入：文件路径、报告对象。
    输出：迭代器 (行号, 对象)。
    预期行为：单行解析失败不打断整个文件；BOM 等文件级错误记为第 0 行问题。
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        report.parse_issues.append(Issue(code=contract.E_JSON_PARSE, path="$", msg=f"不是合法 UTF-8：{exc}"))
        return
    if raw.startswith(b"\xef\xbb\xbf"):
        report.parse_issues.append(Issue(code=contract.E_BOM, path="$", msg="文件带 UTF-8 BOM，必须去掉"))
        text = text.lstrip("\ufeff")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield lineno, jsonio.loads(line)
        except jsonio.StrictJsonError as exc:
            report.total += 1
            report.parse_issues.append(Issue(code=exc.code, path="$", msg=exc.msg, line=lineno))


def validate_rows(rows: Iterable[object]) -> list[Issue]:
    """校验内存中的一批样本（测试与管线内部用）。

    输入：可迭代的样本对象。
    输出：Issue 列表。预期行为：行号从 1 开始，重复 id 会被识别。
    """
    issues: list[Issue] = []
    seen_ids: set[str] = set()
    for lineno, obj in enumerate(rows, start=1):
        issues.extend(validate_sample(obj, line=lineno, seen_ids=seen_ids))
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            seen_ids.add(obj["id"])
    return issues
