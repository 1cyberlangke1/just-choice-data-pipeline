"""契约层与严格读写测试（通用层，与具体 workflow 无关）。

覆盖：严格 JSON 的五个坑、写文件的 LF 约定、契约校验的正例与各类负例（每个错误码至少一次）。
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from jc import contract, jsonio, validate
from jc.jsontypes import JsonValue


def good_sample() -> dict:
    """一份合法样本（各负例都由它变形而来）。"""
    return {
        "id": "maze-train-0000001",
        "state": "state text",
        "questions": {
            "next_step": {
                "type": "choice",
                "instructions": "go where?",
                "criteria": {"up": None, "right": "to the exit"},
            },
            "is_far": {"type": "noul", "instructions": "far?"},
            "size": {
                "type": "score",
                "instructions": "how big?",
                "criteria": ["small", "medium", "large"],
            },
        },
        "answers": {
            "next_step": {"type": "choice", "choice": 1},
            "is_far": {"type": "noul", "noul": True},
            "size": {"type": "score", "score": 2},
        },
    }


def codes(issues: list[validate.Issue]) -> set[str]:
    """把 Issue 列表压成错误码集合，便于断言。"""
    return {issue.code for issue in issues}


def test_good_sample_passes() -> None:
    assert validate.validate_sample(good_sample()) == []


def test_strict_json_rejects_duplicate_key() -> None:
    with pytest.raises(jsonio.StrictJsonError) as exc:
        jsonio.loads('{"a": 1, "a": 2}')
    assert exc.value.code == contract.E_JSON_DUP_KEY


def test_strict_json_rejects_nan_and_infinity() -> None:
    for text in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
        with pytest.raises(jsonio.StrictJsonError) as exc:
            jsonio.loads(text)
        assert exc.value.code == contract.E_JSON_NAN


def test_strict_json_rejects_bom_and_bad_utf8() -> None:
    with pytest.raises(jsonio.StrictJsonError) as exc:
        jsonio.loads_bytes(b"\xef\xbb\xbf{}")
    assert exc.value.code == contract.E_BOM
    with pytest.raises(jsonio.StrictJsonError) as exc:
        jsonio.loads_bytes(b"\xff\xfe{}")
    assert exc.value.code == contract.E_JSON_PARSE


def test_write_jsonl_uses_lf_and_keeps_key_order(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    jsonio.write_jsonl(path, [{"b": 1, "a": "中文"}])
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8")) == {"b": 1, "a": "中文"}
    assert raw.decode("utf-8").index('"b"') < raw.decode("utf-8").index('"a"')


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda s: s.pop("answers"), contract.E_FIELD_MISSING),
        (lambda s: s.update(extra=1), contract.E_FIELD_UNKNOWN),
        (lambda s: s.update(id="bad-id"), contract.E_ID_INVALID),
        (lambda s: s.update(state="   "), contract.E_STATE_INVALID),
        (lambda s: s.update(questions={}), contract.E_QUESTIONS_EMPTY),
        (lambda s: s["questions"]["next_step"].update(type="pick"), contract.E_Q_TYPE),
        (lambda s: s["questions"]["next_step"].update(note="x"), contract.E_Q_FIELD_UNKNOWN),
        (lambda s: s["questions"]["next_step"].update(instructions=""), contract.E_INSTRUCTIONS_INVALID),
        (lambda s: s["questions"]["next_step"].update(criteria=["a", "b"]), contract.E_CHOICE_CRITERIA_TYPE),
        (lambda s: s["questions"]["next_step"].update(criteria={"a": None}), contract.E_CHOICE_OPTION_COUNT),
        (lambda s: s["questions"]["next_step"]["criteria"].update({"  x  ": None}), contract.E_CHOICE_OPTION_NAME),
        (lambda s: s["questions"]["next_step"]["criteria"].update({"12": None}), contract.E_CHOICE_OPTION_NAME),
        (lambda s: s["questions"]["next_step"]["criteria"].update({"UP": None}), contract.E_CHOICE_OPTION_DUP),
        (lambda s: s["questions"]["next_step"]["criteria"].update({"z": 3}), contract.E_DESC_TYPE),
        (lambda s: s["questions"]["is_far"].update(criteria={"yes": "y"}), contract.E_NOUL_CRITERIA_KEY),
        (lambda s: s["questions"]["size"].update(criteria={"a": 1}), contract.E_SCORE_CRITERIA_TYPE),
        (lambda s: s["questions"]["size"].update(criteria=["only"]), contract.E_SCORE_LEVEL_COUNT),
        (lambda s: s["questions"]["size"].update(criteria=["a", None]), contract.E_SCORE_LEVEL_NULL),
        (lambda s: s["answers"].pop("size"), contract.E_ANSWERS_KEYSET),
        (lambda s: s["answers"].update(ghost={"type": "noul", "noul": True}), contract.E_ANSWERS_KEYSET),
        (lambda s: s["answers"]["size"].update(extra=1), contract.E_ANSWER_SHAPE),
        (lambda s: s["answers"]["size"].update(type="noul"), contract.E_ANSWER_TYPE_MISMATCH),
        (lambda s: s["answers"]["size"].update(score=1.0), contract.E_INDEX_NOT_INT),
        (lambda s: s["answers"]["size"].update(score=True), contract.E_INDEX_NOT_INT),
        (lambda s: s["answers"]["size"].update(score=3), contract.E_INDEX_RANGE),
        (lambda s: s["answers"]["is_far"].update(noul=1), contract.E_NOUL_NOT_BOOL),
    ],
)
def test_negative_cases_report_expected_code(
    mutate: Callable[[dict[str, JsonValue]], None], expected: str
) -> None:
    sample = good_sample()
    mutate(sample)
    assert expected in codes(validate.validate_sample(sample))


def test_duplicate_id_is_reported() -> None:
    rows = [good_sample(), copy.deepcopy(good_sample())]
    assert contract.E_ID_DUP in codes(validate.validate_rows(rows))


def test_row_must_be_object() -> None:
    assert contract.E_ROW_NOT_OBJECT in codes(validate.validate_sample([1, 2]))


def test_validate_file_counts_and_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    bad = good_sample()
    bad["id"] = "nope"
    jsonio.write_jsonl(path, [good_sample(), bad])
    report = validate.validate_file(path)
    assert (report.total, report.passed, report.failed) == (2, 1, 1)
    assert report.code_counts() == {contract.E_ID_INVALID: 1}
    assert report.issues[0].line == 2


def test_validate_file_records_parse_errors(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_bytes(b'{"id": "x"}\n{oops\n')
    report = validate.validate_file(path)
    assert report.total == 2
    assert contract.E_JSON_PARSE in report.code_counts()


def test_issue_text_carries_line_id_code_and_path(tmp_path: Path) -> None:
    """输入：一行 id 非法的样本。输出：断言结果。

    预期行为：报错文本必须含**行号 + id + 错误码 + JSON 路径**（PLAN 1.6 定的报错格式），
    这样格式 case 的负例才能靠「文件名即期望」去核对。
    """
    path = tmp_path / "d.jsonl"
    bad = good_sample()
    bad["id"] = "nope"
    jsonio.write_jsonl(path, [good_sample(), bad])
    text = str(validate.validate_file(path).issues[0])
    assert "第 2 行" in text
    assert "id=nope" in text
    assert f"[{contract.E_ID_INVALID}]" in text
    assert "$.id" in text
