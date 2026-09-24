"""契约格式 case 库的验收测试（通用层，与具体 workflow 无关）。

输入：仓库根 `cases/format/` 下的文件。输出：断言结果。
预期行为：
- `valid/*.jsonl` 每一行都必须通过契约校验；
- `invalid/<错误码>.jsonl` 报出的错误码**恰好**是文件名那个（文件名即期望，PLAN 1.6）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import contract, cover, format_cases, validate

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "cases" / "format"

# 两个例外（PLAN 1.6 写明），值是「文件里允许出现的合法行数」：
# - E_BOM 是文件级问题：BOM 属于整份文件，剥掉后那一行本身合法；
# - E_ID_DUP 需要前一行先出现同一个 id，那行前提本身是合法的。
PREREQUISITE_LINES: dict[str, int] = {contract.E_BOM: 1, contract.E_ID_DUP: 1}


def valid_files() -> list[Path]:
    """返回 valid/ 下的 case 文件（文件名是主题，不含期望错误码）。"""
    return sorted((CASES / "valid").glob("*.jsonl"))


def invalid_files() -> list[Path]:
    """返回 invalid/ 下的 case 文件（文件名即期望错误码）。"""
    return sorted((CASES / "invalid").glob("*.jsonl"))


def test_case_library_has_both_halves() -> None:
    """输入：无。输出：断言结果。预期行为：正例与负例两半都必须存在。"""
    assert valid_files(), "cases/format/valid/ 不能是空的"
    assert invalid_files(), "cases/format/invalid/ 不能是空的"


@pytest.mark.parametrize("path", valid_files(), ids=lambda p: p.stem)
def test_valid_case_all_rows_pass(path: Path) -> None:
    """输入：一个正例文件。输出：断言结果。预期行为：每一行都通过，无解析问题。"""
    report = validate.validate_file(path)
    assert report.parse_issues == []
    assert report.issues == []
    assert report.total > 0
    assert report.passed == report.total


@pytest.mark.parametrize("path", invalid_files(), ids=lambda p: p.stem)
def test_invalid_case_reports_exactly_named_code(path: Path) -> None:
    """输入：一个负例文件。输出：断言结果。

    预期行为：报出的错误码集合恰好是文件名；除前提行外每一行都被拒
    （混进合法行会让人误读这个 case 想表达什么）。
    """
    expected = path.stem
    report = validate.validate_file(path)
    assert set(report.code_counts()) == {expected}
    assert report.total > 0
    assert report.passed == PREREQUISITE_LINES.get(expected, 0)


def test_every_error_code_has_an_invalid_case() -> None:
    """输入：无。输出：断言结果。预期行为：契约里的错误码一个都不能漏。"""
    assert {path.stem for path in invalid_files()} == set(contract.ERROR_CODES)


def test_checked_in_files_match_builder(tmp_path: Path) -> None:
    """输入：临时目录。输出：断言结果。

    预期行为：磁盘上的 case 与 `jc.format_cases` 的输出逐字节一致——改了构造器
    就必须重跑 `python -m jc.format_cases cases/format`，两处不能各说各话。
    """
    format_cases.write_all(tmp_path)
    built = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*.jsonl")
    }
    on_disk = {
        path.relative_to(CASES).as_posix(): path.read_bytes()
        for path in CASES.rglob("*.jsonl")
    }
    assert built == on_disk


def test_valid_matrix_covers_every_schema_pair() -> None:
    """输入：无。输出：断言结果。

    预期行为：正例矩阵的维度组合达成**成对覆盖**——任意两维的任意取值组合
    至少出现一次（PLAN 3.3：组合数超阈值时不再全枚举，改用 covering array）。
    """
    assert cover.missing_pairs(format_cases.SCHEMA_DIMS, format_cases.matrix_combos()) == set()
