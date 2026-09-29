"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import jsonio, pipeline
from jc.jsontypes import JsonValue
from tests.workflows.autoformat.conftest import FakeBackend
from workflows.autoformat.workflow import AutoformatConfig, AutoformatWorkflow

CASE_LIMIT = 10

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("state_bad", "join_bad", "block_type_bad", "callout_bad", "ordered_bad")

#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "state_bad": "state 反解不出来",
    "join_bad": "并段答案与文本不自洽",
    "block_type_bad": "块类型答案与文本不自洽",
    "callout_bad": "提示框种类与文本不自洽",
    "ordered_bad": "列表有序性与文本不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="autoformat",
            split="train",
            locales=("zh",),
            run_seed=11,
            per_case=1,
            out_root=tmp_path,
            run_id=run_id,
            config_override=AutoformatConfig(split="train", case_limit=CASE_LIMIT),
            backend=FakeBackend(),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _rows(run_dir: Path) -> list[JsonValue]:
    """输入：run 目录。输出：合格样本行。"""
    return [row for _, row in jsonio.iter_jsonl(run_dir / "train.jsonl")]


def _write(run_dir: Path, rows: list[JsonValue]) -> None:
    """输入：run 目录、样本行。输出：无。预期行为：原样覆盖 train.jsonl（模拟产物被改坏）。"""
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)


def _audit(run_dir: Path, splits: tuple[str, ...] = ("train",)) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return AutoformatWorkflow().audit(run_dir, splits, "zh")


def _find_row(rows: list[JsonValue], family: str) -> JsonValue:
    """输入：样本行表、题族名。输出：第一条含该族题的样本行。预期行为：找不到就报错。"""
    for row in rows:
        for qid in row["questions"]:
            if str(qid) == family or str(qid).startswith(f"{family}_"):
                return row
    raise AssertionError(f"这一批里没有 {family} 那一族")


def test_clean_run_audits_without_missing(tmp_path: Path) -> None:
    """输入：一批干净产物。输出：无。预期行为：四族都出了、五个计数器全零、缺失项为空。"""
    report, missing = _audit(_run(tmp_path))
    counts = report["train"]
    assert missing == []
    assert counts["samples"] == CASE_LIMIT
    for family in ("join", "block_type", "callout_kind", "list_ordered"):
        assert counts["families"][family] > 0, family
    assert not counts["unknown_families"]
    for key in BAD_KEYS:
        assert counts[key] == 0, key


@pytest.mark.parametrize(
    ("family", "counter", "kind"),
    [
        ("join", "join_bad", "noul"),
        ("block_type", "block_type_bad", "choice"),
        ("callout_kind", "callout_bad", "choice"),
        ("list_ordered", "ordered_bad", "noul"),
    ],
)
def test_a_flipped_answer_is_caught_by_its_own_counter(
    tmp_path: Path, family: str, counter: str, kind: str
) -> None:
    """输入：把某一族第一道题的答案改掉。输出：无。

    预期行为：对应的计数器报 1，并进缺失项列表——审计是**纯代码重算**，与构造侧无关。
    """
    run_dir = _run(tmp_path, f"t-{family}")
    rows = _rows(run_dir)
    row = _find_row(rows, family)
    for qid in row["questions"]:
        if str(qid) == family or str(qid).startswith(f"{family}_"):
            answer = row["answers"][qid]
            if kind == "noul":
                answer["noul"] = not answer["noul"]
            else:
                answer["choice"] = (answer["choice"] + 1) % len(row["questions"][qid]["criteria"])
            break
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"][counter] == 1
    assert any(BAD_LABELS[counter] in item for item in missing)
    for other in BAD_KEYS:
        if other != counter:
            assert report["train"][other] == 0, other


def test_unreadable_state_is_caught(tmp_path: Path) -> None:
    """输入：一段行号格式不对的 state。输出：无。预期行为：`state_bad` 报 1，其余计数器不动。"""
    run_dir = _run(tmp_path, "t-state")
    rows = _rows(run_dir)
    rows[0]["state"] = "这里什么行号都没有"
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["state_bad"] == 1
    assert any(BAD_LABELS["state_bad"] in item for item in missing)
    for key in BAD_KEYS[1:]:
        assert report["train"][key] == 0, key


def test_a_wrong_option_text_is_caught(tmp_path: Path) -> None:
    """输入：把答案指向的那个选项的**文案**改掉（编号不动）。输出：无。

    预期行为：`block_type_bad` 报 1——审计比的是选项文案，只比选项名是抓不住这种错的。
    """
    run_dir = _run(tmp_path, "t-opttext")
    rows = _rows(run_dir)
    row = _find_row(rows, "block_type")
    for qid in row["questions"]:
        if str(qid).startswith("block_type"):
            question = row["questions"][qid]
            names = list(question["criteria"])
            index = row["answers"][qid]["choice"]
            question["criteria"][names[index]] = "这里换成了另一套文案"
            break
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["block_type_bad"] == 1
    assert any(BAD_LABELS["block_type_bad"] in item for item in missing)


def test_unknown_family_is_counted(tmp_path: Path) -> None:
    """输入：把一条题的题号改成别的族名。输出：无。预期行为：`unknown_families` 记一笔。

    这条抓的是「产物里混进了别的版本 / 别的 workflow 的题」——静默忽略会让版本串了看不出来。
    """
    run_dir = _run(tmp_path, "t-unknown")
    rows = _rows(run_dir)
    qid = next(iter(rows[0]["questions"]))
    rows[0]["questions"]["nonsense"] = rows[0]["questions"].pop(qid)
    rows[0]["answers"]["nonsense"] = rows[0]["answers"].pop(qid)
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    assert report["train"]["unknown_families"]["nonsense"] == 1


def test_missing_split_is_reported(tmp_path: Path) -> None:
    """输入：只审一个没跑过的切分。输出：无。预期行为：缺失项里点名那个切分。"""
    run_dir = _run(tmp_path, "t-missing")
    _, missing = _audit(run_dir, ("val",))
    assert any("val" in item for item in missing)


def test_missing_family_is_reported(tmp_path: Path) -> None:
    """输入：把某一族的题全删掉。输出：无。预期行为：缺失项里点名那一族一条都没出。"""
    run_dir = _run(tmp_path, "t-nofam")
    rows = _rows(run_dir)
    for row in rows:
        for qid in [qid for qid in row["questions"] if str(qid).startswith("list_ordered")]:
            row["questions"].pop(qid)
            row["answers"].pop(qid)
    _write(run_dir, rows)
    _, missing = _audit(run_dir)
    assert any("list_ordered" in item and "一条都没出" in item for item in missing)


def test_unknown_locale_is_rejected(tmp_path: Path) -> None:
    """输入：不存在的语言。输出：无。预期行为：当场抛错，别拖到解析阶段才报。"""
    run_dir = _run(tmp_path, "t-locale")
    with pytest.raises(FileNotFoundError):
        AutoformatWorkflow().audit(run_dir, ("train",), "nope")
