"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path

from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import flip_choice as _flip_choice, read_rows as _rows, run_config, write_rows as _write
from tests.workflows.chatgate.conftest import CLEAN, FakeBackend
from workflows.chatgate import locales, render
from workflows.chatgate.workflow import ChatGateConfig, ChatGateWorkflow

CASE_LIMIT = 8


#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "content_bad": "认不出来",
    "tag_bad": "逐判据答案与消息不自洽",
    "count_bad": "命中条数与消息不自洽",
    "route_bad": "处置档位与消息不自洽",
    "reply_bad": "干净候选与候选不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="chatgate",
            seed=11,
            override=ChatGateConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return ChatGateWorkflow().audit(run_dir, splits, "zh")


def test_tag_answer_mismatch_is_caught(tmp_path: Path) -> None:
    """输入：把一条逐判据 noul 的答案取反。输出：无。预期行为：`tag_bad` 报 1。"""
    run_dir = _run(tmp_path, "t-tag")
    rows = _rows(run_dir)
    answer = rows[0]["answers"]["tag"]
    answer["noul"] = not answer["noul"]
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["tag_bad"] == 1
    assert any(BAD_LABELS["tag_bad"] in item for item in missing)


def test_two_clean_candidates_are_caught(tmp_path: Path) -> None:
    """输入：把第二条候选也写成干净的。输出：无。

    预期行为：`reply_bad` 报 1——「干净的不止一条」是构造 bug，审计要看得见，不许硬凑一个答案。
    """
    run_dir = _run(tmp_path, "t-two-clean")
    rows = _rows(run_dir)
    parsed = render.parse_state(rows[0]["state"])
    candidates = list(parsed.replies)
    candidates[1] = CLEAN
    draft = render.MessageDraft(text=parsed.message or "", candidates=candidates)
    rows[0]["state"] = render.to_state(draft, locales.load("zh"))
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["reply_bad"] == 1
    assert any(BAD_LABELS["reply_bad"] in item for item in missing)


def test_state_with_only_two_candidates_is_caught(tmp_path: Path) -> None:
    """输入：候选那一段只剩两条。输出：无。预期行为：`content_bad` 报 1（骨架不完整）。"""
    run_dir = _run(tmp_path, "t-short-candidates")
    rows = _rows(run_dir)
    state = str(rows[0]["state"])
    rows[0]["state"] = "\n".join(
        line for line in state.splitlines() if not line.strip().startswith("- 3.")
    )
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    assert report["train"]["content_bad"] == 1

