"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import jsonio, pipeline
from jc.jsontypes import JsonValue

from tests.workflows.causechain.conftest import FRAGMENTS, FakeBackend
from workflows.causechain import features, locales, render
from workflows.causechain.workflow import CauseConfig, CauseChainWorkflow, cases_for

CASE_LIMIT = 8

#: 审计报告里那七个「不自洽」计数器。
BAD_KEYS = ("graph_bad", "log_bad", "root_bad", "hop_bad", "span_bad", "isroot_bad", "count_bad")

#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "graph_bad": "依赖图认不出来",
    "log_bad": "现场日志认不出来",
    "root_bad": "根因与日志不自洽",
    "hop_bad": "跳数与日志不自洽",
    "span_bad": "受影响集合与日志不自洽",
    "isroot_bad": "根因判定与日志不自洽",
    "count_bad": "影响面与日志不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="causechain",
            split="train",
            locales=("zh",),
            run_seed=11,
            per_case=1,
            out_root=tmp_path,
            run_id=run_id,
            config_override=CauseConfig(split="train", case_limit=CASE_LIMIT),
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


def _audit(run_dir: Path) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录。输出：(计数, 缺失项)。"""
    return CauseChainWorkflow().audit(run_dir, ("train",), "zh")


def _flip_answer(row: JsonValue, qid: str) -> None:
    """输入：样本行、题号。输出：无。

    预期行为：把这道题的答案改成另一个——choice 换成另一个选项编号、score 换成相邻档、noul 取反。
    """
    answer = row["answers"][qid]
    if answer["type"] == "choice":
        answer["choice"] = (answer["choice"] + 1) % len(row["questions"][qid]["criteria"])
    elif answer["type"] == "score":
        answer["score"] = (answer["score"] + 1) % len(row["questions"][qid]["criteria"])
    else:
        answer["noul"] = not answer["noul"]


def _first_row_with(rows: list[JsonValue], qid: str) -> JsonValue:
    """输入：样本行、题号。输出：第一条带这个题族的样本行。"""
    return next(row for row in rows if qid in row["questions"])


def _drop_first_edge(state: str) -> str:
    """输入：state。输出：删掉依赖图第一行之后的 state（模拟渲染挂错了图）。"""
    lines = state.splitlines()
    mark = str(locales.load("zh")["state"]["tag_mark"])
    start = lines.index(mark)
    return "\n".join([*lines[: start + 1], *lines[start + 2 :]])


def _all_logs_ok(state: str) -> str:
    """输入：state。输出：把现场日志每一行的正文都换成「正常」的 state。

    预期行为：这样就没有任何服务出错，`observed_labels` 重推不出根因——验的是「日志认不出来」
    这条路，而不是「重推出的根因与答案对不上」那一条。
    """
    loc = locales.load("zh")
    mark = str(loc["state"]["content_mark"])
    sep = str(loc["state"]["tag_sep"])
    out: list[str] = []
    inside = False
    for line in state.splitlines():
        if line.strip().startswith(mark):
            inside = True
        elif inside and line.startswith(render.LINE_PREFIX):
            head = line[len(render.LINE_PREFIX) :].partition(sep)[0]
            out.append(f"{render.LINE_PREFIX}{head}{sep}{FRAGMENTS[features.EVENT_OK]}")
            continue
        out.append(line)
    return "\n".join(out)


def test_clean_run_audits_without_missing(tmp_path: Path) -> None:
    """输入：一批干净产物。输出：无。预期行为：五族都出了、七个计数器全零、缺失项为空。"""
    report, missing = _audit(_run(tmp_path))
    counts = report["train"]
    cases = cases_for(CauseConfig(split="train", case_limit=CASE_LIMIT))
    hop = sum(1 for case in cases if case.probe in ("h0", "h1", "h2"))
    assert missing == []
    assert counts["samples"] == CASE_LIMIT
    for family in ("root", "hop", "span", "isroot", "count"):
        assert counts["families"][family] > 0, family
    assert counts["types"]["noul"] == len(cases) - hop
    assert counts["types"]["choice"] == len(cases) * 2 + hop
    assert counts["types"]["score"] == len(cases)
    for key in BAD_KEYS:
        assert counts[key] == 0, key


@pytest.mark.parametrize(
    ("qid", "counter"),
    [
        ("root", "root_bad"),
        ("hop", "hop_bad"),
        ("span", "span_bad"),
        ("isroot", "isroot_bad"),
        ("count", "count_bad"),
    ],
)
def test_answer_mismatch_is_caught(tmp_path: Path, qid: str, counter: str) -> None:
    """输入：把一条题的答案改掉。输出：无。预期行为：对应的计数器报 1，并进缺失项列表。"""
    run_dir = _run(tmp_path, f"t-{qid}")
    rows = _rows(run_dir)
    _flip_answer(_first_row_with(rows, qid), qid)
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"][counter] == 1
    assert any(BAD_LABELS[counter] in item for item in missing)


def test_a_broken_graph_is_caught(tmp_path: Path) -> None:
    """输入：删掉依赖图里的一条边。输出：无。

    预期行为：记「依赖图认不出来」——边表还原不成四张图里的任何一张（删一条边后它跟剩下
    三张都不相等），后面的重推一律停下。
    """
    run_dir = _run(tmp_path, "t-graph")
    rows = _rows(run_dir)
    rows[0]["state"] = _drop_first_edge(str(rows[0]["state"]))
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["graph_bad"] == 1
    assert any(BAD_LABELS["graph_bad"] in item for item in missing)


def test_logs_that_do_not_pin_a_root_are_caught(tmp_path: Path) -> None:
    """输入：把现场日志全部改成「正常」。输出：无。

    预期行为：记「现场日志认不出来」——六个服务都没出错，重推不出根因；这条路径不该被
    硬凑成一个根因糊过去。
    """
    run_dir = _run(tmp_path, "t-log")
    rows = _rows(run_dir)
    rows[0]["state"] = _all_logs_ok(str(rows[0]["state"]))
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["log_bad"] == 1
    assert any(BAD_LABELS["log_bad"] in item for item in missing)


def test_missing_split_and_unknown_locale_are_reported(tmp_path: Path) -> None:
    """输入：不存在的切分与不存在的语言。输出：无。

    预期行为：切分不存在写进缺失项（报告照出），语言不存在当场抛错——别拖到解析阶段才报。
    """
    run_dir = _run(tmp_path, "t-missing")
    _, missing = CauseChainWorkflow().audit(run_dir, ("val",), "zh")
    assert any("val" in item for item in missing)
    with pytest.raises(FileNotFoundError):
        CauseChainWorkflow().audit(run_dir, ("train",), "nope")
