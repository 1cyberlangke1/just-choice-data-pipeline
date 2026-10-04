"""打分与评分的产物审计测试：能重算的那几条要真查得住，查不住的要说清楚。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import jsonio, pipeline, workflow as workflow_mod
from jc.jsontypes import JsonValue

from workflows.rubric import audit, features, locales, render, workflow as rubric_workflow
from workflows.rubric.features import TagSet

from tests.workflows._scaffold import run_config
from tests.workflows.rubric.conftest import FakeBackend


def make_config(tmp_path: Path, run_id: str, *, split: str = "train") -> pipeline.RunConfig:
    """构造一个小 case 集合的运行配置（指向 tmp 目录）。

    预期行为：**关掉语言闸**——本包的假后端只写英文（夹具局限，理由见
    `tests/workflows/rubric/test_pipeline.make_config`）。
    """
    return run_config(
        tmp_path,
        run_id,
        backend=FakeBackend(),
        workflow="rubric",
        split=split,
        seed=7,
        override=rubric_workflow.RubricConfig(split=split, case_limit=6),
        language_gate=False,
    )


def rewrite(rows: list[JsonValue], target: Path) -> Path:
    """输入：样本行、目标文件。输出：写出后的路径。预期行为：整批重写一遍，供审计复读。"""
    jsonio.write_jsonl(target, rows)
    return target


def tamper_state(tmp_path: Path, run_id: str, mutate: object) -> tuple[Path, int]:
    """输入：运行 id、改 state 的函数。输出：(改过的 run 目录, 行数)。

    预期行为：先把干净产物跑出来，再逐行改 state 并重写到另一个目录——审计只读产物，
    所以「改过的产物」必须真的落盘。
    """
    result = pipeline.run(make_config(tmp_path, run_id))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    for row in rows:
        state = row["state"]
        assert isinstance(state, str)
        row["state"] = mutate(state)
    run_dir = tmp_path / f"tampered-{run_id}"
    rewrite(rows, run_dir / "train.jsonl")
    return run_dir, len(rows)


def test_clean_run_audits_clean(tmp_path: Path) -> None:
    """输入：干净产物。输出：无。预期行为：零缺失、全部计数为零。"""
    result = pipeline.run(make_config(tmp_path, "a1"))
    report, missing = workflow_mod.load("rubric").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 6
    for family in ("met", "band", "gate", "top"):
        assert report["train"]["families"][family] > 0
    for counter in (
        "manifest_bad",
        "body_shape_bad",
        "score_table_bad",
        "band_table_bad",
        "gate_table_bad",
        "score_bad",
        "met_bad",
        "band_bad",
        "gate_bad",
        "top_bad",
    ):
        assert report["train"][counter] == 0, counter


def test_a_tampered_form_token_is_caught(tmp_path: Path) -> None:
    """输入：把清单行的「形式」改成相反的那一档。输出：无。

    预期行为：`body_shape_bad` 记一笔——材料形状是真查得住的，不是「看着像校验」。
    """
    def mutate(state: str) -> str:
        parsed = render.parse_state(state)
        assert parsed.manifest is not None
        other = next(value for value in features.FORMS if value != parsed.manifest.form)
        return state.replace(f"| {parsed.manifest.form} |", f"| {other} |")

    run_dir, count = tamper_state(tmp_path, "a2", mutate)
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["body_shape_bad"] == count
    assert any("材料形状" in item for item in missing)


def test_a_tampered_manifest_token_is_caught(tmp_path: Path) -> None:
    """输入：把清单行的词元换成不认识的。输出：无。预期行为：`manifest_bad` 记一笔。"""
    def mutate(state: str) -> str:
        parsed = render.parse_state(state)
        assert parsed.manifest is not None
        # 篇幅是第一格，前面接的是行前缀（`- `），不是分隔符
        return state.replace(f"- {parsed.manifest.length} |", "- huge |")

    run_dir, count = tamper_state(tmp_path, "a3", mutate)
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["manifest_bad"] == count
    assert any("清单行解析不出来" in item for item in missing)


def test_a_tampered_score_table_is_caught(tmp_path: Path) -> None:
    """输入：把评分单的权重改掉。输出：无。预期行为：`score_table_bad` 记一笔。"""
    run_dir, count = tamper_state(
        tmp_path, "a4", lambda state: state.replace("- complete | 3 |", "- complete | 1 |")
    )
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["score_table_bad"] == count
    assert any("评分单" in item for item in missing)


@pytest.mark.parametrize(
    ("mutate", "counter", "hint"),
    [
        (lambda s: s.replace("- mid | 6 | 7", "- mid | 6 | 8"), "band_table_bad", "档位表"),
        (lambda s: s.replace("- low | revise | reject", "- low | accept | reject"), "gate_table_bad", "处置表"),
    ],
)
def test_a_tampered_table_is_caught(
    tmp_path: Path, mutate: object, counter: str, hint: str
) -> None:
    """输入：把档位表 / 处置表改掉。输出：无。预期行为：对应计数记一笔，缺失项点名那张表。"""
    run_dir, count = tamper_state(tmp_path, f"a5-{counter}", mutate)
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"][counter] == count
    assert any(hint in item for item in missing)


@pytest.mark.parametrize(
    ("qid", "field", "counter", "hint"),
    [
        ("met_1", "noul", "met_bad", "逐条达标"),
        ("band", "score", "band_bad", "档位与总分"),
        ("gate", "choice", "gate_bad", "处置与档位红线"),
    ],
)
def test_a_tampered_answer_is_caught(
    tmp_path: Path, qid: str, field: str, counter: str, hint: str
) -> None:
    """输入：把三种题型之一的答案换掉。输出：无。预期行为：对应计数记一笔。"""
    result = pipeline.run(make_config(tmp_path, f"a6-{qid}"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    for row in rows:
        answers = row["answers"]
        assert isinstance(answers, dict)
        answer = answers[qid]
        assert isinstance(answer, dict)
        if field == "noul":
            answer[field] = not answer[field]
        else:
            answer[field] = (int(answer[field]) + 1) % 3
    run_dir = tmp_path / f"tampered-{qid}"
    rewrite(rows, run_dir / "train.jsonl")
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"][counter] == len(rows)
    assert any(hint in item for item in missing)


def test_a_tampered_top_answer_is_caught(tmp_path: Path) -> None:
    """输入：把「最严重的一条」的答案换掉。输出：无。

    预期行为：`top_bad` 记一笔——这一族的**出不出**与取值都跟重算结果对得上。
    """
    result = pipeline.run(make_config(tmp_path, "a7"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    touched = 0
    for row in rows:
        answers = row["answers"]
        assert isinstance(answers, dict)
        if "top" not in answers:
            continue
        answer = answers["top"]
        assert isinstance(answer, dict)
        answer["choice"] = (int(answer["choice"]) + 1) % len(features.CRITERIA)
        touched += 1
    assert touched > 0, "这批 case 里一条 top 都没出，测不到"
    run_dir = tmp_path / "tampered-top"
    rewrite(rows, run_dir / "train.jsonl")
    report, missing = audit.audit_run(run_dir, ("train",), "zh")
    assert report["train"]["top_bad"] == touched
    assert any("最严重的一条" in item for item in missing)


def test_body_shape_problem_reads_every_rule() -> None:
    """输入：几种材料与清单的组合。输出：无。预期行为：每条口径都真查得住。"""
    short = render.ManifestState(length="short", form="prose", data="none")
    long = render.ManifestState(length="long", form="prose", data="none")
    bulleted = render.ManifestState(length="short", form="bulleted", data="none")
    numbered = render.ManifestState(length="short", form="prose", data="numbers")
    assert audit.body_shape_problem("alpha\nbravo\ncharlie", short) == ""
    assert "句数" in audit.body_shape_problem("alpha\nbravo", long)
    assert "分条" in audit.body_shape_problem("alpha\nbravo\ncharlie", bulleted)
    assert "行文" in audit.body_shape_problem("- alpha\n- bravo\n- charlie", short)
    assert "有数据" in audit.body_shape_problem("alpha\nbravo\ncharlie", numbered)
    assert "无数据" in audit.body_shape_problem("alpha 1\nbravo 2\ncharlie 3", short)


def test_score_table_problem_reads_every_column() -> None:
    """输入：几种评分单。输出：无。预期行为：判据名 / 权重 / 要求 / 观测四列都真查得住。"""
    loc = locales.load("zh")
    tags = TagSet(placeholders="yes", citations="no", length="short", form="prose", data="none")
    manifest = render.ManifestState(length=tags.length, form=tags.form, data=tags.data)
    rows = render.score_rows(tags, loc)
    marks = render.count_marks()
    assert audit.score_table_problem(rows, manifest, marks) == ""
    assert "行，应为" in audit.score_table_problem(rows[:3], manifest, marks)
    bad_weight = (("complete", "1", "3", rows[0][3]), *rows[1:])
    assert "权重" in audit.score_table_problem(bad_weight, manifest, marks)
    bad_need = (("complete", "3", "1", rows[0][3]), *rows[1:])
    assert "要求" in audit.score_table_problem(bad_need, manifest, marks)
    bad_mark = (("complete", "3", "3", "4"), *rows[1:])
    assert "观测该印" in audit.score_table_problem(bad_mark, manifest, marks)
    bad_obs = (*rows[:2], ("concise", "2", "2", "9"), *rows[3:])
    assert "按清单应为" in audit.score_table_problem(bad_obs, manifest, marks)


def test_family_of_strips_the_sequence_suffix() -> None:
    """输入：几种题号。输出：无。预期行为：`met_1` 归到 `met`，其余原样。"""
    assert audit._family_of("met_1") == "met"
    assert audit._family_of("met_5") == "met"
    assert audit._family_of("gate") == "gate"
    assert audit._family_of("top") == "top"
    assert audit.MET_QIDS == ("met_1", "met_2", "met_3", "met_4", "met_5")


