"""浏览器 / GUI 操作的审计测试：交叉校验、篡改检测、缺失产物。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jc import pipeline

from workflows.browser import audit as audit_mod, features, locales, questions, render
from workflows.browser.features import TagSet
from workflows.browser.workflow import BrowserConfig

from tests.workflows.browser.conftest import FakeBackend

CASE_LIMIT = 4


def make_config(tmp_path: Path, run_id: str, backend: object) -> pipeline.RunConfig:
    """构造一个小 case 集合的试跑配置。"""
    return pipeline.RunConfig(
        workflow="browser",
        split="train",
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=BrowserConfig(split="train", case_limit=CASE_LIMIT),
        backend=backend,
    )


def read_rows(run_dir: Path) -> list[dict[str, object]]:
    """输入：run 目录。输出：train.jsonl 里的行（已解析）。"""
    path = run_dir / "train.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_rows(run_dir: Path, rows: list[dict[str, object]]) -> None:
    """输入：run 目录、行。输出：无。预期行为：按产物同样的格式写回（每行一条 + 末尾换行）。"""
    path = run_dir / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def table_of(tags: TagSet) -> tuple[render.ElementRow, ...]:
    """输入：一组特征。输出：它那一屏的规范元素表。"""
    return render.canonical_elements(tags)


def test_expected_from_derives_every_gate() -> None:
    """输入：四类元素表。输出：无。预期行为：可用 / 已是当前 / 被拦住 三档都推得出来。"""
    ready_case = TagSet(intent="fill", state="ready", sensitive="yes", decoy="none", screen="form")
    assert audit_mod.expected_from(table_of(ready_case), "fill") == (
        features.wanted_label(ready_case),
        "type",
        False,
        False,
        True,
        2,
    )
    done_case = TagSet(intent="fill", state="current", sensitive="no", decoy="none", screen="form")
    assert audit_mod.expected_from(table_of(done_case), "fill") == (
        "none",
        "none",
        True,
        False,
        False,
        1,
    )
    blocked_case = TagSet(intent="submit", state="hidden", sensitive="no", decoy="none", screen="form")
    assert audit_mod.expected_from(table_of(blocked_case), "submit") == (
        "none",
        "none",
        False,
        True,
        False,
        1,
    )


def test_expected_from_returns_none_when_it_cannot_tell() -> None:
    """输入：认不出的意图 / 表里没有该角色 / 两行都可用。输出：无。

    预期行为：都返回 None——审计宁可记一笔「算不出来」，也不给一个看着像校验的东西。
    """
    rows = table_of(TagSet(intent="fill", state="ready", sensitive="no", decoy="none", screen="form"))
    assert audit_mod.expected_from(rows, "bogus") is None
    assert audit_mod.expected_from(rows, 3) is None
    no_field = tuple(row for row in rows if row[1] != "field")
    assert audit_mod.expected_from(no_field, "fill") is None
    doubled = rows + ((len(rows) + 1, "field", "用户名", "ready"),)
    assert audit_mod.expected_from(doubled, "fill") is None


def test_expected_from_uses_the_shared_sensitivity_of_the_role_group() -> None:
    """输入：带同角色干扰行的元素表。输出：无。

    预期行为：目标行与干扰行同属一组词表，敏感与否一致——所以干扰行在不在都不影响风险档，
    而「该操作哪个元素」只看那一行可不可用。
    """
    case = TagSet(intent="submit", state="ready", sensitive="yes", decoy="same_role", screen="list")
    expected = audit_mod.expected_from(table_of(case), "submit")
    assert expected is not None
    target, action, done, blocked, confirm, risk = expected
    assert (action, done, blocked, confirm, risk) == ("click", False, False, True, 3)
    assert target == features.wanted_label(case)


def test_family_of_strips_dedupe_suffixes() -> None:
    """输入：题号。输出：无。预期行为：去掉 `_2` 这类后缀。"""
    assert audit_mod._family_of("target") == "target"
    assert audit_mod._family_of("target_2") == "target"
    assert audit_mod._family_of("blocked_3") == "blocked"


def test_audit_passes_a_clean_run(tmp_path: Path) -> None:
    """输入：假后端（从不撒谎）跑出来的产物。输出：无。预期行为：零缺失项、零坏计数。"""
    result = pipeline.run(make_config(tmp_path, "clean", FakeBackend()))
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert missing == []
    counts = report["train"]
    assert counts["samples"] == CASE_LIMIT
    for key in (
        "frame_bad",
        "gates_bad",
        "criteria_bad",
        "intent_bad",
        "elements_bad",
        "target_bad",
        "action_bad",
        "gate_bad",
        "risk_bad",
    ):
        assert counts[key] == 0, key


def test_audit_flags_a_tampered_target_answer(tmp_path: Path) -> None:
    """输入：把「该操作哪个元素」的答案改成另一个选项。输出：无。

    预期行为：审计报出 `target_bad`——它拿元素表 + 意图重算，改一个字节就该暴露。
    """
    result = pipeline.run(make_config(tmp_path, "a1", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        questions_of = row["questions"]
        answers_of = row["answers"]
        assert isinstance(questions_of, dict) and isinstance(answers_of, dict)
        options = questions_of["target"]["criteria"]
        index = answers_of["target"]["choice"]
        answers_of["target"]["choice"] = (index + 1) % len(options)
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["target_bad"] == CASE_LIMIT
    assert any("该操作哪个元素" in item for item in missing)


def test_audit_flags_a_tampered_action_answer(tmp_path: Path) -> None:
    """输入：把动作答案改成另一档。输出：无。预期行为：审计报出 `action_bad`。"""
    result = pipeline.run(make_config(tmp_path, "a2", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        answers_of = row["answers"]
        assert isinstance(answers_of, dict)
        answers_of["action"]["choice"] = 4  # 恒为「不操作」那一档
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["action_bad"] >= 1
    assert any("该做什么动作" in item for item in missing)


def test_audit_flags_a_tampered_gate_answer(tmp_path: Path) -> None:
    """输入：把三档门控里的一档翻过来。输出：无。预期行为：审计报出 `gate_bad`。"""
    result = pipeline.run(make_config(tmp_path, "a3", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        answers_of = row["answers"]
        assert isinstance(answers_of, dict)
        answers_of["blocked"]["noul"] = not answers_of["blocked"]["noul"]
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["gate_bad"] >= 1
    assert any("三档门控" in item for item in missing)


def test_audit_flags_a_tampered_risk_answer(tmp_path: Path) -> None:
    """输入：把风险档挪一档。输出：无。预期行为：审计报出 `risk_bad`。"""
    result = pipeline.run(make_config(tmp_path, "a4", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        answers_of = row["answers"]
        assert isinstance(answers_of, dict)
        answers_of["risk"]["score"] = (answers_of["risk"]["score"] + 1) % features.RISK_LEVELS
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["risk_bad"] == CASE_LIMIT
    assert any("风险档" in item for item in missing)


def test_audit_flags_a_tampered_gate_table(tmp_path: Path) -> None:
    """输入：把 state 里门控表某一行的说明改掉。输出：无。

    预期行为：审计报出 `gates_bad`——门控表是代码渲染的规范表，产物里必须逐行相同。
    """
    result = pipeline.run(make_config(tmp_path, "a5", FakeBackend()))
    rows = read_rows(result.run_dir)
    state = rows[0]["state"]
    assert isinstance(state, str)
    rows[0]["state"] = state.replace("- blocked |", "- blocked | 随便写的", 1)
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["gates_bad"] == 1
    assert any("门控表" in item for item in missing)


def test_audit_flags_a_tampered_criteria_table(tmp_path: Path) -> None:
    """输入：把 state 里意图表某一行的风险改掉。输出：无。预期行为：审计报出 `criteria_bad`。"""
    result = pipeline.run(make_config(tmp_path, "a6", FakeBackend()))
    rows = read_rows(result.run_dir)
    state = rows[0]["state"]
    assert isinstance(state, str)
    rows[0]["state"] = state.replace(
        "- submit | button | click | 1 | 3", "- submit | button | click | 1 | 2", 1
    )
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["criteria_bad"] == 1
    assert any("意图表" in item for item in missing)


def test_audit_flags_a_broken_element_table(tmp_path: Path) -> None:
    """输入：把元素表的行号改成跳号。输出：无。

    预期行为：审计报出 `frame_bad`——元素表解析不出来时，后面的交叉校验也就无从谈起。
    """
    result = pipeline.run(make_config(tmp_path, "a7", FakeBackend()))
    rows = read_rows(result.run_dir)
    state = rows[0]["state"]
    assert isinstance(state, str)
    rows[0]["state"] = state.replace("- 1 | ", "- 9 | ", 1)
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["frame_bad"] == 1
    assert any("元素表解析不出来" in item for item in missing)


def test_audit_flags_an_illegal_intent_answer(tmp_path: Path) -> None:
    """输入：把意图答案的编号指到界外。输出：无。预期行为：审计报出 `intent_bad`。"""
    result = pipeline.run(make_config(tmp_path, "a8", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        answers_of = row["answers"]
        assert isinstance(answers_of, dict)
        answers_of["intent"]["choice"] = 99
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["intent_bad"] == CASE_LIMIT
    assert any("意图答案不是合法取值" in item for item in missing)


def test_audit_flags_a_table_missing_the_intent_role(tmp_path: Path) -> None:
    """输入：把元素表里该意图要求的那个角色全换成另一个角色。输出：无。

    预期行为：审计反推不出该有的六格，报出 `elements_bad` 并点明「反推不出」——
    算不出来就记一笔，不猜（范式 §9.13）。
    """
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="browser",
            split="train",
            locales=("zh",),
            run_seed=7,
            per_case=1,
            out_root=tmp_path,
            run_id="a10",
            config_override=BrowserConfig(split="train", case_limit=1),
            backend=FakeBackend(),
        )
    )
    rows = read_rows(result.run_dir)
    questions_of = rows[0]["questions"]
    answers_of = rows[0]["answers"]
    assert isinstance(questions_of, dict) and isinstance(answers_of, dict)
    intent = list(questions_of["intent"]["criteria"])[answers_of["intent"]["choice"]]
    role = features.role_for_intent(intent)
    other = next(candidate for candidate in features.ROLES if candidate != role)
    state = rows[0]["state"]
    assert isinstance(state, str)
    rows[0]["state"] = state.replace(f"| {role} |", f"| {other} |")
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["elements_bad"] == 1
    assert any("反推不出" in item for item in missing)


def test_audit_reports_a_missing_product(tmp_path: Path) -> None:
    """输入：没有产物的切分。输出：无。预期行为：缺失项里点名它，计数全为零。"""
    report, missing = audit_mod.audit_run(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)
    assert report["train"]["families"] == {}


def test_audit_rejects_an_unknown_locale(tmp_path: Path) -> None:
    """输入：不存在的语言。输出：无。预期行为：当场报错，不拖到解析阶段。"""
    with pytest.raises(FileNotFoundError):
        audit_mod.audit_run(tmp_path, ("train",), "nope")


def test_known_families_are_exactly_the_built_ones() -> None:
    """输入：无。输出：无。预期行为：审计认得的题族就是题族表本身。"""
    assert audit_mod.KNOWN_FAMILIES == frozenset(questions.FAMILIES)
    assert audit_mod.CHECKED_FAMILIES == frozenset(questions.FAMILIES)


def test_audit_flags_an_unknown_family_and_a_missing_one(tmp_path: Path) -> None:
    """输入：把一个题族改名成不认识的。输出：无。

    预期行为：报出 `unknown_families` 一笔，同时点名原来那一族一条都没出——
    产物串版本时这两条会一起出现。
    """
    result = pipeline.run(make_config(tmp_path, "a9", FakeBackend()))
    rows = read_rows(result.run_dir)
    for row in rows:
        questions_of = row["questions"]
        answers_of = row["answers"]
        assert isinstance(questions_of, dict) and isinstance(answers_of, dict)
        questions_of["bogus"] = questions_of.pop("confirm")
        answers_of["bogus"] = answers_of.pop("confirm")
    write_rows(result.run_dir, rows)
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "zh")
    assert report["train"]["unknown_families"]["bogus"] == CASE_LIMIT
    assert any("题族 confirm" in item for item in missing)


def test_audit_accepts_the_english_locale(tmp_path: Path) -> None:
    """输入：英文跑出来的产物。输出：无。

    预期行为：审计用英文那份门控表比对，照样零缺失项——门控表的说明文字按语言写，
    判据词元与语言无关。
    """
    result = pipeline.run(
        pipeline.RunConfig(
            workflow="browser",
            split="train",
            locales=("en",),
            run_seed=7,
            per_case=1,
            out_root=tmp_path,
            run_id="en",
            config_override=BrowserConfig(split="train", case_limit=CASE_LIMIT),
            backend=FakeBackend(lang="en"),
        )
    )
    report, missing = audit_mod.audit_run(result.run_dir, ("train",), "en")
    assert missing == []
    assert report["train"]["gates_bad"] == 0


def test_render_tables_are_language_specific_but_share_the_tokens() -> None:
    """输入：两语言的门控表。输出：无。预期行为：判据词元相同、说明文字不同。"""
    zh = render.gate_rows(locales.load("zh"))
    en = render.gate_rows(locales.load("en"))
    assert [token for token, _ in zh] == [token for token, _ in en]
    assert [text for _, text in zh] != [text for _, text in en]
