"""边界与错误路径：坏方言、坏语言文件、坏产物都要**报人话**，不许静默。

来由：这些分支在正常生成路径上永远走不到（覆盖率实测 90%，缺的全是这里），
但它们是「不崩、不泄密、报错人话」这条纪律的落点——真出问题时，报人话的那行代码
就是唯一的线索。另一类边界是**反解的容错**：换个版式就认不出来的反解等于没有。
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from jc import assemble, jsonio
from workflows.shell import judges, locales, parse, questions, render
from workflows.shell import spec as sp
from workflows.shell import workflow as shell_workflow
from workflows.shell.workflow import OOD_CONFIG, cases_for


def _case(**overrides: Any) -> sp.ShellCase:
    """输入：维度覆盖项。输出：一个 case（默认：bash 上删掉 `build/`、目标一致）。"""
    fields: dict[str, Any] = {
        "dialect": "bash",
        "op": "remove_tree",
        "target": "build_dir",
        "goal_target": "build_dir",
        "wrapper": "plain",
        "goal_style": "imperative",
        "reads": 2,
    }
    fields.update(overrides)
    return sp.ShellCase(**fields)


def _row(case: sp.ShellCase, seed: int = 1, locale: str = "zh") -> tuple[dict[str, Any], sp.CommandSpec]:
    """输入：case、种子、语言。输出：(契约样本, 真值规格)。预期行为：用真的组装器造样本。"""
    loc = locales.load(locale)
    rng = random.Random(seed)
    spec = sp.sample_spec(rng, case, "train", 0, 0)
    labeled = questions.build(spec, loc, spec.phrasing, rng)
    state = render.build(spec, loc)
    return assemble.assemble_sample("shell-train-0000000", state, labeled, rng), spec


# ---------- spec：坏方言、类别不匹配的对象、认不出的命令 ----------


def test_an_unknown_dialect_is_rejected_at_every_door() -> None:
    """输入：不存在的方言。输出：KeyError / ValueError。

    预期行为：路径表、模板表、分类器三处都报人话——静默回落到默认方言会让反解用错模板，
    把好数据判成坏的。
    """
    with pytest.raises(KeyError, match="cmd.exe.*路径"):
        sp.TARGETS[0].path_in("cmd.exe")
    with pytest.raises(KeyError, match="cmd.exe.*模板"):
        sp.OPS[0].template_in("cmd.exe")
    with pytest.raises(ValueError, match="未知方言"):
        sp.classify("ls", "cmd.exe")


def test_an_object_of_the_wrong_kind_is_rejected() -> None:
    """输入：命令作用在类别不匹配的对象上（对数据库跑 `cat`）。输出：ValueError。

    预期行为：报「不接受这个对象」而不是猜一个——猜出来的审批事实会看起来合理但是错的。
    """
    with pytest.raises(ValueError, match="不接受对象"):
        sp.classify("cat orders", "bash")


def test_an_unknown_command_is_rejected() -> None:
    """输入：操作表外的命令。输出：ValueError。预期行为：说清楚是「认不出这条命令」。"""
    with pytest.raises(ValueError, match="认不出这条命令"):
        sp.classify("echo hello", "bash")


# ---------- render / locales：语言文件缺东西 ----------


def test_render_names_the_locale_key_that_is_missing() -> None:
    """输入：缺目标句式 / 缺执行环境文案的语言字典。输出：KeyError。

    预期行为：报缺的是哪一条，而不是 KeyError: 'list_dir' 这种看不出所以然的东西。
    """
    spec = sp.sample_spec(random.Random(1), _case(), "train", 0, 0)
    with pytest.raises(KeyError, match="目标句式"):
        render.goal_text(spec, {"goal": {}})
    with pytest.raises(KeyError, match="执行环境文案"):
        render.env_text(spec, {"env": {}})


def test_a_missing_locale_file_lists_what_is_available() -> None:
    """输入：不存在的语言代码。输出：FileNotFoundError。预期行为：报错里带可用语言。"""
    with pytest.raises(FileNotFoundError, match="可用"):
        locales.load("xx")


# ---------- parse：标签重复、空行、命令与标签同行 ----------


def test_duplicate_labels_across_locales_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两张标签重叠的语言表。输出：ValueError。

    预期行为：合并标签表时当场炸——语言无关的反解靠「标签互不重叠」，重叠会把两种语言的
    标题串成一行（`test_locales.py` 锁的是真语言文件，这里锁的是这道闸本身）。
    """
    monkeypatch.setattr(parse.locales, "available", lambda: ["zh", "en"])
    same = {"meta": {key: key for key in parse.META_KEYS}}
    monkeypatch.setattr(parse.locales, "load", lambda lang: same)
    with pytest.raises(ValueError, match="重复"):
        parse.merged_labels()


def test_parse_tolerates_blank_lines_and_an_inline_command() -> None:
    """输入：带空行、且命令与标签同行的 state。输出：反解结果。

    预期行为：空行跳过、`命令：` 后面直接跟内容也算命令——渲染器目前把命令单独放一行，
    但反解不该依赖这一点：换个版式就认不出来的反解等于没有。
    """
    loc = locales.load("zh")
    spec = sp.sample_spec(random.Random(1), _case(), "train", 0, 0)
    state = render.build(spec, loc)
    assert f"\n  {spec.command}" in state  # 渲染器：命令单独一行、缩进两格
    inline = state.replace(f"\n  {spec.command}", f" {spec.command}")
    inline = inline.replace("\n", "\n\n", 1)  # 抬头后面插一个空行
    parsed = parse.parse_request(inline, parse.label_index(loc))
    assert parsed.dialect == spec.case.dialect
    assert parsed.cwd == spec.cwd
    assert parsed.reads == spec.reads
    assert parsed.command == spec.command


# ---------- judges：五个字段各自对不上都要报得出来 ----------


def test_roundtrip_diff_names_each_mismatched_field() -> None:
    """输入：逐项被改坏的 ParsedRequest。输出：差异列表。

    预期行为：改一项只报一项——判别器得说得出「是哪一项对不上」，人才定位得了。
    """
    loc = locales.load("zh")
    spec = sp.sample_spec(random.Random(1), _case(), "train", 0, 0)
    good = parse.parse_request(render.build(spec, loc), parse.label_index(loc))
    assert judges.roundtrip_diff(spec, good, loc) == []
    for field, broken_value in (
        ("dialect", "powershell"),
        ("cwd", "/tmp/elsewhere"),
        ("goal", "把别的都删掉"),
        ("reads", ("logs/api.log",)),
    ):
        diff = judges.roundtrip_diff(spec, replace(good, **{field: broken_value}), loc)
        assert len(diff) == 1, (field, diff)


def test_label_reports_an_unreadable_state() -> None:
    """输入：反解不出来的 state。输出：Verdict。预期行为：判不通过并说明「反解失败」。"""
    row, _ = _row(_case())
    broken = copy.deepcopy(row)
    broken["state"] = "这不是一份工具调用请求"
    verdict = judges.LabelJudge().judge(broken)
    assert not verdict.ok and "反解失败" in verdict.reason


def test_label_reports_a_goal_that_names_nothing() -> None:
    """输入：用户目标里一个对象都没有。输出：无。预期行为：报「认不出任何对象」。"""
    loc = locales.load("zh")
    row, spec = _row(_case())
    broken = copy.deepcopy(row)
    goal = render.goal_text(spec, loc)
    path = spec.goal_target.path_in(spec.case.dialect)
    broken["state"] = broken["state"].replace(goal, goal.replace(path, "那个目录"))
    assert "那个目录" in broken["state"]
    verdict = judges.LabelJudge().judge(broken)
    assert not verdict.ok
    assert any("认不出任何对象" in problem for problem in verdict.raw["problems"])


def test_label_reports_an_out_of_range_choice_index() -> None:
    """输入：choice 的答案编号越界。输出：无。预期行为：报「答案编号越界」而不是抛异常。"""
    row, _ = _row(_case())
    broken = copy.deepcopy(row)
    qid = next(name for name, q in broken["questions"].items() if q["type"] == "choice")
    broken["answers"][qid]["choice"] = 99
    verdict = judges.LabelJudge().judge(broken)
    assert not verdict.ok
    assert any("答案编号越界" in problem for problem in verdict.raw["problems"])


# ---------- workflow：OOD 档、判别器名单、审计里的坏产物 ----------


def test_the_ood_split_skips_slicing_and_keeps_only_unseen_values() -> None:
    """输入：值留出的采样参数。输出：case 列表。

    预期行为：OOD 档直接收 `build_cases` 筛出来的组合（不切片），且每条都至少含一个
    训练侧没见过的取值。
    """
    ood = cases_for(OOD_CONFIG)
    assert ood
    assert all(case.unseen_dims() for case in ood)
    assert shell_workflow.ShellWorkflow().config_for("test_ood") is OOD_CONFIG


def test_make_judges_rejects_an_unknown_name() -> None:
    """输入：不存在的判别器名。输出：KeyError。预期行为：报可用名单，不静默跳过。"""
    with pytest.raises(KeyError, match="未知判别器"):
        shell_workflow.ShellWorkflow().make_judges(("bogus",), "zh")


def test_available_locales_are_the_language_files() -> None:
    """输入：无。输出：语言列表。预期行为：就是 `locales/` 下的文件名。"""
    assert shell_workflow.ShellWorkflow().available_locales() == locales.available() == ["en", "zh"]


def test_audit_flags_a_product_whose_command_cannot_be_read(tmp_path: Path) -> None:
    """输入：一条命令反解不出来的产物。输出：(报告, 缺失项)。

    预期行为：记成 `label_mismatch` 并在缺失项里说明「反解不出来」——审计看的是产物本身，
    老产物事后被改坏也要能被点名。
    """
    row, _ = _row(_case())
    broken = copy.deepcopy(row)
    broken["state"] = broken["state"].replace("bash（Linux 容器）", "cmd.exe（Windows）")
    assert broken["state"] != row["state"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [broken])
    report, missing = shell_workflow.ShellWorkflow().audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("反解不出来" in item for item in missing)
