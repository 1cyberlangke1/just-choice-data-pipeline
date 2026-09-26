"""判别器的单元测试：好数据通过、坏数据被点出来（两种判别器各管一半）。"""

from __future__ import annotations

import copy
import random
from typing import Any

from jc import assemble
from workflows.shell import judges, locales, questions, render
from workflows.shell import spec as sp


def make_row(case: sp.ShellCase, seed: int = 1, locale: str = "zh") -> tuple[dict[str, Any], sp.CommandSpec]:
    """输入：case、种子、语言。输出：(契约样本, 真值规格)。

    预期行为：用真的组装器造样本，所以判到的东西与管线里跑的一模一样。
    """
    loc = locales.load(locale)
    rng = random.Random(seed)
    spec = sp.sample_spec(rng, case, "train", 0, 0)
    labeled = questions.build(spec, loc, spec.phrasing, rng)
    row = assemble.assemble_sample("shell-train-0000000", render.build(spec, loc), labeled, rng)
    return row, spec


def sample_case(**overrides: Any) -> sp.ShellCase:
    """输入：维度覆盖项。输出：一个 case（默认：bash 上一条与目标一致的删除命令）。"""
    fields = {
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


def test_good_row_passes_both_judges() -> None:
    """输入：一条正常样本。输出：无。预期行为：两个判别器都通过。"""
    row, spec = make_row(sample_case())
    assert judges.RoundtripJudge().judge(row, spec).ok
    assert judges.LabelJudge().judge(row, spec).ok


def test_roundtrip_catches_a_mangled_command() -> None:
    """输入：把 state 里的命令改掉。输出：无。

    预期行为：结构判别器判不通过（它比的是「渲染出来的文本」与真值 spec）。
    """
    row, spec = make_row(sample_case())
    row = copy.deepcopy(row)
    row["state"] = row["state"].replace("rm -rf build/", "rm -rf dist/")
    verdict = judges.RoundtripJudge().judge(row, spec)
    assert not verdict.ok and "命令" in verdict.reason


def test_roundtrip_skips_without_a_spec() -> None:
    """输入：没有 spec。输出：无。预期行为：跳过而不是误判（`skipped` 标记）。"""
    row, _ = make_row(sample_case())
    verdict = judges.RoundtripJudge().judge(row, None)
    assert verdict.ok and verdict.raw.get("skipped") is True


def test_roundtrip_reports_an_unreadable_environment() -> None:
    """输入：执行环境被改坏。输出：无。预期行为：反解失败 → 判不通过并说明原因。"""
    row, spec = make_row(sample_case())
    row = copy.deepcopy(row)
    row["state"] = row["state"].replace("bash（Linux 容器）", "cmd.exe（Windows）")
    verdict = judges.RoundtripJudge().judge(row, spec)
    assert not verdict.ok and "反解失败" in verdict.reason


def test_label_catches_each_tampered_answer() -> None:
    """输入：逐族把答案改错。输出：无。

    预期行为：每一族被改错都要被判出来——判别器只看文本与题面，不读 spec 里的答案。
    """
    row, _ = make_row(sample_case())
    for qid, answer in row["answers"].items():
        broken = copy.deepcopy(row)
        if answer["type"] == "noul":
            broken["answers"][qid]["noul"] = not answer["noul"]
        elif answer["type"] == "choice":
            broken["answers"][qid]["choice"] = (answer["choice"] + 1) % len(
                broken["questions"][qid]["criteria"]
            )
        else:
            broken["answers"][qid]["score"] = (answer["score"] + 1) % len(
                broken["questions"][qid]["criteria"]
            )
        verdict = judges.LabelJudge().judge(broken)
        assert not verdict.ok, qid
        assert any(qid in problem for problem in verdict.raw["problems"]), qid


def test_label_catches_a_goal_that_names_another_object() -> None:
    """输入：把用户目标里的对象换成另一个。输出：无。

    预期行为：`matched_goal` 与 `verdict` 都要跟着变，所以答案会对不上——
    判据是「目标里点名的对象」，不是 spec 里记的字段。
    """
    row, _ = make_row(sample_case(op="list_dir", wrapper="plain"))
    broken = copy.deepcopy(row)
    broken["state"] = broken["state"].replace("查看 build/ 里有哪些文件", "查看 dist/ 里有哪些文件")
    verdict = judges.LabelJudge().judge(broken)
    assert not verdict.ok
    assert any("matched_goal" in problem for problem in verdict.raw["problems"])


def test_label_reports_an_unclassifiable_command() -> None:
    """输入：命令被换成表外的写法。输出：无。预期行为：判不通过并说明认不出来。"""
    row, _ = make_row(sample_case())
    broken = copy.deepcopy(row)
    broken["state"] = broken["state"].replace("rm -rf build/", "rm -rf /etc/passwd")
    verdict = judges.LabelJudge().judge(broken)
    assert not verdict.ok
    assert any("命令认不出来" in problem for problem in verdict.raw["problems"])


def test_label_reports_an_unknown_family() -> None:
    """输入：产物里混进一个不存在的题族。输出：无。预期行为：报「未知题族」。"""
    row, _ = make_row(sample_case())
    broken = copy.deepcopy(row)
    broken["questions"]["made_up"] = {"type": "noul", "instructions": "x", "criteria": {"true": "a", "false": "b"}}
    broken["answers"]["made_up"] = {"type": "noul", "noul": True}
    verdict = judges.LabelJudge().judge(broken)
    assert not verdict.ok
    assert any("未知题族" in problem for problem in verdict.raw["problems"])


def test_named_target_prefers_the_longest_path() -> None:
    """输入：同时含库名与 URL 的目标句。输出：无。

    预期行为：认最长匹配——`orders`（库）是 `…/v1/orders`（URL）的子串，
    只取第一个命中会把 URL 目标认成库目标。
    """
    assert judges.named_target("我想看一下 https://api.example.com/v1/orders 返回什么", "bash").name == "orders_api"
    assert judges.named_target("删掉 orders 表", "bash").name == "orders_db"
    assert judges.named_target("什么都没有", "bash") is None


def test_recompute_matches_the_spec_facts() -> None:
    """输入：一批样本。输出：无。预期行为：文本重算出来的 8 项与 spec 推出来的一模一样。"""
    for index, case in enumerate(
        [
            sample_case(),
            sample_case(op="read_file", target="ssh_key", wrapper="elevated"),
            sample_case(dialect="powershell", op="git_push_force", target="billing_repo", goal_target="auth_repo"),
        ]
    ):
        row, spec = make_row(case, seed=index)
        parsed = judges.parse.parse_request(row["state"], judges.parse.merged_labels())
        got = judges.recompute(parsed)
        assert got["verdict"] == spec.verdict
        assert got["blast_radius"] == spec.blast
        assert got["target_owner"] == spec.flags.owner
        assert got["touches_secrets"] == spec.flags.secret
