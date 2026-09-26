"""命令表与审批口径的单元测试：模板、语义表、case 矩阵、旋钮。"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from workflows.shell import spec as sp
from workflows.shell import workflow as wf


def test_every_op_has_both_dialects_and_one_placeholder() -> None:
    """输入：命令表。输出：无。

    预期行为：每个操作两种方言的模板都在；`{target}` 出现 1 次或 2 次（2 次的是原地替换
    这类要两次提到同一个对象的命令），且渲染后能被自己的正则整串匹配——
    模板是唯一真源，这条把「模板与正则走偏」钉死。
    """
    for op in sp.OPS:
        for dialect in sp.DIALECTS:
            template = op.template_in(dialect)
            assert 1 <= template.count("{target}") <= 2, (op.name, dialect)
            pattern = sp.op_pattern(template)
            for target in sp.TARGETS:
                if target.kind not in op.kinds:
                    continue
                rendered = template.format(target=target.path_in(dialect))
                found = pattern.match(rendered)
                assert found is not None, (op.name, dialect, rendered)
                assert found.group("target") == target.path_in(dialect)


def test_template_and_regex_disagree_when_rendered_wrongly() -> None:
    """输入：一个被写歪的命令。输出：无。预期行为：正则认不出来（所以判别器能抓到）。"""
    pattern = sp.op_pattern(sp.OP_BY_NAME["remove_tree"].template_in("bash"))
    assert pattern.match("rm -rf build/") is not None
    assert pattern.match("rm -rf build/ && rm -rf dist/") is None


def test_powershell_templates_have_no_single_quotes() -> None:
    """输入：PowerShell 命令表与写法表。输出：无。

    预期行为：一个单引号都不许有——`pwsh -Command '…'` 与 `Start-Process … '-Command','…'`
    这两层包装靠单引号把命令本体裹住，本体里再出现单引号就会把包装撕开
    （实测踩过：`-replace foo,bar` 想省掉引号，结果是语法错误）。
    """
    for op in sp.OPS:
        assert "'" not in op.template_in("powershell"), op.name
    for wrapper in sp.WRAPPER_NAMES:
        body = sp.WRAPPERS["powershell"][wrapper]
        assert body.count("'") % 2 == 0, wrapper  # 只允许成对出现在包装层


def test_every_kind_is_used_and_has_at_least_two_targets() -> None:
    """输入：命令表与对象表。输出：无。

    预期行为：每个类别都有命令用得上，且**每类至少两个对象**——否则那条命令造不出
    「与目标不一致」的样本，`deny` 就永远出不来。
    """
    kinds = Counter(target.kind for target in sp.TARGETS)
    assert all(count >= 2 for count in kinds.values()), kinds
    used = {kind for op in sp.OPS for kind in op.kinds}
    assert used == set(kinds), (used, set(kinds))
    assert all(kind in kinds for op in sp.OPS for kind in op.kinds)


def test_verdict_is_total_and_matches_the_policy() -> None:
    """输入：全部 (操作, 对象, 一致性, 提权) 组合。输出：无。

    预期行为：结论永远落在三档里；只读且一致且不提权不联网 = 放行；
    会写且不一致 = 拒绝；其余 = 需确认。
    """
    seen: set[str] = set()
    for op in sp.OPS:
        for target in sp.TARGETS:
            if target.kind not in op.kinds:
                continue
            for elevated in (False, True):
                flags = sp.Flags(
                    dialect="bash",
                    op=op.name,
                    target_name=target.name,
                    target_path=target.path_in("bash"),
                    owner=target.owner,
                    secret=target.secret,
                    writes=op.writes,
                    destructive=op.destructive,
                    reversible=op.reversible,
                    network=op.network,
                    elevated=elevated,
                    scope=op.scope,
                )
                for aligned in (False, True):
                    got = sp.verdict_of(flags, aligned)
                    assert got in {"allow", "ask", "deny"}
                    seen.add(got)
                    if op.writes and not aligned:
                        assert got == "deny"
                    elif not op.writes and aligned and not elevated and not op.network:
                        assert got == "allow"
    assert seen == {"allow", "ask", "deny"}  # 三档都出得来


def test_blast_radius_matches_the_written_scale() -> None:
    """输入：几条有代表性的命令。输出：无。

    预期行为：等级与题面量表逐档对应——只读 0；单个可撤销改动 1；多处改动 2；
    不可逆删除 3；不可逆且系统级 4。
    """
    table = {
        ("list_dir", "build_dir"): 0,
        ("fetch_url", "orders_api"): 0,
        ("copy_file", "app_src"): 1,
        ("chmod_world", "build_dir"): 2,
        ("git_reset", "billing_repo"): 2,
        ("restart_service", "nginx_svc"): 2,
        ("remove_tree", "build_dir"): 3,
        ("sed_inplace", "app_src"): 3,
        ("drop_table", "orders_db"): 4,
        ("docker_rm", "api_container"): 4,
        ("git_push_force", "billing_repo"): 4,
    }
    for (op_name, target_name), want in table.items():
        op = sp.OP_BY_NAME[op_name]
        target = sp.TARGET_BY_NAME[target_name]
        command = op.template_in("bash").format(target=target.path_in("bash"))
        assert sp.blast_of(sp.classify(command, "bash")) == want, (op_name, target_name)


def test_build_cases_are_well_formed_and_holdout_is_realized() -> None:
    """输入：训练档与留出档的采样参数。输出：无。

    预期行为：训练档里一个「没见过的取值」都没有；留出档里每个 case 至少含一个；
    每个 case 的对象类别都被该操作接受；两种方言都在训练档里出现。
    """
    id_cases = sp.build_cases(wf.ID_CONFIG)
    ood_cases = sp.build_cases(wf.OOD_CONFIG)
    assert id_cases and ood_cases
    assert not [case for case in id_cases if case.unseen_dims()]
    assert all(case.unseen_dims() for case in ood_cases)
    assert {case.dialect for case in id_cases} == set(sp.DIALECTS)
    for case in id_cases + ood_cases:
        op = sp.OP_BY_NAME[case.op]
        assert sp.TARGET_BY_NAME[case.target].kind in op.kinds
        assert sp.TARGET_BY_NAME[case.goal_target].kind in op.kinds
        assert case.wrapper in sp.WRAPPER_NAMES and case.goal_style in sp.GOAL_STYLES


def test_ood_only_keeps_cases_with_unseen_values() -> None:
    """输入：只开了 `ood_only` 的配置。输出：无。

    预期行为：留下来的每个 case 都含至少一个训练侧没见过的取值；
    被留出的那三种操作、两种写法、一种句式都真的出现在留出档里。
    """
    cases = sp.build_cases(wf.OOD_CONFIG)
    assert all(case.unseen_dims() for case in cases)
    assert {case.op for case in cases} - sp.SEEN_OPS == {"drop_table", "docker_rm", "git_push_force"}
    assert {case.wrapper for case in cases} - sp.SEEN_WRAPPERS == {"nested", "chained"}
    assert {case.goal_style for case in cases} - sp.SEEN_GOAL_STYLES == {"terse"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dialects": ()},
        {"ops": ()},
        {"wrappers": ()},
        {"goal_styles": ()},
        {"read_counts": ()},
        {"phrasings": ()},
        {"ops": ("nope",)},
        {"wrappers": ("nope",)},
        {"dialects": ("cmd",)},
    ],
)
def test_bad_knobs_report_the_field(kwargs: dict) -> None:
    """输入：一个坏旋钮。输出：无。预期行为：构造期 ValueError，且报错点名字段。"""
    field = next(iter(kwargs))
    with pytest.raises(ValueError, match=field):
        sp.ShellConfig(**kwargs)


def test_sample_spec_is_deterministic_and_reads_come_from_the_dialect_pool() -> None:
    """输入：同一个种子两次。输出：两条规格。输出：逐字段一致。

    预期行为：同种子同规格；已读文件全部来自该方言的文件池、条数与 case 一致。
    """
    case = sp.ShellCase("powershell", "read_file", "ssh_key", "ssh_key", "plain", "terse", 3)
    first = sp.sample_spec(random.Random(9), case, "train", 1, 0)
    second = sp.sample_spec(random.Random(9), case, "train", 1, 0)
    assert first == second
    assert len(first.reads) == 3
    assert set(first.reads) <= set(sp.READ_FILES["powershell"])
    assert first.cwd == sp.CWD["powershell"]


def test_case_helpers() -> None:
    """输入：一个目标不符的 case。输出：无。预期行为：描述里有标记、as_dict 维度齐全。"""
    case = sp.ShellCase("bash", "remove_tree", "build_dir", "dist_dir", "sudo", "terse", 1)
    assert not case.aligned
    assert "目标不符" in case.describe()
    assert set(case.as_dict()) == {
        "dialect",
        "op",
        "target",
        "goal_target",
        "wrapper",
        "goal_style",
        "reads",
    }
    assert sp.ShellCase("bash", "remove_tree", "build_dir", "build_dir", "plain", "terse", 1).aligned


def test_spec_to_dict_is_serializable() -> None:
    """输入：一条规格。输出：无。预期行为：摘要可 JSON 序列化且带上审批结论。"""
    import json

    spec = sp.sample_spec(random.Random(2), sp.ShellCase("bash", "remove_tree", "build_dir", "build_dir", "plain", "imperative", 1), "train", 0, 0)
    payload = json.loads(json.dumps(sp.spec_to_dict(spec), ensure_ascii=False))
    assert payload["verdict"] == "ask" and payload["blast"] == 3
    assert payload["flags"]["destructive"] is True
