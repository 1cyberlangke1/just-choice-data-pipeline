"""命令审批的文本外壳：提示词、state 往返、命令解析、形状校验。"""

from __future__ import annotations

import pytest

from workflows.shell import locales, render
from workflows.shell.features import TagSet

CASE = TagSet(dialect="bash", goal="remove_tree", target="production", elevation="none")
PS_CASE = TagSet(dialect="powershell", goal="list_dir", target="workdir", elevation="elevated")


def test_prompt_lists_every_feature_line() -> None:
    """输入：一组特征。输出：无。

    预期行为：提示词里每个维度各一行（按 `DIM_ORDER`），且带上语言名；不含任何答案。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(CASE, loc, 0)
    for dim in render.DIM_ORDER:
        assert f"{loc['dims'][dim]}" in prompt
    assert loc["language_name"] in prompt
    assert "verdict" not in prompt
    assert "blast_radius" not in prompt


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_state_roundtrip(lang: str) -> None:
    """输入：语言。输出：无。

    预期行为：`to_state` 拼出的 state 能被 `command_of` 原样读回来；`parse_state`
    不认语言也能读（它把各语言的标记都试一遍）。
    """
    loc = locales.load(lang)
    data = {"goal_line": "清掉构建产物", "command": "rm -rf /srv/build", "note": "递归删除"}
    state = render.to_state(data, loc, "bash")
    assert render.command_of(state, loc) == "rm -rf /srv/build"
    assert render.parse_state(state).command == "rm -rf /srv/build"
    assert not render.parse_state(state).is_empty
    assert loc["meta"]["env_label"] in state


def test_command_of_returns_empty_when_missing() -> None:
    """输入：没有命令行的 state。输出：无。预期行为：返回空串，不抛异常。"""
    loc = locales.load("zh")
    assert render.command_of("环境：bash\n", loc) == ""
    assert render.command_of(None, loc) == ""
    assert render.parse_state(None).is_empty


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"command": "   "}, "空"),
        ({"command": "rm -rf /a\nrm -rf /b"}, "多行"),
        ({"command": "```\nrm -rf /a\n```"}, "围栏"),
        ({"command": "{0:N2} MB -f ((Get-ChildItem).Sum / 1MB)"}, "脚本块"),
        ({"command": "} else {"}, "脚本块"),
        ({"command": "查看 $HOME/src 目录下的文件，获取列表信息。"}, "非 ASCII"),
    ],
)
def test_check_command_rejects_broken_shapes(data: dict[str, str], match: str) -> None:
    """输入：坏草稿。输出：无。预期行为：报人话，点名是哪一类问题。"""
    with pytest.raises(ValueError, match=match):
        render.check_command(data)


def test_check_command_accepts_normal() -> None:
    """输入：正常草稿。输出：无。预期行为：不抛异常（尾部换行会被 strip 掉）。"""
    render.check_command({"command": "  ls -la /home/dev  \n"})


def test_check_command_rejects_a_shebang() -> None:
    """输入：以 shebang 开头的命令。输出：无。预期行为：报人话——那是脚本，不是一条命令。"""
    with pytest.raises(ValueError, match="shebang"):
        render.check_command({"command": "#!/bin/sh ls -l /srv/data"})


def _draft(**overrides: str) -> dict[str, str]:
    """输入：要覆盖的字段。输出：一份合规的草稿（实跑里模型写出来的样子）。"""
    base = {
        "goal_line": "看看 /home/dev/app 里有什么",
        "command": "ls -la /home/dev/app",
        "note": "列出该目录下的文件，不会改动任何东西",
    }
    base.update(overrides)
    return base


def test_check_draft_accepts_a_clean_draft() -> None:
    """输入：合规草稿。输出：无。预期行为：不抛异常。"""
    render.check_draft(_draft(), locales.load("zh"))


@pytest.mark.parametrize("field", ["goal_line", "note"])
def test_check_draft_rejects_empty_and_multi_line_text(field: str) -> None:
    """输入：目标或备注是空的 / 多行的草稿。输出：无。预期行为：报人话并点名字段。"""
    loc = locales.load("zh")
    with pytest.raises(ValueError, match="是空的"):
        render.check_draft(_draft(**{field: "   "}), loc)
    with pytest.raises(ValueError, match="是多行"):
        render.check_draft(_draft(**{field: "第一行\n第二行"}), loc)


@pytest.mark.parametrize("prefix", ["用户目标：", "用户目标:", "拟执行命令:"])
def test_check_draft_rejects_a_repeated_field_label(prefix: str) -> None:
    """输入：把字段名自己写进内容的草稿。输出：无。

    预期行为：报人话——实跑里 6 条有 2 条写成 `用户目标：用户目标：查看…`，
    拼进 state 就是一行重复标记，洗不掉。
    """
    with pytest.raises(ValueError, match="把字段名"):
        render.check_draft(_draft(goal_line=f"{prefix}看看 /home/dev/app"), locales.load("zh"))


def test_check_draft_rejects_a_goal_that_is_the_command() -> None:
    """输入：目标那一行直接抄了命令。输出：无。

    预期行为：报人话——这类样本的「目标说明」是假的（实跑 10 条里 4 条这么写）。
    """
    with pytest.raises(ValueError, match="写成了命令"):
        render.check_draft(_draft(goal_line="ls -la /home/dev/app"), locales.load("zh"))


def test_label_leak_reads_the_marker_from_the_locale() -> None:
    """输入：中英文文本 + 对应语言字典。输出：无。

    预期行为：行首标记随语言变——英文里 `User goal:` 也算泄漏，代码里没写死中文；
    正文里正常提到「环境」这类词不算泄漏。
    """
    loc = locales.load("en")
    assert render.label_leak("User goal: list the files", loc) == "User goal"
    assert render.label_leak("list the files in /home/dev", loc) == ""
    assert render.label_leak("环境 变量要改", locales.load("zh")) == ""


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("sudo rm -rf /srv/build", True),
        ("rm -rf /srv/build", False),
        ("ls -la /home/dev", False),
    ],
)
def test_elevated_in(command: str, expected: bool) -> None:
    """输入：命令。输出：无。预期行为：只认提权写法，大小写不敏感。"""
    assert render.elevated_in(command) is expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("Get-ChildItem -Path C:\\", True),
        ("Remove-Item -Path a.txt", True),
        ("ls -la /home/dev", False),
        ('grep -rn "test-" /home/dev', False),
    ],
)
def test_powershell_verb_in(command: str, expected: bool) -> None:
    """输入：命令。输出：无。

    预期行为：`动词-名词` 后面必须真的接字母——`grep -rn "test-"` 这种带引号的词
    不能被误判成 cmdlet（误判会直接拒掉好数据）。
    """
    assert render.powershell_verb_in(command) is expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git reset --hard HEAD~1", True),
        ("docker rm -f web-1", True),
        ("Get-ChildItem -Path C:\\", False),
        ("", False),
    ],
)
def test_external_tool_in(command: str, expected: bool) -> None:
    """输入：命令。输出：无。预期行为：只看第一个词是不是方言无关的外部工具。"""
    assert render.external_tool_in(command) is expected


def test_build_task_carries_tags_and_model() -> None:
    """输入：特征。输出：无。预期行为：任务带 tags、用 `ShellDraft` 当响应模型。"""
    loc = locales.load("zh")
    task = render.build_task(PS_CASE, loc, "shell-train-0000000-render")
    assert task.task_id == "shell-train-0000000-render"
    assert task.spec["tags"] == PS_CASE.as_dict()
    assert task.response_model is render.ShellDraft
    assert "PowerShell" in task.prompt
