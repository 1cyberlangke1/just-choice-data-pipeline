"""判别器测试：四个零 token 的规则判别器 + 一次后端调用的反解判别器。"""

from __future__ import annotations

import pytest

from jc.backend.types import AllProvidersDown, GenerationBackend, TransportError
from workflows.shell import features, judges as judges_mod, locales, render
from workflows.shell.features import TagSet
from workflows.shell.spec import ShellSpec


def spec_for(tags: TagSet, seq: int = 0) -> ShellSpec:
    """输入：一组特征。输出：一个规格。预期行为：切分与问法固定，测试只关心标签。"""
    return ShellSpec(tags=tags, phrasing=0, split="train", seq=seq)


def state_for(command: str, dialect: str = "bash", lang: str = "zh") -> str:
    """输入：一条命令、方言、语言。输出：一条完整的 state 文本。

    预期行为：四行齐全（环境 / 目标 / 命令 / 备注）；零 token 判别器只读「命令」那一行。
    **本文件里的命令全是字符串，从不执行**。
    """
    return render.to_state(
        {"goal_line": "把这件事做掉", "command": command, "note": "客观描述"},
        locales.load(lang),
        dialect,
    )


# ---------- 方言 ----------


def test_dialect_judge_rejects_bash_that_uses_a_cmdlet() -> None:
    """注入 bash，命令里却混了 PowerShell 的动词-名词写法 → 不通过，并点名那条命令。"""
    spec = spec_for(TagSet("bash", "list_dir", "workdir", "none"))
    verdict = judges_mod.DialectJudge().judge(
        {"state": state_for("Get-ChildItem -Path C:\\Users\\dev")}, spec
    )
    assert verdict.ok is False
    assert "Get-ChildItem" in verdict.reason


def test_dialect_judge_accepts_each_dialect_written_in_its_own_way() -> None:
    """bash 写法配 bash 注入通过；PowerShell cmdlet 配 PowerShell 注入通过。"""
    bash = spec_for(TagSet("bash", "list_dir", "workdir", "none"))
    powershell = spec_for(TagSet("powershell", "list_dir", "workdir", "none"))
    assert judges_mod.DialectJudge().judge({"state": state_for("ls -la /home/dev")}, bash).ok is True
    assert (
        judges_mod.DialectJudge()
        .judge(
            {"state": state_for("Get-ChildItem -Path C:\\Users\\dev", "powershell")}, powershell
        )
        .ok
        is True
    )


def test_dialect_judge_accepts_external_tools_in_powershell() -> None:
    """`git` / `docker` 这类外部工具两种方言写法一样，PowerShell 里不该被当成串味。"""
    spec = spec_for(TagSet("powershell", "git_reset", "workdir", "none"))
    verdict = judges_mod.DialectJudge().judge(
        {"state": state_for("git reset --hard HEAD~1", "powershell")}, spec
    )
    assert verdict.ok is True


def test_dialect_judge_rejects_powershell_without_any_cmdlet() -> None:
    """PowerShell 注入 + 一条既没有 cmdlet 也不是外部工具的 bash 命令 → 不通过。"""
    spec = spec_for(TagSet("powershell", "list_dir", "workdir", "none"))
    verdict = judges_mod.DialectJudge().judge(
        {"state": state_for("ls -la /home/dev", "powershell")}, spec
    )
    assert verdict.ok is False
    assert "cmdlet" in verdict.reason


def test_dialect_judge_skips_without_spec() -> None:
    """没有 spec 就没有真值：记成跳过（`skipped=True`），不误判。"""
    verdict = judges_mod.DialectJudge().judge({"state": state_for("ls -la /tmp")})
    assert verdict.ok is True and verdict.skipped is True


# ---------- 破坏性 ----------


def test_destructive_in_matches_whole_words() -> None:
    """输入：一条命令。输出：无。

    预期行为：只认破坏性写法，且**按词边界**——`firm` 里的 `rm` 不算破坏性。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for command, expected in [
        ("rm /home/dev/old.txt", True),
        ("firm handshake --help", False),
        ("git reset --hard HEAD~1", True),
        ("docker rm -f web-1", True),
        (": > /var/log/app.log", True),
        # `> file` 单独成一条命令就是把文件截断（实跑里模型就这么写 truncate_file）
        ("> /home/dev/logs/app.log", True),
        # `>> file` 是追加，不毁数据，不算破坏性
        (">> /home/dev/logs/app.log", False),
        ("ls -la /home/dev", False),
    ]:
        try:
            assert judges_mod.destructive_in(command) is expected
        except AssertionError as exc:
            bad.append(f"{(command, expected)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_destructive_judge_reads_the_command() -> None:
    """零 token：命令里有没有破坏性写法 → 这一族的真值。

    输入：注入的动作、命令、期望的判别结果。输出：无。
    预期行为：判别器比的是「命令读出来的破坏性」与「动作应当有的破坏性」是否**一致**，
    所以矛盾的那两行（只读动作却删东西 / 删除动作却只是列目录）才该被判不通过。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for goal, command, ok in [
        # 命令与注入的破坏性一致 → 通过
        ("remove_file", "rm /home/dev/old.txt", True),
        ("remove_tree", "rm -rf /home/dev/build", True),
        ("truncate_file", "truncate -s 0 /home/dev/app.log", True),
        ("list_dir", "ls -la /home/dev", True),
        ("copy_file", "cp /home/dev/a.txt /home/dev/b.txt", True),
        # 命令与注入的破坏性矛盾 → 不通过
        ("list_dir", "rm -rf /tmp/x", False),
        ("remove_file", "ls -la /home/dev", False),
    ]:
        try:
            tags = TagSet("bash", goal, "workdir", "none")
            verdict = judges_mod.DestructiveJudge().judge({"state": state_for(command)}, spec_for(tags))
            assert verdict.ok is ok, verdict.reason
        except AssertionError as exc:
            bad.append(f"{(goal, command, ok)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_destructive_judge_flags_a_mismatch_and_keeps_the_command() -> None:
    """注入只读动作、命令却删了东西 → 不通过，并把命令原文放进 raw 便于排查。"""
    spec = spec_for(TagSet("bash", "list_dir", "workdir", "none"))
    verdict = judges_mod.DestructiveJudge().judge({"state": state_for("rm -rf /tmp/x")}, spec)
    assert verdict.ok is False
    assert verdict.raw["command"] == "rm -rf /tmp/x"


def test_destructive_judge_skips_without_spec() -> None:
    """没有 spec 就没有真值：记成跳过。"""
    verdict = judges_mod.DestructiveJudge().judge({"state": state_for("rm -rf /tmp/x")})
    assert verdict.ok is True and verdict.skipped is True


# ---------- 提权 ----------


@pytest.mark.parametrize(
    ("elevation", "command", "expected"),
    [
        ("elevated", "sudo ls -la /home/dev", True),
        ("none", "ls -la /home/dev", True),
        ("elevated", "ls -la /home/dev", False),
        ("none", "sudo ls -la /home/dev", False),
    ],
)
def test_elevated_judge_reads_the_elevation_marker(
    elevation: str, command: str, expected: bool
) -> None:
    """零 token：命令里有没有提权写法 → 这一族的真值。"""
    tags = TagSet("bash", "list_dir", "workdir", elevation)
    verdict = judges_mod.ElevatedJudge().judge({"state": state_for(command)}, spec_for(tags))
    assert verdict.ok is expected, verdict.reason


def test_elevated_judge_skips_without_spec() -> None:
    """没有 spec 就没有真值：记成跳过。"""
    verdict = judges_mod.ElevatedJudge().judge({"state": state_for("sudo ls")})
    assert verdict.ok is True and verdict.skipped is True


# ---------- 卫生 ----------


def test_hygiene_judge_accepts_a_clean_state() -> None:
    """四行齐全、命令单行 → 通过。"""
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state_for("ls -la /home/dev")})
    assert verdict.ok is True, verdict.reason


@pytest.mark.parametrize("residue", ["```", "```bash", "<br>"])
def test_hygiene_judge_rejects_format_residue(residue: str) -> None:
    """单独成行的围栏、HTML 换行标签都算格式残留。"""
    state = state_for("ls -la /home/dev") + f"\n{residue}"
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state})
    assert verdict.ok is False
    assert residue in verdict.reason


def test_hygiene_judge_flags_a_multi_line_command() -> None:
    """命令那一行后面还跟着非字段的行 → 那是命令的续行，不通过。

    输入：`拟执行命令：ls -la` 后面紧跟一行 `rm -rf /tmp`。输出：Verdict。
    预期行为：判不通过并点名「多行」——`command_of` 只取一行，第二行起得靠「后面那行
    不是任何字段的行首标记」认出来（否则多行脚本会被当成一条单行命令收下）。
    """
    loc = locales.load("zh")
    state = render.to_state(
        {"goal_line": "看看", "command": "ls -la /home/dev\nrm -rf /tmp", "note": "描述"},
        loc,
        "bash",
    )
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state})
    assert verdict.ok is False
    assert "多行" in verdict.reason


def test_hygiene_judge_flags_an_empty_command() -> None:
    """「拟执行命令」那一行是空的 → 不通过（state 没法当「一条命令」审）。"""
    state = "环境：bash\n用户目标：看看\n拟执行命令：\n备注：描述"
    verdict = judges_mod.HygieneJudge("zh").judge({"state": state})
    assert verdict.ok is False
    assert "空的" in verdict.reason


def test_hygiene_judge_reads_the_marker_from_the_locale() -> None:
    """输入：英文 state（行首标记是 ASCII 的）。输出：Verdict。

    预期行为：认得出「Proposed command:」那一行，代码里没写死中文——换个语言照样认。
    """
    loc = locales.load("en")
    state = render.to_state(
        {"goal_line": "list things", "command": "ls -la /home/dev", "note": "lists files"},
        loc,
        "bash",
    )
    assert judges_mod.HygieneJudge("en").judge({"state": state}).ok is True
    assert judges_mod.HygieneJudge("en").judge({"state": state + "\n```"}).ok is False


def test_hygiene_judge_skips_a_non_text_state() -> None:
    """state 不是文本就放过：那不是「格式残留」，是上游接线错了。

    输入：state 是个字典。输出：Verdict。预期行为：通过且理由写明「不是文本」——
    本 workflow 的渲染器只产字符串，这条分支是产物侧的兜底（与 mail 的 hygiene 同口径）。
    """
    verdict = judges_mod.HygieneJudge("zh").judge({"state": {"not": "text"}})
    assert verdict.ok is True
    assert "不是文本" in verdict.reason


# ---------- 反解 ----------


def test_build_verify_prompt_carries_the_state_and_the_options() -> None:
    """反解提示词要把 state 原样带上，并列出与数据集一致的选项。"""
    prompt = judges_mod.build_verify_prompt(state_for("ls -la /home/dev"), locales.load("zh"))
    assert "ls -la /home/dev" in prompt
    for name in features.TARGETS:
        assert name in prompt


def test_trace_judge_passes_when_reverse_parse_agrees(fake_backend: GenerationBackend) -> None:
    """反解与注入一致 → 通过，并在 raw 里留下反解结果。"""
    spec = spec_for(TagSet("bash", "list_dir", "workdir", "none"))
    judge = judges_mod.TraceJudge(backend=fake_backend(), locale="zh")
    verdict = judge.judge({"state": state_for("ls -la /home/dev")}, spec)
    assert verdict.ok is True and verdict.skipped is False
    assert verdict.raw["traced"]["target_owner"] == "workdir"


def test_trace_judge_rejects_when_reverse_parse_disagrees(fake_backend: GenerationBackend) -> None:
    """反解说成另一个位置 → 不通过，理由里同时列出注入值与反解值。"""
    spec = spec_for(TagSet("bash", "list_dir", "workdir", "none"))
    judge = judges_mod.TraceJudge(backend=fake_backend(trace_lies=True), locale="zh")
    verdict = judge.judge({"state": state_for("ls -la /home/dev")}, spec)
    assert verdict.ok is False
    assert "workdir" in verdict.reason and "production" in verdict.reason


def test_trace_judge_skips_without_spec_or_backend() -> None:
    """没有 spec、或没有后端，都记成跳过（没判成），不是数据不合格。"""
    spec = spec_for(TagSet("bash", "list_dir", "workdir", "none"))
    no_spec = judges_mod.TraceJudge(backend=None, locale="zh").judge({"state": "x"})
    no_backend = judges_mod.TraceJudge(backend=None, locale="zh").judge({"state": "x"}, spec)
    assert no_spec.ok is True and no_spec.skipped is True
    assert no_backend.ok is True and no_backend.skipped is True


def test_trace_judge_skips_an_endpoint_failure(fake_backend: GenerationBackend) -> None:
    """端点故障只算「没判成」——记成「不通过」会把端点抖动写成数据质量问题。"""
    spec = spec_for(TagSet("bash", "list_dir", "workdir", "none"))
    judge = judges_mod.TraceJudge(
        backend=fake_backend(trace_raises=TransportError("网关 504")), locale="zh"
    )
    verdict = judge.judge({"state": state_for("ls -la /home/dev")}, spec)
    assert verdict.ok is True and verdict.skipped is True
    assert "端点故障" in verdict.reason
    assert verdict.raw == {"error": "TransportError"}


def test_trace_judge_reraises_when_every_provider_is_down(
    fake_backend: GenerationBackend,
) -> None:
    """所有 provider 都挂了要冒泡（整批停下是唯一的停机条件），不许吞成「跳过」。"""
    spec = spec_for(TagSet("bash", "list_dir", "workdir", "none"))
    judge = judges_mod.TraceJudge(
        backend=fake_backend(trace_raises=AllProvidersDown("全挂")), locale="zh"
    )
    with pytest.raises(AllProvidersDown):
        judge.judge({"state": state_for("ls -la /home/dev")}, spec)


def test_trace_judge_skips_when_the_draft_misses_fields(fake_backend: GenerationBackend) -> None:
    """反解草稿缺字段 → 判不了，记成跳过并点名缺了哪个字段。"""
    spec = spec_for(TagSet("bash", "list_dir", "workdir", "none"))
    judge = judges_mod.TraceJudge(backend=fake_backend(trace_data={}), locale="zh")
    verdict = judge.judge({"state": state_for("ls -la /home/dev")}, spec)
    assert verdict.ok is True and verdict.skipped is True
    assert "缺字段" in verdict.reason and "target_owner" in verdict.reason
