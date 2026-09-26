"""命令**合法性**：把生成的每条命令交给真的解析器，**只解析、一条都不执行**。

为什么要有：主人明确要求「powershell + shell 都要合成，而且要保证命令确实是合法的」，
并且明确补了一句「不要实际执行任何命令」。所以这里的分工是：

1. bash 侧：每条命令跑 `bash -n`——只做语法分析，**不会**让命令跑起来；
2. PowerShell 侧：每条命令交给 `[System.Management.Automation.Language.Parser]::ParseInput`
   建 AST——同样不执行。这道实测抓到过真错：`-replace foo,bar` 少写引号是**语法错误**
   （`You must provide a value expression following the '-replace' operator`）；
3. 生成的命令本身**一条都不执行**：模板里就躺着 `rm -rf`、`chmod 777`、`git push --force`
   这类破坏性写法，靠「在临时目录里跑一遍」来验合法性，等于把风险押在「模板与 cwd 恰好
   都对」上，换来的却只是「这台机器上恰好能跑」——而静态解析已经能抓到真错，够了。

本机没有 bash / pwsh 时对应测试**跳过**，不假装通过。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from workflows.shell import spec as sp


def all_commands() -> list[tuple[str, str, str]]:
    """输入：无。输出：[(方言, 标签, 命令)]——全部 操作 × 对象 × 写法。"""
    out: list[tuple[str, str, str]] = []
    for dialect in sp.DIALECTS:
        for op in sp.OPS:
            for target in sp.TARGETS:
                if target.kind not in op.kinds:
                    continue
                bare = op.template_in(dialect).format(target=target.path_in(dialect))
                for wrapper in sp.WRAPPER_NAMES:
                    cmd = sp.WRAPPERS[dialect][wrapper].format(cmd=bare)
                    out.append((dialect, f"{op.name}/{target.name}/{wrapper}", cmd))
    return out


def find_bash() -> str | None:
    """输入：无。输出：能用的 bash 路径，或 None。

    预期行为：优先 MSYS 的原生 bash（快、`/tmp` 就是本机目录），退回 PATH 上的
    `bash`（本机是 WSL 的启动器）。两者都要先跑一次 `echo` 才算数。
    """
    candidates = [r"C:\msys64\usr\bin\bash.exe", shutil.which("bash") or ""]
    for path in candidates:
        if not path or not Path(path).exists():
            continue
        done = subprocess.run([path, "-c", "echo ok"], capture_output=True, text=True)
        if done.returncode == 0 and "ok" in done.stdout:
            return path
    return None


def find_pwsh() -> str | None:
    """输入：无。输出：能用的 pwsh 路径，或 None。"""
    found = shutil.which("pwsh") or r"C:\Program Files\PowerShell\7\pwsh.exe"
    if found and Path(found).exists():
        return found
    return None


def test_classify_roundtrips_every_generated_command() -> None:
    """输入：全部命令。输出：无。

    预期行为：每条命令都能反推回「它本来就是哪个操作、哪个对象、哪种方言、是否提权」——
    这是判别器能独立重算答案的前提。
    """
    checked = 0
    for dialect, label, cmd in all_commands():
        op_name, target_name, wrapper = label.split("/")
        flags = sp.classify(cmd, dialect)
        assert flags.op == op_name, label
        assert flags.target_name == target_name, label
        assert flags.dialect == dialect, label
        assert flags.elevated == (wrapper == "elevated"), label
        checked += 1
    assert checked > 500


def test_every_bash_command_is_syntactically_valid() -> None:
    """输入：bash 侧全部命令。输出：无。预期行为：`bash -n` 全部通过。"""
    bash = find_bash()
    if bash is None:
        pytest.skip("本机没有可用的 bash")
    bad: list[str] = []
    for _dialect, label, cmd in all_commands():
        if _dialect != "bash":
            continue
        done = subprocess.run([bash, "-n", "-c", cmd], capture_output=True, text=True)
        if done.returncode != 0:
            bad.append(f"{label}：{cmd} -> {done.stderr.strip()}")
    assert not bad, bad


PS_PARSE = r"""
$items = Get-Content -LiteralPath $args[0] -Raw -Encoding UTF8 | ConvertFrom-Json
$bad = @()
foreach ($it in $items) {
  $errs = $null
  [void][System.Management.Automation.Language.Parser]::ParseInput($it.cmd, [ref]$null, [ref]$errs)
  if ($errs -and $errs.Count -gt 0) { $bad += [pscustomobject]@{ label = $it.label; msg = $errs[0].Message } }
}
@($bad) | ConvertTo-Json -Depth 3 -Compress
"""


def test_every_powershell_command_parses(tmp_path: Path) -> None:
    """输入：PowerShell 侧全部命令。输出：无。预期行为：AST 解析零错误。"""
    pwsh = find_pwsh()
    if pwsh is None:
        pytest.skip("本机没有可用的 pwsh")
    items = [
        {"label": label, "cmd": cmd}
        for dialect, label, cmd in all_commands()
        if dialect == "powershell"
    ]
    # 载荷写到 pytest 的临时目录：写在仓库树里，被中断时就会留下一个**未跟踪**的垃圾文件，
    # `git add -A` 一不小心就把它提交进去了（实测在 `git status` 里撞到过）。
    payload = tmp_path / "_ps_payload.json"
    payload.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    try:
        done = subprocess.run(
            [pwsh, "-NoProfile", "-Command", PS_PARSE, str(payload)],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        text = done.stdout.strip()
        bad = [] if not text or text == "null" else json.loads(text)
        assert not bad, bad
    finally:
        payload.unlink(missing_ok=True)
