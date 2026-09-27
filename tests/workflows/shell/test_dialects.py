"""命令**合法性**：把生成的每条命令交给真的解析器，**只解析、一条都不执行**。

为什么要有：主人明确要求「powershell + shell 都要合成，而且要保证命令确实是合法的」，
并且明确补了一句「不要实际执行任何命令」。所以这里的分工是：

1. bash 侧：每条命令跑 `bash -n`——只做语法分析，**不会**让命令跑起来；
2. PowerShell 侧：每条命令交给 `[System.Management.Automation.Language.Parser]::ParseInput`
   建 AST——同样不执行；少一个右括号这类语法错误会被报出来
   （由 `test_the_parse_harness_reports_a_real_error` 锁住：坏命令**必须**被报出来）；
3. 生成的命令本身**一条都不执行**：模板里就躺着 `rm -rf`、`chmod 777`、`git push --force`
   这类破坏性写法，靠「在临时目录里跑一遍」来验合法性，等于把风险押在「模板与 cwd 恰好
   都对」上，换来的却只是「这台机器上恰好能跑」——而静态解析已经能抓到真错，够了。
4. **解析脚本用 `-File` 传参，不用 `-Command` + 尾随参数**（2026-09-27 的教训）：
   `pwsh -Command <脚本> <路径>` 里那个路径**不会**进 `$args`，而是被拼到脚本文本末尾；
   脚本以换行结尾时它就成了**独立语句**，被 PowerShell 交给默认程序去「执行」——本机
   `.json` 关联 Notepad4，于是每跑一次弹一个记事本，而脚本因为 `$args[0]` 是 null 什么都没
   解析、测试**空转通过**。现在改成 `-File`，并加了「解析条数必须等于载荷条数」的空转闸。

本机没有 bash / pwsh 时对应测试**跳过**，不假装通过。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
[pscustomobject]@{ parsed = @($items).Count; bad = @($bad) } | ConvertTo-Json -Depth 3 -Compress
"""


def run_ps_parse(pwsh: str, tmp_path: Path, items: list[dict[str, str]]) -> dict[str, Any]:
    """输入：pwsh 路径、临时目录、载荷条目。输出：`{"parsed": 解析条数, "bad": 报错列表}`。

    预期行为：载荷与解析脚本都写进 pytest 的临时目录（写仓库树会留下**未跟踪**的垃圾文件，
    `git add -A` 一不小心就提交进去了），并用 **`-File`** 把载荷路径交给脚本的 `$args[0]`。
    `-File` 是唯一能把尾随参数真正送进 `$args` 的写法（见模块 docstring 第 4 条）。
    """
    payload = tmp_path / "_ps_payload.json"
    script = tmp_path / "_ps_parse.ps1"
    payload.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    script.write_text(PS_PARSE, encoding="utf-8")
    done = subprocess.run(
        [pwsh, "-NoProfile", "-File", str(script), str(payload)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    text = done.stdout.strip()
    assert text, f"解析脚本什么都没输出（stderr：{done.stderr.strip()}）"
    report: dict[str, Any] = json.loads(text)
    return report


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
    report = run_ps_parse(pwsh, tmp_path, items)
    # 空转闸：解析条数必须等于载荷条数。少了就说明脚本根本没读到载荷，「零错误」不算数
    # （2026-09-27 实测踩过：`-Command` + 尾随参数让 `$args[0]` 为 null，测试空转通过）。
    assert report["parsed"] == len(items), report
    assert not report["bad"], report["bad"][:5]


def test_the_parse_harness_reports_a_real_error(tmp_path: Path) -> None:
    """输入：一条**故意写坏**的 PowerShell 命令。输出：无。

    预期行为：解析器必须把它报出来。这条锁的是「空转闸」本身——脚本一旦读不到载荷、
    或断言只看「没有错误」，这里就会退化成通过，那是 2026-09-27 修掉的那种假绿。
    """
    pwsh = find_pwsh()
    if pwsh is None:
        pytest.skip("本机没有可用的 pwsh")
    report = run_ps_parse(pwsh, tmp_path, [{"label": "broken", "cmd": "if ($true { Write-Output 1 }"}])
    assert report["parsed"] == 1, report
    assert report["bad"], "坏命令没被报出来——解析器没真的干活"
