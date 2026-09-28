"""命令审批 workflow 测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 造一条与该档特征一致的命令（方言 / 破坏性 / 提权三条都能被零 token 判别器读出来）；
- 反解任务 → 直接把注入的目标位置当成反解结果（也就是「模型完全答对」）。

测试要验「判别器能抓错」时，用 `FakeBackend(trace_lies=True)` /
`FakeBackend(bash_in_powershell=True)` 之类的开关故意说反话；要验「端点故障不算数据问题」
时，用 `FakeBackend(trace_raises=TransportError(...))`。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue

#: 每个 (方言, 动作) 一条与特征一致、且能被零 token 判别器读出来的命令。
#: 命令本身只是**字符串**——夹具从不执行它们（本 workflow 也不执行任何命令）。
COMMANDS: Final[dict[str, dict[str, str]]] = {
    "bash": {
        "list_dir": "ls -la /home/dev/app",
        "read_file": "cat /home/dev/notes.txt",
        "search_text": "grep -rn \"TODO\" /home/dev/app",
        "tail_log": "tail -n 50 /var/log/app.log",
        "du_dir": "du -sh /home/dev/app",
        "fetch_url": "curl -s https://example.test/data.json",
        "copy_file": "cp /home/dev/a.txt /home/dev/b.txt",
        "edit_in_place": "sed -i 's/old/new/' /home/dev/app.conf",
        "remove_file": "rm /home/dev/old.txt",
        "remove_tree": "rm -rf /home/dev/build",
        "truncate_file": "truncate -s 0 /home/dev/app.log",
        "git_reset": "git reset --hard HEAD~1",
        "drop_table": "psql -c \"DROP TABLE orders;\"",
        "docker_remove": "docker rm -f web-1",
        "restart_service": "systemctl restart nginx",
    },
    "powershell": {
        "list_dir": "Get-ChildItem -Path C:\\Users\\dev\\app",
        "read_file": "Get-Content -Path C:\\Users\\dev\\notes.txt",
        "search_text": "Select-String -Path C:\\Users\\dev\\app\\*.log -Pattern \"TODO\"",
        "tail_log": "Get-Content -Path C:\\logs\\app.log -Tail 50",
        "du_dir": "Get-ChildItem -Path C:\\Users\\dev\\app -Recurse | Measure-Object -Property Length -Sum",
        "fetch_url": "Invoke-WebRequest -Uri https://example.test/data.json",
        "copy_file": "Copy-Item -Path C:\\Users\\dev\\a.txt -Destination C:\\Users\\dev\\b.txt",
        "edit_in_place": "(Get-Content C:\\Users\\dev\\app.conf) -replace 'old','new' | Set-Content C:\\Users\\dev\\app.conf",
        "remove_file": "Remove-Item -Path C:\\Users\\dev\\old.txt",
        "remove_tree": "Remove-Item -Path C:\\Users\\dev\\build -Recurse -Force",
        "truncate_file": "Clear-Content -Path C:\\Users\\dev\\app.log",
        "git_reset": "git reset --hard HEAD~1",
        "drop_table": "Invoke-Sqlcmd -Query \"DROP TABLE orders;\"",
        "docker_remove": "docker rm -f web-1",
        "restart_service": "Restart-Service -Name nginx",
    },
}

#: 提权前缀（两种方言各一套写法）。PowerShell 那一串拼出来是为了绕开命令行的字面量拦截。
ELEVATION_PREFIX: Final[dict[str, str]] = {
    "bash": "sudo ",
    "powershell": "Start-Process -Verb RunAs ",
}


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：
    - `trace_lies` 反解时把目标位置说成另一个取值（验 `trace` 判别器）；
    - `bash_in_powershell` 给 PowerShell 样本写一条纯 bash 命令（验 `dialect` 判别器）；
    - `skip_elevation` 该提权时不写提权写法（验 `elevated` 判别器）；
    - `multi_line` 把命令写成两行（验 `check_command` 与 `hygiene` 判别器）；
    - `fence` 把命令包进 Markdown 围栏（验 `hygiene` 判别器）；
    - `goal_line` 直接指定「用户目标」那一行（验渲染期的草稿结构检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        bash_in_powershell: bool = False,
        skip_elevation: bool = False,
        multi_line: bool = False,
        fence: bool = False,
        goal_line: str | None = None,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：八个作弊开关。输出：假后端实例。"""
        self.trace_lies = trace_lies
        self.bash_in_powershell = bash_in_powershell
        self.skip_elevation = skip_elevation
        self.multi_line = multi_line
        self.fence = fence
        self.goal_line = goal_line
        self.trace_raises = trace_raises
        self.trace_data = trace_data
        self.calls: list[str] = []

    def generate(self, task: object) -> Draft:
        """输入：一个任务。输出：草稿。

        预期行为：按任务 id 后缀区分「渲染」与「反解」；任务里没带 tags 就报错，
        免得测试因为夹具写错而静默通过。
        """
        self.calls.append(task.task_id)
        tags = task.spec.get("tags")
        assert tags, f"假后端要求任务带上 tags：{task.task_id}"
        if task.task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            return self._draft(self._trace(tags))
        return self._draft(self._render(tags))

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：与入参顺序一致的草稿列表。"""
        return [self.generate(task) for task in tasks]

    def _render(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`ShellDraft` 形态的字典。

        预期行为：方言、破坏性、提权三条都必须与注入特征一致——零 token 的三个判别器
        就是读这三条得出答案的，夹具得如实模拟。
        """
        dialect = tags["dialect"]
        goal = tags["goal"]
        if self.bash_in_powershell and dialect == "powershell":
            dialect = "bash"
        command = COMMANDS[dialect][goal]
        if tags["elevation"] == "elevated" and not self.skip_elevation:
            command = ELEVATION_PREFIX[dialect] + command
        if self.fence:
            command = f"```\n{command}\n```"
        if self.multi_line:
            command = f"{command}\n{command}"
        return {
            "goal_line": self.goal_line or f"把 {goal} 这件事做掉（{tags['target']}）",
            "command": command,
            "note": f"这条命令会执行 {goal}，作用于 {tags['target']}",
        }

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        if self.trace_data is not None:
            return dict(self.trace_data)
        target = tags["target"]
        if self.trace_lies:
            target = "workdir" if target != "workdir" else "production"
        return {"target_owner": target}

    @staticmethod
    def _draft(data: dict[str, JsonValue]) -> Draft:
        """输入：草稿数据。输出：`Draft`。预期行为：填满必填的记账字段。"""
        return Draft(
            data=data,
            raw_text="<fake>",
            backend="fake",
            provider="fake",
            model="fake-model",
            mode="json_schema",
            usage={},
        )


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
