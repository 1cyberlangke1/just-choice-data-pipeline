"""行级规则检查测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec` 造草稿，所以它天然「知道」注入特征与行计划：

- 渲染任务 → 按行计划拼出 8 行：计划说命中的行用「必然命中该规则」的样板行，计划说干净的
  行用「一条规则都不命中」的样板行（其中几行刻意写得可疑，用来钉住「干净行必须真的干净」）；
- 反解任务 → 直接把计划里的命中行数当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies_hits=True)` / `FakeBackend(fence_line=True)`
之类的开关故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from tests.workflows._fakes import draft as _draft
from workflows.lint import features

#: 每条规则一条「必然命中」的样板行。规则表一改这里就得改——`test_conftest` 逐条验过。
HIT_LINES: Final[dict[str, str]] = {
    features.RULE_SECRET: 'API_KEY = "sk-live-9f2b2c"',
    features.RULE_TODO: "# TODO: retry the upload",
    features.RULE_DEBUG: 'print("loading config")',
    features.RULE_PATH: 'CONFIG = "/etc/app/config.yml"',
    features.RULE_LONG: (
        "缓存失效之后请求会回源到数据库，如果这时候并发量正好很高，回源压力就会成倍放大，"
        "所以这里必须再加一层本地缓存放住热点键，否则整条链路都会被拖慢。"
    ),
}

#: 干净行池：**一条规则都不命中**，其中前几行刻意写得可疑（提到 password 却走环境变量、
#: 调用 `logger.info` 而不是 `logger.debug`），用来钉住「干净行必须真的干净」这条口径。
CLEAN_LINES: Final[tuple[str, ...]] = (
    "def load_config():",
    "import os",
    "return dict(retries=3)",
    "# read the local configuration",
    'raise ValueError("bad input")',
    "class Client:",
    "for item in items:",
    'logger.info("cache warm")',
    'PASSWORD = os.environ["PASSWORD"]',
    'CONFIG_PATH = os.environ.get("APP_CONFIG")',
)


def draft_lines(
    rule: str,
    hit_rows: list[int],
    *,
    unrealized: bool = False,
    extra_hit: bool = False,
    drop_line: bool = False,
    fence_line: bool = False,
) -> list[str]:
    """输入：规则名、命中行下标、四个「故意写歪」的开关。输出：8 行草稿。

    预期行为：默认**完全兑现计划**；四个开关各自破坏一件事——
    `unrealized` 让一个该命中的行不命中、`extra_hit` 让一个干净行命中、
    `drop_line` 少给一行、`fence_line` 让一行带围栏。
    `unrealized` 在「一行都不该命中」的计划上没有可破坏的对象，这时退化成 `extra_hit`
    （不然 `hits=0` 的那几条会照常通过，参数化的反例测试就会漏掉它们）。
    """
    hits = set(hit_rows)
    out: list[str] = []
    clean_seen = 0
    for row in range(features.SIZE):
        if row in hits:
            out.append(HIT_LINES[rule])
            continue
        out.append(CLEAN_LINES[clean_seen % len(CLEAN_LINES)])
        clean_seen += 1
    if unrealized:
        if hit_rows:
            out[hit_rows[0]] = CLEAN_LINES[0]
        else:
            extra_hit = True
    if extra_hit:
        spare = next((row for row in range(features.SIZE) if row not in hits), None)
        if spare is not None:
            out[spare] = HIT_LINES[rule]
    if drop_line:
        out = out[:-1]
    if fence_line:
        out[0] = "```" + out[0]
    return out


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies_hits` 反解时把命中行数说成另一个数（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `unrealized` / `extra_hit` / `drop_line` / `fence_line` 见 `draft_lines`。
    """

    def __init__(
        self,
        *,
        trace_lies_hits: bool = False,
        trace_raises: Exception | None = None,
        unrealized: bool = False,
        extra_hit: bool = False,
        drop_line: bool = False,
        fence_line: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies_hits = trace_lies_hits
        self.trace_raises = trace_raises
        self.unrealized = unrealized
        self.extra_hit = extra_hit
        self.drop_line = drop_line
        self.fence_line = fence_line
        self.tasks: list[str] = []

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        case = spec["case"]
        hit_rows = [int(row) for row in spec["hit_rows"]]
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            hits = len(hit_rows)
            if self.trace_lies_hits:
                hits = (hits + 1) % (features.SIZE + 1)
            return _draft({"hits": hits}, f"hits={hits}")
        lines = draft_lines(
            str(case["rule"]),
            hit_rows,
            unrealized=self.unrealized,
            extra_hit=self.extra_hit,
            drop_line=self.drop_line,
            fence_line=self.fence_line,
        )
        return _draft({"lines": list(lines)}, "\n".join(lines))

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：草稿列表。预期行为：逐个走 `generate`。"""
        return [self.generate(task) for task in tasks]


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies_hits=True)` 造一个假后端。"""
    return FakeBackend
