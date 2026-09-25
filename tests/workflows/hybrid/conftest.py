"""会议日程 workflow 测试的公共夹具：一个**不发网络请求**的假后端 + 小规格构造器。

假后端按 `task.spec["count"]` 从固定词池里取标题 / 议题，所以它天然「会写文本」；
测试要验判别器能抓错时，用 `FakeBackend(drift=1)` 让标题整体错位、
或 `FakeBackend(dirty=True)` 写一段夹带时刻的文本。
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from jc.backend.types import Draft
from workflows.hybrid import locales, render
from workflows.hybrid import spec as sp

#: 假后端的标题池：条数够多，且**没有**时刻 / 日期 / 星期 / 会议室名（否则会被卫生检查拒掉）。
TITLES: tuple[str, ...] = (
    "季度预算评审",
    "产品路线图对齐",
    "招聘计划讨论",
    "客户反馈复盘",
    "数据中心扩容",
    "年度合规检查",
    "供应链风险排查",
    "培训体系改版",
)
#: 假后端的议题池：同上，条数够多且干净。
AGENDAS: tuple[str, ...] = (
    "核对各部门本季度的预算执行情况并确认下季度额度",
    "确认下季度的功能优先级与交付节奏",
    "讨论新增岗位的职责边界与招聘节奏",
    "汇总近期客户投诉并确定改进项",
    "评估机柜与带宽的扩容方案",
    "逐条核对合规检查清单里的未闭合项",
    "梳理关键供应商的交付风险与备选方案",
    "确定新版培训课程的模块划分与排期",
)
#: 故意夹带骨架事实的一段文本（用来验卫生检查与 `text` 判别器）。
DIRTY_TITLE = "下午 3 点的预算评审"


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec["count"]` 推出，不联网。

    字段：`drift` 标题 / 议题整体错位几格（用来验判别器能抓错）；
    `dirty` 第一场写一段夹带时刻的标题；`count_delta` 故意多写 / 少写几场；
    `broken` 返回一个没有 meetings 数组的草稿。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        drift: int = 0,
        dirty: bool = False,
        count_delta: int = 0,
        broken: bool = False,
    ) -> None:
        """输入：四个作弊开关。输出：假后端实例。"""
        self.drift = drift
        self.dirty = dirty
        self.count_delta = count_delta
        self.broken = broken
        self.calls: list[str] = []

    def generate(self, task: Any) -> Draft:
        """输入：一个任务。输出：草稿。预期行为：任务里没带 count 就报错（夹具写错要炸）。"""
        self.calls.append(task.task_id)
        assert "count" in task.spec, f"假后端要求任务带上 count：{task.task_id}"
        count = max(1, task.spec["count"] + self.count_delta)
        if self.broken:
            return self._draft({"meetings": "不是数组"})
        data: dict[str, Any] = {
            "meetings": [
                {
                    "title": TITLES[(index + self.drift) % len(TITLES)],
                    "agenda": AGENDAS[(index + self.drift) % len(AGENDAS)],
                }
                for index in range(count)
            ]
        }
        if self.dirty:
            data["meetings"][0]["title"] = DIRTY_TITLE
        return self._draft(data)

    def generate_many(self, tasks: list[Any]) -> list[Draft]:
        """输入：任务列表。输出：与入参顺序一致的草稿列表。"""
        return [self.generate(task) for task in tasks]

    @staticmethod
    def _draft(data: dict[str, Any]) -> Draft:
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


#: 默认维度：一场 3 场会、中午开、不冲突、列表版式、周三、短时长的普通日程。
CASE_DEFAULTS: dict[str, Any] = {
    "meetings": 3,
    "opening": "early",
    "clash": "none",
    "layout": "bullet",
    "weekday": 3,
    "duration_mix": "short",
}


def _make_case(**overrides: Any) -> sp.HybridCase:
    """输入：维度覆盖项。输出：一个 case。预期行为：未覆盖的维度取 `CASE_DEFAULTS`。"""
    fields = dict(CASE_DEFAULTS)
    fields.update(overrides)
    return sp.HybridCase(**fields)


def fill_text(spec: sp.HybridSpec, locale: str = "zh", **backend_kwargs: Any) -> sp.HybridSpec:
    """输入：规格、语言、假后端开关。输出：同一份规格（`written` 已填好）。

    预期行为：走与真实渲染完全一样的路径（`build_task` → `check_draft`），
    这样测试里的文本与生产里的文本同源；出题与判别器都依赖 `written`。
    """
    loc = locales.load(locale)
    backend = FakeBackend(**backend_kwargs)
    draft = backend.generate(render.build_task(spec, loc, spec.text_task_id))
    spec.written.update(render.check_draft(spec, loc, draft.data))
    return spec


def _make_spec(seed: int = 5, *, locale: str = "zh", **overrides: Any) -> sp.HybridSpec:
    """输入：种子、语言、case 覆盖项。输出：文本已填好的真值规格。

    预期行为：同种子必得同一份规格（构造器只用传进来的 RNG，不碰全局随机源）。
    """
    spec = sp.sample_spec(random.Random(seed), _make_case(**overrides), "train", 0, 0)
    return fill_text(spec, locale)


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(dirty=True)` 造一个假后端。"""
    return FakeBackend


@pytest.fixture
def make_case() -> Any:
    """工厂夹具：`make_case(clash="same_room", meetings=3)` 造一个 case。"""
    return _make_case


@pytest.fixture
def make_spec() -> Any:
    """工厂夹具：`make_spec(7, meetings=4)` 造一份文本已填好的真值规格。"""
    return _make_spec
