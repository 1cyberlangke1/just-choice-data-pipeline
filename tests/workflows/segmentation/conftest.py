"""segmentation 的假后端与草稿构造器：脚本化出草稿，不联网。

假后端按 `task.spec` 现造页面正文——**默认完全兑现计划**（每一页的抬头由注入特征推出来），
另外几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from workflows.segmentation import features, locales, render
from workflows.segmentation.features import SegmentationCase
from workflows.segmentation.spec import SegmentationSpec

#: 每一类的抬头片段：命中哪一类就往页面里拼哪一段。中英标记都在 `features.PATTERNS` 里，
#: 所以同一份片段在 zh / en 两种语言下都算得出来（假后端因此不必按语言分叉）。
MARKERS: Final[dict[str, str]] = {
    "form": "表单编号：A-1001。填报人：张三，联系电话 138-0000-0000。",
    "letter": "尊敬的收件人：来信已收到。此致 敬礼。",
    "receipt": "收据编号：R-2002。收款人：李四，已收款 128.00 元。",
    "contract": "合同编号：C-3003。甲方：王五。乙方：赵六，签约日期 2026-09-30。",
}

#: 每一种来源的痕迹片段。**与 `MARKERS` 刻意不重叠**：痕迹不能把某一页算成别的类。
TRACES: Final[dict[str, str]] = {
    "scan": "扫描页码：第 1 页 / 共 3 页。",
    "fax": "传真抬头：FAX 021-0000 2026-09-30 09:12。",
    "photo": "拍摄时间：IMG_20260930_091233.jpg。",
}

#: 凑长度的填充句（不含任何一类的抬头，也不含任何来源痕迹）。
PAD: Final[str] = "正文内容按真实材料的写法铺开，把上面的抬头交代清楚，其余细节自行补足。"


def page_text(cls: str, source: str) -> str:
    """输入：种类编号、来源编号。输出：一页完全兑现它的正文（长度落在区间内）。"""
    return f"{TRACES[source]}{MARKERS[cls]}{PAD}{PAD}"


def pages_of(plan: features.SegmentationPlan, source: str) -> list[str]:
    """输入：切分计划、来源。输出：逐页正文（按计划里的页序）。"""
    return [page_text(cls, source) for cls in plan.shape]


def spec_for(
    case: SegmentationCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> SegmentationSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return SegmentationSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
        plan=features.plan_for(case, phrasing),
    )


def sample_of(pages: list[str], lang: str = "zh") -> dict[str, JsonValue]:
    """输入：逐页正文、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.PacketDraft(pages=pages), locales.load(lang))}


def faithful(
    case: SegmentationCase, lang: str = "zh"
) -> tuple[SegmentationSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：逐页正文完全兑现计划。"""
    spec = spec_for(case)
    pages = pages_of(spec.plan, case.source)
    render.check_draft(render.PacketDraft(pages=pages), spec.plan, case.source, locales.load(lang))
    return spec, sample_of(pages, lang)


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意写歪」的开关，默认全关）：

    - `read_lies` 反解时漏报最后一类（验 `read` 判别器）；
    - `read_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `drop_marker` 把最后一页的抬头删掉（那一页一条判据都不命中）；
    - `extra_marker` 给最后一页添上别的类的抬头（那一页同时命中两类）；
    - `no_trace` 把最后一页的来源痕迹删掉；
    - `wrong_pages` 少写一页、`too_short` 把首页缩到一个字符、`fence` 给首页套上围栏。
    """

    def __init__(
        self,
        *,
        read_lies: bool = False,
        read_raises: Exception | None = None,
        drop_marker: bool = False,
        extra_marker: bool = False,
        no_trace: bool = False,
        wrong_pages: bool = False,
        too_short: bool = False,
        fence: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.read_lies = read_lies
        self.read_raises = read_raises
        self.drop_marker = drop_marker
        self.extra_marker = extra_marker
        self.no_trace = no_trace
        self.wrong_pages = wrong_pages
        self.too_short = too_short
        self.fence = fence
        self.tasks: list[str] = []

    def pages_for(self, case: SegmentationCase) -> list[str]:
        """输入：注入特征。输出：按开关写歪之后的一包页面。"""
        plan = features.plan_for(case, 0)
        pages = pages_of(plan, case.source)
        if self.wrong_pages and pages:
            pages = pages[:-1]
        if not pages:
            return pages
        last = len(pages) - 1
        if self.drop_marker:
            pages[last] = pages[last].replace(MARKERS[plan.shape[last]], "")
        if self.extra_marker:
            spare = [cls for cls in features.CLASS_IDS if cls != plan.shape[last]]
            pages[last] = pages[last] + MARKERS[spare[0]]
        if self.no_trace:
            pages[last] = pages[last].replace(TRACES[case.source], "")
        if self.too_short:
            pages[0] = "短"
        if self.fence:
            pages[0] = f"```{pages[0]}```"
        return pages

    def read_report(self, classes: list[str]) -> list[dict[str, str]]:
        """输入：计划里出现过的种类。输出：反解草稿的 classes（每条带编号与原样引文）。"""
        wanted = list(classes)
        if self.read_lies and wanted:
            wanted = wanted[:-1]
        return [{"class_id": cls, "quote": MARKERS[cls]} for cls in wanted]

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        if task_id.endswith("-read"):
            if self.read_raises is not None:
                raise self.read_raises
            report = self.read_report([str(cls) for cls in spec["classes"]])
            return _draft(
                {"classes": report}, "classes=" + ",".join(item["class_id"] for item in report)
            )
        case = SegmentationCase(
            classes=str(spec["case"]["classes"]),
            pages=str(spec["case"]["pages"]),
            source=str(spec["case"]["source"]),
        )
        pages = self.pages_for(case)
        return _draft({"pages": pages}, "\n".join(pages))

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：草稿列表。预期行为：逐个走 `generate`。"""
        return [self.generate(task) for task in tasks]


def _draft(data: dict[str, JsonValue], raw: str) -> Draft:
    """输入：解析后的数据、原始文本。输出：`Draft`。预期行为：记账字段填假值，不进产物。"""
    return Draft(
        data=data,
        raw_text=raw,
        backend="fake",
        provider="fake",
        model="fake",
        mode="fake",
    )


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(read_lies=True)` 造一个假后端。"""
    return FakeBackend
