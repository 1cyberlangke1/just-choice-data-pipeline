"""receiptcat 的假后端与草稿构造器：脚本化出草稿，不联网。

假后端按 `task.spec` 现造收据行——**默认完全兑现计划**（关键词、金额、诱饵都由注入特征推
出来），另外几个开关各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from workflows.receiptcat import features, locales, render
from workflows.receiptcat.features import ReceiptCase
from workflows.receiptcat.spec import ReceiptCatSpec

#: 收据行里的填充词：只为了让条目读起来像一条真收据，且长度够过 `MIN_CONTENT_CHARS`。
#: 按语言分开——zh 侧写中文、en 侧写英文，真机与假后端的行为才一致（范式 §9.32）。
FILLERS: Final[dict[str, str]] = {"zh": "报销明细", "en": "reimbursement detail"}


def text_of(case: ReceiptCase, lang: str = "zh") -> str:
    """输入：注入特征、语言。输出：一条完全兑现它的收据行。"""
    loc = locales.load(lang)
    word = features.keyword_table(loc)[case.category][0]
    parts = [f"{word} {FILLERS[lang]}"]
    if case.decoy == features.DECOY_NEAR:
        parts.append(render.decoy_word_for(case, loc))
    parts.append(case.amount)
    return " ".join(parts)


def spec_for(
    case: ReceiptCase, phrasing: int = 0, split: str = "train", seq: int = 1
) -> ReceiptCatSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return ReceiptCatSpec(
        case=case,
        phrasing=phrasing,
        split=split,
        seq=seq,
    )


def sample_of(text: str, lang: str = "zh") -> dict[str, JsonValue]:
    """输入：收据行、语言。输出：只带 state 的样本。"""
    return {"state": render.to_state(render.ContentDraft(text=text), locales.load(lang))}


def faithful(
    case: ReceiptCase, lang: str = "zh"
) -> tuple[ReceiptCatSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：收据行完全兑现计划。"""
    spec = spec_for(case)
    text = text_of(case, lang)
    render.check_draft(render.ContentDraft(text=text), case, locales.load(lang))
    return spec, sample_of(text, lang)


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `locale` 草稿用哪一种语言写（默认 zh）；
    - `trace_lies` 反解时把类别与档位说反（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `wrong_category` 写另一类的关键词、`wrong_amount` 另写一个金额；
    - `decoy_missing` 该埋诱饵时不埋、`decoy_extra` 不该埋诱饵时埋上；
    - `name_leak` 正文里带上类别名、`too_short` 缩到一个词、`fence` 套上围栏。
    """

    def __init__(
        self,
        *,
        locale: str = "zh",
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
        wrong_category: bool = False,
        wrong_amount: bool = False,
        decoy_missing: bool = False,
        decoy_extra: bool = False,
        name_leak: bool = False,
        too_short: bool = False,
        fence: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.locale = locale
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.wrong_category = wrong_category
        self.wrong_amount = wrong_amount
        self.decoy_missing = decoy_missing
        self.decoy_extra = decoy_extra
        self.name_leak = name_leak
        self.too_short = too_short
        self.fence = fence

    def _case(self, task: object) -> ReceiptCase:
        """输入：渲染任务。输出：任务里带的注入特征。预期行为：缺字段直接报错。"""
        raw = task.spec
        return ReceiptCase(
            category=str(raw["category"]), band=str(raw["band"]), decoy=str(raw["decoy"])
        )

    def _text(self, case: ReceiptCase) -> str:
        """输入：注入特征。输出：按开关拼出来的收据行。"""
        loc = locales.load(self.locale)
        other = features.CATEGORY_IDS[
            (features.CATEGORY_IDS.index(case.category) + 1) % len(features.CATEGORY_IDS)
        ]
        category = other if self.wrong_category else case.category
        word = features.keyword_table(loc)[category][0]
        parts = [f"{word} {FILLERS[self.locale]}"]
        if self.name_leak:
            parts.append(render.category_name(loc, case.category))
        wants_decoy = case.decoy == features.DECOY_NEAR
        if wants_decoy and not self.decoy_missing:
            parts.append(render.decoy_word_for(case, loc))
        elif not wants_decoy and self.decoy_extra:
            parts.append(render.decoy_word_for(case, loc))
        parts.append("9999.99" if self.wrong_amount else case.amount)
        text = " ".join(parts)
        if self.too_short:
            text = word
        if self.fence:
            text = f"```text\n{text}\n```"
        return text

    def generate(self, task: object) -> Draft:
        """输入：渲染任务或反解任务。输出：草稿。

        预期行为：按 `task.response_model` 分派——反解任务回一个 JSON 形状的类别名 + 档位
        下标（`trace_lies` 时说反话），渲染任务回一条收据行。
        """
        loc = locales.load(self.locale)
        case = self._case(task)
        if task.task_id.endswith("-trace"):
            # 端点故障只该发生在反解那一次调用上——渲染任务照样得给出收据行，否则整批会
            # 因为「写不出来」被 `E_ASSEMBLE` 拦下，验不到「端点故障不算数据问题」。
            if self.trace_raises is not None:
                raise self.trace_raises
            want = features.label_for(case)
            name = render.category_names(loc)[want.category]
            index = features.TIERS.index(want.tier)
            if self.trace_lies:
                name = render.category_names(loc)[features.CATEGORY_IDS[0]]
                index = 0
            return Draft(
                data={"category": name, "tier": index},
                raw_text="",
                backend="fake",
                provider="fake",
                model="fake",
                mode="fake",
            )
        return Draft(
            data={"text": self._text(case)},
            raw_text="",
            backend="fake",
            provider="fake",
            model="fake",
            mode="fake",
        )


backend = _fakes.instance_fixture(FakeBackend)
fake_backend = _fakes.factory_fixture(FakeBackend)
