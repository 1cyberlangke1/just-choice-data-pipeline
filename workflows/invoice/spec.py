"""单据 workflow 的真值规格：一张单据的全部字段与注入的「对不上」位置。

输入/输出：只有不可变数据类与纯函数；不含领域词表（词表在 `locales/`，这里只存下标）。
预期行为：**写进单据的数字就是这里的数字**——答案由构造决定，既不问模型，
也不靠事后解析文本；文本解析只用于「反解校验」（`parse.py` + `judges.py`）。
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 维度取值 ----------

LAYOUTS: Final[tuple[str, ...]] = ("labeled_lines", "pipe_table", "aligned_table", "paragraph")
LABEL_STYLES: Final[tuple[str, ...]] = ("canonical", "aliased")
CURRENCIES: Final[tuple[str, ...]] = ("cny", "usd", "eur")
DATE_FORMATS: Final[tuple[str, ...]] = ("dashed", "slashed", "dotted")
TAX_STATES: Final[tuple[str, ...]] = ("none", "ok", "off")
LINE_COUNTS: Final[tuple[int, ...]] = (2, 3, 4)
DEFECT_COUNTS: Final[tuple[int, ...]] = (0, 1, 2, 3)
SUBTOTAL_FLAGS: Final[tuple[bool, ...]] = (True, False)

#: 训练侧见过的取值。`test_ood` 只收「至少含一个不在下面集合里的取值」的 case，
#: 这样留出的是**取值**而不是「另一批同分布样本」（PLAN 3.3 的值留出）。
SEEN_LAYOUTS: Final[frozenset[str]] = frozenset(("labeled_lines", "pipe_table", "aligned_table"))
SEEN_LABEL_STYLES: Final[frozenset[str]] = frozenset(("canonical",))
SEEN_CURRENCIES: Final[frozenset[str]] = frozenset(("cny", "usd"))
SEEN_DATE_FORMATS: Final[frozenset[str]] = frozenset(("dashed", "slashed"))

#: 各类槽位（题族的「选项」就是同类的别的槽，所以干扰项天然来自同一份单据）。
DATE_SLOTS: Final[tuple[str, ...]] = ("issue", "due", "received")
#: 金额槽。`shipping`（运费）保证「总计」永远不等于「小计」——否则没有税额行时
#: 两个选项的描述会一模一样，那道题就有两个说得通的答案了。
MONEY_SLOTS: Final[tuple[str, ...]] = ("subtotal", "shipping", "tax", "total")
PARTY_SLOTS: Final[tuple[str, ...]] = ("issuer", "payer", "ship_to")
ID_SLOTS: Final[tuple[str, ...]] = ("doc", "po", "contract")

#: 可能出现的税率（百分数）。
TAX_RATES: Final[tuple[int, ...]] = (3, 6, 9, 13)
#: 明细行的列（渲染表头用；反解不靠它认列，靠「后三格是数字」）。
LINE_COLS: Final[tuple[str, ...]] = ("name", "qty", "unit", "amount")
#: spec 里存的是词表**下标**，所以每种语言的词表长度都不能小于这个数（测试锁住）。
WORD_POOL: Final[int] = 12


def percent_of(value: int, rate: int) -> int:
    """输入：金额、税率（百分数）。输出：税额。

    预期行为：四舍五入（half-up），**不用 Python 的 `round`**——后者是银行家舍入，
    同一个式子在不同地方算会得到不同结果，判别器与构造必须完全一致。
    """
    return (value * rate + 50) // 100


def _shift(date: tuple[int, int, int], days: int) -> tuple[int, int, int]:
    """输入：日期、天数。输出：偏移后的日期。预期行为：用 `datetime` 算，跨月跨年自动对。"""
    base = dt.date(date[0], date[1], date[2])
    moved = base + dt.timedelta(days=days)
    return (moved.year, moved.month, moved.day)


@dataclass(frozen=True)
class InvoiceCase:
    """一个 case：8 个维度的取值组合（组合数 ≲10⁴，直接全枚举 + 过滤不可解组合）。"""

    layout: str
    labels: str
    currency: str
    date_format: str
    lines: int
    defects: int
    subtotal_ok: bool
    tax_state: str

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}，顺序与声明一致（哈希分桶按它算，必须稳定）。"""
        return {
            "layout": self.layout,
            "labels": self.labels,
            "currency": self.currency,
            "date_format": self.date_format,
            "lines": self.lines,
            "defects": self.defects,
            "subtotal_ok": self.subtotal_ok,
            "tax_state": self.tax_state,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        subtotal = "小计对" if self.subtotal_ok else "小计错"
        return (
            f"{self.layout}/{self.labels}/{self.currency}/{self.date_format}"
            f"/{self.lines}行/{self.defects}处错/{subtotal}/税额{self.tax_state}"
        )

    @property
    def has_tax(self) -> bool:
        """输入：无。输出：单据里有没有税额行。"""
        return self.tax_state != "none"

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个 case 里训练侧没见过的维度名（空 = 全部见过）。"""
        checks = (
            ("layout", self.layout, SEEN_LAYOUTS),
            ("labels", self.labels, SEEN_LABEL_STYLES),
            ("currency", self.currency, SEEN_CURRENCIES),
            ("date_format", self.date_format, SEEN_DATE_FORMATS),
        )
        return tuple(name for name, value, seen in checks if value not in seen)


def compatible(case: InvoiceCase) -> bool:
    """输入：一个 case。输出：这个组合写得出来吗。

    预期行为：只有一条硬约束——「有几行对不上」不能超过明细行数
    （2 行的单据不可能有 3 行错）。无解的组合既不生成、也不算没覆盖。
    """
    return case.defects <= case.lines


@dataclass(frozen=True)
class InvoiceConfig:
    """一个切分的采样参数。

    字段：各维度的取值表（留出时收窄）、问法编号池、`ood_only` 是否只留「含没见过的取值」的 case。
    """

    split: str = "train"
    layouts: tuple[str, ...] = LAYOUTS
    label_styles: tuple[str, ...] = LABEL_STYLES
    currencies: tuple[str, ...] = CURRENCIES
    date_formats: tuple[str, ...] = DATE_FORMATS
    line_counts: tuple[int, ...] = LINE_COUNTS
    defect_counts: tuple[int, ...] = DEFECT_COUNTS
    subtotal_flags: tuple[bool, ...] = SUBTOTAL_FLAGS
    tax_states: tuple[str, ...] = TAX_STATES
    phrasings: tuple[int, ...] = (0, 1, 2)
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：空取值表在构造期就报人话，不拖到采样阶段。

        以前空表会活到 `rng.choice`，抛 `Cannot choose from an empty sequence`
        （校验规则见 `jc.knobs`）。
        """
        for field, values in (
            ("layouts", self.layouts),
            ("label_styles", self.label_styles),
            ("currencies", self.currencies),
            ("date_formats", self.date_formats),
            ("line_counts", self.line_counts),
            ("defect_counts", self.defect_counts),
            ("subtotal_flags", self.subtotal_flags),
            ("tax_states", self.tax_states),
            ("phrasings", self.phrasings),
        ):
            require_pool(field, values)


def build_cases(config: InvoiceConfig) -> list[InvoiceCase]:
    """枚举 case 矩阵。

    输入：采样参数。输出：`InvoiceCase` 列表（顺序稳定、可复算）。
    预期行为：全枚举后剔掉无解组合；`ood_only` 时只留「至少含一个训练侧没见过的取值」的 case。
    """
    out: list[InvoiceCase] = []
    for layout in config.layouts:
        for labels in config.label_styles:
            for currency in config.currencies:
                for date_format in config.date_formats:
                    for lines in config.line_counts:
                        for defects in config.defect_counts:
                            for subtotal_ok in config.subtotal_flags:
                                for tax_state in config.tax_states:
                                    case = InvoiceCase(
                                        layout=layout,
                                        labels=labels,
                                        currency=currency,
                                        date_format=date_format,
                                        lines=lines,
                                        defects=defects,
                                        subtotal_ok=subtotal_ok,
                                        tax_state=tax_state,
                                    )
                                    if not compatible(case):
                                        continue
                                    if config.ood_only and not case.unseen_dims():
                                        continue
                                    out.append(case)
    return out


@dataclass(frozen=True)
class LineItem:
    """一条明细：名称取词表下标；`amount` 可能与 `qty * unit` 不等（注入的错）。"""

    name_index: int
    qty: int
    unit: int
    amount: int

    @property
    def consistent(self) -> bool:
        """输入：无。输出：这一行的「数量 × 单价」与金额是否一致。"""
        return self.amount == self.qty * self.unit


@dataclass(frozen=True)
class InvoiceSpec:
    """一条样本的真值规格：单据上写的每一个值都在这里。"""

    case: InvoiceCase
    split: str
    seq: int
    doc_no: str
    po_no: str
    contract_no: str
    issue_date: tuple[int, int, int]
    due_date: tuple[int, int, int]
    received_date: tuple[int, int, int]
    parties: tuple[int, int, int]
    handler_index: int
    items: tuple[LineItem, ...]
    subtotal: int
    shipping: int
    tax_rate: int
    tax_amount: int
    total: int
    phrasing: int

    @property
    def has_tax(self) -> bool:
        """输入：无。输出：这份单据有没有税额行。"""
        return self.case.has_tax

    @property
    def date_slots(self) -> tuple[str, ...]:
        """输入：无。输出：本单据可问的日期槽（`date_field` 的选项集合）。"""
        return DATE_SLOTS

    @property
    def money_slots(self) -> tuple[str, ...]:
        """输入：无。输出：本单据可问的金额槽（没有税额行时不含 `tax`）。"""
        return MONEY_SLOTS if self.has_tax else ("subtotal", "shipping", "total")

    @property
    def bad_lines(self) -> int:
        """输入：无。输出：几行的「数量 × 单价」与金额对不上，封顶 3（score 的答案）。"""
        return min(3, sum(1 for item in self.items if not item.consistent))

    @property
    def subtotal_consistent(self) -> bool:
        """输入：无。输出：小计是否等于明细金额之和（noul 的答案）。"""
        return self.subtotal == sum(item.amount for item in self.items)

    @property
    def tax_consistent(self) -> bool:
        """输入：无。输出：税额是否等于「小计 × 税率」（noul 的答案）。"""
        return self.tax_amount == percent_of(self.subtotal, self.tax_rate)

    def date_of(self, slot: str) -> tuple[int, int, int]:
        """输入：日期槽名。输出：该槽的日期。预期行为：未知槽名直接报错。"""
        table = {
            "issue": self.issue_date,
            "due": self.due_date,
            "received": self.received_date,
        }
        if slot not in table:
            raise KeyError(f"未知日期槽：{slot!r}（可用：{sorted(table)}）")
        return table[slot]

    def money_of(self, slot: str) -> int:
        """输入：金额槽名。输出：该槽的金额。预期行为：没有税额行时问 `tax` 直接报错。"""
        table = {
            "subtotal": self.subtotal,
            "shipping": self.shipping,
            "tax": self.tax_amount,
            "total": self.total,
        }
        if slot not in table or (slot == "tax" and not self.has_tax):
            raise KeyError(f"本单据没有金额槽：{slot!r}（可用：{sorted(self.money_slots)}）")
        return table[slot]

    def party_of(self, slot: str) -> int:
        """输入：当事方槽名。输出：词表下标。预期行为：未知槽名直接报错。"""
        table = {"issuer": self.parties[0], "payer": self.parties[1], "ship_to": self.parties[2]}
        if slot not in table:
            raise KeyError(f"未知当事方槽：{slot!r}（可用：{sorted(table)}）")
        return table[slot]

    def id_of(self, slot: str) -> str:
        """输入：编号槽名。输出：编号字符串。预期行为：未知槽名直接报错。"""
        table = {"doc": self.doc_no, "po": self.po_no, "contract": self.contract_no}
        if slot not in table:
            raise KeyError(f"未知编号槽：{slot!r}（可用：{sorted(table)}）")
        return table[slot]


def sample_spec(
    rng: random.Random, case: InvoiceCase, split: str, seq: int, phrasing: int
) -> InvoiceSpec:
    """输入：RNG、case、切分名、样本序号、问法编号。输出：这一条的真值规格。

    预期行为：金额与日期全部由构造决定——`defects` 决定几行的「数量 × 单价 ≠ 金额」，
    `subtotal_ok` 决定小计是否等于明细之和，`tax_state` 决定税额行「没有 / 对 / 错」。
    金额都用整数（不做小数），免得舍入把「对不上」和「格式」搅在一起；
    同一份单据里**同类取值两两不同**（单价、金额、日期），否则一道选择题会有两个正确答案。
    """
    year = rng.choice((2025, 2026))
    issue = (year, rng.randint(1, 12), rng.randint(1, 28))
    received = _shift(issue, -rng.randint(1, 14))
    due = _shift(issue, rng.randint(15, 60))
    broken = set(rng.sample(range(case.lines), case.defects))
    items: list[LineItem] = []
    used_units: set[int] = set()
    for index in range(case.lines):
        qty = rng.randint(1, 9)
        unit = rng.randint(120, 980)
        while unit in used_units:  # 单价两两不同：它是 `line_field` 的选项描述
            unit += 1
        used_units.add(unit)
        amount = qty * unit
        if index in broken:
            amount += rng.randint(3, 97) * rng.choice((-1, 1))
        items.append(LineItem(name_index=rng.randrange(WORD_POOL), qty=qty, unit=unit, amount=amount))
    written = sum(item.amount for item in items)
    subtotal = written if case.subtotal_ok else written + rng.randint(3, 97) * rng.choice((-1, 1))
    rate = rng.choice(TAX_RATES)
    expected_tax = percent_of(subtotal, rate)
    if case.tax_state == "none":
        tax_amount = 0
    elif case.tax_state == "ok":
        tax_amount = expected_tax
    else:
        tax_amount = expected_tax + rng.randint(2, 60)
    shipping = rng.randint(12, 98)
    while shipping in {subtotal, tax_amount}:  # 运费也不能与别的金额撞成同一个数
        shipping += 1
    return InvoiceSpec(
        case=case,
        split=split,
        seq=seq,
        doc_no=f"INV-{year}-{rng.randint(1000, 9999)}",
        po_no=f"PO-{rng.randint(10000, 99999)}",
        contract_no=f"CT-{year}{rng.randint(100, 999)}",
        issue_date=issue,
        due_date=due,
        received_date=received,
        parties=tuple(rng.sample(range(WORD_POOL), 3)),
        handler_index=rng.randrange(WORD_POOL),
        items=tuple(items),
        subtotal=subtotal,
        shipping=shipping,
        tax_rate=rate,
        tax_amount=tax_amount,
        total=subtotal + tax_amount + shipping,
        phrasing=phrasing,
    )


def spec_to_dict(spec: InvoiceSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "split": spec.split,
        "seq": spec.seq,
        "doc_no": spec.doc_no,
        "subtotal": spec.subtotal,
        "shipping": spec.shipping,
        "tax_rate": spec.tax_rate,
        "tax_amount": spec.tax_amount,
        "total": spec.total,
        "items": [
            {"qty": item.qty, "unit": item.unit, "amount": item.amount, "ok": item.consistent}
            for item in spec.items
        ],
    }
