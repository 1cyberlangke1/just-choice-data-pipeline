"""文档包切分（segmentation）的特征层：文档种类表 + 页序 + 切分口径（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这一包材料该怎么切」的样本由三个**注入特征**描述——
`classes` 这一包里出现了哪几类文档、`pages` 这一包一共几页、`source` 这一包从哪种来源来
（扫描仪 / 传真 / 手机拍照）。**标签先行**：先定这三个取值，再由本模块的纯函数算出结论
（每一页属于哪一类、整包切成几份、该走哪一档切分），模型碰不到标签。

**为什么要这一族**：社区里 `tax-doc-classifier`（★314）把一页 PDF 的文本送进 Jev，判它属于
261 种 IRS 表格里的哪一种、7 种页面类别里的哪一种；`docjev`（★307）更进一步——一包文档进去，
出来的是**有序的类别与页区间**，也就是把包切成一份份文档。两者的共同点是「一叠纸进来，要一个
能直接落地的分堆结论」，而不是让人把每页从头读到尾再拍脑袋。本 workflow 取后者的形态：包里每
一页属于哪一类由页面上的**机械抬头**定，切分规则由种类表定，标签全是算出来的。

**判据为什么写成机械词表**：真实材料里「这是哪一种文档」是有固定形状的——申请表有申请单号、
收据有收款人、合同有甲乙双方。把这些形状写成词表之后，「这一页属于哪一类」就是**算**出来的
（`class_of` 按 `PATTERNS` 逐条匹配），模型只负责写出一页读得通、带那个抬头的正文。表由代码写、
内容由模型写（范式 §9.22 / §9.23）。

**切分口径**（`docs_for`）：相邻两页换了类 → 一定属于两份不同的文档；相邻两页同类且该类**可并**
（`MERGEABLE`）→ 并成同一份；相邻两页同类但该类**不可并** → 一页一份。

**`route` 为什么不按「单一文档 / 同类相邻要切 / 多类混合」分三档**：这一族先按那个口径把分布
算了一遍（4 类、子集大小 1~3、页数 1~4，共 42 组组合）：单一文档 2 组 / 同类相邻要切 2 组 /
多类混合 38 组——多类那一档占 90%，答案分布当场退化，均衡闸留不住。改成按**实际份数**分档
（`one` 1 份 / `two` 2 份 / `many` 3 份及以上）之后是 10 / 14 / 18（23.8% / 33.3% / 42.9%），
三档都站得住，问的还是同一件事（要切几份）。**先验证分布再定档位**这条留在了范式 §9.70。

**`source` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一包页写在扫描件、传真件还是
手机拍照的正文里，算出来的份数一个字都不差。训练侧只见过 `scan` / `fax`，留出侧是 `photo`，
用来验「读法不靠来源先验」。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从页面正文重推
标签（审计侧，走 `PATTERNS` 的词表）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from jc.cover import cover_dims
from jc.jsontypes import JsonValue

#: 文档种类的编号（**结构键**，不是文案：种类名与判据原文全在 `locales/`）。
CLASS_IDS: Final[tuple[str, ...]] = ("form", "letter", "receipt", "contract")

#: 允许「相邻同类页并成同一份」的种类。表单与信函是**多页一份**的材料（一页申请表接着一页附表），
#: 收据与合同是**一页一份**（一张收据就是一份凭证，一份合同就是一份合同）。
MERGEABLE: Final[frozenset[str]] = frozenset({"form", "letter"})

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面（页面里出现这个形状就算命中这一类），中英各一组，两种语言共用一份口径。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "form": re.compile(
        r"(表单编号|申请表|申请单|填报人)"
        r"|\b(form no|form id|application form|applicant name)\b",
        re.IGNORECASE,
    ),
    "letter": re.compile(
        r"(尊敬的|此致|敬礼|顺颂)"
        r"|\b(dear sir|dear madam|yours sincerely|sincerely yours)\b",
        re.IGNORECASE,
    ),
    "receipt": re.compile(
        r"(收据编号|收款人|已收款|收讫)"
        r"|\b(receipt no|paid in full|amount received|cashier)\b",
        re.IGNORECASE,
    ),
    "contract": re.compile(
        r"(合同编号|甲方|乙方|签约日期)"
        r"|\b(contract no|party a|party b|signed on)\b",
        re.IGNORECASE,
    ),
}

#: 包的来源（结构键）。**只影响渲染、不影响标签**，所以它是这一族的留出维。
SOURCE_SCAN: Final[str] = "scan"
SOURCE_FAX: Final[str] = "fax"
SOURCE_PHOTO: Final[str] = "photo"
SOURCES: Final[tuple[str, ...]] = (SOURCE_SCAN, SOURCE_FAX, SOURCE_PHOTO)

#: 来源 → 「这一包是怎么来的」留在页面上的痕迹。渲染期要求**每一页**都带本来源的痕迹；
#: 痕迹本身不是任何一类文档的判据（两套词表刻意不重叠），所以它不会把标签带偏。
SOURCE_TRACES: Final[dict[str, re.Pattern[str]]] = {
    "scan": re.compile(r"(扫描页码|扫描件|扫描仪)|\b(scanned|scan no|scanner)\b", re.IGNORECASE),
    "fax": re.compile(r"(传真抬头|传真号|传真件)|\bfax\b", re.IGNORECASE),
    "photo": re.compile(r"(拍摄时间|手机拍照|拍摄于)|\b(IMG_\d{8}|taken at)\b", re.IGNORECASE),
}

#: 切分档位名（按份数分档）。
ROUTE_LEVELS: Final[tuple[str, ...]] = ("one", "two", "many")

#: 一包里最多出现几类文档。**刻意留一档不取满**（种类表有 4 类，这里封顶 3）：
#: 4 类全中的包只有「4 页 4 类」一种组合，`count` 的选项表会多出一个几乎取不到的档位，
#: 均衡闸会把那个空档当成结构性不可达报出来。
MAX_PACKET_CLASSES: Final[int] = 3

#: 一包的页数取值（字符串，与 `JsonValue` 对齐）。
PAGE_COUNTS: Final[tuple[str, ...]] = ("1", "2", "3", "4")

# 每页正文的长度上下限**不在这里**：同样一页纸，英文的字符数天然比中文多（真机实测 zh 中位数
# 131、en 中位数 212），一份共用上下限会把英文顶在上限上（首轮 en 侧 1 条 295 字符被拒）。
# 它是**按语言校准的渲染参数**，所以放在 `locales/<lang>.toml` 的 `state.page_min` / `page_max`，
# 由 `render.page_bounds` 取。


def parse_classes(value: str) -> tuple[str, ...]:
    """输入：种类组合串（`"form+receipt"`）。输出：种类编号元组。预期行为：空串给空元组。"""
    return tuple(part for part in value.split("+") if part)


def join_classes(classes: Iterable[str]) -> str:
    """输入：种类编号。输出：按 `CLASS_IDS` 规范顺序排好的组合串。预期行为：同一集合只有一种写法。"""
    wanted = set(classes)
    return "+".join(cls for cls in CLASS_IDS if cls in wanted)


def _all_class_sets() -> tuple[str, ...]:
    """输入：无。输出：种类表的全部 1~`MAX_PACKET_CLASSES` 元子集（14 个）。"""
    out: list[str] = []
    for size in range(1, MAX_PACKET_CLASSES + 1):
        for combo in combinations(CLASS_IDS, size):
            out.append("+".join(combo))
    return tuple(out)


#: 全部种类子集（`cover_dims` 的取值表）。
CLASS_SETS: Final[tuple[str, ...]] = _all_class_sets()

#: 维度表。全空间 14 × 页数(受约束) × 3 = 126 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "classes": CLASS_SETS,
    "pages": PAGE_COUNTS,
    "source": SOURCES,
}


def pages_allowed(dim_a: str, value_a: object, dim_b: str, value_b: object) -> bool:
    """取值对约束：一包至少每类一页，所以**页数不能少于类数**。

    输入：两个维度名与各自取值（顺序不保证）。输出：这一对能否共存。
    预期行为：只认 `classes` / `pages` 这一对；其它组合一律允许。约束交给枚举器之后，
    「1 页 3 类」这种不可能的组合根本不会进 case 库，也不用在采样期再挡一遍。
    """
    pairs = ((dim_a, value_a), (dim_b, value_b))
    classes_value = next((str(value) for name, value in pairs if name == "classes"), None)
    pages_value = next((str(value) for name, value in pairs if name == "pages"), None)
    if classes_value is None or pages_value is None:
        return True
    return int(pages_value) >= len(parse_classes(classes_value))


def matches(text: str) -> tuple[str, ...]:
    """输入：一页正文。输出：按 `CLASS_IDS` 顺序命中的种类编号（可能一条都不命中，也可能多条）。"""
    return tuple(cls for cls in CLASS_IDS if PATTERNS[cls].search(text))


def class_of(text: str) -> str | None:
    """输入：一页正文。输出：这一页所属的种类编号；**判不出来给 None**。

    预期行为：命中恰好一类才算判得出来——一条都不命中（这一页没有任何可识别的抬头）与命中多类
    （抬头混着写，读者也分不清）都算判不出来，渲染期据此重渲染。
    """
    hits = matches(text)
    return hits[0] if len(hits) == 1 else None


def source_trace(text: str, source: str) -> bool:
    """输入：一页正文、来源编号。输出：这一页带不带本来源的痕迹。"""
    return bool(SOURCE_TRACES[source].search(text))


def shape_for(classes: tuple[str, ...], pages: int) -> tuple[str, ...]:
    """输入：种类元组（按 `CLASS_IDS` 顺序）、总页数。输出：逐页的种类（页序）。

    预期行为：按类均分页数，**余数给排在前的类**（同一入参必得同一页序）；同类页在页序里是
    连续的，所以「相邻同类」这条规则才有东西可判。
    """
    base, extra = divmod(pages, len(classes))
    out: list[str] = []
    for index, cls in enumerate(classes):
        out.extend([cls] * (base + (1 if index < extra else 0)))
    return tuple(out)


def docs_for(shape: tuple[str, ...]) -> int:
    """输入：页序。输出：这一包按切分口径该切成几份文档。

    预期行为：逐段（同类相邻的连续页）累加——可并的那一类整段算一份，不可并的那一类一段里
    有几页就算几份。空页序给 0。
    """
    total = 0
    index = 0
    while index < len(shape):
        cls = shape[index]
        end = index
        while end < len(shape) and shape[end] == cls:
            end += 1
        run = end - index
        total += 1 if cls in MERGEABLE else run
        index = end
    return total


def route_for(docs: int) -> str:
    """输入：份数。输出：切分档位名。预期行为：1 份 / 2 份 / 3 份及以上。"""
    if docs <= 1:
        return "one"
    if docs == 2:
        return "two"
    return "many"


def _distractors(classes: tuple[str, ...], rng: random.Random) -> tuple[tuple[str, ...], ...]:
    """输入：真值种类、随机源。输出：三个干扰组合（`combo` 那一族的其余选项）。

    预期行为：只取**与真值同大小、且恰好差一处**的组合（换掉其中一个种类）——大小相同，
    考生没法靠「数一数几个」把它划掉，只能真去读页面。4 类下这个池子恒为 3~4 个：
    |真值|=1 时 3 个、|=2 时 4 个、|=3 时 3 个，抽三个不会撞车。
    """
    inside = list(classes)
    outside = [cls for cls in CLASS_IDS if cls not in classes]
    pool = sorted(
        {
            tuple(sorted([*[cls for cls in inside if cls != drop], add], key=CLASS_IDS.index))
            for drop in inside
            for add in outside
        }
    )
    picked = rng.sample(pool, 3)
    return tuple(sorted(picked, key=lambda combo: tuple(CLASS_IDS.index(cls) for cls in combo)))


def plan_for(case: SegmentationCase, phrasing: int) -> SegmentationPlan:
    """输入：case、问法编号。输出：切分计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都得到
    同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    classes = parse_classes(case.classes)
    return SegmentationPlan(
        shape=shape_for(classes, int(case.pages)),
        distractors=_distractors(classes, random.Random(seed)),
    )


@dataclass(frozen=True)
class SegmentationCase:
    """一个注入特征组合：包里有哪些类 + 一共几页 + 这一包从哪来。

    字段：`classes` 种类组合串；`pages` 页数（字符串）；`source` 来源（`SOURCES` 里的一个）。
    """

    classes: str
    pages: str
    source: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"classes": self.classes, "pages": self.pages, "source": self.source}

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.classes}-{self.pages}页-{self.source}"

    def class_ids(self) -> tuple[str, ...]:
        """输入：无。输出：该包的种类编号元组（按 `CLASS_IDS` 顺序）。"""
        return parse_classes(self.classes)

    @property
    def page_count(self) -> int:
        """输入：无。输出：该包的页数。"""
        return int(self.pages)


@dataclass(frozen=True)
class SegmentationPlan:
    """一条样本的切分计划：逐页的种类 + 组合题的三个干扰项。

    字段：`shape` 逐页种类（页序）；`distractors` 组合题的三个干扰组合（与真值同大小、差一处）。
    """

    shape: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def classes(self) -> tuple[str, ...]:
        """输入：无。输出：这一包出现过的种类（按 `CLASS_IDS` 顺序）。"""
        return tuple(cls for cls in CLASS_IDS if cls in set(self.shape))

    @property
    def count(self) -> int:
        """输入：无。输出：出现过的种类数。"""
        return len(self.classes)

    @property
    def docs(self) -> int:
        """输入：无。输出：按切分口径算出来的份数。"""
        return docs_for(self.shape)

    @property
    def route(self) -> str:
        """输入：无。输出：切分档位名。"""
        return route_for(self.docs)


@dataclass(frozen=True)
class SegmentationLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`classes` 出现过的种类；`shape` 逐页种类（页序）。
    """

    classes: tuple[str, ...]
    shape: tuple[str, ...]

    def contains(self, cls: str) -> bool:
        """输入：种类编号。输出：这一包有没有出现这一类。"""
        return cls in self.classes

    @property
    def count(self) -> int:
        """输入：无。输出：出现过的种类数。"""
        return len(self.classes)

    @property
    def docs(self) -> int:
        """输入：无。输出：按切分口径算出来的份数。"""
        return docs_for(self.shape)

    @property
    def route(self) -> str:
        """输入：无。输出：切分档位名。"""
        return route_for(self.docs)


def label_for(plan: SegmentationPlan) -> SegmentationLabels:
    """输入：切分计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——页面正文是后写的，标签先定。`render.check_draft`
    负责保证正文真的兑现这份计划（每一页按字面算出来的种类与计划逐页相同）。
    """
    return SegmentationLabels(classes=plan.classes, shape=plan.shape)


def observed_labels(pages: tuple[str, ...]) -> SegmentationLabels:
    """输入：逐页正文。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认正文、只认字面。判不出种类的页记成空串，
    这样它与计划对不上时（审计会看见）不会被悄悄当成某一类。
    """
    shape = tuple(class_of(text) or "" for text in pages)
    return SegmentationLabels(
        classes=tuple(cls for cls in CLASS_IDS if cls in set(shape)), shape=shape
    )


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[SegmentationCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`SegmentationCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(
        DIMS if dims is None else dims, max_full=max_full, allow_pair=pages_allowed
    )
    return [
        SegmentationCase(
            classes=str(row["classes"]), pages=str(row["pages"]), source=str(row["source"])
        )
        for row in rows
    ]
