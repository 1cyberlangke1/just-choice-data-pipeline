"""引文核查 workflow 的真值规格：一份来源文档 + 一条待核查的引文。

输入/输出：不可变数据类与纯函数；不含句子文本（文案在 `locales/`，这里只放结构）。
预期行为：核查结论**只从渲染出来的文本反推**（`relation_of` / `verdict_of` / `answers_of`），
构造与判别共用同一套规则——表格写错与渲染写歪于是变成同一类错误，不会「两边各错一半
还互相印证」。

链条（对应官方 `citation_check` cookbook）：引文在不在文档里（字符串整句匹配）→
在不在它声称的小节里（定位）→ 声称的小节与主张是什么关系（读上下文）→ 结论 → 处置。
官方用模型算「关系」那一步；这里把关系**做成可构造的**：文档里的每句话都是某个主题的
第 N 句的正/反极性，主张是同一句的**转述**，于是「支持 / 反驳 / 没提」是纯集合判断，
零 token、可无限放量。

为什么不做成 LLM 生成：这里要训的是「按文本判引文」，不是「编一段文档」；文档由表格造，
答案由集合运算定，判别器就能把文本反解回来独立重算。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 取值表 ----------

#: 文档主题：每个主题在语言文件里有一份「句子表」（原文 / 反面 / 两种主张转述）与一组小节标题。
TOPIC_NAMES: Final[tuple[str, ...]] = ("auth", "billing", "retention", "rollout")

#: 核查结论档。前四档是官方 `citation_check` 的口径，`mislocated` 是官方流程里
#: 「引文在文档里、但不在声称的小节里」那一档（官方示例里叫 section-only）。
STATUSES: Final[tuple[str, ...]] = (
    "verified",
    "contradicted",
    "unsupported",
    "mislocated",
    "fabricated",
)

#: 引文写法：`verbatim` 照抄原文；`truncated` 是原文的**前半句**；`reworded` 是主张的转述。
QUOTE_FORMS: Final[tuple[str, ...]] = ("verbatim", "truncated", "reworded")

#: 干扰小节的性质：`plain` 只有同主题的无关句；`decoy` 另有一节放主张的另一面；
#: `off_topic` 有一整节讲别的主题（标题也来自那个主题）。
NOISES: Final[tuple[str, ...]] = ("plain", "decoy", "off_topic")

#: 声称出处的位置。
WHERES: Final[tuple[str, ...]] = ("first", "middle", "last")

#: 小节编号的写法：`[1] 标题` / `1. 标题`。
DOC_STYLES: Final[tuple[str, ...]] = ("bracketed", "dotted")

#: 每份文档的小节数（每节两句）。
SECTION_CHOICES: Final[tuple[int, ...]] = (2, 3)

#: 主张的措辞（语言文件里的 `claim_frames`）。
CLAIM_STYLES: Final[tuple[int, ...]] = (0, 1)

#: 每个主题的句子键——顺序固定，取值稳定；句子文本在语言文件里。
#: 每份文档最多用到 `3 小节 × 2 句`，加上「引文取自哪一句」，6 句够用且留有余量。
SENTENCE_IDS: Final[dict[str, tuple[str, ...]]] = {
    "auth": (
        "token_rotation",
        "session_scope",
        "key_storage",
        "auth_log",
        "mfa_admin",
        "ip_allow",
    ),
    "billing": (
        "invoice_monthly",
        "refund_window",
        "tax_included",
        "proration",
        "dunning",
        "credit_note",
    ),
    "retention": (
        "log_90d",
        "backup_daily",
        "deletion_30d",
        "backup_encrypt",
        "region_lock",
        "pii_masking",
    ),
    "rollout": (
        "canary",
        "rollback_plan",
        "freeze_friday",
        "migration_review",
        "feature_flag",
        "smoke_test",
    ),
}

#: 每个主题在语言文件里的小节标题数。
HEADING_COUNT: Final[int] = 6


# ---------- 句子与文档 ----------


@dataclass(frozen=True)
class Fact:
    """一句原子陈述：`topic` 主题里的第 `sentence` 句，极性 `negated`。

    文本不在这里（在语言文件里）——所以「同一句的正反两面」是结构上的关系，
    判别器从文本反查回 (主题, 句, 极性) 之后，用的还是同一个 `key()`。
    """

    topic: str
    sentence: str
    negated: bool = False

    @property
    def opposite(self) -> Fact:
        """输入：无。输出：同一句的反面。预期行为：只翻极性，不动主题与句号。"""
        return Fact(self.topic, self.sentence, not self.negated)

    def key(self) -> tuple[str, str, bool]:
        """输入：无。输出：可直接比大小的三元组键。"""
        return (self.topic, self.sentence, self.negated)


@dataclass(frozen=True)
class Section:
    """一个小节：编号（1 起）、标题（主题 + 下标）、按顺序的两句话。"""

    index: int
    heading_topic: str
    heading: int
    facts: tuple[Fact, ...]


# ---------- case（8 个维度的取值组合） ----------


@dataclass(frozen=True)
class CitationCase:
    """一个 case：8 个维度的取值组合。"""

    topic: str
    status: str
    quote_form: str
    sections: int
    noise: str
    claim_style: int
    where: str
    doc_style: str

    @property
    def count(self) -> int:
        """输入：无。输出：这份文档的小节数。"""
        return self.sections

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}，顺序与声明一致（哈希分桶按它算，必须稳定）。"""
        return {
            "topic": self.topic,
            "status": self.status,
            "quote_form": self.quote_form,
            "sections": self.sections,
            "noise": self.noise,
            "claim_style": self.claim_style,
            "where": self.where,
            "doc_style": self.doc_style,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"{self.topic}/{self.status}/{self.quote_form}/{self.sections}节"
            f"/{self.noise}/措{self.claim_style}/{self.where}/{self.doc_style}"
        )

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个 case 里训练侧没见过的维度名（空 = 全部见过）。"""
        checks: tuple[tuple[str, Any, frozenset[Any]], ...] = (
            ("topic", self.topic, SEEN_TOPICS),
            ("status", self.status, SEEN_STATUSES),
            ("quote_form", self.quote_form, SEEN_QUOTE_FORMS),
            ("sections", self.sections, SEEN_SECTION_COUNTS),
            ("noise", self.noise, SEEN_NOISES),
            ("claim_style", self.claim_style, SEEN_CLAIM_STYLES),
            ("where", self.where, SEEN_WHERES),
            ("doc_style", self.doc_style, SEEN_DOC_STYLES),
        )
        return tuple(name for name, value, seen in checks if value not in seen)


def where_fits(where: str, sections: int) -> bool:
    """输入：位置档、小节数。输出：这个位置在这份文档里存不存在。

    预期行为：两节没有「中间」；三节才有。不成立的组合在 `build_cases` 里剔掉。
    """
    return where != "middle" or sections >= 3


def quote_form_fits(status: str, quote_form: str) -> bool:
    """输入：结论档、引文写法。输出：这个组合成不成立。

    预期行为：只有 `fabricated` 才允许引文不是原文（截断 / 转述）；其余档引文必须照抄，
    否则「引文在不在文档里」与结论就自相矛盾了。
    """
    if status == "fabricated":
        return quote_form != "verbatim"
    return quote_form == "verbatim"


def _needs_free_section(noise: str) -> bool:
    """输入：干扰档。输出：它是不是需要「一节既不是声称出处、也不是引文老家」的小节。

    预期行为：`decoy` / `off_topic` 都要往第三个地方塞东西；`plain` 不要。
    """
    return noise in ("decoy", "off_topic")


@dataclass(frozen=True)
class CitationConfig:
    """一个切分的采样参数。

    字段：各维度的取值表（留出时收窄）、问法池、`ood_only` 是否只留「含没见过的取值」的 case。
    """

    split: str = "train"
    topics: tuple[str, ...] = TOPIC_NAMES
    statuses: tuple[str, ...] = STATUSES
    quote_forms: tuple[str, ...] = QUOTE_FORMS
    section_counts: tuple[int, ...] = SECTION_CHOICES
    noises: tuple[str, ...] = NOISES
    claim_styles: tuple[int, ...] = CLAIM_STYLES
    wheres: tuple[str, ...] = WHERES
    doc_styles: tuple[str, ...] = DOC_STYLES
    phrasings: tuple[int, ...] = (0, 1, 2)
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        for field, values in (
            ("topics", self.topics),
            ("statuses", self.statuses),
            ("quote_forms", self.quote_forms),
            ("section_counts", self.section_counts),
            ("noises", self.noises),
            ("claim_styles", self.claim_styles),
            ("wheres", self.wheres),
            ("doc_styles", self.doc_styles),
            ("phrasings", self.phrasings),
        ):
            require_pool(field, values)
        for field, values, allowed in (
            ("topics", self.topics, TOPIC_NAMES),
            ("statuses", self.statuses, STATUSES),
            ("quote_forms", self.quote_forms, QUOTE_FORMS),
            ("noises", self.noises, NOISES),
            ("wheres", self.wheres, WHERES),
            ("doc_styles", self.doc_styles, DOC_STYLES),
        ):
            unknown = [value for value in values if value not in allowed]
            if unknown:
                raise ValueError(f"{field} 里有不存在的取值：{unknown}（可用：{list(allowed)}）")
        bad_sections = [value for value in self.section_counts if value not in SECTION_CHOICES]
        if bad_sections:
            raise ValueError(f"section_counts 里有不支持的小节数：{bad_sections}")


def build_cases(config: CitationConfig) -> list[CitationCase]:
    """枚举 case 矩阵。

    输入：采样参数。输出：`CitationCase` 列表（顺序稳定、可复算）。
    预期行为：全枚举后剔掉两类不成立的组合——「两节却要放中间」「两节 + `mislocated`
    却还要往第三个地方塞干扰」；`ood_only` 时只留「至少含一个训练侧没见过的取值」的 case。
    """
    out: list[CitationCase] = []
    for topic in config.topics:
        for status in config.statuses:
            for quote_form in config.quote_forms:
                if not quote_form_fits(status, quote_form):
                    continue
                for sections in config.section_counts:
                    for noise in config.noises:
                        # 两节时 `mislocated` 已经占满两节（声称出处 + 引文老家），
                        # 没有第三处可塞干扰——这一格无解，直接剔掉。
                        if _needs_free_section(noise) and sections < 2 + (status == "mislocated"):
                            continue
                        for claim_style in config.claim_styles:
                            for where in config.wheres:
                                if not where_fits(where, sections):
                                    continue
                                for doc_style in config.doc_styles:
                                    case = CitationCase(
                                        topic=topic,
                                        status=status,
                                        quote_form=quote_form,
                                        sections=sections,
                                        noise=noise,
                                        claim_style=claim_style,
                                        where=where,
                                        doc_style=doc_style,
                                    )
                                    if config.ood_only and not case.unseen_dims():
                                        continue
                                    out.append(case)
    return out


# ---------- spec（渲染前的真值） ----------


@dataclass(frozen=True)
class CitationSpec:
    """一条样本的真值：整份文档 + 主张 + 引文的来历。

    字段：`sections` 文档；`claim` 主张的那一句；`quote_source` 引文取自哪一句
    （`truncated` 是它的前半句、`reworded` 没有来源句）；`quote_home` 引文**真正**在的小节
    （None = 不在文档里）；`claimed` 引文**声称**的出处（1 起）。
    """

    case: CitationCase
    split: str
    seq: int
    phrasing: int
    sections: tuple[Section, ...]
    claim: Fact
    quote_source: Fact | None
    quote_home: int | None
    claimed: int

    def section(self, index: int) -> Section:
        """输入：小节编号（1 起）。输出：那个小节。预期行为：越界直接报错。"""
        for section in self.sections:
            if section.index == index:
                return section
        raise KeyError(f"没有编号为 {index} 的小节（共 {len(self.sections)} 节）")

    @property
    def claimed_section(self) -> Section:
        """输入：无。输出：引文声称的那个小节。"""
        return self.section(self.claimed)

    def truth(self) -> tuple[bool, bool, str]:
        """输入：无。输出：(引文在不在文档里, 在不在声称的小节里, 声称小节与主张的关系)。

        预期行为：三条都只从 spec 的结构算——`quote_home is None` 就是「不在文档里」，
        判别侧会从文本重算一遍同样的三条（见 `judges.recompute`），两边对不上由测试当场抓。
        """
        found = self.quote_home is not None
        located = found and self.quote_home == self.claimed
        return found, located, relation_of(self.claim, self.claimed_section.facts)


# ---------- 规则引擎（构造侧与判别侧共用） ----------


def relation_of(claim: Fact, facts: tuple[Fact, ...]) -> str:
    """输入：主张、一个小节的句子。输出：`supports` / `contradicts` / `says_nothing`。

    预期行为：只看**这个小节**——有句子与主张同句同极性 = 支持；有句子与主张同句反极性 =
    反驳；否则没提。判据就是题面里写的那三条。
    """
    keys = {fact.key() for fact in facts}
    if claim.key() in keys:
        return "supports"
    if claim.opposite.key() in keys:
        return "contradicts"
    return "says_nothing"


def section_of_fact(sections: tuple[Section, ...], fact: Fact | None) -> int | None:
    """输入：文档、一句话。输出：它在哪个小节（1 起），或 None。

    预期行为：构造期保证**同一句话在文档里只出现一次**，所以「第一个命中的小节」就是唯一解。
    """
    if fact is None:
        return None
    for section in sections:
        if any(item.key() == fact.key() for item in section.facts):
            return section.index
    return None


def source_of(found: bool, located: bool) -> str:
    """输入：前两条事实。输出：引文的实际出处档。

    预期行为：不在文档里 = `not_in_doc`；在、且在声称的小节里 = `cited_section`；
    在、但在别的小节里 = `other_section`。
    """
    if not found:
        return "not_in_doc"
    return "cited_section" if located else "other_section"


def verdict_of(found: bool, located: bool, relation: str) -> str:
    """输入：前三条事实。输出：四档结论之一。

    预期行为：就是题面里写的那张表——不在文档里 = 编造；在文档里但不在声称的小节里 =
    证据不足；在声称的小节里按该小节的态度分三档（支持 = 通过、反驳 = 反驳、没提 = 证据不足）。
    """
    if not found:
        return "fabricated"
    if not located:
        return "unsupported"
    return {"supports": "verified", "contradicts": "contradicted", "says_nothing": "unsupported"}[
        relation
    ]


def action_of(verdict: str) -> str:
    """输入：结论档。输出：处置档。

    预期行为：只有「证据不足」要人看——「编造」由字符串匹配直接定案，也不需要人。
    """
    return "human_review" if verdict == "unsupported" else "auto_accept"


def strength_of(found: bool, located: bool, relation: str) -> int:
    """输入：前三条事实。输出：证据强度 0~3。

    预期行为：就是题面里那张量表——不在文档里 0；在但不在声称小节 1；在声称小节但没提 2；
    在声称小节且有明确态度 3。
    """
    if not found:
        return 0
    if not located:
        return 1
    return 2 if relation == "says_nothing" else 3


def answers_of(found: bool, located: bool, relation: str) -> dict[str, Any]:
    """输入：前三条事实。输出：{题族名: 答案}——七个题族一次算全。

    预期行为：结论、处置、证据强度都**只**由这三条推出来，任何一侧都不许另起一套判断。
    键**就是** `questions.FAMILIES` 里那七个名字（判别器拿题号反查，见 `judges._check`）——
    少一个或写错一个名字，答案判别器会把每一条样本都判成「未知题族」。
    """
    verdict = verdict_of(found, located, relation)
    return {
        "found": found,
        "located": located,
        "relation": relation,
        "quote_source": source_of(found, located),
        "verdict": verdict,
        "action": action_of(verdict),
        "evidence_strength": strength_of(found, located, relation),
    }


# ---------- 采样 ----------


def case_rng(case: CitationCase) -> random.Random:
    """输入：case。输出：**由 case 派生**的 RNG。

    预期行为：整份文档完全由 case 决定，不掷骰子——所以同一 case 两次生成逐字节一致，
    「两次生成一样」这条能直接查。不用内置 `hash()`（它对字符串每进程随机化）。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    seed = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def _position_of(where: str, sections: int) -> int:
    """输入：位置档、小节数。输出：小节编号（1 起）。"""
    if where == "first":
        return 1
    if where == "last":
        return sections
    if where == "middle":
        if sections < 3:
            raise ValueError(f"{sections} 节没有「中间」这一档")
        return (sections + 1) // 2
    raise ValueError(f"未知位置档：{where!r}")


def _fillers(
    pool: tuple[str, ...], claim: Fact, used: set[tuple[str, str, bool]], count: int, rng: random.Random
) -> list[Fact]:
    """输入：句子键池、主张、已用键、要几句、RNG。输出：若干句无关句子。

    预期行为：**避开主张那一句**（否则它会把「没提」变成「支持 / 反驳」），
    并且不重复已经用过的键——同一句话在文档里出现两次会让「引文在哪个小节」失去唯一解。
    """
    candidates = [
        Fact(claim.topic, sentence, negated)
        for sentence in pool
        if sentence != claim.sentence
        for negated in (False, True)
    ]
    rng.shuffle(candidates)
    out: list[Fact] = []
    for fact in candidates:
        if len(out) == count:
            break
        if fact.key() in used:
            continue
        used.add(fact.key())
        out.append(fact)
    if len(out) != count:
        raise ValueError(f"句子池不够：要 {count} 句无关句，只凑出 {len(out)} 句")
    return out


def sample_spec(
    rng: object, case: CitationCase, split: str, seq: int, phrasing: int
) -> CitationSpec:
    """输入：RNG（不用，留着满足签名）、case、切分名、样本序号、问法编号。输出：真值规格。

    预期行为：文档、主张、引文来历**完全由 case 决定**（见 `case_rng`）。构造顺序：
    定主张 → 定小节数与标题 → 定「引文老家」与「声称出处」→ 按结论档摆那一句 →
    其余位置填无关句 → 按干扰档加料。
    """
    del rng  # 引文核查页不需要外部随机：多样性由 8 个维度给足
    picker = case_rng(case)
    pool = SENTENCE_IDS[case.topic]
    claim = Fact(case.topic, picker.choice(pool), negated=bool(picker.getrandbits(1)))
    sections_count = case.sections
    claimed = _position_of(case.where, sections_count)
    #: 引文老家：`mislocated` 是「别的小节」，其余档就是声称的那个小节。
    if case.status == "mislocated":
        home = picker.choice([index for index in range(1, sections_count + 1) if index != claimed])
    else:
        home = claimed

    used: set[tuple[str, str, bool]] = set()
    special: dict[int, Fact] = {}
    quote_source: Fact | None = None
    quote_home: int | None = None
    if case.status == "verified":
        special[claimed] = claim
        quote_source, quote_home = claim, home
    elif case.status == "contradicted":
        special[claimed] = claim.opposite
        quote_source, quote_home = claim.opposite, home
    elif case.status == "mislocated":
        special[home] = claim
        quote_source, quote_home = claim, home
    #: `unsupported` 与 `fabricated` 的「引文老家」都只是一句无关句。
    else:
        filler = _fillers(pool, claim, used, 1, picker)[0]
        special[claimed if case.status == "unsupported" else home] = filler
        if case.status == "unsupported":
            quote_source, quote_home = filler, home
        elif case.quote_form == "truncated":
            #: 截断写法得有「被截的原句」才渲染得出来——否则渲染层拿到 None 直接炸
            #: （`fact_text(None)` 抛「要渲染的句子是空的」）。这里取那句**真实躺在文档里**
            #: 的无关句：引文是它的前半句，整句匹配照样找不到，所以 `quote_home` 仍留 None
            #: （结论仍是编造）。`reworded` 写的是主张的转述，用不到来源句。
            quote_source = filler
    for fact in special.values():
        used.add(fact.key())

    #: 干扰：`decoy` 在第三个地方放主张的另一面（被占就放主张本身，保证不重复、
    #: 也保证这一档真的改变页面）；`off_topic` 把第三个地方整节换成别的主题。
    free = [index for index in range(1, sections_count + 1) if index not in {claimed, home}]
    decoy_at = picker.choice(free) if case.noise == "decoy" and free else None
    off_topic_at = picker.choice(free) if case.noise == "off_topic" and free else None
    other_topic = picker.choice([name for name in TOPIC_NAMES if name != case.topic])

    headings = picker.sample(range(HEADING_COUNT), sections_count)
    other_headings = picker.sample(range(HEADING_COUNT), sections_count)
    sections: list[Section] = []
    for index in range(1, sections_count + 1):
        if index == off_topic_at:
            facts = _fillers(
                SENTENCE_IDS[other_topic],
                Fact(other_topic, SENTENCE_IDS[other_topic][0]),
                used,
                2,
                picker,
            )
            sections.append(
                Section(
                    index=index,
                    heading_topic=other_topic,
                    heading=other_headings[index - 1],
                    facts=tuple(facts),
                )
            )
            continue
        facts = []
        if index in special:
            facts.append(special[index])
        elif index == decoy_at:
            decoy = claim.opposite if claim.opposite.key() not in used else claim
            used.add(decoy.key())
            facts.append(decoy)
        facts.extend(_fillers(pool, claim, used, 2 - len(facts), picker))
        sections.append(
            Section(
                index=index,
                heading_topic=case.topic,
                heading=headings[index - 1],
                facts=tuple(facts),
            )
        )
    return CitationSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        sections=tuple(sections),
        claim=claim,
        quote_source=quote_source,
        quote_home=quote_home,
        claimed=claimed,
    )


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的主题：`rollout` 整个留出（句子表、标题一起没见过）。
SEEN_TOPICS: Final[frozenset[str]] = frozenset(
    name for name in TOPIC_NAMES if name != "rollout"
)
#: 训练侧见过的结论档：`mislocated` 留出（引文在文档里、但不在声称的小节里）。
SEEN_STATUSES: Final[frozenset[str]] = frozenset(
    name for name in STATUSES if name != "mislocated"
)
#: 训练侧见过的引文写法：`reworded`（引文写成主张的转述）留出。
SEEN_QUOTE_FORMS: Final[frozenset[str]] = frozenset(
    name for name in QUOTE_FORMS if name != "reworded"
)
#: 训练侧见过的小节数：两节与三节都见过——版式类留出换来的 OOD 与「同一批数据换个长度」
#: 几乎等价，信息量太低，不如把预算留给结论档与引文写法。
SEEN_SECTION_COUNTS: Final[frozenset[int]] = frozenset(SECTION_CHOICES)
#: 训练侧见过的干扰档：`off_topic`（有一整节讲别的主题）留出。
SEEN_NOISES: Final[frozenset[str]] = frozenset(
    name for name in NOISES if name != "off_topic"
)
#: 训练侧见过的主张措辞：第一种留出。
SEEN_CLAIM_STYLES: Final[frozenset[int]] = frozenset({1})
#: 训练侧见过的位置与编号写法：都见过（同上，版式类留出信息量太低）。
SEEN_WHERES: Final[frozenset[str]] = frozenset(WHERES)
SEEN_DOC_STYLES: Final[frozenset[str]] = frozenset(DOC_STYLES)
