"""内容审核（护栏）workflow 的真值规格：一条待审内容 + 一份政策判据。

输入/输出：不可变数据类与纯函数；**不含任何领域词表**（条款名、危害级别、措辞模板、
正文模板、选项描述与判据原文全在 `locales/`）。
预期行为：所有答案都**只从渲染出来的那屏内容反推**——每条政策条款是否被违反、整体严重度、
处置动作，全由已知的表算出来；构造侧与判别侧**共用同一套规则函数**，所以「表写错」与
「渲染写歪」会变成同一类错误，不会两边各错一半还互相印证。

为什么不做成 LLM 生成：待审内容由「违规句 + 擦边句 + 填充句」拼出来，判据是已知的表，
让模型去编只会引入抖动；这里要训的是「按政策读一条内容并处置」，不是「编一条内容」。

与 routing 的区别：routing 那几个检测题族共用**一张规则表**，这里的六条条款是**互相独立**的
（官方 `llm_guardrails` 的形态：一批独立 noul + 一个 score + 阈值路由），
所以「同一条内容里哪几条同时成立」才是难点。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 政策条款 ----------

#: 六条互相独立的政策条款（顺序即 `top_clause` 并列时的取用顺序，必须稳定）。
CLAUSES: Final[tuple[str, ...]] = ("harass", "meds", "scam", "privacy", "spam", "cheat")

#: 条款 → 危害级别。级别只进严重度矩阵，**不进正文**（正文只说「发生了什么」）。
#: 三档 3 / 2 / 1 条：高危占三条不是随手定的——`top_clause` 只在最高级别里挑，
#: 最高级别只有一两条时那个答案会被垄断（实测两条时会到 45%）。
HARM_LEVELS: Final[dict[str, str]] = {
    "meds": "high",
    "scam": "high",
    "privacy": "high",
    "harass": "medium",
    "cheat": "medium",
    "spam": "low",
}
LEVEL_ORDER: Final[tuple[str, ...]] = ("low", "medium", "high")

# ---------- 违规条款子集表 ----------
#: 「这条内容违反了哪几条」的十一种组合。四个目标一起满足（复算见 `tmp/pick_subsets4.py`）：
#: 1. 六条条款**各出现 5 次**（真值占比 0.455，离 50% 只差 0.045）；
#: 2. **没有任何一对条款总是同时出现**——否则「有 A 就一定有 B」是送答案的捷径；
#: 3. 最高危害级别落在三档的组合数 6 / 3 / 1：高危那一档故意占多数，因为 `top_clause`
#:    只在最高级别里挑，最高级别那几条够多才摊得平（配合正文句序并列规则，七个取值全活）；
#: 4. 配合严重度矩阵，五档严重度占比 4 / 5 / 9 / 9 / 6（最大偏差 0.073）。
VIOLATION_SETS: Final[tuple[tuple[str, ...], ...]] = (
    (),
    ("meds", "scam", "privacy", "harass"),
    ("meds", "scam", "privacy", "cheat"),
    ("meds", "scam", "privacy", "harass", "cheat", "spam"),
    ("meds", "scam", "harass"),
    ("meds", "privacy", "cheat"),
    ("scam", "privacy", "spam"),
    ("harass", "spam"),
    ("cheat", "spam"),
    ("harass", "cheat"),
    ("spam",),
)

#: 「只是提到、按判据不算违规」的擦边句组合。它**不影响任何答案**，
#: 只让正文里多出一句看起来像违规的话——这正是要压掉的「见关键词就报警」捷径。
#: 与 `VIOLATION_SETS` 允许重叠：同一条款既可以有擦边句、也可以另有一句真的违规。
MENTION_SETS: Final[tuple[tuple[str, ...], ...]] = (
    (),
    ("meds",),
    ("privacy", "spam"),
    ("harass", "scam", "cheat"),
)

# ---------- 其余维度 ----------

REACHES: Final[tuple[str, ...]] = ("single", "group", "public")
DIRECTIONS: Final[tuple[str, ...]] = ("inbound", "outbound")
AUDIENCES: Final[tuple[str, ...]] = ("general", "minors")
HISTORIES: Final[tuple[str, ...]] = ("clean", "warned", "repeat")
WORDINGS: Final[tuple[str, ...]] = ("plain", "quoted", "softened")

# ---------- 严重度矩阵 ----------

#: 严重度矩阵：最高危害级别 × 传播面 → 0~4 档。九格全写（判据原文照抄进题面）。
#: 语义：级别越高、看得人越多越严重；「低危 + 单人可见」才落到 0。
SEVERITY_TABLE: Final[dict[tuple[str, str], int]] = {
    ("low", "single"): 0,
    ("low", "group"): 1,
    ("low", "public"): 1,
    ("medium", "single"): 1,
    ("medium", "group"): 2,
    ("medium", "public"): 3,
    ("high", "single"): 2,
    ("high", "group"): 3,
    ("high", "public"): 4,
}

# ---------- 处置动作 ----------

ACTIONS: Final[tuple[str, ...]] = ("pass", "review", "block", "route")


def max_harm_of(violations: tuple[str, ...]) -> str | None:
    """输入：违规条款。输出：其中最高的危害级别（空集给 None）。

    预期行为：空集不是「low」——「一条都没违规」与「违规但很轻」是两回事，
    混成同一个值会让严重度矩阵的 0 档与 1 档糊在一起。
    """
    if not violations:
        return None
    return max((HARM_LEVELS[clause] for clause in violations), key=LEVEL_ORDER.index)


def severity_of(violations: tuple[str, ...], reach: str) -> int:
    """输入：违规条款、传播面。输出：严重度 0~4。预期行为：查表（不做加权求和）。

    一条都没违规时固定 0 档——矩阵里没有「无违规」这一行，所以这一档单独定义。
    """
    level = max_harm_of(violations)
    if level is None:
        return 0
    return SEVERITY_TABLE[(level, reach)]


def violated(clause: str, violations: tuple[str, ...]) -> bool:
    """输入：条款名、违规集合。输出：该条款是否被违反。预期行为：集合判断。"""
    return clause in violations


def top_clause_of(violations: tuple[str, ...], order: tuple[str, ...]) -> str:
    """输入：违规条款、违规句在正文里的出现顺序。输出：危害级别最高的那一条。

    预期行为：空集给 `"none"`；**并列时取正文里最先出现的那一句**（`order` 里靠前的）。
    这条并列规则必须写死在题面里，否则「级别相同取哪一条」没有唯一答案，标签就不合法。
    用正文句序而不是条款表顺序来破并列，是为了让七个取值都能摊到差不多的比例：
    条款表顺序是固定的，最高级别那几条里排前面的会垄断这个答案。
    """
    level = max_harm_of(violations)
    if level is None:
        return "none"
    return next(clause for clause in order if HARM_LEVELS[clause] == level)


def minor_exposure_of(audience: str, violations: tuple[str, ...]) -> bool:
    """输入：受众、违规条款。输出：是否构成「未成年人暴露」。

    预期行为：受众含未成年人 **且** 命中 medium / high 级别的条款才为真——低危
    （垃圾推广）单独出现不算暴露，这条分界线写进题面。
    **故意比处置规则①更宽**：处置规则①只对高危内容转专项。两族口径若强行统一，
    总有一族被均衡闸标红——统一到中危时「转专项」占 0.409，统一到高危时这一族只剩
    0.273（变体扫描见 `tmp/gr_actions.py`）。分开之后两族分别在 0.409 与 0.273，
    都落在容差里。
    """
    return audience == "minors" and max_harm_of(violations) in ("medium", "high")


def repeat_offender_of(history: str, severity: int) -> bool:
    """输入：历史、严重度。输出：是否算「累犯」。预期行为：历史是多次违规且严重度 ≥ 2。"""
    return history == "repeat" and severity >= 2


def action_of(
    *,
    severity: int,
    direction: str,
    reach: str,
    audience: str,
    history: str,
    violations: tuple[str, ...],
) -> str:
    """输入：严重度、方向、传播面、受众、历史、违规条款。输出：处置动作。

    预期行为：**六条有序规则，按顺序取第一个命中**（判据原文照抄进题面）：
    ① 受众含未成年人且命中高危条款 → 转专项；② 模型输出且严重度 ≥ 3 → 拦截；
    ③ 公开可见且严重度 ≥ 2 → 拦截；④ 累犯且严重度 ≥ 2 → 拦截；
    ⑤ 严重度 ≥ 2 → 复核；⑥ 其余 → 放行。
    """
    if audience == "minors" and max_harm_of(violations) == "high":
        return "route"
    if direction == "outbound" and severity >= 3:
        return "block"
    if reach == "public" and severity >= 2:
        return "block"
    if repeat_offender_of(history, severity):
        return "block"
    if severity >= 2:
        return "review"
    return "pass"


# ---------- case 与采样参数 ----------

CONTENT_PREFIX: Final[str] = "MSG-"


@dataclass(frozen=True)
class GuardCase:
    """一条待审内容的七个维度（顺序与 `as_dict` 一致，哈希分桶按它算，必须稳定）。"""

    reach: str
    direction: str
    violations: tuple[str, ...]
    mentions: tuple[str, ...]
    audience: str
    history: str
    wording: str

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}（顺序稳定；两个集合维度摊成 `+` 连接的串）。"""
        return {
            "reach": self.reach,
            "direction": self.direction,
            "violations": label_of(self.violations),
            "mentions": label_of(self.mentions),
            "audience": self.audience,
            "history": self.history,
            "wording": self.wording,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"{self.reach}/{self.direction}/违规={label_of(self.violations)}"
            f"/提及={label_of(self.mentions)}/{self.audience}/{self.history}/{self.wording}"
        )

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个 case 里训练侧没见过的维度名（空 = 全部见过）。"""
        checks: tuple[tuple[str, Any, frozenset[Any]], ...] = (
            ("mentions", self.mentions, SEEN_MENTIONS),
            ("history", self.history, SEEN_HISTORIES),
            ("wording", self.wording, SEEN_WORDINGS),
        )
        return tuple(name for name, value, seen in checks if value not in seen)


def label_of(subset: tuple[str, ...]) -> str:
    """输入：条款子集。输出：可读标签（空集给 `none`）。

    预期行为：子集在 `as_dict` / `describe` 里必须有个稳定且唯一的写法——
    直接 `str(tuple)` 会因为顺序不同而把同一个集合写成两种样子，统计会裂开。
    所以先按 `CLAUSES` 顺序归一，再用 `+` 连接。
    """
    if not subset:
        return "none"
    return "+".join(clause for clause in CLAUSES if clause in subset)


def material_of(case: GuardCase) -> str:
    """输入：case。输出：素材指纹——决定**正文文本**的那几维拼成的串。

    预期行为：七个维度全部进指纹（`wording` 只改措辞、不改判据，但它确实改文本，
    所以也算素材）。切分按它分桶（`workflow._slice`），同一条内容整组落在同一档。
    """
    return "|".join(f"{key}={value}" for key, value in case.as_dict().items())


#: 正文里固定垫几句与判据无关的话（不给任何答案提供线索，只把正文撑成一段话）。
FILLER_SLOTS: Final[int] = 2

#: 决定**答案**的那几维（句序按它派生）。`mentions` / `wording` 只改文本，不进这里。
ANSWER_DIMS: Final[tuple[str, ...]] = (
    "reach",
    "direction",
    "violations",
    "audience",
    "history",
)


def answer_material(case: GuardCase) -> str:
    """输入：case。输出：只含决定答案那几维的指纹（句序用它，切分不用它）。

    预期行为：把 `mentions` / `wording` 排除在外——它们只改文本。若混进来，
    「加一句擦边句」或「换个措辞」都会重排违规句，而 `top_clause` 的并列规则吃的
    正是正文顺序，于是这两维就会**偷偷改答案**（实测踩过：改措辞后 top_clause 变了一个值）。
    """
    values = case.as_dict()
    return "|".join(f"{key}={values[key]}" for key in ANSWER_DIMS)


def slot_order(case: GuardCase) -> tuple[tuple[str, str], ...]:
    """输入：case。输出：正文句子的槽位顺序 `[(类型, 条款或编号), ...]`。

    预期行为：槽位 = 每个违规条款一句 + 每个擦边条款一句 + `FILLER_SLOTS` 句填充；
    顺序由 `answer_material` 派生——同一 case 必得同一顺序，顺序与条款名无关
    （不是字母序、也不是条款表顺序），而且**加减擦边句、换措辞都不会重排违规句**。
    渲染器与判别器都调它，两边看到的是同一个顺序。
    """
    slots: list[tuple[str, str]] = [
        ("violation", clause) for clause in CLAUSES if clause in case.violations
    ]
    slots += [("mention", clause) for clause in CLAUSES if clause in case.mentions]
    slots += [("filler", str(index)) for index in range(FILLER_SLOTS)]
    material = answer_material(case)
    return tuple(
        sorted(
            slots,
            key=lambda slot: hashlib.sha256(
                f"{material}|{slot[0]}|{slot[1]}".encode("utf-8")
            ).hexdigest(),
        )
    )


def violation_order(case: GuardCase) -> tuple[str, ...]:
    """输入：case。输出：违规句在正文里的出现顺序（`top_clause_of` 的并列规则要用它）。"""
    return tuple(clause for kind, clause in slot_order(case) if kind == "violation")


@dataclass(frozen=True)
class GuardConfig:
    """一个切分的采样参数：各维度取值表 + 问法池 + `ood_only`。"""

    split: str = "train"
    reaches: tuple[str, ...] = REACHES
    directions: tuple[str, ...] = DIRECTIONS
    violations: tuple[tuple[str, ...], ...] = VIOLATION_SETS
    mentions: tuple[tuple[str, ...], ...] = MENTION_SETS
    audiences: tuple[str, ...] = AUDIENCES
    histories: tuple[str, ...] = HISTORIES
    wordings: tuple[str, ...] = WORDINGS
    phrasings: tuple[int, ...] = (0, 1, 2)
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        pools: tuple[tuple[str, tuple[Any, ...]], ...] = (
            ("reaches", self.reaches),
            ("directions", self.directions),
            ("violations", self.violations),
            ("mentions", self.mentions),
            ("audiences", self.audiences),
            ("histories", self.histories),
            ("wordings", self.wordings),
            ("phrasings", self.phrasings),
        )
        for field, values in pools:
            require_pool(field, values)
        known: tuple[tuple[str, tuple[Any, ...], tuple[Any, ...]], ...] = (
            ("reaches", self.reaches, REACHES),
            ("directions", self.directions, DIRECTIONS),
            ("violations", self.violations, VIOLATION_SETS),
            ("mentions", self.mentions, MENTION_SETS),
            ("audiences", self.audiences, AUDIENCES),
            ("histories", self.histories, HISTORIES),
            ("wordings", self.wordings, WORDINGS),
        )
        for field, values, allowed in known:
            unknown = [value for value in values if value not in allowed]
            if unknown:
                raise ValueError(f"{field} 里有不存在的取值：{unknown}（可用：{list(allowed)}）")


def build_cases(config: GuardConfig) -> list[GuardCase]:
    """枚举 case 矩阵。

    输入：采样参数。输出：`GuardCase` 列表（顺序稳定、可复算）。
    预期行为：全枚举（**不做成立性过滤**——七个维度两两不矛盾：擦边句与违规句允许
    指向同一条款，那时该条款照样算违规，判据唯一）；`ood_only` 时只留「至少含一个
    训练侧没见过的取值」的 case，所以 OOD 与同分布侧**按构造就不重叠**。
    """
    out: list[GuardCase] = []
    for reach in config.reaches:
        for direction in config.directions:
            for violations in config.violations:
                for mentions in config.mentions:
                    for audience in config.audiences:
                        for history in config.histories:
                            for wording in config.wordings:
                                case = GuardCase(
                                    reach,
                                    direction,
                                    violations,
                                    mentions,
                                    audience,
                                    history,
                                    wording,
                                )
                                if config.ood_only and not case.unseen_dims():
                                    continue
                                out.append(case)
    return out


# ---------- 真值规格 ----------


@dataclass(frozen=True)
class GuardSpec:
    """一条待审内容的完整真值：case + 由规则算出来的一切。"""

    case: GuardCase
    split: str
    seq: int
    phrasing: int
    content_id: str
    severity: int
    action: str
    top_clause: str
    minor_exposure: bool
    repeat_offender: bool

    def violated(self, clause: str) -> bool:
        """输入：条款名。输出：该条款是否被违反。预期行为：转调模块级 `violated`。"""
        return violated(clause, self.case.violations)


def content_id_of(case: GuardCase) -> str:
    """输入：case。输出：内容号（形如 `MSG-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**：
    用序号或切分名拼内容号会让同一条内容在不同切分里拿到不同号，
    等于把切分信息写进 state。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{CONTENT_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(rng: Any, case: GuardCase, split: str, seq: int, phrasing: int) -> GuardSpec:
    """输入：RNG（**不用**）、case、切分、序号、问法。输出：真值规格。

    预期行为：不掷骰子——同一 case 必得同一份规格（内容号也由 case 指纹派生），
    这样「同素材同题必须同答」这条一致性校验才成立。
    """
    del rng  # 规格完全由 case 决定；签名里的 RNG 只为与协议一致
    severity = severity_of(case.violations, case.reach)
    return GuardSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        content_id=content_id_of(case),
        severity=severity,
        action=action_of(
            severity=severity,
            direction=case.direction,
            reach=case.reach,
            audience=case.audience,
            history=case.history,
            violations=case.violations,
        ),
        top_clause=top_clause_of(case.violations, violation_order(case)),
        minor_exposure=minor_exposure_of(case.audience, case.violations),
        repeat_offender=repeat_offender_of(case.history, severity),
    )


def spec_to_dict(spec: GuardSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "split": spec.split,
        "seq": spec.seq,
        "content_id": spec.content_id,
        "severity": spec.severity,
        "action": spec.action,
        "top_clause": spec.top_clause,
        "minor_exposure": spec.minor_exposure,
        "repeat_offender": spec.repeat_offender,
    }


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的擦边句组合：最后一种留出（OOD 侧才出现「辱骂 + 诱导转账 + 作弊」一起擦边）。
SEEN_MENTIONS: Final[frozenset[tuple[str, ...]]] = frozenset(
    value for value in MENTION_SETS if value != ("harass", "scam", "cheat")
)
#: 训练侧见过的历史：`warned` 留出（OOD 侧才出现「曾被警告一次」这个取值）。
#: 不留 `repeat`：那一档是 `repeat_offender` 与处置规则④的正例，留掉会让这一族
#: 在同分布侧退化成**常量假**——均衡闸当场标 0.500，等于用「坏数据」换「更难 OOD」。
SEEN_HISTORIES: Final[frozenset[str]] = frozenset(
    value for value in HISTORIES if value != "warned"
)
#: 训练侧见过的措辞：`softened` 留出（委婉写法只出现在 OOD 侧）。
#: 三条留出**都落在表面形式上**（擦边句组合 / 历史取值 / 措辞），刻意不动规则分支：
#: 试过留出 `reach = single`，严重度当场塌成 0.409 挤在 3 档——矩阵里「一对一」那一列
#: 是 0/1/2 三档的主要来源，拿「更难 OOD」换「坏数据」不值（对比见 `tmp/gen-check`）。
SEEN_WORDINGS: Final[frozenset[str]] = frozenset(
    value for value in WORDINGS if value != "softened"
)
