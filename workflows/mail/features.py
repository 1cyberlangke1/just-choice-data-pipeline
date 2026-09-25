"""邮件安全分诊的特征层：注入维度 + 标签规则（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」与 3.3「tag 组合」：
- 一封邮件由 5 个**注入特征**描述（发件人身份 / 诱因 / 索取 / 链接 / 语气），
  其中「链接」这一维把「域名像不像」与「链接文字有没有伪装」合成一个取值，
  这样就没有跨维约束要事后过滤；
- **标签先行**：先定这 5 维取值，再由本模块的纯函数算出全部金标签，
  模型只负责把特征写成自然文本，碰不到答案；
- 组合数 3×5×5×9×4 = 2700，全枚举对 LLM 太贵，交给 `jc.cover` 的**成对覆盖**
  （几十条就覆盖任意两维的取值组合）。只差一维的组合天然成对出现，
  所以「最小对」是 case 矩阵的性质，不需要额外造。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、不含随机源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims

# ---------- 维度表（PLAN 3.3 的 tags） ----------

SENDER: Final[tuple[str, ...]] = (
    "known_vendor",  # 认识的正规商家
    "unknown_personal",  # 陌生个人
    "forged_brand",  # 冒充知名品牌（显示名与真实域名不符）
)

LURE: Final[tuple[str, ...]] = (
    "none",  # 没有额外诱因，就是正常沟通
    "urgency",  # 制造紧迫（限时、即将停用）
    "reward",  # 意外奖励（中奖、返现、免费额度）
    "fear",  # 恐吓（已违规、将冻结、已立案）
    "curiosity",  # 勾起好奇（有人提到你、你被选上了）
)

ASK: Final[tuple[str, ...]] = (
    "none",
    "credentials",  # 要账号凭据
    "payment",  # 要付款 / 改收款信息
    "click_link",  # 只要求点链接
    "attachment",  # 要求打开附件
)

#: 「什么都不索取」那一档（反解提示词里要把它排除在「额外索取」的选项之外）。
ASK_NONE: Final[str] = "none"

# 链接维：`none` + 「域名可信度 × 链接文字是否伪装」的 8 种组合。
# 合成一维是为了让「没有链接就不该问链接文字」这条约束**天然成立**，
# 不用事后过滤（过滤会让成对覆盖掉取值对）。
LINK_NONE: Final[str] = "none"
LINK_TARGETS: Final[tuple[str, ...]] = ("official", "lookalike", "ip", "shortener")
ANCHORS: Final[tuple[str, ...]] = ("honest", "mismatch")
LINK: Final[tuple[str, ...]] = (
    LINK_NONE,
    *tuple(f"{target}_{anchor}" for target in LINK_TARGETS for anchor in ANCHORS),
)

# 语气维**只描述语域**，不承载任何内容信号。原先的 `threatening` 取值把「施压」又
# 表达了一遍，于是 `lure=none` + `tone=threatening` 的清单自相矛盾：一边要求
# 「通篇没有任何施压」，一边要求「用警告与后果的措辞」，模型只能摆烂。
# 施压信号统一归 `lure` 维承担，这一维就与内容正交了（2026-09-25 实跑发现的坑）。
TONE: Final[tuple[str, ...]] = ("formal", "friendly", "casual", "curt")

DIMS: Final[dict[str, tuple[str, ...]]] = {
    "sender": SENDER,
    "lure": LURE,
    "ask": ASK,
    "link": LINK,
    "tone": TONE,
}

# 会把链接域名判成「有欺骗性」的取值前缀。
DECEPTIVE_TARGETS: Final[frozenset[str]] = frozenset({"lookalike", "ip"})
# 「在向收件人索取东西」的索取取值：钓鱼判定的前半截。
WANTING_ASKS: Final[frozenset[str]] = frozenset({"credentials", "payment", "click_link"})
# 制造压力的诱因。
PRESSURE_LURES: Final[frozenset[str]] = frozenset({"urgency", "fear"})
# 陌生发件人用来钓人的诱因（利益 / 好奇）。
PROMO_LURES: Final[frozenset[str]] = frozenset({"reward", "curiosity"})

#: 天生不能同时出现的取值对（每项是若干 `(维度名, 取值)`）。
#: 现在只有一条：清单同时说「要求点击链接」和「邮件里没有任何链接」时无解——
#: 模型只能瞎写，反解必错（2026-09-25 实跑抓到的那条）。
FORBIDDEN: Final[frozenset[frozenset[tuple[str, str]]]] = frozenset(
    {frozenset({("ask", "click_link"), ("link", LINK_NONE)})}
)


def allow_pair(dim_a: str, value_a: str, dim_b: str, value_b: str) -> bool:
    """输入：两个维度的取值。输出：这一对能否同时出现。

    预期行为：只查 `FORBIDDEN`，与两个维度谁先谁后无关——覆盖器按声明顺序传参，
    这里不做顺序假设。给 `jc.cover` 当取值对约束用。
    """
    return frozenset({(dim_a, value_a), (dim_b, value_b)}) not in FORBIDDEN


def split_link(value: str) -> tuple[str, str]:
    """把链接维的取值拆成 (域名形态, 链接文字形态)。

    输入：`LINK` 里的一个取值。输出：`(target, anchor)`；没有链接时是 `("none", "n_a")`。
    预期行为：未知取值直接报错，不静默兜底。
    """
    if value == LINK_NONE:
        return "none", "n_a"
    target, _, anchor = value.rpartition("_")
    if target not in LINK_TARGETS or anchor not in ANCHORS:
        raise ValueError(f"未知的链接取值：{value!r}")
    return target, anchor


#: 每道 choice 题的选项名（= 该题的答案取值域），顺序就是选项书写顺序。
#: `link_text` 刻意不含 `no_link`：没有链接的邮件**不出这道题**（见 `questions.families_for`），
#: 所以 `no_link` 永远不是数据里的答案；但 locale 里保留它，反解提示词才完整。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "requested": ASK,
    "pressure": LURE,
    "link_text": ("matches", "mismatched"),
    "sender_identity": ("matches", "no_claim", "forged"),
    "action": ("deliver", "review", "quarantine", "block"),
}


@dataclass(frozen=True)
class TagSet:
    """一组注入特征（= PLAN 3.3 的一个 tag 组合）。

    字段：`sender` / `lure` / `ask` / `link`（合成维）/ `tone`。
    """

    sender: str
    lure: str
    ask: str
    link: str
    tone: str

    def as_dict(self) -> dict[str, str]:
        """输入：无。输出：`{维度名: 取值}`，顺序与 `DIMS` 一致。"""
        return {
            "sender": self.sender,
            "lure": self.lure,
            "ask": self.ask,
            "link": self.link,
            "tone": self.tone,
        }

    @property
    def link_target(self) -> str:
        """输入：无。输出：域名形态（`none` / `official` / `lookalike` / `ip` / `shortener`）。"""
        return split_link(self.link)[0]

    @property
    def anchor(self) -> str:
        """输入：无。输出：链接文字形态（`n_a` / `honest` / `mismatch`）。"""
        return split_link(self.link)[1]

    def label(self) -> Labels:
        """输入：无。输出：这一组特征对应的全部金标签。"""
        return label_for(self)


@dataclass(frozen=True)
class Labels:
    """一封信的全部金标签：字段名就是题族名，`questions.py` 直接按名字取。

    前四个是「读出来」的原子特征（答案就是某个注入维度的取值），后三个是「算出来」的
    汇总判定。原子题做成 choice 而不是 noul 是有意的：choice 的答案就是维度取值，
    在均匀 tag 空间里**天然均衡**；noul 的答案是一个子集，必然偏斜。
    """

    requested: str  # ask 的取值
    pressure: str  # lure 的取值
    link_text: str  # no_link / matches / mismatched
    sender_identity: str  # matches / no_claim / forged
    is_phishing: bool
    is_spam: bool
    severity: int
    action: str


#: 四档处置的轻重，用来取「分类判出来的动作」与「严重度下限」里更重的那个。
ACTION_RANK: Final[dict[str, int]] = {
    "deliver": 0,
    "review": 1,
    "quarantine": 2,
    "block": 3,
}

#: 严重度 → 处置的**下限**：3 级（高度可疑）至少要人工复核，4 级（风险极高）至少要隔离。
#: 存在的理由：严重度是**风险量表**，钓鱼 / 垃圾是**分类判定**，两者会分叉——
#: 「冒名 + 伪链 + 施压，却什么都没索取」够不上钓鱼（钓鱼是合取，见 `label_for`），
#: 却实打实高风险。只按分类判处置时，这种信会拿到 `deliver`（「正常投递，不做额外处理」），
#: 与同一封信的「风险很高 / 极高」直接打架（2026-09-25 全量 case 扫描抓到 128 组）。
ACTION_FLOOR: Final[dict[int, str]] = {3: "review", 4: "quarantine"}


def action_for(tags: TagSet, *, is_phishing: bool, is_spam: bool, severity: int) -> str:
    """输入：特征、钓鱼判定、垃圾判定、严重度。输出：处置动作（四档，有序）。

    预期行为：取「按分类判出来的动作」与「严重度要求的下限」（`ACTION_FLOOR`）里
    更重的那个。分类这一路是「能拦就拦、拦不了就查、没事就放」：钓鱼 → 拦截 / 隔离、
    垃圾 → 复核。推广类只降级到 `review` 而不是直接拦——收件人可能真的订阅过，
    直接拦会误伤。
    """
    if is_phishing and severity >= 3:
        by_class = "block"
    elif is_phishing:
        by_class = "quarantine"
    elif is_spam:
        by_class = "review"
    else:
        by_class = "deliver"
    floor = ACTION_FLOOR.get(severity, "deliver")
    return floor if ACTION_RANK[floor] > ACTION_RANK[by_class] else by_class


def link_text_for(tags: TagSet) -> str:
    """输入：一组特征。输出：链接文字题的答案（`no_link` / `matches` / `mismatched`）。

    预期行为：`no_link` 只对「本来就没有链接」的样本成立，而那种样本**不会出这道题**
    （见 `questions.families_for`），所以它永远不是数据里的答案，只是对样本的完整描述。
    """
    target, anchor = split_link(tags.link)
    if target == "none":
        return "no_link"
    return "mismatched" if anchor == "mismatch" else "matches"


def sender_identity_for(tags: TagSet) -> str:
    """输入：一组特征。输出：发件人身份题的答案（`matches` / `no_claim` / `forged`）。"""
    if tags.sender == "forged_brand":
        return "forged"
    if tags.sender == "unknown_personal":
        return "no_claim"
    return "matches"


def label_for(tags: TagSet) -> Labels:
    """输入：一组注入特征。输出：全部金标签。

    预期行为：**纯函数、只看特征、不看文本**——文本是后写的，标签先定。

    钓鱼判据是**合取**而不是析取：「要东西」且「有可疑迹象」才算钓鱼。
    这不是为了好看——单一特征就判钓鱼会让 tag 空间里 78% 的样本都是钓鱼，
    训出来的先验直接歪掉；合取同时更贴近真实判断（同事来信要个文档不是钓鱼）。
    """
    credential_theft = tags.ask == "credentials"
    deceptive_link = tags.link_target in DECEPTIVE_TARGETS
    deceptive_anchor = tags.anchor == "mismatch"
    impersonation = tags.sender == "forged_brand"
    wants_something = tags.ask in WANTING_ASKS
    suspicious = deceptive_link or impersonation or tags.lure in PRESSURE_LURES
    is_phishing = wants_something and suspicious
    is_spam = tags.lure in PROMO_LURES or (
        tags.sender == "unknown_personal" and tags.ask != "none"
    )
    severity = min(
        4,
        (2 if credential_theft else 0)
        + (1 if tags.ask == "payment" else 0)
        + (1 if deceptive_link else 0)
        + (1 if deceptive_anchor and deceptive_link else 0)
        + (1 if impersonation else 0)
        + (1 if tags.lure in PRESSURE_LURES else 0)
    )
    return Labels(
        requested=tags.ask,
        pressure=tags.lure,
        link_text=link_text_for(tags),
        sender_identity=sender_identity_for(tags),
        is_phishing=is_phishing,
        is_spam=is_spam,
        severity=severity,
        action=action_for(tags, is_phishing=is_phishing, is_spam=is_spam, severity=severity),
    )


def build_cases(max_full: int, dims: dict[str, tuple[str, ...]] | None = None) -> list[TagSet]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖；LLM workflow 要把它压到几十）、
    可选的维度表覆盖（留出某几个取值时用，例如 OOD 的「值留出」）。
    输出：`TagSet` 列表，顺序稳定。
    预期行为：同一入参必得同一输出；成对覆盖保证任意两维的取值组合至少出现一次；
    被 `FORBIDDEN` 否掉的取值对既不出现、也不算没覆盖。
    """
    rows = cover_dims(
        DIMS if dims is None else dims, max_full=max_full, allow_pair=allow_pair
    )
    return [TagSet(**row) for row in rows]


def compatible(tags: TagSet) -> bool:
    """输入：一组特征。输出：这组特征写得出来吗（没有任何被禁的取值对）。

    预期行为：纯函数、不看文本；`label_flip_axes` 用它剔除无解的邻居组合。
    """
    values = tags.as_dict()
    names = list(DIMS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if not allow_pair(left, values[left], right, values[right]):
                return False
    return True


def label_flip_axes(tags: TagSet) -> list[TagSet]:
    """列出「只改一维」的邻居组合（用来查最小对覆盖率）。

    输入：一组特征。输出：把每个维度换成同维其它取值后的组合列表（不含自己）。
    预期行为：结果顺序与 `DIMS` 的维度顺序、取值顺序一致；无解的邻居（撞上
    `FORBIDDEN` 的那些）直接跳过，不给出写不出来的组合；用于审计
    「标签翻转的兄弟是否也在数据集里」。
    """
    values = tags.as_dict()
    out: list[TagSet] = []
    for name, items in DIMS.items():
        for value in items:
            if value == values[name]:
                continue
            candidate = TagSet(**{**values, name: value})
            if compatible(candidate):
                out.append(candidate)
    return out
