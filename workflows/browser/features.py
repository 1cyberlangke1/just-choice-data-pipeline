"""浏览器 / GUI 操作（browser）的特征层：注入维度 + 元素表 + 判据（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一次「这一屏上下一步该怎么做」的判断由 5 个**注入特征**
描述，**标签先行**——先定这 5 维取值，再由本模块的纯函数算出元素表与全部金标签。

五维分成两拨（这条分法是设计的第一步）：
- **只由那句话承载**（提示词给、元素表不给）：用户这一轮想干什么 `intent`——真实 agent 里
  这件事只能从用户那句话读出来；
- **只由元素表承载**（提示词不给）：目标那一行的状态 `state`、它是不是敏感操作 `sensitive`、
  表里有没有一行同角色的干扰元素 `decoy`、这一屏是什么界面 `screen`——这四件事在真实
  agent 里是界面快照里的字段，不是那句请求里的措辞。

**元素表由代码渲染，用户那句话由后端写**：表里每一行的标签按（角色 × 敏感）从 `LABELS`
里取，所以「敏感」这件事握在代码手里，模型碰不到标签——这是「标签先行」的落地
（范式 §9.2 / §9.16）。`rows_for` 是纯函数，同一组特征必得同一张表，所以元素表、
`target` 题族的选项与全部答案都能从 tags 复算（产物可审计、可逐字节重建）。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、**绝不执行任何东西**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims, row_allowed

# ---------- 维度表 ----------

#: 用户这一轮想干什么。**只由那句话承载**（也是 `intent` 题族的选项顺序）。
INTENTS: Final[tuple[str, ...]] = (
    "fill",  # 填写
    "open",  # 打开
    "switch",  # 切换到
    "submit",  # 点按
)

#: 目标那一行的状态。**只由元素表承载**。
STATES: Final[tuple[str, ...]] = (
    "ready",  # 可用
    "current",  # 已是当前
    "disabled",  # 置灰不可点
    "hidden",  # 不可见
)

#: 目标那一行是不是敏感操作（会动钱 / 删东西 / 发布 / 退订）。**只由元素表承载**。
SENSITIVES: Final[tuple[str, ...]] = (
    "yes",  # 敏感
    "no",  # 不敏感
)

#: 表里有没有一行**同角色**的干扰元素。**只由元素表承载**。
DECOYS: Final[tuple[str, ...]] = (
    "none",  # 没有
    "same_role",  # 有一行同角色、但状态不可用的元素
)

#: 这一屏是什么界面。**只由元素表承载**（决定填充行的角色构成）。
SCREENS: Final[tuple[str, ...]] = (
    "form",  # 表单页
    "list",  # 列表页
    "settings",  # 设置页
)

DIMS: Final[dict[str, tuple[str, ...]]] = {
    "intent": INTENTS,
    "state": STATES,
    "sensitive": SENSITIVES,
    "decoy": DECOYS,
    "screen": SCREENS,
}

#: 意图 → 要操作的角色。判据表里那行「意图表」的代码版本。
ROLE_OF_INTENT: Final[dict[str, str]] = {
    "fill": "field",
    "open": "link",
    "switch": "tab",
    "submit": "button",
}

#: 意图 → 该做的动作。**不变量：`ACTION_OF_INTENT` 的值域去掉 `none` 就是 `ACTIONS`**。
ACTION_OF_INTENT: Final[dict[str, str]] = {
    "fill": "type",
    "open": "open",
    "switch": "switch",
    "submit": "click",
}

#: 角色（元素表里那一列）。
ROLES: Final[tuple[str, ...]] = ("field", "link", "tab", "button")

#: `action` 题族的选项顺序：四个动作 + 「不操作」。
ACTIONS: Final[tuple[str, ...]] = ("type", "open", "switch", "click", "none")

#: 敏感清单。**题面里原样印出来**，标签里出现其中任一个词就算敏感操作。
SENSITIVE_WORDS: Final[tuple[str, ...]] = (
    "支付",
    "删除",
    "发布",
    "退订",
    "注销",
    "订单",
    "密码",
    "银行卡",
    "身份证",
)

#: 元素标签词表，按（角色 × 是否敏感）分组。
#: **不变量：同一格里每个标签都命中 `SENSITIVE_WORDS` 当且仅当敏感那一档是 `yes`**，
#: 由 `tests/workflows/browser/test_features.py` 锁住；跨组的字符串两两不同（免得同一张
#: 表里出现两个同名选项）。标签是**界面快照里的字段**（结构化素材），不是模型的发挥；
#: 那句自然语言由后端写。
LABELS: Final[dict[tuple[str, str], tuple[str, ...]]] = {
    ("field", "no"): ("收件地址", "手机号", "用户名", "项目名称", "搜索关键词"),
    ("field", "yes"): ("支付密码", "银行卡号", "身份证号"),
    ("link", "no"): ("帮助中心", "使用说明", "历史记录", "概览视图", "操作手册"),
    ("link", "yes"): ("退订页面", "注销账号", "删除记录"),
    ("tab", "no"): ("概览", "明细", "统计", "草稿"),
    # 「退订设置」而不是「退订」：`("button", "yes")` 里已经有一个「退订」，
    # 而本表的**不变量**是跨组字符串两两不同（免得同一张表里出现两个同名选项）。
    ("tab", "yes"): ("退订设置", "注销", "删除归档"),
    ("button", "no"): ("刷新", "保存草稿", "下一步", "返回"),
    ("button", "yes"): ("提交订单", "删除", "发布", "支付", "退订"),
}

#: 每种屏型给的填充行角色顺序。取的时候会**剔掉目标行那一角色**——填充行绝不能跟目标行
#: 抢「该操作哪个元素」的答案。
FILLER_ROLES: Final[dict[str, tuple[str, ...]]] = {
    "form": ("field", "link", "tab", "button"),
    "list": ("link", "tab", "button", "field"),
    "settings": ("button", "field", "tab", "link"),
}

#: 填充行的状态池：按 `salt` 轮着取。**只有「可用」以外的状态也能当填充**，
#: 这样一屏里不会到处都是可点的东西（真实界面就是这样）。
FILLER_STATES: Final[tuple[str, ...]] = ("ready", "ready", "current", "disabled", "hidden")

#: 填充行最多几行（加上目标行与干扰行，一屏 4~6 行）。
FILLER_LIMIT: Final[int] = 3

#: 「下一步该操作哪个元素」在没有可操作元素时的答案（题面里的选项名保持 ASCII）。
NONE_TARGET: Final[str] = "none"

#: 风险表：行是意图、列是敏感与否，值 0~3。判据就这一张表，没有别的算法。
RISK_TABLE: Final[dict[tuple[str, str], int]] = {
    ("fill", "no"): 1,
    ("fill", "yes"): 2,
    ("open", "no"): 0,
    ("open", "yes"): 2,
    ("switch", "no"): 0,
    ("switch", "yes"): 1,
    ("submit", "no"): 1,
    ("submit", "yes"): 3,
}

#: 风险档的档数（0 ~ 3）。
RISK_LEVELS: Final[int] = 4

#: 天生不能同时出现的取值对。本 workflow **一对都没有**：五维是「意图 × 界面快照」，
#: 任意组合都写得出来（没有哪一格是判据自相矛盾的）。
FORBIDDEN: Final[frozenset[tuple[str, str, str, str]]] = frozenset()


@dataclass(frozen=True)
class Row:
    """元素表里的一行。

    字段：`label` 元素名（也是 `target` 题族的选项名）；`role` 角色；`state` 状态。
    """

    label: str
    role: str
    state: str


@dataclass(frozen=True)
class Labels:
    """一条样本的全部金标签。字段名与题族名一一对应。"""

    intent: str
    target: str
    action: str
    done: bool
    blocked: bool
    confirm: bool
    risk: int


@dataclass(frozen=True)
class TagSet:
    """一组注入特征。字段名与 `DIMS` 的键一一对应。"""

    intent: str
    state: str
    sensitive: str
    decoy: str
    screen: str

    def as_dict(self) -> dict[str, str]:
        """输入：无。输出：`{维度名: 取值}`（顺序按 `DIMS`）。"""
        return {name: getattr(self, name) for name in DIMS}

    def label(self) -> Labels:
        """输入：无。输出：这组特征对应的全部金标签。"""
        return label_for(self)


def allow_pair(left: str, left_value: str, right: str, right_value: str) -> bool:
    """输入：两个维度名与各自取值。输出：这一对写得出来吗。

    预期行为：与顺序无关——`(a, x, b, y)` 与 `(b, y, a, x)` 都查同一张禁用表。
    本 workflow 的禁用表是空的，所以恒为真；留着这个函数是为了与其它 workflow 同形。
    """
    if left > right:
        left, left_value, right, right_value = right, right_value, left, left_value
    return (left, left_value, right, right_value) not in FORBIDDEN


def is_sensitive(label: str) -> bool:
    """输入：一个元素标签。输出：它是不是敏感操作。

    预期行为：标签里出现 `SENSITIVE_WORDS` 里任一个词就算敏感——判据与题面里印的那份
    清单是同一份常量，不靠人抄。
    """
    return any(word in label for word in SENSITIVE_WORDS)


def role_for_intent(intent: str) -> str:
    """输入：意图。输出：要操作的角色。预期行为：未知意图直接抛 KeyError。"""
    return ROLE_OF_INTENT[intent]


def action_for_intent(intent: str) -> str:
    """输入：意图。输出：该做的动作。预期行为：未知意图直接抛 KeyError。"""
    return ACTION_OF_INTENT[intent]


def _salt(tags: TagSet) -> int:
    """输入：一组特征。输出：一个稳定的整数（用来在词表里挑标签）。

    预期行为：只看 tag 取值、不看 rng——同一组特征必得同一张元素表，元素表与答案才能
    从 tags 复算（产物可审计、可逐字节重建）。
    """
    total = 0
    for name, values in DIMS.items():
        total = total * 8 + values.index(getattr(tags, name))
    return total


def pick_label(role: str, sensitive: str, salt: int) -> str:
    """输入：角色、是否敏感、一个整数。输出：词表里挑出来的标签。

    预期行为：同一入参必得同一标签；`salt` 差 1 时必得**不同**标签（干扰行与目标行
    不能同名，否则选项名重复会被契约拒掉）。
    """
    pool = LABELS[(role, sensitive)]
    return pool[salt % len(pool)]


def filler_rows(screen: str, target_role: str, salt: int) -> list[Row]:
    """输入：屏型、目标行的角色、一个整数。输出：填充行。

    预期行为：角色顺序按 `FILLER_ROLES[screen]`，**剔掉目标行那一角色**，最多
    `FILLER_LIMIT` 行；状态按 `FILLER_STATES` 从 `salt` 处轮取；标签一律取**不敏感**那一格
    ——填充行不该把「敏感」这个信号搅浑（敏感只有目标行与干扰行可能带）。
    """
    roles = [role for role in FILLER_ROLES[screen] if role != target_role]
    out: list[Row] = []
    for index, role in enumerate(roles[:FILLER_LIMIT]):
        state = FILLER_STATES[(salt + index) % len(FILLER_STATES)]
        out.append(Row(pick_label(role, "no", salt + 2 + index), role, state))
    return out


def logical_rows(tags: TagSet) -> list[Row]:
    """输入：一组特征。输出：这一屏的元素表，**目标行在第一行**（顺序稳定）。

    预期行为：第一行是目标行（角色由意图决定、状态就是注入的 `state`）；`decoy=same_role`
    时第二行是**同角色的干扰行**，它的状态一定不可用（注入的状态是 `disabled` 就写
    `hidden`，否则写 `disabled`）；其余是填充行。所以「角色对得上意图」的行最多两行，
    其中只有目标行可能可用——答案唯一。
    """
    salt = _salt(tags)
    role = role_for_intent(tags.intent)
    rows = [Row(pick_label(role, tags.sensitive, salt), role, tags.state)]
    if tags.decoy == "same_role":
        decoy_state = "hidden" if tags.state == "disabled" else "disabled"
        rows.append(Row(pick_label(role, tags.sensitive, salt + 1), role, decoy_state))
    return rows + filler_rows(tags.screen, role, salt)


def rows_for(tags: TagSet) -> list[Row]:
    """输入：一组特征。输出：这一屏的元素表，**按显示顺序**（顺序稳定）。

    预期行为：把 `logical_rows` 按 `salt` 旋转一下再显示——目标行老钉在第一行，会让
    「用户指的是第几行」这种反解问题退化成常数，一屏元素表的读法也就不真实了。
    旋转量只看 tags，所以同一组特征必得同一张表（可复算、可逐字节重建）。
    """
    rows = logical_rows(tags)
    offset = _salt(tags) % len(rows)
    return rows[offset:] + rows[:offset]


def wanted_row(tags: TagSet) -> Row:
    """输入：一组特征。输出：用户想操作的那一行（目标行，不管它现在什么状态）。"""
    return logical_rows(tags)[0]


def wanted_index(tags: TagSet) -> int:
    """输入：一组特征。输出：目标行在**显示顺序**里的位置（从 1 数起）。

    预期行为：给反解判别器用——它拿到的表是显示顺序，答案就是这个 1 起的行号。
    """
    return rows_for(tags).index(wanted_row(tags)) + 1


def wanted_label(tags: TagSet) -> str:
    """输入：一组特征。输出：用户想操作的那个元素的标签。

    预期行为：**与 `Labels.target` 不是一回事**——这一档答的是「用户指的是谁」（那句
    请求里必须原样出现它），`Labels.target` 答的是「下一步该操作谁」（不可用 / 已完成时
    是 `NONE_TARGET`）。反解判别器比的是前者，题族问的是后者。
    """
    return wanted_row(tags).label


def target_row(tags: TagSet) -> Row | None:
    """输入：一组特征。输出：下一步该操作的那一行（没有就 None）。

    预期行为：只在**目标行状态是「可用」**时给出那一行——已完成（`current`）或被拦住
    （`disabled` / `hidden`）时没有可操作的元素。
    """
    if tags.state != "ready":
        return None
    return wanted_row(tags)


def label_for(tags: TagSet) -> Labels:
    """输入：一组特征。输出：全部金标签。

    预期行为：**纯函数、只看特征、不看文本**——文本是后写的，标签先定。三档门控
    互斥且穷尽：`done`（已是当前）/ `blocked`（置灰或不可见）/ `confirm`（可用且敏感），
    三个都为假时按意图表直接操作。

    全空间（4×4×2×2×3 = 192 组）上的分布实测：`intent` 四档均匀；`done` / `blocked` 各自
    占四分之一；`confirm` 占八分之一（要同时满足「可用」与「敏感」）；`risk` 0~3 四档都
    到得了（`open`/`switch` 不敏感是 0 档，`submit` 敏感是 3 档）。要的是**每一档都到得了**，
    不是每档一样多。
    """
    wanted = wanted_row(tags)
    ready = tags.state == "ready"
    return Labels(
        intent=tags.intent,
        target=wanted.label if ready else NONE_TARGET,
        action=action_for_intent(tags.intent) if ready else "none",
        done=tags.state == "current",
        blocked=tags.state in {"disabled", "hidden"},
        confirm=ready and is_sensitive(wanted.label),
        risk=RISK_TABLE[(tags.intent, tags.sensitive)],
    )


def build_cases(max_full: int, dims: dict[str, tuple[str, ...]] | None = None) -> list[TagSet]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖；LLM workflow 要把它压到几十）、
    可选的维度表覆盖（留出某几个取值时用，例如 OOD 的「值留出」）。
    输出：`TagSet` 列表，顺序稳定。
    预期行为：同一入参必得同一输出；成对覆盖保证任意两维的取值组合至少出现一次；
    被 `FORBIDDEN` 否掉的取值对既不出现、也不算没覆盖（本 workflow 没有禁用对）。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full, allow_pair=allow_pair)
    return [TagSet(**row) for row in rows]


def compatible(tags: TagSet) -> bool:
    """输入：一组特征。输出：这组特征写得出来吗（没有任何被禁的取值对）。"""
    return row_allowed(list(DIMS), tags.as_dict(), allow_pair)


#: choice 题族里**选项由文案固定**的那几族（`target` 的选项是每屏元素表，不在这里）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "intent": INTENTS,
    "action": ACTIONS,
}
