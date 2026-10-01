"""根因定位的题族：把现场计划翻成契约形态的题目。

输入：现场计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——依赖图、判读规则与事件形状表都印在 state 里，题面只需要指明问的是
哪一条，只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `root`（choice，6 档）：这次故障的根因是哪个服务。选项就是六个服务位，答案不退化；
- `hop`（choice，3 档）：**指定的那个出错服务**离根因几跳（0 跳 / 1 跳 / 2 跳以上）。
  只在探针是 `h0` / `h1` / `h2` 的样本上出题；
- `span`（choice，4 档）：这次故障**受影响的服务集合**是哪一组。选项 = 真值 + 三个「只差一两处」
  的邻近集合（`features.CausePlan.distractors`）——差得越少越难分辨，题目才有区分度；
- `isroot`（noul）：**指定的那个服务**是不是根因。只在探针是 `ryes` / `rno` 的样本上出题，
  两种探针各占一半，所以这一族 True / False 按构造就是 50% / 50%。
- `count`（score，6 档）：这次故障**一共影响了几个服务**（含根因）。这是本 workflow 唯一的
  score 题族，也是唯一不问「哪一个服务」的题——它问的是**影响面有多大**（值班先要的那个数）。
  答案 = 出错服务数 − 1 当下标，判据只有一条：`R4` 说的「受影响的服务 = 所有出错的服务」，
  所以读法是「六行日志逐行数，非正常的有几行」。

**为什么 `count` 不算 `span` 的重复**：`span` 要在真值 + 三个邻近集合里挑**是哪一组**（选项里
含只差一两个服务的干扰项），`count` 只要一个数。前者错一个服务就全错，后者错一个服务仍可能
对——两族的**错法**不一样，问的粒度也不一样。至于「哪一跳断了」那类问法（§5 提过），
在本 workflow 的传播模型里都能化归成「是哪个服务」——那正是 `root` / `hop` / `isroot`
已经在问的，单开一族只会是换词，所以不做。

**为什么 `hop` 与 `isroot` 分走不同的探针**：它们问的是同一件事的两个侧面（「隔多远」与
「是不是它」），但答案空间一个是 3 档、一个是布尔。共用同一个探针的话，布尔那一族的 True
只剩三分之一。拆成五个探针、两族各取三个 / 两个之后，两族各自只受自己的**可行性上限**约束：
跳数三档 24 / 20 / 12、布尔两边 24 / 20（上限怎么来的见 `features.PROBES` 那段）。

**为什么 `span` 不做「答案位置固定」的优化**：它的真值永远放在 `c0`，编号由组装器最后一步
打乱（`jc.assemble.choice` 的 `shuffle=True`），所以选项位置不携带信息。
"""

from __future__ import annotations

from typing import Final

from jc.assemble import choice as _choice
from jc.assemble import noul as _noul
from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import SERVICES, CausePlan, hop_class

#: 题族的固定顺序：先问根因，再问跳数、问集合、问是不是它，最后问影响面。
FAMILIES: Final[tuple[str, ...]] = ("root", "hop", "span", "isroot", "count")


def root_text(loc: dict[str, JsonValue], graph: str, service: str) -> str:
    """输入：语言字典、图名、服务。输出：`root` 那一族的选项描述（审计也用它比对）。

    预期行为：`服务编号（角色）`——角色按依赖图算（没人调用=入口、不调用别人=末端、其余=中间），
    所以描述随图变，六个选项两两不同。
    """
    role = features.role_of(graph, service)
    return str(loc["fam"]["root"]["opt"]).format(
        service=service, role=str(loc["role"][role])
    )


def named_service(instructions: str) -> str | None:
    """输入：一道题的题面。输出：题面里点名的那个服务（一个都没点名、或点了多个就给 None）。

    预期行为：`hop` 与 `isroot` 两族的题面模板里恰好有一个服务编号，审计拿它把「这次问的是谁」
    从产物里还原出来——不然那两族只能比个数，比不出对错。
    """
    hits = [service for service in SERVICES if service in instructions]
    return hits[0] if len(hits) == 1 else None


def hop_text(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、跳数档位。输出：`hop` 那一族的选项描述。"""
    return str(loc["fam"]["hop"]["opt"][level])


def span_text(loc: dict[str, JsonValue], services: tuple[str, ...]) -> str:
    """输入：语言字典、服务集合。输出：集合的可读描述（`span` 的选项与审计共用）。

    预期行为：**先按 `SERVICES` 的规范顺序归一**，再拼；空集合用该语言里那句「一个都没受影响」。
    选项描述要**两两不同**（`_choice` 会当场验这一点），而 `choice` 的选项是字典、比对时按值比，
    所以一个集合只能有一种写法——归一放在这里，调用方传什么顺序都得到同一串。
    """
    spec = loc["fam"]["span"]
    order = {service: index for index, service in enumerate(SERVICES)}
    ordered = tuple(sorted(services, key=lambda service: order.get(service, len(SERVICES))))
    if not ordered:
        return str(spec["none"])
    return str(spec["joiner"]).join(ordered)


def _root(loc: dict[str, JsonValue], phrasing: int, plan: CausePlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、现场计划。输出：一道 choice 题（六个服务全出）。"""
    spec = loc["fam"]["root"]
    options = {service: root_text(loc, plan.graph, service) for service in SERVICES}
    return _choice("root", locales.pick(spec["q"], phrasing), options, plan.root)


def _hop(loc: dict[str, JsonValue], phrasing: int, plan: CausePlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、现场计划。输出：一道 choice 题（三档跳数全出）。

    预期行为：题面点名这次问的服务，考生照着依赖图数最少步数即可。
    """
    spec = loc["fam"]["hop"]
    options = {level: hop_text(loc, level) for level in features.HOP_CLASSES}
    assert plan.probed is not None  # 采样阶段已按可行性过滤，这里不可能为空
    answer = hop_class(plan.distances[plan.probed])
    return _choice(
        "hop",
        locales.pick(spec["q"], phrasing).format(service=plan.probed),
        options,
        answer,
    )


def _span(loc: dict[str, JsonValue], phrasing: int, plan: CausePlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、现场计划。输出：一道 choice 题（真值 + 三个邻近干扰）。"""
    spec = loc["fam"]["span"]
    combos = [plan.errored, *plan.distractors]
    options = {f"c{index}": span_text(loc, combo) for index, combo in enumerate(combos)}
    return _choice("span", locales.pick(spec["q"], phrasing), options, "c0")


def _isroot(loc: dict[str, JsonValue], phrasing: int, plan: CausePlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、现场计划。输出：一道 noul 题。

    预期行为：题面点名这次问的服务；判据两句写清「是根因」与「不是根因」分别指什么。
    """
    spec = loc["fam"]["isroot"]
    assert plan.probed is not None
    criteria = {"true": str(spec["criteria_true"]), "false": str(spec["criteria_false"])}
    return _noul(
        "isroot",
        locales.pick(spec["q"], phrasing).format(service=plan.probed),
        criteria,
        plan.probed == plan.root,
    )


def _score(
    family: str, loc: dict[str, JsonValue], instructions: str, answer: int
) -> LabeledQuestion:
    """输入：题族、语言字典、题面、金答案（档位下标）。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `features` 算出来的数一一对应；
    档位有自然顺序（1 个服务 < 2 个服务 < ……），所以不打乱。
    """
    question: dict[str, JsonValue] = {
        "type": "score",
        "instructions": instructions,
        "criteria": [str(level) for level in loc["fam"][family]["levels"]],
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def count_text(loc: dict[str, JsonValue], index: int) -> str:
    """输入：语言字典、答案下标。输出：该档的档位描述（审计也用它比对）。

    预期行为：越界的下标当场报人话——审计拿它比产物，静默给个空串会把「档位抄错」藏起来。
    """
    levels = [str(level) for level in loc["fam"]["count"]["levels"]]
    if not 0 <= index < len(levels):
        raise ValueError(f"count 的下标越界：{index}（共 {len(levels)} 档）")
    return levels[index]


def _count(loc: dict[str, JsonValue], phrasing: int, plan: CausePlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、现场计划。输出：一道 score 题（影响面几档）。"""
    spec = loc["fam"]["count"]
    return _score(
        "count",
        loc,
        locales.pick(spec["q"], phrasing),
        features.count_index(plan.errored),
    )


def build(plan: CausePlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：现场计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：`root` 与 `span` 每条样本都出；`hop` 只在跳数探针上出、`isroot` 只在布尔探针上出。
    choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer` 是布尔。
    """
    out: list[LabeledQuestion] = [_root(loc, phrasing, plan)]
    if plan.probe in ("h0", "h1", "h2"):
        out.append(_hop(loc, phrasing, plan))
    out.append(_span(loc, phrasing, plan))
    if plan.probe in ("ryes", "rno"):
        out.append(_isroot(loc, phrasing, plan))
    out.append(_count(loc, phrasing, plan))
    return out
