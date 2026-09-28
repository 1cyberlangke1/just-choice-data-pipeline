"""完成声明核验（verify）的特征层：注入维度 + 标签规则（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」与 3.3「tag 组合」：
- 一条样本由 5 个**注入特征**描述（声称做完的事 / 口气 / 检查结果 / 覆盖程度 / 是否落盘）；
- **标签先行**：先定这 5 维取值，再由本模块的纯函数算出全部金标签；
- 「自报事实」与「验收门槛」两行由**代码**渲染进 state —— 所以「该判成什么」是表上
  一次可复核的推导，不是模型的自由发挥（模型只写那句声明，见 `render.py`）。

六个题族里只有两个由文本承载（`kind` 声称做完了什么、`tone` 口气），它们交给
`judges.TraceJudge` 反解校验；另外四个（`verdict` / `gap` / `confidence` / `committed`）
全部能从「自报事实」那一行重算出来，由 `audit.py` 逐条核对。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims

#: 声称做完的事。**每一档都必须是一件「活儿」，不能是「事实行管的那三件事」**——
#: 第一版这里写的是 `tests_pass`（说测试跑通了），结果提示词自己跟自己打架：一边要它说
#: 测试跑通，一边要它别陈述检查结果，模型每次都照前一条写，`tone` 反解当场被带偏
#: （实跑 10 条里拒了 2 条，全是这一条）。改成「补了用例」这种活儿就没了矛盾。
CLAIM_KINDS: Final[tuple[str, ...]] = (
    "tests_added",  # 说给某块代码补了测试用例
    "bug_fixed",  # 说某个缺陷修好了
    "perf_improved",  # 说性能提上去了
    "refactor_done",  # 说重构收尾了
    "api_added",  # 说新接口加上了
)

#: 那句声明的口气。这一维**只由文本承载**，所以它既是题族的答案，也要被反解校验。
TONES: Final[tuple[str, ...]] = (
    "plain",  # 平铺直叙：只说做了什么
    "hedged",  # 留有余地：大概、应该、看上去
    "absolute",  # 把话说满：肯定、绝对、不可能有问题
)

#: 自报的检查结果。
CHECKS: Final[tuple[str, ...]] = ("all_pass", "partial_pass", "all_fail", "not_run")

#: 自报的覆盖程度。
COVERAGES: Final[tuple[str, ...]] = ("covered", "partial", "uncovered")

#: 改动是否已经落盘。
ARTIFACTS: Final[tuple[str, ...]] = ("committed", "uncommitted")

DIMS: Final[dict[str, tuple[str, ...]]] = {
    "claim_kind": CLAIM_KINDS,
    "tone": TONES,
    "checks": CHECKS,
    "coverage": COVERAGES,
    "artifact": ARTIFACTS,
}

#: 天生不能同时出现的取值对：**没跑过检查，就不可能声称覆盖到了**。
#: 不把这两对禁掉，全空间里会有一批自相矛盾的事实表，而「判成什么」在那种表上没有意义。
FORBIDDEN: Final[frozenset[frozenset[tuple[str, str]]]] = frozenset(
    {
        frozenset({("checks", "not_run"), ("coverage", "covered")}),
        frozenset({("checks", "not_run"), ("coverage", "partial")}),
    }
)

#: 核验结论的四种取值（也是 `verdict` 题族的选项）。
VERDICTS: Final[tuple[str, ...]] = ("verified", "partial", "unverified", "refuted")

#: 「还差什么」的五种取值（也是 `gap` 题族的选项），顺序与判据里的关卡顺序一致。
GAPS: Final[tuple[str, ...]] = (
    "run_the_checks",
    "fix_the_failing_checks",
    "tighten_scope",
    "commit_the_change",
    "nothing_missing",
)

#: 三条验收门槛的个数（`confidence` 是 0~3 的评分）。
GATE_COUNT: Final[int] = 3


def allow_pair(dim_a: str, value_a: str, dim_b: str, value_b: str) -> bool:
    """输入：两个维度的取值。输出：这一对能否同时出现。

    预期行为：只查 `FORBIDDEN`，与两个维度谁先谁后无关——覆盖器按声明顺序传参，
    这里不做顺序假设。给 `jc.cover` 当取值对约束用。
    """
    return frozenset({(dim_a, value_a), (dim_b, value_b)}) not in FORBIDDEN


@dataclass(frozen=True)
class Facts:
    """「自报事实」那一行的三格。

    这是标签计算与**事后审计**共用的中间形态：渲染器把它写进 state，审计再从 state 里
    解析出同构的 `Facts`，两边各算一次结论——不一致就说明文本与标签脱节了。
    """

    checks: str
    coverage: str
    artifact: str


def facts_for(tags: TagSet) -> Facts:
    """输入：一组注入特征。输出：自报事实。**不看文本**。"""
    return Facts(checks=tags.checks, coverage=tags.coverage, artifact=tags.artifact)


def gates_from(facts: Facts) -> tuple[bool, bool, bool]:
    """输入：自报事实。输出：三条门槛各自过没过（检查全绿 / 覆盖完整 / 已落盘）。

    预期行为：门槛是**固定**的三条——口径写在题面里，不随样本变；变的只是事实过不过。
    """
    return (
        facts.checks == "all_pass",
        facts.coverage == "covered",
        facts.artifact == "committed",
    )


def verdict_from(facts: Facts) -> str:
    """输入：自报事实。输出：核验结论。

    预期行为：
    - 检查全挂 → `refuted`（证据直接否掉声明）；
    - 检查没跑、或覆盖留了空白 → `unverified`（既没否掉也没撑住）；
    - 检查全绿且覆盖完整 → `verified`（证据撑得住）；
    - 其余（检查只过了一部分）→ `partial`。
    """
    if facts.checks == "all_fail":
        return "refuted"
    if facts.checks == "not_run" or facts.coverage == "uncovered":
        return "unverified"
    if facts.checks == "all_pass" and facts.coverage == "covered":
        return "verified"
    return "partial"


def gap_from(facts: Facts) -> str:
    """输入：自报事实。输出：**第一个**还没过的关卡对应的动作。

    预期行为：按「检查 → 覆盖 → 落盘」的固定顺序取第一个没过的关卡：
    - 检查没跑 → `run_the_checks`；检查挂了（全挂或只过一部分）→ `fix_the_failing_checks`；
    - 覆盖不完整 → `tighten_scope`（把结论收窄到证据撑得住的范围）；
    - 改动没落盘 → `commit_the_change`；
    - 三条全过 → `nothing_missing`。
    """
    if facts.checks == "not_run":
        return "run_the_checks"
    if facts.checks in ("all_fail", "partial_pass"):
        return "fix_the_failing_checks"
    if facts.coverage != "covered":
        return "tighten_scope"
    if facts.artifact != "committed":
        return "commit_the_change"
    return "nothing_missing"


def confidence_from(facts: Facts) -> int:
    """输入：自报事实。输出：三条门槛过了几条（0~3）。"""
    return sum(gates_from(facts))


def committed_from(facts: Facts) -> bool:
    """输入：自报事实。输出：改动落盘了没有（第三条门槛的直读）。"""
    return facts.artifact == "committed"


@dataclass(frozen=True)
class TagSet:
    """一组注入特征（= PLAN 3.3 的一个 tag 组合）。

    字段：`claim_kind` / `tone` / `checks` / `coverage` / `artifact`。
    """

    claim_kind: str
    tone: str
    checks: str
    coverage: str
    artifact: str

    def as_dict(self) -> dict[str, str]:
        """输入：无。输出：`{维度名: 取值}`，顺序与 `DIMS` 一致。"""
        return {
            "claim_kind": self.claim_kind,
            "tone": self.tone,
            "checks": self.checks,
            "coverage": self.coverage,
            "artifact": self.artifact,
        }

    @property
    def facts(self) -> Facts:
        """输入：无。输出：这组特征对应的自报事实。"""
        return facts_for(self)

    def label(self) -> Labels:
        """输入：无。输出：这一组特征对应的全部金标签。"""
        return label_for(self)


@dataclass(frozen=True)
class Labels:
    """一条样本的全部金标签：字段名就是题族名，`questions.py` 直接按名字取。

    `kind` 与 `tone` 由文本承载（模型写的那句声明必须看得出来是什么事、什么口气，
    由 `judges.TraceJudge` 反解校验）；其余四个都能从 state 里那两行重算出来。
    """

    kind: str
    tone: str
    verdict: str
    gap: str
    confidence: int
    committed: bool


def label_for(tags: TagSet) -> Labels:
    """输入：一组注入特征。输出：全部金标签。

    预期行为：**纯函数、只看特征、不看文本**——文本是后写的，标签先定。

    全空间（5×3×4×3×2 = 360 组，剔除禁用对后 300 组）上的分布实测如下：

    - `verdict`：unverified 30%、refuted 30%、partial 30%、verified 10%；
    - `gap`：fix_the_failing_checks 60%、tighten_scope 20%、run_the_checks 10%、
      commit_the_change 5%、nothing_missing 5%；
    - `confidence`：1 档 45%、0 档 25%、2 档 25%、3 档 5%；
    - `committed`：正好五五开（这一维不与 `FORBIDDEN` 相交）。

    偏斜来自判据本身（「还差什么」天然扎堆在「去修挂掉的检查」上），照实标出不掩盖；
    要的是**每一档都到得了**，不是每档一样多。同分布侧（留出 `api_added` 与 `absolute`）
    每族每档都还在，没有死选项。
    """
    facts = facts_for(tags)
    return Labels(
        kind=tags.claim_kind,
        tone=tags.tone,
        verdict=verdict_from(facts),
        gap=gap_from(facts),
        confidence=confidence_from(facts),
        committed=committed_from(facts),
    )


def build_cases(max_full: int, dims: dict[str, tuple[str, ...]] | None = None) -> list[TagSet]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖；LLM workflow 要把它压到几十）、
    可选的维度表覆盖（留出某几个取值时用，例如 OOD 的「值留出」）。
    输出：`TagSet` 列表，顺序稳定。
    预期行为：同一入参必得同一输出；成对覆盖保证任意两维的取值组合至少出现一次；
    被 `FORBIDDEN` 否掉的取值对既不出现、也不算没覆盖。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full, allow_pair=allow_pair)
    return [TagSet(**row) for row in rows]


def compatible(tags: TagSet) -> bool:
    """输入：一组特征。输出：这组特征写得出来吗（没有任何被禁的取值对）。"""
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
    预期行为：无解的邻居（撞上 `FORBIDDEN` 的那些）直接跳过。
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


#: choice 题族允许出现的选项名（题面只收本题允许的，别的取值留给反解提示词用）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "kind": CLAIM_KINDS,
    "tone": TONES,
    "verdict": VERDICTS,
    "gap": GAPS,
}
