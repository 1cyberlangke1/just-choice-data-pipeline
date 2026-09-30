"""打分与评分（rubric）的特征层：注入维度 + 评分单 / 档位表 / 处置表（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一次「这份材料该打几分」的判断由 5 个**注入特征**描述，
**标签先行**——先定这 5 维取值，再由本模块的纯函数算出全部金标签。

五维分成两拨：

- **材料正文承载**（读者只能从那几行材料里看出来）：`placeholders` 材料里有没有留空占位、
  `citations` 材料里有没有出处标记。这两维交给 `judges.TraceJudge` 反解校验——答不回来
  就说明那段材料与标签脱节了；
- **材料清单承载**（读者从「材料清单」那一行字段里读）：`length` 篇幅、`form` 形式、
  `data` 有没有量化数据。这三维由 `render.check_draft` 在渲染期机械校验、由
  `judges.HygieneJudge` 再查一遍、由 `audit.py` 拿产物交叉校验。

评分单的五条判据**权重与要求是固定的**，**观测那一格由上面五维查表得出**——所以整张评分单
都能由注入特征推出来，读者不用猜：

- 达标 = 观测 ≥ 要求；
- 总分 = 达标那几条的权重之和；
- 档位 = 总分落在哪一段（低 / 中 / 高）；
- 处置 = 「档位 × 有没有踩红线」查表（通过 / 返修 / 退回）；
- 最严重的一条 = 没达标的判据里权重最大的那条，**并列或全达标时没有唯一答案**（不出题）。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、**绝不执行任何代码**。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims, row_allowed

# ---------- 维度表 ----------

#: 材料里有没有留空占位（`TODO` / `待补` 这类）。**只由材料正文承载**。
PLACEHOLDERS: Final[tuple[str, ...]] = ("no", "yes")

#: 材料里有没有出处标记（`[1]` 这种）。**只由材料正文承载**。
CITATIONS: Final[tuple[str, ...]] = ("no", "yes")

#: 材料篇幅。**只由「材料清单」那一行承载**。
LENGTHS: Final[tuple[str, ...]] = ("short", "medium", "long")

#: 材料形式：整段行文 / 分条列出。**只由「材料清单」那一行承载**。
FORMS: Final[tuple[str, ...]] = ("prose", "bulleted")

#: 材料里有没有量化数据。**只由「材料清单」那一行承载**。
DATAS: Final[tuple[str, ...]] = ("none", "numbers")

DIMS: Final[dict[str, tuple[str, ...]]] = {
    "placeholders": PLACEHOLDERS,
    "citations": CITATIONS,
    "length": LENGTHS,
    "form": FORMS,
    "data": DATAS,
}

#: 观测**只能从材料正文里数出来**的那几维：评分单上那两行不印数字，印 `meta.count_mark`。
#: 其余三维的观测由代码按「材料清单」那一行算好印出来——读者不读材料也能填那三行，
#: 但这两行填不出来，所以材料不是装饰（`render.check_draft` 与 `audit.py` 共用这份口径）。
BODY_DIMS: Final[frozenset[str]] = frozenset({"placeholders", "citations"})

# ---------- 评分单 ----------

#: 五条判据，顺序固定（就是评分单的行序）。
CRITERIA: Final[tuple[str, ...]] = (
    "complete",
    "grounded",
    "concise",
    "formatted",
    "quantified",
)

#: 每条判据的权重（三档：1 / 2 / 3）。
WEIGHT_OF: Final[dict[str, int]] = {
    "complete": 3,
    "grounded": 3,
    "concise": 2,
    "formatted": 2,
    "quantified": 1,
}

#: 每条判据的要求（观测不低于它才算达标）。
NEED_OF: Final[dict[str, int]] = {
    "complete": 3,
    "grounded": 2,
    "concise": 2,
    "formatted": 2,
    "quantified": 2,
}

#: 每条判据的观测由哪一维查表得来。
OBS_DIM: Final[dict[str, str]] = {
    "complete": "placeholders",
    "grounded": "citations",
    "concise": "length",
    "formatted": "form",
    "quantified": "data",
}

#: 观测表：观测 = `OBS_TABLE[维度][那一维的取值]`，取值 0~4。
#:
#: 刻意让 `length=medium` 落在**观测 == 要求**（2 == 2）这一格上：它是 OOD 的留出取值
#: （`workflow.OOD_LENGTHS`），考的就是「正好压线算不算达标」这条边界。
#: 另外「完整」那条的要求是 3、观测是 4 / 0，所以它也是压线口径（4 ≥ 3）。
OBS_TABLE: Final[dict[str, dict[str, int]]] = {
    "placeholders": {"no": 4, "yes": 0},
    "citations": {"no": 1, "yes": 4},
    "length": {"short": 4, "medium": 2, "long": 0},
    "form": {"prose": 1, "bulleted": 4},
    "data": {"none": 0, "numbers": 3},
}

#: 档位（低 / 中 / 高）。总分范围 0~11（五条判据权重之和）。
BANDS: Final[tuple[str, ...]] = ("low", "mid", "high")

#: 档位门槛：(下界, 上界)，闭区间。渲染时写成两格数字，读者直接找区间。
BAND_LINES: Final[dict[str, tuple[int, int]]] = {
    "low": (0, 5),
    "mid": (6, 7),
    "high": (8, 11),
}

#: 踩红线的那条判据：`grounded`（没给出处的材料不可信）。
REDLINE: Final[str] = "grounded"

#: 红线两态，顺序就是处置表的列序。
REDLINE_STATES: Final[tuple[str, ...]] = ("clean", "redline")

#: 三种处置。
GATES: Final[tuple[str, ...]] = ("accept", "revise", "reject")

#: 处置表：(档位, 红线态) → 处置。**查表，不是算出来的**——缺格直接抛 KeyError。
GATE_TABLE: Final[dict[tuple[str, str], str]] = {
    ("high", "clean"): "accept",
    ("mid", "clean"): "accept",
    ("low", "clean"): "revise",
    ("high", "redline"): "revise",
    ("mid", "redline"): "reject",
    ("low", "redline"): "reject",
}

# ---------- 正文的机械口径（渲染期 / 判别器 / 审计三处共用这一份）----------

#: 留空占位的写法。刻意收得窄而具体：命中就是「这一处还没写完」。
PLACEHOLDER_MARKS: Final[tuple[str, ...]] = ("TODO", "FIXME", "待补", "待填", "待定")

#: 出处标记：`[1]` / `[12]` 这种方括号编号。
CITATION_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[\d+\]")

#: 量化数据：阿拉伯数字。中文数字（「三条」）不算——口径必须机械可判。
DIGIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9]")

#: 分条的标记：`- ` 开头的行。
BULLET_PREFIX: Final[str] = "- "

#: 句子 / 行的边界：中英的句号 / 问号 / 叹号 / 分号，**外加换行**——材料里换了一行就是
#: 新的一句。英文句点只在**后面不是数字**时算（`3.5`、`v1.2` 里的点不断句）；出处标记
#: `[1]` 本来就不含句点。
SENTENCE_END_PATTERN: Final[re.Pattern[str]] = re.compile(r"[。！？；!?;\n]|\.(?!\d)")

#: 篇幅的**目标**句数区间（提示词里那句「短 3~6 句」就是它）。
#:
#: 为什么按**句**而不是按**行**数：真机第一批 20 条里 16 条被「行数」白拒了——模型把
#: 材料写成**一整段**（中文一段本来就是一行），而「短 3~6 行」这种要求它压根没法满足。
#: 材料本来就是自然语言，句子才是它的自然单位；行数只在「分条」那一维才有意义。
#: 这与范式 §9.19 同一条教训：验收口径要跟着数据真实形态走，别拿一个模型做不到的写法
#: 去白拒好数据。
LENGTH_SENTENCES: Final[dict[str, tuple[int, int]]] = {
    "short": (3, 6),
    "medium": (6, 12),
    "long": (10, 18),
}

#: 验收时上界的宽容倍数：`LENGTH_SENTENCES` 既是给模型的写作指引也是题面口径，但**材料
#: 清单那一行才是权威**——正文比指引多写几句，读者照样按清单回答篇幅，标签一点没错，
#: 按上界硬拒是白拒好数据（review 真机上「大 6~12 行」被写到 14 行，连撞两次；范式 §9.19）。
LENGTH_SENTENCES_SLACK: Final[int] = 2


def length_bounds(length: str) -> tuple[int, int]:
    """输入：篇幅取值。输出：**验收**用的 (下界, 上界)。

    预期行为：下界照 `LENGTH_SENTENCES` 不放，上界乘 `LENGTH_SENTENCES_SLACK`。渲染期、
    判别器与产物审计都用这一份，免得三处各写各的。
    """
    low, high = LENGTH_SENTENCES[length]
    return low, high * LENGTH_SENTENCES_SLACK


def count_sentences(text: str) -> int:
    """输入：材料正文。输出：句子数。

    预期行为：按 `SENTENCE_END_PATTERN`（终止符或换行）切段，非空段各算一句——**结尾
    没有终止符的最后一段也算一句**（「TODO：补上压测数据」就是一句，不能因为没写句号
    就当成零句；整段中文散文按 。！？ 数，换行写的一行也算一句）。
    """
    return sum(1 for chunk in SENTENCE_END_PATTERN.split(text) if chunk.strip())


def has_placeholder(text: str) -> bool:
    """输入：材料正文。输出：有没有留空占位。"""
    return any(mark in text for mark in PLACEHOLDER_MARKS)


def has_citation(text: str) -> bool:
    """输入：材料正文。输出：有没有出处标记（`[1]` 这种）。"""
    return CITATION_PATTERN.search(text) is not None


def has_number(text: str) -> bool:
    """输入：材料正文。输出：有没有阿拉伯数字。

    预期行为：先抹掉出处标记再找数字——`[1]` 里的那个 1 是出处编号，不是量化数据。
    不抹的话「有出处 × 无数据」这一格永远写不出来，两个维度会互相打架。
    """
    return DIGIT_PATTERN.search(CITATION_PATTERN.sub("", text)) is not None


def bullet_lines(lines: Sequence[str]) -> list[str]:
    """输入：材料正文的每一行。输出：分条行（`- ` 开头的那些，行首空白不算）。"""
    return [line for line in lines if line.lstrip().startswith(BULLET_PREFIX)]


# ---------- 由特征推出全部金标签 ----------


@dataclass(frozen=True)
class Score:
    """一份评分单算出来的结论（不含两维注入特征本身）。

    字段：`met` 逐条达标；`total` 总分；`band` 档位；`gate` 处置；`top` 最严重的那条。

    为什么要单独有这个类：**算分这一套只有一个实现**——构造侧（`label_for`）与审计侧
    （`audit.py` 拿产物里那张评分单重算）都走 `score_from_obs`，所以「表写错」与「渲染写歪」
    会变成同一类错误，不会两边各算各的还互相印证。
    """

    met: tuple[bool, ...]
    total: int
    band: str
    gate: str
    top: str | None

    def met_at(self, index: int) -> bool:
        """输入：判据序号（从 1 开始）。输出：这条判据达标吗。"""
        return self.met[index - 1]


@dataclass(frozen=True)
class Labels:
    """一条样本的全部金标签。

    字段：`placeholders` / `citations` 是两维注入特征本身（反解判别器比对的就是它俩）；
    其余五个字段就是 `Score` 的五个字段（逐条达标 / 总分 / 档位 / 处置 / 最严重的那条）。
    """

    placeholders: str
    citations: str
    met: tuple[bool, ...]
    total: int
    band: str
    gate: str
    top: str | None

    def met_at(self, index: int) -> bool:
        """输入：判据序号（从 1 开始）。输出：这条判据达标吗。"""
        return self.met[index - 1]


@dataclass(frozen=True)
class TagSet:
    """一组注入特征。字段名与 `DIMS` 的键一一对应。"""

    placeholders: str
    citations: str
    length: str
    form: str
    data: str

    def as_dict(self) -> dict[str, str]:
        """输入：无。输出：`{维度名: 取值}`（顺序按 `DIMS`）。"""
        return {name: getattr(self, name) for name in DIMS}

    def label(self) -> Labels:
        """输入：无。输出：这组特征对应的全部金标签。"""
        return label_for(self)


#: 天生不能同时出现的取值对。本 workflow **一个都没有**：五个维度说的是材料的五件互不
#: 相干的事（有没有留空 / 有没有出处 / 多长 / 分不分条 / 有没有数字），任意组合都写得出来。
#: 留着这张表是为了与别的 workflow 同构——真机批次若暴露某个格子没有唯一答案，按同样口径
#: 补进来（范式 §9.15 第二条教训：能禁就禁，不能禁才改提示词）。
FORBIDDEN: Final[frozenset[tuple[str, str, str, str]]] = frozenset()

#: 维度名 → 声明下标。`allow_pair` 用它把取值对归一化到同一个顺序上。
_ORDER: Final[dict[str, int]] = {name: index for index, name in enumerate(DIMS)}


def allow_pair(left: str, left_value: str, right: str, right_value: str) -> bool:
    """输入：两个维度名与各自取值。输出：这一对写得出来吗。

    预期行为：与调用顺序无关——按 `DIMS` 的声明顺序归一化后再查禁用表，
    所以 `(a, x, b, y)` 与 `(b, y, a, x)` 落到同一个四元组上。
    """
    if _ORDER[left] > _ORDER[right]:
        left, left_value, right, right_value = right, right_value, left, left_value
    return (left, left_value, right, right_value) not in FORBIDDEN


def obs_for(criterion: str, tags: TagSet) -> int:
    """输入：判据名、一组注入特征。输出：这条判据的观测（0~4）。"""
    dim = OBS_DIM[criterion]
    return OBS_TABLE[dim][getattr(tags, dim)]


def obs_map(tags: TagSet) -> dict[str, int]:
    """输入：一组注入特征。输出：`{判据名: 观测}`（顺序按 `CRITERIA`）。"""
    return {name: obs_for(name, tags) for name in CRITERIA}


def band_of_total(total: int) -> str:
    """输入：总分。输出：档位。

    预期行为：按 `BAND_LINES` 找总分落在哪一段；一段都不落在里面说明三档没盖满 0~11，
    那是表写错了，直接抛 KeyError（静默兜底会把「表漏了」伪装成「算出来了」）。
    """
    for band in BANDS:
        low, high = BAND_LINES[band]
        if low <= total <= high:
            return band
    raise KeyError(f"总分 {total} 不落在任何档位区间里——`BAND_LINES` 没盖满")


def top_of_met(met: Sequence[bool]) -> str | None:
    """输入：逐条达标（顺序按 `CRITERIA`）。输出：最严重的那条判据（并列或全达标时 None）。

    预期行为：只看没达标的判据，取权重最大的那条；**权重并列时这道题没有唯一解**，
    返回 None——`questions.families_for` 据此不出这一族，不硬造一个「随便挑一条」的答案。
    """
    unmet = [name for name, ok in zip(CRITERIA, met, strict=True) if not ok]
    if not unmet:
        return None
    best = max(WEIGHT_OF[name] for name in unmet)
    winners = [name for name in unmet if WEIGHT_OF[name] == best]
    return winners[0] if len(winners) == 1 else None


def score_from_obs(obs: Mapping[str, int]) -> Score:
    """输入：`{判据名: 观测}`。输出：这份评分单的结论。

    预期行为：五步全是纯函数——达标（观测 ≥ 要求）、总分（达标那几条的权重之和）、
    档位（查区间）、红线（红线那条没达标）、处置（查二维表）、最严重的一条（唯一才给）。
    判据名认不出或缺格直接抛 KeyError：那是产物串版本了，不能悄悄当成「算出来了」。
    """
    met = tuple(obs[name] >= NEED_OF[name] for name in CRITERIA)
    total = sum(WEIGHT_OF[name] for name, ok in zip(CRITERIA, met, strict=True) if ok)
    band = band_of_total(total)
    state = "redline" if not met[CRITERIA.index(REDLINE)] else "clean"
    return Score(
        met=met,
        total=total,
        band=band,
        gate=GATE_TABLE[(band, state)],
        top=top_of_met(met),
    )


def met_of(criterion: str, tags: TagSet) -> bool:
    """输入：判据名、一组注入特征。输出：这条判据达标吗（观测 ≥ 要求）。"""
    return obs_for(criterion, tags) >= NEED_OF[criterion]


def total_for(tags: TagSet) -> int:
    """输入：一组注入特征。输出：总分（达标那几条的权重之和）。"""
    return score_from_obs(obs_map(tags)).total


def band_for(tags: TagSet) -> str:
    """输入：一组注入特征。输出：档位。"""
    return score_from_obs(obs_map(tags)).band


def redline_hit(tags: TagSet) -> bool:
    """输入：一组注入特征。输出：有没有踩红线（红线那条判据没达标）。"""
    return not met_of(REDLINE, tags)


def redline_state(tags: TagSet) -> str:
    """输入：一组注入特征。输出：红线态（`clean` / `redline`），就是处置表的列名。"""
    return "redline" if redline_hit(tags) else "clean"


def gate_for(tags: TagSet) -> str:
    """输入：一组注入特征。输出：处置（通过 / 返修 / 退回）。"""
    return score_from_obs(obs_map(tags)).gate


def top_for(tags: TagSet) -> str | None:
    """输入：一组注入特征。输出：最严重的那条判据（并列或全达标时 None）。"""
    return score_from_obs(obs_map(tags)).top


def label_for(tags: TagSet) -> Labels:
    """输入：一组注入特征。输出：全部金标签。

    预期行为：**纯函数、只看特征、不看文本**——文本是后写的，标签先定。

    全空间 2×2×3×2×2 = 48 组（没有禁用对）。要的是**每一档都到得了**，不是每档一样多：
    `band` 三档、`gate` 三种处置、五条判据的达标与不达标、`top` 的唯一与并列都验过
    （`tests/workflows/rubric/test_features.py`）。
    """
    score = score_from_obs(obs_map(tags))
    return Labels(
        placeholders=tags.placeholders,
        citations=tags.citations,
        met=score.met,
        total=score.total,
        band=score.band,
        gate=score.gate,
        top=score.top,
    )


def build_cases(max_full: int, dims: dict[str, tuple[str, ...]] | None = None) -> list[TagSet]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某几个取值时用）。
    输出：`TagSet` 列表，顺序稳定。
    预期行为：同一入参必得同一输出；成对覆盖保证任意两维的取值组合至少出现一次；
    被 `FORBIDDEN` 否掉的取值对既不出现、也不算没覆盖。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full, allow_pair=allow_pair)
    return [TagSet(**row) for row in rows]


def compatible(tags: TagSet) -> bool:
    """输入：一组特征。输出：这组特征写得出来吗（没有任何被禁的取值对）。"""
    return row_allowed(list(DIMS), tags.as_dict(), allow_pair)


#: choice 题族允许出现的选项名（题面只收本题允许的，别的取值留给反解提示词用）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "gate": GATES,
    "top": CRITERIA,
}
