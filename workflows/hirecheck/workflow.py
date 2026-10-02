"""招聘初筛门禁 workflow：**特征注入式**（定级规则与档位表由代码写，一屏候选人材料由后端写）。

流程与 `alertrisk` / `sqlreview` / `depcheck` 同构（`plan → sample → render → ask`），区别在「判
什么」：

- **判据表、定级规则、处置档位表、材料抬头与格式规格是结构化事实**，由代码渲染进 state——让模型去
  写只会写歪；
- 后端只写**一屏候选人材料**（真实的姓名、技能组合、学历写法与备注长尾得很，模板拼出来的脱离现实
  分布）；
- **哪几条判据命中由代码按字面算**（`features.PATTERNS` 的四条正则，各占一行），等级再由权重公式
  算出来，所以标签是注入特征经纯函数算出来的（`features.plan_for` + `label_for`），模型碰不到；写
  出来的材料必须兑现这份计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` /
  `FrameJudge` 再各查一遍）。

**为什么这一族的中心题型是 `score`**：初筛定级天生是有序的多档判断（1~5 级），硬塞进 choice 会把
「3 级与 4 级差多少」这个序丢掉。`grade` 用 score 题型、等级表不打乱，答案就是等级下标。

**为什么要这一族**：社区里 `hiresignal` 明写「用 TypeSafe Jev 做简历首轮匹配和面试筛选」，
`Open Medical Jev` 那一簇（先逐项评分、再用门控决定放行还是升级给人工）是同一形状的另一个落地——
机器把一份材料摆出来，**要不要进下一轮**这个判断交给类型化决策。与 `depcheck` / `alertrisk` 不重
复：那两家判的是一个依赖的准入与一条安全警报的定级，本家判的是**一份候选人材料的首轮去留**。

**这一族只写不查**：要的是**文本**——姓名、技能、许可状态与学历都只写在材料里，判据认的也只是字面，
没有任何一步会去查这个人的背景、连招聘系统或背调服务。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按命中组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`channel=campus` 这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("hirecheck")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import hash_bucket, slice_by_bucket, restricted_dims
from jc.jsontypes import JsonValue
from jc.knobs import require_pool
from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import HireCheckCase
from .spec import HireCheckSpec

WORKFLOW_NAME = "hirecheck"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `channel=campus`：它是**只影响渲染、不影响标签**的那一维——等级由四条判据算
# （`channel` 一个字符都不参与），渠道只决定这份材料长成什么样（备注里怎么提材料是怎么来的）。
# 同分布侧一次都见不到它，模型必须真的按字面判，而不是靠「校招来的简历一般都很朴素」这类先验蒙。
#
# 反过来，留出 `flags` 会让 `grade` / `combo` / `act` 三族各少几档（16 组少一半），那些档位在同分布
# 侧就取不到了——均衡闸与审计都会当场报出来。所以留出的是 `channel`。
OOD_CHANNELS: Final[frozenset[str]] = frozenset({features.CHANNEL_CAMPUS})

# 同分布切分的哈希分桶边界（组合留出：桶按整条 case 算，同一条 case 永远同切分）。
#
# 种子是**挑过的**，不是随手写的：五档等级里最低那一档只有一个命中组合（四条判据全不命中），它落在
# 哪个切分，哪个切分才见得到 1 级。训练侧见不到 1 级，`grade` 那一族的最低档就永远学不到——所以种子
# 按「训练侧五档齐全、小批量挑选（10 条）也五档齐全」挑定，`test_pipeline` 里有一条测试把这件事钉
# 死；换种子时它会当场报出来。
SPLIT_SEED: Final[int] = 20261008
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "flags": "命中了哪几条初筛判据",
    "channel": "这份材料从哪条渠道进来",
    "level": "按权重公式算出来的等级",
}


@dataclass(frozen=True)
class HireCheckConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 16×3 = 48 组，512 让枚举器稳稳走**全枚举**；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池；
    - `roles` 岗位方向池（**样本级旋钮**：只改 `skills:` 怎么列与 `note:` 怎么写，不改 case、标签与
      切分）。
    - `years` 年限池（**第二个样本级旋钮**：只改 `years:` 那一行怎么写，同样不改 case、标签与切分）。
      为什么要有它：`years:` 是这一族最后一个没有样本级变量的自由字段，不钉住的话模型会落回自己的
      先验（真机上 `depcheck` 的 `version:` 就是这么塌的，范式 §9.60）。池子要**两侧都有货**：
      `H1` 命中的样本只从「不足三年」那一半取、不命中的只从三年以上那一半取（见 `features.year_pool`）。
    - `names` 姓名池（**第三个样本级旋钮**：只改 `candidate:` 那一行怎么写，同样不改 case、标签与
      切分）。为什么要有它：`candidate:` 是这一族最后一个没有样本级变量的自由字段——真机 `hire-10b`
      的 zh 侧 10 条里「张伟」出现了 9 次（范式 §9.60 / §9.61）。存的是**下标**，池子按语言在
      `locales/*.toml` 的 `prompt.names` 里，所以两张表必须等长（`test_locales` 锁住）。
    """

    split: str = "train"
    cover_limit: int = 512
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1)
    roles: tuple[int, ...] = features.ROLE_IDS
    years: tuple[int, ...] = features.YEAR_HINTS_SHORT + features.YEAR_HINTS_LONG
    names: tuple[int, ...] = features.NAME_IDS

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasing_pool", self.phrasing_pool)
        require_pool("roles", self.roles)
        require_pool("years", self.years)
        require_pool("names", self.names)
        bad = [item for item in self.years if not isinstance(item, int) or item < 0]
        if bad:
            raise ValueError(f"years 里的取值要是非负整数：{bad}")
        # 年限池要**两侧都有货**：`H1` 命中与不命中的样本各取一半（见 `features.year_pool`），哪一半
        # 空着都会在采样期报错。这种错在构造期就能看出来，不用拖到真跑（范式 §9.61）。
        for side, half in (
            ("不足三年", features.YEAR_HINTS_SHORT),
            ("三年以上", features.YEAR_HINTS_LONG),
        ):
            if not any(item in half for item in self.years):
                raise ValueError(
                    f"years 里一个「{side}」的取值都没有（要有 {list(half)} 里的至少一个）："
                    f"{list(self.years)}"
                )
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


def _restricted(**overrides: tuple[JsonValue, ...]) -> dict[str, tuple[JsonValue, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    return restricted_dims(features.DIMS, overrides)


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        channel=tuple(value for value in features.CHANNELS if value not in OOD_CHANNELS)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含 OOD 取值的维度表列表（留出切分用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时只改这
    里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(
            channel=tuple(value for value in features.CHANNELS if value in OOD_CHANNELS)
        )
    ]


def bucket_of(case: HireCheckCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _dims_of(case: HireCheckCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多一个「等级」）。

    预期行为：`as_dict` 的两个维度之外再加 `level`（权重公式算出来的等级）——小批量里最容易被忽略的
    正是它：只按 `flags` / `channel` 铺开的话，10 条会落在「5 级一条都出不来」上，`grade` 那一族的
    答案分布当场偏掉，均衡闸标红。
    """
    return {
        "flags": case.flags,
        "channel": case.channel,
        "level": str(features.level_of(case.tags())),
    }


def _cover_pick(cases: list[HireCheckCase], limit: int) -> list[HireCheckCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：命中组合 → 渠道 → 等级），同分时取原顺序里靠前的；
    名额没用完就按原顺序接着填。

    为什么不是直接截断：`flags` 有 16 个取值、`level` 有 5 个，纯截断的话 10 条的小批量很可能一条
    「5 级」都出不来，`grade` 与 `act` 的某些档位也取不到，均衡闸与审计都会当场报出来。
    """
    names = ("flags", "channel", "level")
    chosen: list[HireCheckCase] = []
    seen: dict[str, set[str]] = {name: set() for name in names}
    remaining = list(cases)
    while remaining and len(chosen) < limit:
        keys = [
            (sum(1 for name in names if _dims_of(case)[name] not in seen[name]), -index)
            for index, case in enumerate(remaining)
        ]
        best = max(range(len(remaining)), key=lambda index: keys[index])
        case = remaining.pop(best)
        chosen.append(case)
        for name in names:
            seen[name].add(_dims_of(case)[name])
    return chosen


def cases_for(config: HireCheckConfig) -> list[HireCheckCase]:
    """输入：采样参数。输出：该切分的命中组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], HireCheckCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()), config.split, bucket_of, SPLIT_BUCKETS
        )
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class HireCheckWorkflow(BackendHolder):
    """招聘初筛门禁：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 一屏候选人材料由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = tuple(SPLIT_BUCKETS) + ("test_ood",)
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> HireCheckConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return HireCheckConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: HireCheckConfig, case: HireCheckCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法编号与岗位方向由 RNG 从池里抽（它们不是 case 维度），判定计划在 `sample` 阶段
        按 case 指纹算出来——与切分、语言、进程都无关。

        岗位方向是**样本级旋钮**（范式 §9.57 / §9.58：写文本的提示词里要有样本级变量，否则同维取值
        下的提示词逐字相同、`note:` 会塌成一句话）。它不进 `stats_dimensions`——那不是数据维度，报
        进均衡表只会把「答案分布」这件事冲淡。年限同理，而且它**按判据分半**：`H1` 命中的 case 只能
        从「不足三年」那一半取，不然渲染期的 `check_draft` 会把「文本兑现不了计划」的样本当场打回
        （范式 §9.61）。姓名是第三个同类旋钮，只改 `candidate:` 那一行怎么写。
        """
        return SplitSpec(
            name=config.split,
            params={
                "case": case,
                "phrasing": rng.choice(config.phrasing_pool),
                "role": rng.choice(config.roles),
                "years": features.pick_years(rng, config.years, case.tags()),
                "name": rng.choice(config.names),
                "seq": seq,
            },
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> HireCheckSpec:
        """输入：生成计划。输出：真值规格。预期行为：类型不对当场报错，不拖到渲染期。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        role = split.params["role"]
        years = split.params["years"]
        name = split.params["name"]
        if (
            not isinstance(case, HireCheckCase)
            or not isinstance(phrasing, int)
            or not isinstance(role, int)
            or not isinstance(years, int)
            or not isinstance(name, int)
        ):
            raise TypeError(
                f"生成计划里的 case / phrasing / role / years / name 类型不对：{split.params!r}"
            )
        return HireCheckSpec(
            case=case,
            phrasing=phrasing,
            role=role,
            years=years,
            name=name,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: HireCheckSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那屏候选人材料，先按注入特征校验（不合格抛错，管线据此重渲染），再与代码
        渲染的判据表 / 定级规则 / 档位表 / 抬头拼起来；后端没注入就报错——`needs` 非空时管线一定会
        注入，这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("hirecheck workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        packet = draft.data.get("packet")
        if not isinstance(packet, str):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.ContentDraft(packet=packet)
        render.check_draft(parsed, spec.case, loc)
        return render.to_state(parsed, loc)

    def ask(self, spec: HireCheckSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（候选人备注）。

        预期行为：只报**一段自然语言正文**——`note:` 那一行的备注。键名、姓名、年限、技能、许可状态
        与学历都是格式（机器写的），把它们报上去只会把汉字占比稀释掉，把合格的中文产物误报成「语言对
        不上」（范式 §9.50）。备注取不到时报空表——那把尺子对空表不发表意见。
        """
        parsed = render.parse_state(sample.get("state"))
        note = render.note_text(parsed.text or "")
        return {"候选人备注": note} if note else {}



@register(WORKFLOW_NAME)
def _factory() -> HireCheckWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return HireCheckWorkflow()
