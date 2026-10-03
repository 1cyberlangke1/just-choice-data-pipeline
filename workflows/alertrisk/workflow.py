"""安全警报风险定级 workflow：**特征注入式**（定级矩阵与档位表由代码写，一屏警报记录由后端写）。

流程与 `promptinject` / `logtriage` / `turntake` / `adblock` / `debugstep` / `subqa` / `sqlreview`
同构（`plan → sample → render → ask`），区别在「判什么」：

- **因子表、定级矩阵、处置档位表、记录抬头与格式规格是结构化事实**，由代码渲染进 state——让模型
  去写只会写歪；
- 后端只写**一屏警报记录**（真实的规则名、主机名、进程与请求长尾得很，模板拼出来的脱离现实分布）；
- **哪几个因子取到高危档由代码按字面算**（`features.PATTERNS` 的三条正则，各占一行），等级再由
  矩阵查出来，所以标签是注入特征经纯函数算出来的（`features.plan_for` + `label_for`），模型碰不到；
  写出来的记录必须兑现这份计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` /
  `FrameJudge` 再各查一遍）。

**为什么这一族的中心题型是 `score`**：定级天生是有序的多档判断（1~5 级），硬塞进 choice 会把
「3 级与 4 级差多少」这个序丢掉。`grade` 用 score 题型、等级表不打乱，答案就是等级下标。

**为什么要这一族**：社区里 `triagedy`（安全警报分类过滤器，★1）明写「模型返回类型化配置、
**策略路由保留在代码里**」——本 workflow 的定级矩阵与处置档位表就是那份「保留在代码里」的策略；
`typesafe-triage-guard`（可观测性告警分诊 + 部署风险门控）与 `captaincore`（★71，把恶意软件扫描
结果做成 noul / choice / score 三类决策）是同一形状的另外两个落地。

**这一族只写不执行**：要的是**文本**——规则名、主机数、可达性与凭据要求都只写在记录里，判据认的
也只是字面，没有任何一步会去连机器、跑命令或查漏洞库。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按命中组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`source=cloud` 这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("alertrisk")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import hash_bucket, slice_by_bucket, restrictor
from jc.jsontypes import JsonValue
from jc.knobs import require_pool
from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import AlertRiskCase
from .spec import AlertRiskSpec

WORKFLOW_NAME = "alertrisk"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `source=cloud`：它是**只影响渲染、不影响标签**的那一维——等级由三个因子算
# （`source` 一个字符都不参与），报警通道只决定这屏记录长成什么样（终端检测是进程与脚本、
# WAF 是 HTTP 请求、云配置审计是不合规配置项）。同分布侧一次都见不到它，模型必须真的按字面判，
# 而不是靠「云审计的告警一般都不严重」这类先验蒙。
#
# 反过来，留出 `highs` 会让 `grade` / `highs` / `act` 三族各少几档（8 组少一半），那些档位在同
# 分布侧就取不到了——均衡闸与审计都会当场报出来。所以留出的是 `source`。
OOD_SOURCES: Final[frozenset[str]] = frozenset({features.SOURCE_CLOUD})

# 同分布切分的哈希分桶边界（组合留出：桶按整条 case 算，同一条 case 永远同切分）。
#
# 种子是**挑过的**，不是随手写的：五档等级里最低那一档只有一个命中组合（三个因子全为低），
# 它落在哪个切分，哪个切分才见得到 1 级。训练侧见不到 1 级，`grade` 那一族的最低档就永远学不到
# ——所以种子按「训练侧五档齐全、小批量挑选（8 条）也五档齐全」挑定，`test_pipeline` 里有一条
# 测试把这件事钉死；换种子时它会当场报出来。
SPLIT_SEED: Final[int] = 20261006
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "highs": "哪几个因子取到高危档",
    "source": "这条警报从哪条通道来",
    "level": "查矩阵定出来的等级",
}


@dataclass(frozen=True)
class AlertRiskConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 8×3 = 24 组，512 让枚举器稳稳走**全枚举**；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    - `directions` 手法方向池（**样本级旋钮**：只改 `detail:` 怎么写，不改 case、标签与切分）。
    """

    split: str = "train"
    cover_limit: int = 512
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1)
    directions: tuple[int, ...] = features.DIRECTION_IDS

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasing_pool", self.phrasing_pool)
        require_pool("directions", self.directions)
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


_restricted = restrictor(features.DIMS)


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        source=tuple(value for value in features.SOURCES if value not in OOD_SOURCES)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含 OOD 取值的维度表列表（留出切分用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(source=tuple(value for value in features.SOURCES if value in OOD_SOURCES))
    ]


def bucket_of(case: AlertRiskCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _dims_of(case: AlertRiskCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多一个「等级」）。

    预期行为：`as_dict` 的两个维度之外再加 `level`（矩阵算出来的等级）——小批量里最容易被忽略的
    正是它：只按 `highs` / `source` 铺开的话，10 条会落在「5 级一条都出不来」上，`grade` 那一族
    的答案分布当场偏掉，均衡闸标红。
    """
    return {
        "highs": case.highs,
        "source": case.source,
        "level": str(features.level_of(case.tags())),
    }


def _cover_pick(cases: list[AlertRiskCase], limit: int) -> list[AlertRiskCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：命中组合 → 报警通道 → 等级），同分时取原顺序
    里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`highs` 有 8 个取值、`level` 有 5 个，纯截断的话 10 条的小批量很可能一条
    「5 级」都出不来，`grade` 与 `act` 的某些档位也取不到，均衡闸与审计都会当场报出来。
    """
    names = ("highs", "source", "level")
    chosen: list[AlertRiskCase] = []
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


def cases_for(config: AlertRiskConfig) -> list[AlertRiskCase]:
    """输入：采样参数。输出：该切分的命中组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], AlertRiskCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()), config.split, bucket_of, SPLIT_BUCKETS
        )
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class AlertRiskWorkflow(BackendHolder):
    """安全警报风险定级：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 一屏警报记录由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = tuple(SPLIT_BUCKETS) + ("test_ood",)
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> AlertRiskConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return AlertRiskConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: AlertRiskConfig, case: AlertRiskCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法编号由 RNG 从池里抽（它不是 case 维度），判定计划在 `sample` 阶段按 case
        指纹算出来——与切分、语言、进程都无关。

        手法方向也是**每条样本抽一个**（范式 §9.7：写文本的提示词里要有样本级变量）。同一条 case
        在两次 run 里抽到的方向不同，所以提示词不再逐字相同；它不进 `stats_dimensions`——那不是
        数据维度，报进均衡表只会把「答案分布」这件事冲淡。
        """
        return SplitSpec(
            name=config.split,
            params={
                "case": case,
                "phrasing": rng.choice(config.phrasing_pool),
                "direction": rng.choice(config.directions),
                "seq": seq,
            },
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> AlertRiskSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        direction = split.params["direction"]
        if (
            not isinstance(case, AlertRiskCase)
            or not isinstance(phrasing, int)
            or not isinstance(direction, int)
        ):
            raise TypeError(f"生成计划里的 case / phrasing / direction 类型不对：{split.params!r}")
        return AlertRiskSpec(
            case=case,
            phrasing=phrasing,
            direction=direction,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: AlertRiskSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那屏警报记录，先按注入特征校验（不合格抛错，管线据此重渲染），再与代码
        渲染的因子表 / 矩阵 / 档位表 / 抬头拼起来；后端没注入就报错——`needs` 非空时管线一定会
        注入，这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("alertrisk workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        packet = draft.data.get("packet")
        if not isinstance(packet, str):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.ContentDraft(packet=packet)
        render.check_draft(parsed, spec.case, loc)
        return render.to_state(parsed, loc)

    def ask(self, spec: AlertRiskSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（告警描述）。

        预期行为：只报**一段自然语言正文**——`detail:` 那一行的描述。键名、规则名、主机数、可达性
        与凭据要求都是格式（机器写的），把它们报上去只会把汉字占比稀释掉，把合格的中文产物误报成
        「语言对不上」（范式 §9.50）。描述取不到时报空表——那把尺子对空表不发表意见。
        """
        parsed = render.parse_state(sample.get("state"))
        detail = render.detail_text(parsed.text or "")
        return {"告警描述": detail} if detail else {}



@register(WORKFLOW_NAME)
def _factory() -> AlertRiskWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return AlertRiskWorkflow()
