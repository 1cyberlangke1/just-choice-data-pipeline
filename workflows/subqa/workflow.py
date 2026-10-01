"""字幕对质检 workflow：**特征注入式**（判据表与档位表由代码写，一条字幕对由后端写）。

流程与 `promptinject` / `logtriage` / `turntake` / `adblock` / `debugstep` 同构
（`plan → sample → render → ask`），区别在「判什么」：

- **判据表、处置档位表、字幕对抬头与格式规格是结构化事实**，由代码渲染进 state——让模型去写只会
  写歪；
- 后端只写**一条双语字幕对**（真实字幕长尾得很，模板拼出来的脱离现实分布）；
- **命中哪几条由代码按字面算**（`features.PATTERNS` 的四条正则，各占一行），所以标签是注入特征
  经纯函数算出来的（`features.plan_for` + `label_for`），模型碰不到；写出来的字幕对必须兑现这份
  计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` / `FrameJudge` 再各查一遍）。

**为什么要这一族**：社区里 `transcript-lens`（★16）逐句分析字幕，`jev-subtitle-translator`
（★1）「用 Noul 判断标记需要人工复核的字幕对，但不改写字幕」。共同点是**每一对字幕都要一次
「能不能直接发」的判断，判断完立刻要落地一个动作**（直接发 / 抽查 / 逐句复核 / 打回重译）。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`track=lecture` 这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("subqa")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import hash_bucket, slice_by_bucket
from jc.jsontypes import JsonValue
from jc.knobs import require_pool
from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import SubQaCase
from .spec import SubQaSpec

WORKFLOW_NAME = "subqa"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `track=lecture`：它是**只影响渲染、不影响标签**的那一维——标签由 `hits` 算
# （`track` 一个字符都不参与），素材类型只决定这条字幕长成什么样（电影的对白短促、剧集有角色名、
# 讲座是长句与术语）。同分布侧一次都见不到它，模型必须真的按字面判，而不是靠「讲座字幕里出现 X
# 就拦掉」的先验蒙。
#
# 反过来，留出 `hits` 会让 `count` / `combo` / `act` 三族各少几档（16 组少一半），那些档位在
# 同分布侧就取不到了——均衡闸与审计都会当场报出来。所以留出的是 `track`。
OOD_TRACKS: Final[frozenset[str]] = frozenset({features.TRACK_LECTURE})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20261004
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "hits": "该命中哪几条判据",
    "track": "这条字幕对来自哪种素材",
}


@dataclass(frozen=True)
class SubQaConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 16×3 = 48 组，512 让枚举器稳稳走**全枚举**
      （成对覆盖在这里只会白扔组合）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池；
    - `timecodes` 开始时间码池（**样本级旋钮**：只改 `Timing:` 那一行怎么写，不改 case、标签与
      切分）。为什么要有它：`Timing:` 是这一族唯一一个符号类自由字段，真机上整批共用同一串数字
      （`sq-10c-zh` 9 条里 4 条一模一样，范式 §9.61）。
    """

    split: str = "train"
    cover_limit: int = 512
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1)
    timecodes: tuple[str, ...] = features.TIMECODE_HINTS

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasing_pool", self.phrasing_pool)
        require_pool("timecodes", self.timecodes)
        # 形状不对的时间码写进提示词只会让写手拿到一句没法照做的要求（真跑起来是静默的数据走样），
        # 而且 `S3` 的正则就按 `HH:MM:SS,mmm` 认——所以在构造期逐条拦下。
        bad = [item for item in self.timecodes if not features.TIMECODE_SHAPE.match(item)]
        if bad:
            raise ValueError(f"timecodes 里不是 `HH:MM:SS,mmm` 形状的取值：{bad}")
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


def _restricted(**overrides: tuple[JsonValue, ...]) -> dict[str, tuple[JsonValue, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    dims: dict[str, tuple[JsonValue, ...]] = dict(features.DIMS)
    dims.update(overrides)
    return dims


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        track=tuple(value for value in features.TRACKS if value not in OOD_TRACKS)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含 OOD 取值的维度表列表（留出切分用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(track=tuple(value for value in features.TRACKS if value in OOD_TRACKS))
    ]


def bucket_of(case: SubQaCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _dims_of(case: SubQaCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多一个「命中条数」）。

    预期行为：`as_dict` 的两个维度之外再加 `count`（命中条数）——小批量里最容易被忽略的正是它：
    只按 `hits` / `track` 铺开的话，10 条会落在「0 条命中 2 条、1 条命中 8 条」上，`count` 与
    `act` 两族的答案分布当场偏到 80%，均衡闸标红。
    """
    return {"hits": case.hits, "track": case.track, "count": str(len(case.tags()))}


def _cover_pick(cases: list[SubQaCase], limit: int) -> list[SubQaCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：命中组合 → 素材类型 → 命中条数），同分时取原
    顺序里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`hits` 有 16 个取值、`count` 有 5 个，纯截断的话 10 条的小批量很可能一条
    「4 条全中」都出不来，`combo` 与 `act` 的某些档位也取不到，均衡闸与审计都会当场报出来。
    """
    names = ("hits", "track", "count")
    chosen: list[SubQaCase] = []
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


def cases_for(config: SubQaConfig) -> list[SubQaCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], SubQaCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()), config.split, bucket_of, SPLIT_BUCKETS
        )
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class SubQaWorkflow(BackendHolder):
    """字幕对质检：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 一条字幕对由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = tuple(SPLIT_BUCKETS) + ("test_ood",)
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> SubQaConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return SubQaConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: SubQaConfig, case: SubQaCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法编号由 RNG 从池里抽（它不是 case 维度），判定计划在 `sample` 阶段按
        「case 指纹 + 问法编号」确定性地算出来。
        开始时间码同理是**样本级旋钮**（范式 §9.61）：它不进 `stats_dimensions`——那不是数据维度，
        报进均衡表只会把「答案分布」这件事冲淡。
        """
        return SplitSpec(
            name=config.split,
            params={
                "case": case,
                "phrasing": rng.choice(config.phrasing_pool),
                "timecode": rng.choice(config.timecodes),
                "seq": seq,
            },
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> SubQaSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        timecode = split.params["timecode"]
        if (
            not isinstance(case, SubQaCase)
            or not isinstance(phrasing, int)
            or not isinstance(timecode, str)
        ):
            raise TypeError(f"生成计划里的 case / phrasing / timecode 类型不对：{split.params!r}")
        return SubQaSpec(
            case=case,
            phrasing=phrasing,
            timecode=timecode,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: SubQaSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那条字幕对，先按注入特征校验（不合格抛错，管线据此重渲染），再与代码
        渲染的判据表 / 档位表 / 抬头拼起来；后端没注入就报错——`needs` 非空时管线一定会注入，
        这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("subqa workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        cue = draft.data.get("cue")
        if not isinstance(cue, str):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.ContentDraft(cue=cue)
        render.check_draft(parsed, spec.case, loc)
        return render.to_state(parsed, loc)

    def ask(self, spec: SubQaSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（译文与译者备注）。

        预期行为：只报**两段自然语言正文**——`Target:` 那一行的译文与 `Note:` 那一行的备注。键名、
        时间码、`-->` 这些是格式，原文那一行用的是源语言（不是目标语言），把它们报上去只会把汉字
        占比稀释掉，把合格的中文产物误报成「语言对不上」（范式 §9.50）。两段都空时报空表——那把
        尺子对空表不发表意见。
        """
        parsed = render.parse_state(sample.get("state"))
        cue = parsed.text or ""
        texts = {"译文": render.target_text(cue), "译者备注": render.note_text(cue)}
        return {field: text for field, text in texts.items() if text}



@register(WORKFLOW_NAME)
def _factory() -> SubQaWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return SubQaWorkflow()
