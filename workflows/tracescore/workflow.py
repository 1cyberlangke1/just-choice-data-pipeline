"""代理轨迹自评 workflow：**特征注入式**（信号表与档位表由代码写，执行轨迹由后端写）。

流程与日志行分诊（`workflows/logtriage`）、提示注入（`workflows/promptinject`）同构
（`plan → sample → render → ask`），区别在「判什么」：

- **行为信号表、处置档位表、轨迹抬头是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**一段执行轨迹**（命令行代理、编辑器内代理、对话式代理的会话片段读起来都是长尾的，
  模板拼出来的脱离现实分布）；
- **命中哪几条由代码按字面算**（`features.PATTERNS` 的四条正则），所以标签是注入特征经纯函数
  算出来的（`features.plan_for` + `label_for`），模型碰不到；写出来的轨迹必须兑现这份计划
  （`render.check_draft` 在渲染期查、`judges.HygieneJudge` / `FrameJudge` 再各查一遍）。

**为什么要这一族**：社区里 `claude-code-trace`（★370）拿 Jev 从 Claude Code 的会话日志里给
「代理进度、工具使用、焦点、探索、恢复与令牌效率」逐维打分；`summarize`（★223）用 Jev 当
**摘要之前的预过滤器**给转录片段打分；`jevmeter`（★70）逐句评分并渲染成片。共同点是一段**已经
发生过的执行轨迹**进来、要一个**立刻能落地**的结论（继续 / 换方案 / 停下问人），而不是让人把
轨迹从头读到尾再拍脑袋。

**与 `rubric` 的区别**：`rubric` 判的是「一份材料按清单逐条打分」（清单由代码写、材料是任意
文本）；本 workflow 判的是**执行轨迹**这一种材料，计分口径是四条行为信号的加权（跑过验证、
按观察调整是加分项，原地打转、反复翻旧账是扣分项），分数直接决定处置档位。材料形态、计分口径
与结论语义都不一样，不重复。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`source=chat`（对话式代理）这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("tracescore")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import hash_bucket, slice_by_bucket, restrictor
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import TraceScoreCase
from .spec import TraceScoreSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "tracescore"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `source=chat`：它是**只影响渲染、不影响标签**的那一维——标签由 `hits` 算
# （`source` 一个字符都不参与），入口只决定这段轨迹长成什么样（命令行代理 / 编辑器内代理 /
# 对话式代理）。同分布侧一次都见不到它，模型必须真的按字面判，而不是靠「这种入口里出现 X 就
# 该换方案」的先验蒙。
#
# 反过来，留出 `hits` 会让 `count` / `combo` / `route` 三族各少几档（16 组少一半），那些档位在
# 同分布侧就取不到了——均衡闸与审计都会当场报出来。所以留出的是 `source`。
OOD_SOURCES: Final[frozenset[str]] = frozenset({features.SOURCE_CHAT})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "hits": "该命中哪几条信号",
    "source": "轨迹从哪一种入口来",
}


@dataclass(frozen=True)
class TraceScoreConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 16×3 = 48 组，512 让枚举器稳稳走**全枚举**
      （成对覆盖在这里只会白扔组合）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 512
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1)


_restricted = restrictor(features.DIMS)


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        source=tuple(value for value in features.SOURCES if value not in OOD_SOURCES)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含 OOD 取值的维度表（`test_ood` 用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(source=tuple(value for value in features.SOURCES if value in OOD_SOURCES))
    ]


def bucket_of(case: TraceScoreCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _dims_of(case: TraceScoreCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多「命中条数」与「推进分」）。

    预期行为：`as_dict` 的两个维度之外再加 `count`（命中条数）与 `score`（推进分）——小批量里
    最容易被忽略的正是这两个：只按 `hits` / `source` 铺开的话，10 条会落在「0 条命中 2 条、
    1 条命中 8 条」上，`count` 的 5 档与 `route` 的 3 档当场缺档，均衡闸标红。

    为什么两个都要：`count` 与 `score` 不是一一对应（同样命中两条，`S1+S3` 是 5 分而 `S2+S4`
    是 1 分），覆盖了条数并不等于覆盖了档位。
    """
    labels = features.label_for(features.plan_for(case, 0))
    return {
        "hits": case.hits,
        "source": case.source,
        "count": str(labels.count),
        "score": str(labels.score),
    }


def _cover_pick(cases: list[TraceScoreCase], limit: int) -> list[TraceScoreCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：命中组合 → 入口 → 推进分 → 命中条数），同分
    时取原顺序里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`hits` 有 16 个取值、`score` 有 5 个，纯截断的话 10 条的小批量很可能
    一条「4 条全中」都出不来，`combo` 与 `route` 的某些档位也取不到，均衡闸与审计都会当场报
    出来。覆盖优先把这种「小批量必然缺一档」的坑填掉，代价只是 case 顺序不再等于桶序（切分归属
    仍是桶算的，没变）。
    """
    names = ("hits", "source", "score", "count")
    chosen: list[TraceScoreCase] = []
    seen: dict[str, set[str]] = {name: set() for name in names}
    remaining = list(cases)
    while remaining and len(chosen) < limit:
        keys = [
            (
                sum(1 for name in names if _dims_of(case)[name] not in seen[name]),
                -index,
            )
            for index, case in enumerate(remaining)
        ]
        best = max(range(len(remaining)), key=lambda index: keys[index])
        case = remaining.pop(best)
        chosen.append(case)
        for name in names:
            seen[name].add(_dims_of(case)[name])
    return chosen


def cases_for(config: TraceScoreConfig) -> list[TraceScoreCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], TraceScoreCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()), config.split, bucket_of, SPLIT_BUCKETS
        )
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class TraceScoreWorkflow(BackendHolder):
    """代理轨迹自评：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 执行轨迹由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> TraceScoreConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return TraceScoreConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: TraceScoreConfig, case: TraceScoreCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> TraceScoreSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, TraceScoreCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return TraceScoreSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: TraceScoreSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那段执行轨迹，先按注入特征校验（不合格抛错，管线据此重渲染），再与代码
        渲染的信号表 / 档位表 / 抬头拼起来；后端没注入就报错——`needs` 非空时管线一定会注入，
        这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("tracescore workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        text = draft.data.get("text")
        if not isinstance(text, str):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.ContentDraft(text=text)
        render.check_draft(parsed, spec.case, loc)
        return render.to_state(parsed, loc)

    def ask(self, spec: TraceScoreSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「执行轨迹」）。

        预期行为：只报模型写的那段轨迹；`tags` 是信号的编号（由语言表写），不报。
        """
        parsed = render.parse_state(sample.get("state"))
        text = parsed.text
        return {"执行轨迹": text} if text and text.strip() else {}



@register(WORKFLOW_NAME)
def _factory() -> TraceScoreWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return TraceScoreWorkflow()
