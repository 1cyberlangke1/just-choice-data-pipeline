"""情感与情绪（sentiment）workflow：**构造式**（情绪弧线与四个答案都由代码算出来，零 token）。

覆盖的决策形状（社区清单）：`yanwai`（★15）在聊天模块里显示每条消息的情绪与沟通建议、
`jev-dev` 比较情感变化的偏差。要判的是同一件事：**一串对话的情绪怎么走，哪一回合变了调**。

**这一族的轴是「带符号的序列怎么走」**：`quoteread` 把每个读数换算成基准点再比档（逐点独立）、
`clipcut` 判线性前缀的去留、`planstep` 判偏序上的下一步——都是**结构**判断；这里判的是
符号、符号变化的位置、变化幅度与变化的方向。四个答案（整段倾向 / 首次转折在哪一回合 /
最大落差 / 有没有回升）全部由四条规则算出来，零 token。

切分按 PLAN 3.3：
- train / val / test_iid：**组合留出**——按 case（弧线 + 场景）的哈希分桶切片，
  同一组取值整组不跨切分；
- test_ood：**值留出**——弧线下标 15~18（来回摆 / 高开后一路偏差 / 先平后高再翻负 /
  开局很差后一路偏好）在训练侧一个都不出现。

导入本模块即自注册（`jc.workflow.load("sentiment")` 可用）。
"""

from __future__ import annotations

import random
from collections import Counter

from pathlib import Path
from typing import Final, cast

from jc import jsonio
from jc.backend.types import GenerationBackend
from jc.cover import cover_pick, hash_bucket, slice_by_bucket, restrictor
from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion, SplitSpec, StandardMethods, family_of, plan_with_phrasing, register, config_for_split

from . import features, judges as judges_mod, locales, parse, questions, render
from . import spec as spec_mod
from .features import SentimentCase
from .spec import SentimentConfig, SentimentSpec

WORKFLOW_NAME = "sentiment"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（组合留出：桶按 case 算，同一组取值永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 同分布侧配置：只收同分布弧线。
ID_CONFIG: Final[SentimentConfig] = SentimentConfig(split="train")
#: 留出侧配置：只收留出弧线（`ood_dims` 定的），不切片。
OOD_CONFIG: Final[SentimentConfig] = SentimentConfig(split="test_ood")

DIMENSION_TITLES: Final[dict[str, str]] = {
    "arc": "情绪弧线",
    "topic": "场景",
}


_restricted = restrictor(features.DIMS)


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：同分布侧的维度表（弧线只留同分布那一批）。"""
    return _restricted(arc=features.ID_ARCS)


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含留出弧线的维度表（`test_ood` 用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(arc=features.OOD_ARCS)]


def bucket_of(case: SentimentCase) -> int:
    """输入：一个 case。输出：它的哈希桶号（同一组取值永远同桶）。"""
    return hash_bucket(case, SPLIT_SEED)


def _cover_pick(cases: list[SentimentCase], limit: int) -> list[SentimentCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：弧线 → 场景），同分时取原顺序里靠前的；
    名额没用完就按原顺序接着填。

    为什么不是直接截断：四个题族的分布靠「15 条弧线逐族铺开」撑起来，纯截断的话小批量很可能
    某个档位一条都出不来（均衡闸与审计会当场报出来）。
    """
    names = ("arc", "topic")
    return cover_pick(cases, limit, names, lambda case: case.as_dict())


def cases_for(config: SentimentConfig) -> list[SentimentCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 收留出弧线的组合（并集去重）；其余切分先剔除留出取值，
    再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], SentimentCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()),
            config.split,
            bucket_of,
            SPLIT_BUCKETS,
        )
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


@register(WORKFLOW_NAME)
class SentimentWorkflow(StandardMethods):
    """情感与情绪：情绪弧线 → 整段倾向 / 首次转折 / 最大落差 / 有没有回升。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 构造式：弧线与答案都由代码算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")
    split_names_tuple = tuple(SPLIT_BUCKETS) + ("test_ood",)
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_judges = judges_mod
    pkg_locales = locales

    def config_for(self, split: str) -> SentimentConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用 OOD 配置；其余切分用同分布配置并把 `split` 换成调用方要的
        那一档（不换会切出三份一模一样的数据）；未知切分报人话。
        实现见 `jc.workflow.config_for_split`（12 家逐字同构，只留那一份）。
        """
        return cast(
            "SentimentConfig",
            config_for_split(split, OOD_CONFIG, ID_CONFIG, SPLIT_BUCKETS, self.split_names()),
        )

    def plan_sample(
        self, rng: random.Random, config: SentimentConfig, case: SentimentCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、序号。输出：带问法的切分计划。"""
        return plan_with_phrasing(config, rng, case, seq, config.phrasings)

    def sample(self, rng: random.Random, split: SplitSpec) -> SentimentSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（请求号与四个答案都在这里定死）。"""
        return spec_mod.sample_spec(
            rng, split.params["case"], split.name, split.params["seq"], split.params["phrasing"]
        )

    def render(self, spec: SentimentSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件必得同一段
        文本，所以「两次生成逐字节一致」这条能直接查。
        """
        del rng
        return render.build(spec, locales.load(locale))

    def ask(self, spec: SentimentSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: object, *, case: object = None) -> parse.ParsedArc:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的结构反解
        （各语言的标签互不重叠，由 `tests/workflows/sentiment/test_locales.py` 锁住）。
        """
        del case
        return parse.parse_state(state, parse.merged_tables())

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

        预期行为：只看产物本身——题族是否齐全、choice 的答案编号是否越界、以及**答案能不能由
        state 重算出来**（复用 `judges.check_sample_any_locale`，与判别器同一把尺，这样
        「老产物」也能事后查）。
        """
        locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段（判定按「任一语言」走）
        # 用**合并表**反解：产物按 `locales[seq % len]` 轮转渲染，调用方给的语言未必是产物那一种。
        tables = parse.merged_tables()
        known = set(questions.FAMILIES)
        report: dict[str, JsonValue] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, JsonValue] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "tone": Counter(),
                "pivot": Counter(),
                "drop": Counter(),
                "rebound": Counter(),
                "index_bad": 0,
                "label_mismatch": 0,
                "unknown_families": Counter(),
            }
            if not path.exists():
                missing.append(f"{split}: 产物不存在（{path.name}）")
                report[split] = counts
                continue
            for _, row in jsonio.iter_jsonl(path):
                counts["samples"] += 1
                try:
                    parsed = parse.parse_state(row["state"], tables)
                    facts = judges_mod.facts_of(parsed)
                except (ValueError, KeyError) as exc:
                    counts["label_mismatch"] += 1
                    missing.append(f"{split}: 有一条请求反解不出来（{exc}）")
                    continue
                for key in ("tone", "pivot", "drop", "rebound"):
                    counts[key][facts[key]] += 1
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`tone_2` -> `tone`）
                    counts["families"][family] += 1
                    counts["types"][question["type"]] += 1
                    if family not in known:
                        counts["unknown_families"][family] += 1
                    if question["type"] == "choice":
                        options = list(question["criteria"])
                        index = row["answers"][qid]["choice"]
                        if not isinstance(index, int) or not 0 <= index < len(options):
                            counts["index_bad"] += 1
                problems = judges_mod.check_sample_any_locale(row, facts)
                counts["label_mismatch"] += len(problems)
                if problems:
                    missing.append(f"{split}: {problems[0]}")
            for family in questions.FAMILIES:
                if counts["families"][family] == 0:
                    missing.append(f"{split}: 题族 {family} 一条都没出")
            if counts["index_bad"]:
                missing.append(f"{split}: {counts['index_bad']} 处编号异常")
            if counts["label_mismatch"]:
                missing.append(f"{split}: {counts['label_mismatch']} 条答案与 state 重算不一致")
            if counts["unknown_families"]:
                missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
            report[split] = counts
        return report, missing

    def bind_backend(self, backend: GenerationBackend) -> None:
        """空操作：情绪判读是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("情感与情绪 workflow 是纯构造式的，不该被绑定生成后端")
