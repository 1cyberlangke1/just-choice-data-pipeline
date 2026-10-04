"""有界任务推进（planstep）workflow：**构造式**（任务表与四个答案都由代码算出来，零 token）。

覆盖的决策形状（社区清单）：`foreman`（★406）用一个监督器让编程 agent 始终保持任务方向、
`stanley-code`（★113）把 agent 的工作流收进有界决策、`glowbom-oss`（★163）让 Jev 判断
「这一步该不该继续」、`jev-agent-design-with-topk-logits-choices`（★20）在候选提案里挑下一步。
它们要解决的是同一件事：**手里有一张有依赖关系的任务表，下一步该推进哪一项**。

**这一族的轴是「偏序上的贪心推进」**：`clipcut` 的「顺序折叠」是线性前缀，`cachereuse` 是
两条轨迹的逐步对齐，`maze` / `invoice` 是逐条独立判断；这里每一项的处境取决于它的**前置集合**
是否全部完成——「前面」是一个下闭集，不是一条线。四个答案（还剩几项 / 下一步是哪一项 /
还要几轮 / 做完下一步会不会立刻解锁别的任务）全部由四条规则算出来，零 token。

切分按 PLAN 3.3：
- train / val / test_iid：**组合留出**——按 case（图形状 + 已完成项数）的哈希分桶切片，
  同一组取值整组不跨切分；
- test_ood：**值留出**——图形状下标 10 / 11（「迟合流」与「分叉再汇」）在训练侧一个都不出现。

导入本模块即自注册（`jc.workflow.load("planstep")` 可用）。
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
from .features import PlanCase
from .spec import PlanConfig, PlanSpec

WORKFLOW_NAME = "planstep"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（组合留出：桶按 case 算，同一组取值永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 同分布侧配置：只收同分布图形状。
ID_CONFIG: Final[PlanConfig] = PlanConfig(split="train")
#: 留出侧配置：只收留出图形状（`ood_dims` 定的），不切片。
OOD_CONFIG: Final[PlanConfig] = PlanConfig(split="test_ood")

DIMENSION_TITLES: Final[dict[str, str]] = {
    "graph": "依赖图形状",
    "done": "已完成项数",
}


_restricted = restrictor(features.DIMS)


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：同分布侧的维度表（图形状只留同分布那一批）。"""
    return _restricted(graph=features.ID_GRAPHS)


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含留出图形状的维度表（`test_ood` 用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(graph=features.OOD_GRAPHS)]


def bucket_of(case: PlanCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _cover_pick(cases: list[PlanCase], limit: int) -> list[PlanCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：图形状 → 已完成项数），同分时取原顺序里
    靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`next` 与 `remaining` 两族的分布靠「每张图的完成顺序都是 0~4 的一个
    排列」撑起来，纯截断的话 10 条的小批量很可能一条 `done=5`（全部做完）都出不来，
    均衡闸与审计会当场报出来。
    """
    names = ("graph", "done")
    return cover_pick(cases, limit, names, lambda case: case.as_dict())


def cases_for(config: PlanConfig) -> list[PlanCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 收留出图形状的组合（并集去重）；其余切分先剔除留出取值，
    再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], PlanCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()), config.split, bucket_of, SPLIT_BUCKETS
        )
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


@register(WORKFLOW_NAME)
class PlanStepWorkflow(StandardMethods):
    """有界任务推进：任务表 → 还剩几项 / 下一步 / 还要几轮 / 解不解锁。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 构造式：任务表与答案都由代码算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")
    split_names_tuple = tuple(SPLIT_BUCKETS) + ("test_ood",)
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_judges = judges_mod
    pkg_locales = locales

    def config_for(self, split: str) -> PlanConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用 OOD 配置；其余切分用同分布配置并把 `split` 换成调用方要的
        那一档（不换会切出三份一模一样的数据）；未知切分报人话。
        实现见 `jc.workflow.config_for_split`（12 家逐字同构，只留那一份）。
        """
        return cast(
            "PlanConfig",
            config_for_split(split, OOD_CONFIG, ID_CONFIG, SPLIT_BUCKETS, self.split_names()),
        )

    def plan_sample(
        self, rng: random.Random, config: PlanConfig, case: PlanCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasings)

    def sample(self, rng: random.Random, split: SplitSpec) -> PlanSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（请求号与四个答案都在这里定死）。"""
        return spec_mod.sample_spec(
            rng, split.params["case"], split.name, split.params["seq"], split.params["phrasing"]
        )

    def render(self, spec: PlanSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件必得同一段
        文本，所以「两次生成逐字节一致」这条能直接查。
        """
        del rng
        return render.build(spec, locales.load(locale))

    def ask(self, spec: PlanSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: object, *, case: object = None) -> parse.ParsedPlan:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的结构反解
        （各语言的标签互不重叠，由 `tests/workflows/planstep/test_locales.py` 锁住）。
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
                "remaining": Counter(),
                "next": Counter(),
                "rounds": Counter(),
                "unlocks": Counter(),
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
                for key in ("remaining", "next", "rounds", "unlocks"):
                    counts[key][facts[key]] += 1
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`next_2` -> `next`）
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
        """空操作：任务推进是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("任务推进 workflow 是纯构造式的，不该被绑定生成后端")
