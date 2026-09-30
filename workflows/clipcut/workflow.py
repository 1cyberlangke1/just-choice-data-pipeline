"""片段筛选与去冗余（clipcut）workflow：**构造式**（片段表与四个答案都由代码算出来，零 token）。

覆盖的决策形状（社区清单）：`summarize`（★223）在把转录片段送去摘要器之前先给片段打分筛选、
`jevmeter`（★70）逐句给视频评分、`jevclip`（★21）判断字幕片段以筛选高亮内容并删去冗余。
它们要解决的是同一件事：**从一串片段里挑出该留下的那几段**。这里把「挑哪几段」做成**可判定
的算术**，规则只有两条，全印在题面里：一段被保留当且仅当它含高亮点、且它是它那个主题第一次
出现的位置；主题重复时只有首现有资格。

**这一族的轴是「顺序折叠」**：`cachereuse` 比的是两条轨迹的**逐步对齐**（成对比较），
`maze` / `invoice` 那些是**逐条独立判断**，而这里每一段的去留都取决于**它前面的片段**
（同一个主题有没有先出现过）——答案依赖前缀，不只看本条。

切分按 PLAN 3.3：
- train / val / test_iid：**组合留出**——按 case（主题序列 + 高亮序列）的哈希分桶切片，
  同一份素材整组不跨切分；
- test_ood：**值留出**——含留出主题 `T5` 的那条主题序列（下标 4）在训练侧一个都不出现
  （它只把某几段的主题名换成同分布侧见不到的名字，不动任何规则分支，所以留出它是安全的）。

导入本模块即自注册（`jc.workflow.load("clipcut")` 可用）。
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Final

from jc import jsonio
from jc.backend.types import GenerationBackend
from jc.judge import Judge
from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion, SplitSpec, family_of, register

from . import features, judges as judges_mod, locales, parse, questions, render
from . import spec as spec_mod
from .features import ClipCase
from .spec import ClipConfig, ClipSpec

WORKFLOW_NAME = "clipcut"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（组合留出：桶按 case 算，同一份素材永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 同分布侧配置：只收不含留出主题的序列。
ID_CONFIG: Final[ClipConfig] = ClipConfig(split="train")
#: 留出侧配置：只收含留出主题的序列（`ood_dims` 定的），不切片。
OOD_CONFIG: Final[ClipConfig] = ClipConfig(split="test_ood")

DIMENSION_TITLES: Final[dict[str, str]] = {
    "topics": "主题序列",
    "flags": "高亮序列",
}


def _restricted(**overrides: tuple[JsonValue, ...]) -> dict[str, tuple[JsonValue, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    dims: dict[str, tuple[JsonValue, ...]] = dict(features.DIMS)
    dims.update(overrides)
    return dims


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：同分布侧的主题序列（剔除含留出主题的那一条）。"""
    return _restricted(topics=features.ID_TOPIC_PATTERNS)


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含留出主题序列的维度表（`test_ood` 用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(topics=features.OOD_TOPIC_PATTERNS)]


def bucket_of(case: ClipCase) -> int:
    """输入：一个 case。输出：它的哈希桶号（0~99）。

    预期行为：只按 case（主题序列 + 高亮序列）算，与切分名无关——同一份素材永远落在同一个
    桶里，所以「同一素材不跨切分」是算出来的，不是靠人排的。不用内置 `hash()`（它对字符串
    每进程随机化），否则切分会随进程漂移。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{SPLIT_SEED}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[ClipCase], split: str) -> list[ClipCase]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的；切片用**位置**而不是桶号
    过滤，只按桶号过滤时哈希方差会让 val 只剩个位数。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def _dims_of(case: ClipCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表。"""
    return {"topics": str(case.topics), "flags": str(case.flags)}


def _cover_pick(cases: list[ClipCase], limit: int) -> list[ClipCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：主题序列 → 高亮序列），同分时取原顺序里
    靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：主题序列 4 个取值、高亮序列 8 个取值，纯截断的话 10 条的小批量很可能
    一条「第 3 段才保留」都出不来，`first_kept` 的某些档位取不到，均衡闸与审计都会当场报出来。
    """
    names = ("topics", "flags")
    chosen: list[ClipCase] = []
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


def cases_for(config: ClipConfig) -> list[ClipCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 收留出主题序列的组合（并集去重）；其余切分先剔除留出取值，
    再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], ClipCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(features.build_cases(config.cover_limit, dims=id_dims()), config.split)
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


@register(WORKFLOW_NAME)
class ClipCutWorkflow:
    """片段筛选与去冗余：片段表 → 保留段数 / 最早保留的段 / 有没有被挡下 / 有没有主题重复。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 构造式：片段表与答案都由代码算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：支持的切分名。"""
        return tuple(SPLIT_BUCKETS) + ("test_ood",)

    def config_for(self, split: str) -> ClipConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用留出主题序列；其余切分用同分布表，**并且把 `split` 换成调用方
        要的那一档**——`ID_CONFIG` 是个单例，直接返回它会让 train/val/test_iid 三次都按同一档
        切片（切出来三份一模一样的数据）。
        """
        if split == "test_ood":
            return OOD_CONFIG
        if split in SPLIT_BUCKETS:
            return replace(ID_CONFIG, split=split)
        raise KeyError(f"未知切分：{split!r}（可用：{self.split_names()}）")

    def build_cases(self, config: ClipConfig) -> list[ClipCase]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。"""
        return cases_for(config)

    def case_label(self, case: ClipCase) -> str:
        """输入：一个 case。输出：一行可读描述（报告与文件名用）。"""
        return case.describe()

    def stats_dimensions(self, case: ClipCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：{维度名: 取值}（两个维度全摊平）。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: ClipConfig, case: ClipCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法在这里抽（同一 case 的多条样本会用不同问法）；case 与序号一起放进
        params，`sample` 阶段不再做任何随机决策。
        """
        return SplitSpec(
            name=config.split,
            params={"case": case, "phrasing": rng.choice(config.phrasings), "seq": seq},
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> ClipSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（请求号与四个答案都在这里定死）。"""
        return spec_mod.sample_spec(
            rng, split.params["case"], split.name, split.params["seq"], split.params["phrasing"]
        )

    def render(self, spec: ClipSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件必得同一段
        文本，所以「两次生成逐字节一致」这条能直接查。
        """
        del rng
        return render.build(spec, locales.load(locale))

    def ask(self, spec: ClipSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: object, *, case: object = None) -> parse.ParsedClip:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的结构反解
        （各语言的标签互不重叠，由 `tests/workflows/clipcut/test_locales.py` 锁住）。
        「原文 → 键」留给判别器，它手上有语言文件。
        """
        del case
        return parse.parse_state(state, parse.merged_tables())

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。预期行为：未知名字直接报错。"""
        out: list[Judge] = []
        for name in names:
            if name == "roundtrip":
                out.append(judges_mod.RoundtripJudge(locale=locale))
            elif name == "label":
                out.append(judges_mod.LabelJudge(locale=locale))
            else:
                raise KeyError(f"未知判别器：{name!r}（可用：roundtrip / label）")
        return tuple(out)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：空表（纯构造族，没有「模型写的正文」）。

        预期行为：`needs` 为空，state 全由代码按语言模板拼，天然一致。
        """
        del sample
        return {}

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
        words = locales.words()
        known = set(questions.FAMILIES)
        report: dict[str, JsonValue] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, JsonValue] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "keep_count": Counter(),
                "first_kept": Counter(),
                "blocked_any": Counter(),
                "dup_any": Counter(),
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
                    facts = judges_mod.facts_of(parsed, words)
                except (ValueError, KeyError) as exc:
                    counts["label_mismatch"] += 1
                    missing.append(f"{split}: 有一条请求反解不出来（{exc}）")
                    continue
                counts["keep_count"][facts["keep_count"]] += 1
                counts["first_kept"][facts["first_kept"]] += 1
                counts["blocked_any"][facts["blocked_any"]] += 1
                counts["dup_any"][facts["dup_any"]] += 1
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`keep_count_2` -> `keep_count`）
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

    def available_locales(self) -> list[str]:
        """输入：无。输出：支持的语言列表。"""
        return locales.available()

    def bind_backend(self, backend: GenerationBackend) -> None:
        """空操作：片段筛选是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("片段筛选 workflow 是纯构造式的，不该被绑定生成后端")
