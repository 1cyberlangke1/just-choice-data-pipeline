"""会议日程 workflow：**混合式**（构造骨架 + 模型补表述）。

与另外三个 workflow 的关系（PLAN 3.2 的分类表）：
- `maze` / `invoice` 纯构造式：连文本都是算出来的（零 token）；
- `mail` 特征注入式：标签先行，模型只写文本外壳；
- `hybrid`（本 workflow）介于两者之间：日期 / 时刻 / 时长 / 会议室 / 主持人由构造器算，
  **每场会的标题与议题交给后端写**——这一半是模型「无中生有」的，
  也正是真实日程表里最不可枚举的部分。

正确性靠两条：**答案只来自骨架**（`title_of` / `agenda_of` 问的文本，真值就是后端写出来的
原文，由 `render` 记回 `spec.written`）；**文本不许夹带骨架事实**（夹带就拒）。

切分按 PLAN 3.3：
- train / val / test_iid：**组合留出**——按 6 维组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——版式 `table` 与冲突档 `same_room` 在训练侧一个都不出现，
  `test_ood` 只收「至少含一个没见过的取值」的 case。

导入本模块即自注册（`jc.workflow.load("hybrid")` 可用）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import hashlib
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Final

from jc import jsonio
from jc.backend.types import GenerationBackend
from jc.judge import Judge
from jc.workflow import LabeledQuestion, SplitSpec, family_of, register

from . import judges as judges_mod
from . import locales, questions, render
from . import spec as spec_mod
from .spec import HybridCase, HybridConfig, HybridSpec

WORKFLOW_NAME = "hybrid"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（组合留出：桶按 case 的 6 维取值算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260925
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 训练侧取值：只留 spec 里标了「见过」的那几档。排序是为了顺序稳定
#: （frozenset 的迭代顺序随进程变化，不能直接拿来枚举）。
ID_CONFIG: Final[HybridConfig] = HybridConfig(
    split="train",
    layouts=tuple(sorted(spec_mod.SEEN_LAYOUTS)),
    clashes=tuple(sorted(spec_mod.SEEN_CLASHES)),
)
#: 留出档：取值表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的 case。
OOD_CONFIG: Final[HybridConfig] = HybridConfig(split="test_ood", ood_only=True)
#: 注意：训练侧没有 `same_room`，所以同分布切分里**根本不会出现会议室冲突**，
#: `room_free` 恒为 True（均衡报告会把它标成偏差 0.5）。这是值留出的代价，
#: 不是数据错——`room_free` 的「假」那一半只在 `test_ood` 里出现。

DIMENSION_TITLES: Final[dict[str, str]] = {
    "meetings": "会议场数",
    "opening": "日程起始时段",
    "clash": "冲突档",
    "layout": "版式",
    "weekday": "星期几",
    "duration_mix": "时长档",
}


def bucket_of(case: HybridCase, seed: int = SPLIT_SEED) -> int:
    """输入：case、种子。输出：0~99 的桶号。

    预期行为：同一组合必得同一桶，不同组合近似均匀——切分单位是**组合**，不是样本；
    不用内置 `hash()`（它对字符串每进程随机化），否则切分会随进程漂移。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[HybridCase], split: str) -> list[HybridCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的——只按桶号过滤时，
    哈希方差会让 val 只剩个位数；同一组合永远落在同一段。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def cases_for(config: HybridConfig) -> list[HybridCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合；
    其余切分按 6 维组合的哈希桶切片，同组合不跨切分；`case_limit` 在**切片之后**截取，
    与同族 workflow 的语义一致（小批量试跑用，见 `jc.knobs` 的记账口径）。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        out = cases
    else:
        out = _slice(cases, config.split)
    return out[: config.case_limit] if config.case_limit else out


@register(WORKFLOW_NAME)
class HybridWorkflow:
    """会议日程：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    #: 要后端写标题与议题；后端只出文本，骨架与答案都由构造器算。
    needs: frozenset[str] = frozenset({"text"})
    judges: tuple[str, ...] = ("skeleton", "answers", "text")

    def __init__(self) -> None:
        """输入：无。输出：实例。预期行为：后端先为空，由 `bind_backend` 注入。"""
        self.backend: GenerationBackend | None = None

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：四个切分（同分布三档 + 一个值留出的 OOD 档）。"""
        return ("train", "val", "test_iid", "test_ood")

    def config_for(self, split: str) -> HybridConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用全取值表 + `ood_only`；其余切分用训练侧取值表。
        未知切分直接报错（不猜、不回落）。
        """
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        if split == "test_ood":
            return OOD_CONFIG
        return replace(ID_CONFIG, split=split)

    def build_cases(self, config: HybridConfig) -> list[HybridCase]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。"""
        return cases_for(config)

    def case_label(self, case: HybridCase) -> str:
        """输入：一个 case。输出：一行可读描述（报告与文件名用）。"""
        return case.describe()

    def stats_dimensions(self, case: HybridCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：{维度名: 取值}（6 个维度全摊平）。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: HybridConfig, case: HybridCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法在这里抽（同一 case 的多条样本会用不同问法）；
        主题方向也在这里抽（提示词里只有场数一个变量的话，同场数的样本提示词会逐字相同，
        标题就会塌成套话，见 `spec.THEME_IDS` 的来由）；
        case 与序号一起放进 params，`sample` 阶段不再做任何随机决策。
        """
        return SplitSpec(
            name=config.split,
            params={
                "case": case,
                "phrasing": rng.choice(config.phrasings),
                "theme": rng.choice(config.themes),
                "seq": seq,
            },
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> HybridSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（骨架定死，文本留空等后端）。"""
        return spec_mod.sample_spec(
            rng,
            split.params["case"],
            split.name,
            split.params["seq"],
            split.params["phrasing"],
            split.params["theme"],
        )

    def render(self, spec: HybridSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（整份日程文本）。

        预期行为：向后端要每场会的标题与议题，校验通过后**记回 `spec.written`**
        （`ask` 与判别器都靠它拿文本真值），再拼成日程。
        后端没注入就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，
        绝不能悄悄降级成模板文本（那会产出一份「看起来像样、其实没有模型参与」的数据）。
        """
        if self.backend is None:
            raise RuntimeError("hybrid workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc, spec.text_task_id))
        spec.written.update(render.check_draft(spec, loc, draft.data))
        return render.build(spec, loc)

    def ask(self, spec: HybridSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: object, *, case: object = None) -> render.ParsedSchedule:
        """输入：渲染后的 state。输出：反解出来的日程。

        预期行为：`Workflow.reparse` 没有语言参数，所以反解**语言无关**——
        版式靠结构认，文本字段靠合并标签表（各语言的标签互不重叠，由 test_locales.py 锁住）。
        """
        return render.parse_state(state)

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。预期行为：未知名字直接报错。"""
        out: list[Judge] = []
        for name in names:
            if name == "skeleton":
                out.append(judges_mod.SkeletonJudge(locale=locale))
            elif name == "answers":
                out.append(judges_mod.AnswerJudge(locale=locale))
            elif name == "text":
                out.append(judges_mod.TextJudge(locale=locale))
            else:
                raise KeyError(f"未知判别器：{name!r}（可用：skeleton / answers / text）")
        return tuple(out)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（每场会的标题与议题）。

        预期行为：只报模型写的两段文字；起止时间 / 会议室 / 主持人由代码排，不报。反解不出来
        （坏 state）给空表——那由契约校验报。
        """
        parsed = render.parse_state(sample.get("state"))
        texts: dict[str, str] = {}
        for index, meeting in enumerate(parsed.meetings, start=1):
            texts[f"第 {index} 场标题"] = meeting.title
            texts[f"第 {index} 场议题"] = meeting.agenda
        return texts

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

        预期行为：只看产物本身能查的东西——题族是否齐全、题型分布、choice 的答案编号是否越界，
        以及**选项个数与日程场数是否自洽**（反解 state 数出几场会，再核对各题族的选项数）。
        反解语言无关，所以不需要具体语言文件。
        """
        known = set(questions.FAMILIES)
        report: dict[str, JsonValue] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, JsonValue] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "index_bad": 0,
                "options_bad": 0,
                "unknown_families": Counter(),
            }
            if not path.exists():
                missing.append(f"{split}: 产物不存在（{path.name}）")
                report[split] = counts
                continue
            for _, row in jsonio.iter_jsonl(path):
                counts["samples"] += 1
                meetings = len(render.parse_state(row["state"]).meetings)
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`title_of_2` -> `title_of`）
                    counts["families"][family] += 1
                    counts["types"][question["type"]] += 1
                    if family not in known:
                        counts["unknown_families"][family] += 1
                    if question["type"] != "choice":
                        continue
                    options = list(question["criteria"])
                    answer = row["answers"][qid]["choice"]
                    if not 0 <= answer < len(options):
                        counts["index_bad"] += 1
                    if not _options_fit(family, len(options), meetings):
                        counts["options_bad"] += 1
            for family in questions.FAMILIES:
                if counts["families"][family] == 0:
                    missing.append(f"{split}: 题族 {family} 一条都没出")
            if counts["index_bad"]:
                missing.append(f"{split}: {counts['index_bad']} 条 choice 答案编号越界")
            if counts["options_bad"]:
                missing.append(
                    f"{split}: {counts['options_bad']} 条 choice 的选项个数与日程场数对不上"
                )
            if counts["unknown_families"]:
                missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
            report[split] = counts
        _ = locale  # 反解语言无关，不需要具体语言文件
        return report, missing

    def available_locales(self) -> list[str]:
        """输入：无。输出：支持的语言列表。"""
        return locales.available()

    def bind_backend(self, backend: GenerationBackend) -> None:
        """输入：生成后端。输出：无。预期行为：存下来，`render` 阶段用它写标题与议题。"""
        self.backend = backend


def _options_fit(family: str, options: int, meetings: int) -> bool:
    """输入：题族、选项个数、日程场数。输出：这个选项个数说得通吗。

    预期行为：按题族各自的构造规则核对——问「哪一场」的题族选项数 = 场数；
    `room_of` 的选项是**去重后**用到的会议室，所以是 2~场数；`duration_of` 恒为 4；
    `meeting_day` 恒为 7。
    """
    if family == "duration_of":
        return options == len(questions.DURATION_KEYS)
    if family == "meeting_day":
        return options == len(questions.WEEKDAY_KEYS)
    if family == "room_of":
        return 2 <= options <= meetings
    return options == meetings
