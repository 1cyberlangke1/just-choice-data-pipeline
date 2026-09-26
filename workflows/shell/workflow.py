"""命令审批 workflow：**构造式**（命令与审批事实都由表格算出来，`needs` 为空、零 token）。

它在 PLAN 3.2 的分类里属于「抽取式」的近亲——素材（一条待执行的命令 + 上下文）由构造器
造好，答案是从素材**按策略推**出来的，所以判别器能把文本反解回来独立重算。

覆盖的决策形状（官方 use-case map）：
- **Routing**：`verdict`（放行 / 需确认 / 拒绝）；
- **Detection**：`destructive` / `network` / `elevated` / `touches_secrets` 这些属性判断；
- **Scoring**：`blast_radius`（影响范围 0~4 级）；
- **Verification**：`matched_goal`（命令与声明的意图是否一致）。

两种方言（bash / PowerShell）都在同分布侧出现，命令的合法性由
`tests/workflows/shell/test_dialects.py` 用真的 bash 与 PowerShell 解析器把全部命令过一遍。

切分按 PLAN 3.3：
- train / val / test_iid：**组合留出**——按 7 维组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`drop_table` / `docker_rm` / `git_push_force` 三种操作、
  `nested` / `chained` 两种写法、`terse` 一种句式在训练侧一个都不出现。

**关于 `verdict` 的分布**：只读命令只占 8/19，而 `allow` 只能是只读且与目标一致的那一支，
所以三档天然不均（`ask` 最多）。题族照样**打乱选项**——档位由选项描述承载、不靠位置承载，
打乱之后「答案位置」不再泄露标签（范式 §5.8）。均衡报告会如实标出这个偏斜，它是数据
本身的形状，不是缺陷。

导入本模块即自注册（`jc.workflow.load("shell")` 可用）。
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

from jc import jsonio
from jc.judge import Judge
from jc.workflow import LabeledQuestion, SplitSpec, family_of, register

from . import judges as judges_mod
from . import locales, parse, questions, render
from . import spec as spec_mod
from .spec import CommandSpec, ShellCase, ShellConfig

WORKFLOW_NAME = "shell"
WORKFLOW_VERSION = "0.1.0"

#: 同分布切分的哈希分桶边界（组合留出：桶按 case 的 7 维取值算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260926
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 训练侧取值：各维度只留 spec 里标了「见过」的那几档。
#: 排序是为了顺序稳定（frozenset 的迭代顺序随进程变化，不能直接拿来枚举）。
ID_CONFIG: Final[ShellConfig] = ShellConfig(
    split="train",
    ops=tuple(sorted(spec_mod.SEEN_OPS)),
    wrappers=tuple(sorted(spec_mod.SEEN_WRAPPERS)),
    goal_styles=tuple(sorted(spec_mod.SEEN_GOAL_STYLES)),
)
#: 留出档：取值表全开，靠 `ood_only` 只留「至少含一个没见过的取值」的 case。
OOD_CONFIG: Final[ShellConfig] = ShellConfig(split="test_ood", ood_only=True)

DIMENSION_TITLES: Final[dict[str, str]] = {
    "dialect": "方言",
    "op": "操作",
    "target": "命令操作的对象",
    "goal_target": "用户目标点名的对象",
    "wrapper": "执行写法",
    "goal_style": "目标句式",
    "reads": "已读文件条数",
}


def bucket_of(case: ShellCase, seed: int = SPLIT_SEED) -> int:
    """输入：case、种子。输出：0~99 的桶号。

    预期行为：同一组合必得同一桶，不同组合近似均匀——切分单位是**组合**，不是样本；
    不用内置 `hash()`（它对字符串每进程随机化），否则切分会随进程漂移。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[ShellCase], split: str) -> list[ShellCase]:
    """输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。

    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的——只按桶号过滤时，
    哈希方差会让 val 只剩个位数；同一组合永远落在同一段。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def cases_for(config: ShellConfig) -> list[ShellCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 直接收 `build_cases` 按 `ood_only` 筛出来的组合；
    其余切分按 7 维组合的哈希桶切片，同组合不跨切分。
    """
    cases = spec_mod.build_cases(config)
    if config.split == "test_ood":
        return cases
    return _slice(cases, config.split)


@register(WORKFLOW_NAME)
class ShellWorkflow:
    """命令审批：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 构造式：命令与答案都由表格算出来，不需要后端
    judges: tuple[str, ...] = ("roundtrip", "label")

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：支持的切分名。"""
        return tuple(SPLIT_BUCKETS) + ("test_ood",)

    def config_for(self, split: str) -> ShellConfig:
        """输入：切分名。输出：该切分的采样参数。

        预期行为：`test_ood` 用全取值表 + `ood_only`；其余切分用训练侧取值表，
        **并且把 `split` 换成调用方要的那一档**——`ID_CONFIG` 是个单例，直接返回它会让
        train/val/test_iid 三次都按同一档切片（切出来三份一模一样的数据，
        实测踩过：三个切分各 3225 条，一模一样）。未知切分直接报错。
        """
        if split == "test_ood":
            return OOD_CONFIG
        if split in SPLIT_BUCKETS:
            return replace(ID_CONFIG, split=split)
        raise KeyError(f"未知切分：{split!r}（可用：{self.split_names()}）")

    def build_cases(self, config: ShellConfig) -> list[ShellCase]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。"""
        return cases_for(config)

    def case_label(self, case: ShellCase) -> str:
        """输入：一个 case。输出：一行可读描述（报告与文件名用）。"""
        return case.describe()

    def stats_dimensions(self, case: ShellCase) -> dict[str, Any]:
        """输入：一个 case。输出：{维度名: 取值}（7 个维度全摊平）。"""
        return case.as_dict()

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: ShellConfig, case: ShellCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。

        预期行为：问法在这里抽（同一 case 的多条样本会用不同问法）；
        case 与序号一起放进 params，`sample` 阶段不再做任何随机决策。
        """
        return SplitSpec(
            name=config.split,
            params={"case": case, "phrasing": rng.choice(config.phrasings), "seq": seq},
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> CommandSpec:
        """输入：独立 RNG、生成计划。输出：真值规格（命令文本与目标都在这里定死）。"""
        return spec_mod.sample_spec(
            rng, split.params["case"], split.name, split.params["seq"], split.params["phrasing"]
        )

    def render(self, spec: CommandSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（纯字符串）。

        预期行为：**纯函数**，不用 RNG、不碰后端——同一份 spec + 同一份语言文件
        必得同一段文本，所以「两次生成逐字节一致」这条能直接查。
        """
        return render.build(spec, locales.load(locale))

    def ask(
        self, spec: CommandSpec, rng: random.Random, locale: str
    ) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec, locales.load(locale), spec.phrasing, rng)

    def reparse(self, state: Any, *, case: Any = None) -> parse.ParsedRequest:
        """输入：渲染后的 state。输出：反解结果。

        预期行为：`Workflow.reparse` 没有语言参数，所以用**合并标签表**做语言无关的反解
        （各语言的标签互不重叠，由 `tests/workflows/shell/test_locales.py` 锁住）。
        """
        return parse.parse_request(state, parse.merged_labels())

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

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, Any], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

        预期行为：只看产物本身——题族是否齐全、两种方言是否都出现、choice 的答案编号是否
        越界，以及**答案能不能由 state 重算出来**（把 `LabelJudge` 的判据搬到产物级，
        这样「老产物」也能事后查）。反解用合并标签表，所以不需要具体语言文件。
        """
        labels = parse.merged_labels()
        known = set(questions.FAMILIES)
        report: dict[str, Any] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, Any] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "dialects": Counter(),
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
                    parsed = parse.parse_request(row["state"], labels)
                    expected = judges_mod.recompute(parsed)
                except ValueError as exc:
                    counts["label_mismatch"] += 1
                    missing.append(f"{split}: 有一条命令反解不出来（{exc}）")
                    continue
                counts["dialects"][parsed.dialect] += 1
                for qid, question in row["questions"].items():
                    family = family_of(qid)  # 去掉同族后缀（`verdict_2` -> `verdict`）
                    counts["families"][family] += 1
                    counts["types"][question["type"]] += 1
                    if family not in known:
                        counts["unknown_families"][family] += 1
                    answer = row["answers"][qid]
                    if question["type"] == "choice":
                        options = list(question["criteria"])
                        index = answer["choice"]
                        if not 0 <= index < len(options):
                            counts["index_bad"] += 1
                            continue
                        got: Any = options[index]
                    elif question["type"] == "noul":
                        got = answer["noul"]
                    else:
                        got = answer["score"]
                    want = expected.get(family)
                    if want is not None and got != want:
                        counts["label_mismatch"] += 1
            for family in questions.FAMILIES:
                if counts["families"][family] == 0:
                    missing.append(f"{split}: 题族 {family} 一条都没出")
            for dialect in spec_mod.DIALECTS:
                if counts["dialects"][dialect] == 0:
                    missing.append(f"{split}: 方言 {dialect} 一条都没出（两种都要合成）")
            if counts["index_bad"]:
                missing.append(f"{split}: {counts['index_bad']} 条 choice 答案编号越界")
            if counts["label_mismatch"]:
                missing.append(f"{split}: {counts['label_mismatch']} 条答案与命令重算不一致")
            if counts["unknown_families"]:
                missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
            report[split] = counts
        _ = locale  # 反解用合并标签表（语言无关），不需要具体语言文件
        return report, missing

    def available_locales(self) -> list[str]:
        """输入：无。输出：支持的语言列表。"""
        return locales.available()

    def bind_backend(self, backend: Any) -> None:
        """空操作：命令审批是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("命令审批 workflow 是纯构造式的，不该被绑定生成后端")
