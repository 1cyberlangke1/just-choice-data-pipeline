"""邮件安全分诊 workflow：**特征注入式**（标签先行，模型只写文本外壳）。

流程与迷宫完全一致（`plan → sample → render → ask`），区别只在「答案从哪来」：
- 迷宫的答案由构造算法保证；
- 本 workflow 的答案由**注入特征**经纯函数算出，文本是后端按特征清单后写的。

所以这里的正确性靠「可反解」：`judges.TraceJudge` 把写好的邮件交回后端重新读出特征，
对不上就丢。模型碰不到标签，标签也不会被文本带偏。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`curiosity` 手法与短链接在训练侧一个都不出现。

导入本模块即自注册（`jc.workflow.load("mail")` 可用）。
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from jc import jsonio
from jc.judge import Judge
from jc.workflow import LabeledQuestion, SplitSpec, register

from . import features, judges as judges_mod, locales, questions, render
from .features import TagSet
from .spec import MailSpec

WORKFLOW_NAME = "mail"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
OOD_LURES: Final[frozenset[str]] = frozenset({"curiosity"})
OOD_LINK_TARGETS: Final[frozenset[str]] = frozenset({"shortener"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260925
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "sender": "发件人身份",
    "lure": "手法",
    "ask": "索取",
    "link_target": "链接域名",
    "anchor": "链接文字",
    "tone": "语气",
}


@dataclass(frozen=True)
class MailConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——mail 的空间有几千组，这个值实际总是让枚举器走**成对覆盖**；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 32
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)


def _restricted(**overrides: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    dims = dict(features.DIMS)
    dims.update(overrides)
    return dims


def _without_ood(values: tuple[str, ...], dropped: frozenset[str], *, by_target: bool) -> tuple[str, ...]:
    """输入：取值序列、要丢掉的集合、是否按域名形态比对。输出：剩下的取值。"""
    if by_target:
        return tuple(v for v in values if features.split_link(v)[0] not in dropped)
    return tuple(v for v in values if v not in dropped)


def _only_targets(values: tuple[str, ...], targets: frozenset[str]) -> tuple[str, ...]:
    """输入：链接维取值、要保留的域名形态。输出：只含这些形态的取值。

    预期行为：`none` 的形态是 `none`，不在 `targets` 里就一并丢掉——
    留出短链接时不能顺手把「没有链接」也当成留出取值。
    """
    return tuple(v for v in values if features.split_link(v)[0] in targets)


def id_dims() -> dict[str, tuple[str, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        lure=_without_ood(features.LURE, OOD_LURES, by_target=False),
        link=_without_ood(features.LINK, OOD_LINK_TARGETS, by_target=True),
    )


def ood_dims() -> list[dict[str, tuple[str, ...]]]:
    """输入：无。输出：两组只含 OOD 取值的维度表（手法留出一组、链接留出一组）。"""
    return [
        _restricted(lure=tuple(sorted(OOD_LURES))),
        _restricted(link=_only_targets(features.LINK, OOD_LINK_TARGETS)),
    ]


def bucket_of(case: TagSet, seed: int = SPLIT_SEED) -> int:
    """输入：tag 组合、种子。输出：0~99 的桶号。

    预期行为：同一组合必得同一桶，不同组合近似均匀——切分单位是**组合**，不是样本。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def cases_for(config: MailConfig) -> list[TagSet]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的两组组合；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, str], ...], TagSet] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(
            features.build_cases(config.cover_limit, dims=id_dims()), config.split
        )
    return cases[: config.case_limit] if config.case_limit else cases


def _slice(cases: list[TagSet], split: str) -> list[TagSet]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的——只按桶号过滤时，
    48 个组合的哈希方差会让 val 只剩个位数；同一组合永远落在同一段。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


class MailWorkflow:
    """邮件安全分诊：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 文本外壳由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("link", "hygiene", "trace")

    def __init__(self) -> None:
        """输入：无。输出：实例。预期行为：后端要等 `bind_backend` 才注入。"""
        self.backend: Any = None

    def bind_backend(self, backend: Any) -> None:
        """输入：生成后端。输出：无。预期行为：存下后端，`render` 与 `trace` 判别器要用。"""
        self.backend = backend

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：支持的切分名。"""
        return ("train", "val", "test_iid", "test_ood")

    def config_for(self, split: str) -> MailConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return MailConfig(split=split)

    def build_cases(self, config: MailConfig) -> list[TagSet]:
        """输入：采样参数。输出：tag 组合列表。"""
        return cases_for(config)

    def case_label(self, case: TagSet) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: TagSet) -> dict[str, Any]:
        """输入：一个 case。输出：统计维度（把链接维拆成域名与链接文字两栏更好读）。"""
        return {
            "sender": case.sender,
            "lure": case.lure,
            "ask": case.ask,
            "link_target": case.link_target,
            "anchor": case.anchor,
            "tone": case.tone,
        }

    def dimension_titles(self) -> dict[str, str]:
        """输入：无。输出：{维度名: 中文标题}。"""
        return dict(DIMENSION_TITLES)

    def plan_sample(
        self, rng: random.Random, config: MailConfig, case: TagSet, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。"""
        return SplitSpec(
            name=config.split,
            params={
                "tags": case,
                "phrasing": rng.choice(config.phrasing_pool),
                "seq": seq,
            },
        )

    def sample(self, rng: random.Random, split: SplitSpec) -> MailSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return MailSpec(
            tags=split.params["tags"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: MailSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一封邮件；后端没注入就报错——`needs` 非空时管线一定会注入，
        这里没有后端说明接线错了，绝不能悄悄降级成模板。
        """
        if self.backend is None:
            raise RuntimeError("mail workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.tags, loc, spec.render_task_id))
        render.check_official_domain(spec.tags, draft.data)
        return render.to_state(draft.data, loc)

    def ask(self, spec: MailSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec.tags, locales.load(locale), spec.phrasing)

    def reparse(self, state: Any, *, case: Any = None) -> render.ParsedMail:
        """输入：渲染后的 state。输出：纯代码能反解的部分（有没有链接）。"""
        return render.parse_state(state)

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。预期行为：未知名字直接报错。"""
        out: list[Judge] = []
        for name in names:
            if name == "link":
                out.append(judges_mod.LinkJudge())
            elif name == "hygiene":
                out.append(judges_mod.HygieneJudge(locale))
            elif name == "trace":
                out.append(judges_mod.TraceJudge(self.backend, locale))
            else:
                raise KeyError(f"未知判别器：{name!r}（可用：link / hygiene / trace）")
        return tuple(out)

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, Any], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

        预期行为：只读产物本身能查的东西——题族是否齐全、题型分布、
        以及**链接有无与 `link_text` 这道题是否自洽**（这条不需要 spec 就能查）。
        """
        known = set(questions.FAMILIES)
        report: dict[str, Any] = {}
        missing: list[str] = []
        for split in splits:
            path = run_dir / f"{split}.jsonl"
            counts: dict[str, Any] = {
                "samples": 0,
                "families": Counter(),
                "types": Counter(),
                "link_ok": 0,
                "link_bad": 0,
                "unknown_families": Counter(),
            }
            if not path.exists():
                missing.append(f"{split}: 产物不存在（{path.name}）")
                report[split] = counts
                continue
            for _, row in jsonio.iter_jsonl(path):
                counts["samples"] += 1
                has_url = render.parse_state(row["state"]).has_url
                for qid, question in row["questions"].items():
                    family = "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")
                    counts["families"][family] += 1
                    counts["types"][question["type"]] += 1
                    if family not in known:
                        counts["unknown_families"][family] += 1
                    if family == "link_text":
                        # 有链接的邮件才该出这道题；出了就说明文本里必须有链接
                        counts["link_ok" if has_url else "link_bad"] += 1
            for family in questions.FAMILIES:
                if counts["families"][family] == 0:
                    missing.append(f"{split}: 题族 {family} 一条都没出")
            if counts["link_bad"]:
                missing.append(f"{split}: {counts['link_bad']} 条 link_text 答案与文本链接有无矛盾")
            if counts["unknown_families"]:
                missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
            report[split] = counts
        _ = locale  # 本 workflow 的审计只用产物本身，不需要语言文件
        return report, missing

    def available_locales(self) -> list[str]:
        """输入：无。输出：支持的语言列表。"""
        return locales.available()


register(WORKFLOW_NAME)(MailWorkflow)
