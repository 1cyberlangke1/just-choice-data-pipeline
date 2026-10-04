"""结构恢复 workflow：**结构由代码定、句子由模型写**。

对应官方 structure-recovery 那类用例：一份丢掉了排版标记、正文被硬换行折断的纯文本，让读者把
「内容块」的切分与类型读回来。流程与内容审核（`workflows/guardrail`）同构
（`plan → sample → render → ask`），区别在「判什么」：

- **块的顺序、类型、列表有没有序、提示框是哪一类**由 `features.build_plan` 定死——让模型写只会
  写歪，而且标签必须由代码算；
- 后端只写**每一块的文字**（一份备忘的正文是长尾的，模板拼出来的脱离现实分布）；
- 写出来的文字必须兑现这份计划：`render.check_draft` 把文本按渲染规则排好再**反解一遍**，块的
  类型 / 提示框种类 / 列表有序性必须与计划逐项相同，否则当场重渲染。

切分按 PLAN 3.3 的泛化留出：

- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——主题 `incident` / `migration`、行宽 `56`、变体 `tip`、命令族
  `container` 这四种取值训练侧一次都不出现（`features.SEEN_*` 是唯一的口径）。

导入本模块即自注册（`jc.workflow.load("autoformat")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_pick, hash_bucket, slice_by_bucket
from jc.judge import BACKEND, LOCALE
from jc.jsontypes import JsonValue
from jc.workflow import (
    BackendHolder,
    DraftError,
    LabeledQuestion,
    SplitSpec,
    plan_with_phrasing,
    register,
)

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import DocCase, DocConfig
from .spec import DocSpec

WORKFLOW_NAME = "autoformat"
WORKFLOW_VERSION = "0.1.0"

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260929
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "topic": "主题",
    "wrap": "行宽",
    "shape": "文档形状",
    "variant": "变体",
    "code_family": "命令族",
}


@dataclass(frozen=True)
class AutoformatConfig(DocConfig):
    """一个切分的采样参数：六张取值表 + 全枚举上限 + 小规模上限。

    字段：`cover_limit` 全枚举上限——全空间是「主题 × 行宽 × 形状 × 变体 × 命令族」，
    512 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；`case_limit` >0 时每个切分只取
    前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）。
    """

    cover_limit: int = 512
    case_limit: int = 0

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话。"""
        super().__post_init__()
        if self.cover_limit < 1:
            raise ValueError(f"cover_limit 必须 >= 1：{self.cover_limit}")
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


def bucket_of(case: DocCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _dims_of(case: DocCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多一个「块数」）。

    预期行为：`as_dict` 的五个维度之外再加 `blocks`（这份文档有多少个块）——小批量里最容易被忽略
    的正是它：只按形状铺开的话，10 条会全落在 `mixed` 上，`join` 那一族的题量当场偏到一边。
    """
    dims = {key: str(value) for key, value in case.as_dict().items()}
    dims["blocks"] = str(len(features.build_plan(case).slots))
    return dims


def _cover_pick(cases: list[DocCase], limit: int) -> list[DocCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：形状 → 变体 → 命令族 → 主题 → 行宽 → 块数），
    同分时取原顺序里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：全组合里 `mixed` 形状占多数，纯截断的话 10 条的小批量很可能一个 `quote`
    或 `callout` 形状都出不来，`callout_kind` 那一族当场「一条都没出」，审计会报出来。
    """
    names = ("shape", "variant", "code_family", "topic", "wrap", "blocks")
    return cover_pick(cases, limit, names, _dims_of)


def cases_for(config: AutoformatConfig) -> list[DocCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 收「至少含一个训练侧没见过的取值」的组合（并集去重）；其余切分先剔除
    留出取值（`ood_only=False` 的构造本身就只从见过的那几档里取），再按哈希桶分。
    """
    if config.split == "test_ood":
        cases = features.build_cases(config)
        cases = [case for case in cases if case.unseen_dims()]
    else:
        cases = [case for case in features.build_cases(config) if not case.unseen_dims()]
        cases = slice_by_bucket(cases, config.split, bucket_of, SPLIT_BUCKETS)
    if config.case_limit:
        cases = _cover_pick(cases, config.case_limit)
    return cases


class AutoformatWorkflow(BackendHolder):
    """结构恢复：纯文本的块结构由代码定，正文由后端写。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 每一块的文字由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "roundtrip", "trace")
    #: 判别器名单表：名字 → （`judges` 里的类名，怎么造）。
    judge_table = {
        "hygiene": ("HygieneJudge", LOCALE),
        "roundtrip": ("RoundtripJudge", LOCALE),
        "trace": ("TraceJudge", BACKEND),
    }
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> AutoformatConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return AutoformatConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: AutoformatConfig, case: DocCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> DocSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, DocCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return DocSpec(
            case=case,
            split=split.name,
            seq=int(split.params["seq"]),
            phrasing=phrasing,
            plan=features.build_plan(case),
        )

    def render(self, spec: DocSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要每一块的文字，先按块计划校验，再排成「丢掉了标记、硬换行折过」的
        纯文本。**校验失败时带着失败理由原地重写一次**，仍不合格才抛错（管线那层会再重渲染）。
        校验通过后把文字**回填**进 spec——「并段」那一族的题面要引用行号，而行号由折行结果决定。

        为什么在 workflow 里再重写一次：`check_draft` 拦下来的大多是**字数**（真机
        `af-cnt-b-zh` 那 5 条 `E_ASSEMBLE` 全是「长度 58 / 55 / 73 不在 74~240」），而
        「至少 74 个字」对模型是**计数任务**——管线那层重渲染用的是**同一份提示词**，等于把骰子
        再掷一次；把「第 10 块只有 58 个字」这个确定性信号回灌，才修得掉。
        """
        del rng
        loc = locales.load(locale)
        texts = self._draft(spec, loc, None)
        try:
            render.check_draft(spec.plan, texts, loc)
        except DraftError as exc:
            texts = self._draft(spec, loc, str(exc))
            render.check_draft(spec.plan, texts, loc)
        spec.texts = texts
        return render.render_state(spec.plan, texts, loc)

    def _draft(self, spec: DocSpec, loc: dict[str, JsonValue], feedback: str | None) -> list[str]:
        """输入：规格、语言字典、可选的上一次失败理由。输出：各块文字。

        预期行为：后端没注入就报错：`needs` 非空时管线一定会注入，这里没有后端说明接线错了，
        绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）；`texts` 不是数组也报错，
        不猜。
        """
        if self.backend is None:
            raise RuntimeError("autoformat workflow 需要生成后端，但还没被 bind_backend 注入")
        draft = self.backend.generate(render.build_task(spec, loc, feedback))
        raw = draft.data.get("texts")
        if not isinstance(raw, list):
            raise ValueError(f"渲染草稿缺 texts 字段：{draft.data!r}")
        return [str(item) for item in raw]

    def ask(self, spec: DocSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec, locales.load(locale), spec.phrasing)

    def reparse(self, state: object, *, case: object = None) -> render.parse.ParsedDoc:
        """输入：渲染后的 state。输出：纯代码能反解的部分（块结构）。"""
        del case
        return render.parse_state(str(state))

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（块序号 → 块文字）。

        预期行为：只报**散文块**——代码块里是命令行（工具名与选项，与语言无关），报进去会把
        「一份以代码为主的文档」误判成写错了语言。反解不出来（坏 state）给空表：那由契约校验报。
        """
        doc = render.parse_state(str(sample.get("state", "")))
        return {
            f"第 {index} 块": block.text
            for index, block in enumerate(doc.blocks, start=1)
            if block.kind != features.KIND_CODE and block.text.strip()
        }



@register(WORKFLOW_NAME)
def _factory() -> AutoformatWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return AutoformatWorkflow()
