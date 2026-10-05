"""行级规则检查 workflow：**特征注入式**（规则表与处置办法由代码写，片段原文由后端写）。

流程与打分评分（`workflows/rubric`）同构（`plan → sample → render → ask`），区别在「判什么」：

- **片段抬头、规则名与判据、处置办法表是结构化事实**，由代码渲染进 state——让模型去写只会
  写歪；
- 后端只写**那 8 行原文**（真实代码 / 日志 / 文档里的行是长尾的，模板拼的脱离现实）；
- 哪几行命中由**注入特征经纯函数**算出（`features.plan_for`），模型碰不到；写出来的行必须
  兑现这份计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` 再查一遍）。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「体裁 = 文档」这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("lint")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_pick, hash_bucket, restrictor, slice_by_bucket
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import LintCase
from .spec import LintSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "lint"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是 `genre=doc`：训练侧只见过源码与日志两种体裁，OOD 考的就是「同一套规则判据换到
# 文档体裁还认不认」。而三条体裁的**答案分布完全一样**（命中行数仍取 0~8 均匀、规则表仍是
# 那五条），所以留出它不会让任何一族的答案在训练侧退化成常量——逐档枚举验过
# （`tests/workflows/lint/test_pipeline.py`）。
#
# 反过来，留出某条**规则**（比如 `long`）会把「按模式判命中」与「按长度判命中」劈成两半，
# 训练侧再也见不到长度判据那一类，`line` 族就少了一整种考法；留出某个**命中行数**更不行，
# `count` 的 9 个取值会缺一格。这两类都不能留。
OOD_GENRES: Final[frozenset[str]] = frozenset({"doc"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260929
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "rule": "查哪条规则",
    "genre": "片段体裁",
    "hits": "命中行数",
}


@dataclass(frozen=True)
class LintConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 5×2×9 = 90 组、全空间 5×3×9 = 135 组，
      512 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；
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
    return _restricted(genre=tuple(v for v in features.GENRES if v not in OOD_GENRES))


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（体裁留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(genre=tuple(v for v in features.GENRES if v in OOD_GENRES))]


def bucket_of(case: LintCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _cover_pick(cases: list[LintCase], limit: int) -> list[LintCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：命中行数 → 规则 → 体裁），同分时取原顺序
    里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`hits` 有九个取值，而 `spot` 只在「恰好一行命中」时才出——纯截断的
    话，10 条的小批量有约三成概率一条 `spot` 都出不来，审计会当场报「题族 spot 一条都没出」。
    覆盖优先把这种「小批量必然缺一族」的坑填掉，代价只是 case 顺序不再等于桶序（切分归属
    仍是桶算的，没变）。
    """
    names = ("hits", "rule", "genre")
    return cover_pick(cases, limit, names, lambda case: case.as_dict())


def cases_for(config: LintConfig) -> list[LintCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], LintCase] = {}
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
class LintWorkflow(BackendHolder):
    """行级规则检查：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 片段原文由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> LintConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return LintConfig(split=split)

    def stats_dimensions(self, case: LintCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "rule": case.rule,
            "genre": case.genre,
            "hits": case.hits,
        }

    def plan_sample(
        self, rng: random.Random, config: LintConfig, case: LintCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> LintSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, LintCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return LintSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: LintSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那 8 行原文，先按行计划校验（不合格抛错，管线据此重渲染），再与
        代码渲染的抬头 / 判据 / 处置办法表拼起来；后端没注入就报错——`needs` 非空时管线一定
        会注入，这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「片段由模型写」这个
        前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("lint workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        raw = draft.data.get("lines")
        if not isinstance(raw, list):
            raise ValueError(f"渲染草稿里没有 lines 字段：{raw!r}")
        lines = [str(item) for item in raw]
        render.check_draft(lines, spec.plan)
        return render.to_state(lines, spec.plan, loc)

    def ask(self, spec: LintSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)


