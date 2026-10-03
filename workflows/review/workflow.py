"""改动审查 workflow：**特征注入式**（清单行与三张表由代码写，改动正文由后端写）。

流程与工具调用门控（`workflows/toolgate`）同构（`plan → sample → render → ask`），区别在
「判什么」：

- **改动清单、审查规则表、严重度表、门控表是结构化事实**，由代码渲染进 state——让模型去写
  只会写歪；
- 后端只写**改动正文**（真实改动里那几行代码是长尾的，模板拼的脱离现实）；
- 标签由**注入特征**经纯函数算出，模型碰不到。五个题族里两个由那段正文承载
  （有没有凭据字面量 / 有没有调试输出），交给 `judges.TraceJudge` 反解校验——答不回来就
  说明代码与标签脱节了。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「变更类型 = 新增」这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("review")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final, cast

from jc.cover import hash_bucket, restrictor
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register, sliced_cases

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import TagSet
from .spec import ReviewSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "review"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是 `change_kind=added`：变更类型只进**门控表**那一列，而门控表的三档在「新增」这一列
# 上全都到得了（重 → block、中 → request_changes、轻 → allow），所以留出它**不杀任何一档**，
# 逐档枚举验过（`tests/workflows/review/test_pipeline.py`）。
#
# 反过来，留出 `secret=yes` 会杀掉凭据规则、留出 `debug=yes` 会杀掉调试规则、留出
# `size=large` 会杀掉规模规则与「轻」那一档、留出 `file_kind=docs` 会让整个 OOD 侧只剩
# 「不提」一种标签——那四个都不能留。
OOD_CHANGE_KINDS: Final[frozenset[str]] = frozenset({"added"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "secret": "改动的新增行里有没有凭据字面量",
    "debug": "改动的新增行里有没有调试输出",
    "file_kind": "文件类型",
    "change_kind": "变更类型",
    "size": "改动规模",
}


@dataclass(frozen=True)
class ReviewConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 2×2×4×2×3 = 96 组、剔除 2 对禁用对后 60 组，
      512 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；不带维度限制的全空间
      144 组剔除 2 对禁用对后 108 组，同样在 512 之下、也走全枚举
      （`tests/workflows/review/test_features.py` 用它验覆盖器）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 512
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)


_restricted = restrictor(features.DIMS)


def id_dims() -> dict[str, tuple[str, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        change_kind=tuple(v for v in features.CHANGE_KINDS if v not in OOD_CHANGE_KINDS)
    )


def ood_dims() -> list[dict[str, tuple[str, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（变更类型留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(change_kind=tuple(v for v in features.CHANGE_KINDS if v in OOD_CHANGE_KINDS))
    ]


def bucket_of(case: TagSet) -> int:
    return hash_bucket(case, SPLIT_SEED)


def cases_for(config: ReviewConfig) -> list[TagSet]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    实现见 `jc.workflow.sliced_cases`（11 家逐字同构，只留那一份）。
    """
    return cast(
        "list[TagSet]",
        sliced_cases(config, features, ood_dims, id_dims, bucket_of, SPLIT_BUCKETS),
    )


class ReviewWorkflow(BackendHolder):
    """改动审查：正文、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 改动正文由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> ReviewConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return ReviewConfig(split=split)

    def case_label(self, case: TagSet) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: TagSet) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "secret": case.secret,
            "debug": case.debug,
            "file_kind": case.file_kind,
            "change_kind": case.change_kind,
            "size": case.size,
        }

    def plan_sample(
        self, rng: random.Random, config: ReviewConfig, case: TagSet, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool, key="tags")

    def sample(self, rng: random.Random, split: SplitSpec) -> ReviewSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return ReviewSpec(
            tags=split.params["tags"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: ReviewSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一段改动正文，再与代码渲染的清单行 / 三张表拼起来；
        后端没注入就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，
        绝不能悄悄降级成模板（那会让「正文由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("review workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.tags, loc, spec.render_task_id))
        render.check_draft(draft.data, spec.tags, loc)
        return render.to_state(draft.data, spec.tags, loc)

    def ask(self, spec: ReviewSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.tags, locales.load(locale), spec.phrasing)


@register(WORKFLOW_NAME)
def _factory() -> ReviewWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return ReviewWorkflow()
