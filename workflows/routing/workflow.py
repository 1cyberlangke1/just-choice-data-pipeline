"""工单分诊 workflow：**特征注入式**（属性行与三张表由代码写，工单正文由后端写）。

流程与工具调用门控（`workflows/toolgate`）同构（`plan → sample → render → ask`），区别在
「判什么」：

- **属性行、派单表、优先级表、附加规则是结构化事实**，由代码渲染进 state——让模型去写
  只会写歪；
- 后端只写**工单正文**（真实工单里那段话是长尾的，模板拼的脱离现实）；
- 标签由**注入特征**经纯函数算出，模型碰不到。六个题族里两个由那段正文承载
  （是什么问题 / 有多急），交给 `judges.TraceJudge` 反解校验——答不回来就说明文本与
  标签脱节了。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「产品域 = 身份」这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("routing")` 可用）。
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
from .spec import RoutingSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "routing"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是 `area=identity`：产品域**不参与任何一族的选项集**（六族分别是问题类型 / 紧急度 /
# 派单规则 / 队列 / 优先级 / 要不要人工），它只出现在 `needs_human` 的一条分支里，而那一条
# 还有「队列是事故响应组」这另一半撑着——所以留出它**不杀任何一档**，逐档枚举验过。
# 反过来，留出 `compliance=security` 会杀掉 `security_report` 规则与 `security_incident`
# 队列，留出 `urgency=deadline` 会杀掉优先级 3 档，留出 `tier=enterprise` 会杀掉优先级 3/1
# 两档——那三条都不能留。
OOD_AREAS: Final[frozenset[str]] = frozenset({"identity"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260929
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "issue": "这条工单是什么问题",
    "urgency": "有多急",
    "tier": "客户等级",
    "compliance": "合规标记",
    "area": "产品域",
}


@dataclass(frozen=True)
class RoutingConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 5×3×3×3×3 = 405 组、剔除 3 对禁用对后 324 组，
      1024 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；不带维度限制的全空间
      540 组剔除 3 对禁用对后 432 组，同样在 1024 之下、也走全枚举
      （`tests/workflows/routing/test_features.py` 用它验覆盖器）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 1024
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)


_restricted = restrictor(features.DIMS)


def id_dims() -> dict[str, tuple[str, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(area=tuple(v for v in features.AREAS if v not in OOD_AREAS))


def ood_dims() -> list[dict[str, tuple[str, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（产品域留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(area=tuple(sorted(OOD_AREAS)))]


def bucket_of(case: TagSet, seed: int = SPLIT_SEED) -> int:
    return hash_bucket(case, seed)


def cases_for(config: RoutingConfig) -> list[TagSet]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    实现见 `jc.workflow.sliced_cases`（11 家逐字同构，只留那一份）。
    """
    return cast(
        "list[TagSet]",
        sliced_cases(config, features, ood_dims, id_dims, bucket_of, SPLIT_BUCKETS),
    )


@register(WORKFLOW_NAME)
class RoutingWorkflow(BackendHolder):
    """工单分诊：正文、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 工单正文由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> RoutingConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return RoutingConfig(split=split)

    def case_label(self, case: TagSet) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: TagSet) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "issue": case.issue,
            "urgency": case.urgency,
            "tier": case.tier,
            "compliance": case.compliance,
            "area": case.area,
        }

    def plan_sample(
        self, rng: random.Random, config: RoutingConfig, case: TagSet, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool, key="tags")

    def sample(self, rng: random.Random, split: SplitSpec) -> RoutingSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return RoutingSpec(
            tags=split.params["tags"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: RoutingSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一段工单正文，再与代码渲染的属性行 / 三张表拼起来；
        后端没注入就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，
        绝不能悄悄降级成模板（那会让「正文由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("routing workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.tags, loc, spec.render_task_id))
        render.check_draft(draft.data, loc)
        return render.to_state(draft.data, spec.tags, loc)

    def ask(self, spec: RoutingSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.tags, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「工单正文」）。

        预期行为：只报模型写的那段正文；「工单属性」那一行与几张表里的取值词元由代码写，不报。
        """
        parsed = render.parse_state(sample.get("state"))
        return {"工单正文": parsed.body} if parsed.body.strip() else {}
