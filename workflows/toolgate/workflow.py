"""工具调用门控 workflow：**特征注入式**（策略表与状态行由代码写，那句调用意图由后端写）。

流程与完成声明核验（`workflows/verify`）同构（`plan → sample → render → ask`），区别在
「判什么」：

- **策略表与「策略与状态」那一行是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写一句**调用意图**（真实 agent 的调用意图是长尾的，模板拼的脱离现实）；
- 标签由**注入特征**经纯函数算出，模型碰不到。六个题族里只有 `tool`（要调哪个工具）与
  `payload`（动的是什么内容）由那句意图承载，交给 `judges.TraceJudge` 反解校验——
  答不回来就说明文本与标签脱节了。

与命令审批（`workflows/shell`）的分工：那个 workflow 判**一条命令**该不该放行（方言 /
动作 / 落点 / 提权），文本就是命令本身；这里判**一次工具调用**（工具 / 作用范围 /
内容性质 / 策略档位 / 审批状态），文本只是意图，命令与参数不在题面上。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「发消息」这个工具与「permissive」这个策略档位，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("toolgate")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final, cast

from jc.cover import hash_bucket, restricted_dims
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register, sliced_cases

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import TagSet
from .spec import ToolGateSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "toolgate"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是这两个：留出 `tool=send_message` 让「发消息」这类工具在训练侧一次都不出现；
# 留出 `policy=permissive` 让最松的档位整个没见过。
#
# 为什么只留出这两个：留出别的取值会**连带**杀掉别的族的好几档，同分布侧就出现死选项了。
# 都拿枚举逐档验过（见 `tests/workflows/toolgate/test_pipeline.py` 的不变量测试）：
# - 留出 `payload=harmful` → 策略表第一条 `harmful_payload` 再也命中不了（那条规则是
#   `rule` 族的一个选项），直接死一格；
# - 留出 `reach=external` → `external_under_strict` 与风险 3 档一起消失；
# - 留出 `policy=strict` → `external_under_strict` / `workspace_write_strict` 一起消失；
# - 留出 `approval=granted` → `overridden` 的答案在同分布侧恒为 false（均衡闸会当场标它）；
# - 留出 `tool` 的某个取值（`send_message` 除外也行）只杀掉它**自己那一族**的那一档，
#   但它会顺带把「策略表第 3 条」的命中面收窄——只有 `send_message` 收窄后仍留得下
#   另外两个会改动的工具，所以挑它；
# - `policy=permissive` 只影响状态行本身，不参与任何题族取值，留出它最干净。
OOD_TOOLS: Final[frozenset[str]] = frozenset({"send_message"})
OOD_POLICIES: Final[frozenset[str]] = frozenset({"permissive"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260928
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "tool": "要调的工具",
    "reach": "作用范围",
    "payload": "内容性质",
    "policy": "策略档位",
    "approval": "审批状态",
}


@dataclass(frozen=True)
class ToolGateConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 4×3×3×2×2 = 144 组、剔除禁用对后 100 组，
      256 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；不带维度限制的全空间
      270 组 > 256，仍走成对覆盖（`tests/workflows/toolgate/test_features.py` 用它验覆盖器）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 256
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)


def _restricted(**overrides: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    return restricted_dims(features.DIMS, overrides)


def id_dims() -> dict[str, tuple[str, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        tool=tuple(v for v in features.TOOLS if v not in OOD_TOOLS),
        policy=tuple(v for v in features.POLICIES if v not in OOD_POLICIES),
    )


def ood_dims() -> list[dict[str, tuple[str, ...]]]:
    """输入：无。输出：两组只含 OOD 取值的维度表（工具留出一组、策略档位留出一组）。

    预期行为：本 workflow 的禁用对发生在 `tool` 与 `reach` / `payload` 之间，留出
    `tool=send_message` 时那一维只剩一个取值，禁用对会自然把 `reach=workspace` 剔掉；
    返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第三个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(tool=tuple(sorted(OOD_TOOLS))),
        _restricted(policy=tuple(sorted(OOD_POLICIES))),
    ]


def bucket_of(case: TagSet, seed: int = SPLIT_SEED) -> int:
    return hash_bucket(case, seed)


def cases_for(config: ToolGateConfig) -> list[TagSet]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    实现见 `jc.workflow.sliced_cases`（11 家逐字同构，只留那一份）。
    """
    return cast(
        "list[TagSet]",
        sliced_cases(config, features, ood_dims, id_dims, bucket_of, SPLIT_BUCKETS),
    )


class ToolGateWorkflow(BackendHolder):
    """工具调用门控：意图、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 那句调用意图由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> ToolGateConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return ToolGateConfig(split=split)

    def case_label(self, case: TagSet) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: TagSet) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "tool": case.tool,
            "reach": case.reach,
            "payload": case.payload,
            "policy": case.policy,
            "approval": case.approval,
        }

    def plan_sample(
        self, rng: random.Random, config: ToolGateConfig, case: TagSet, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool, key="tags")

    def sample(self, rng: random.Random, split: SplitSpec) -> ToolGateSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return ToolGateSpec(
            tags=split.params["tags"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: ToolGateSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一句调用意图，再与代码渲染的策略表 / 状态行拼起来；后端没注入
        就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，绝不能悄悄
        降级成模板（那会让「意图由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("toolgate workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.tags, loc, spec.render_task_id))
        render.check_draft(draft.data, loc)
        return render.to_state(draft.data, spec.tags, loc)

    def ask(self, spec: ToolGateSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.tags, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「调用意图」）。

        预期行为：只报模型写的那句话；工具名与参数是标识符，不报。
        """
        parsed = render.parse_state(sample.get("state"))
        return {"调用意图": parsed.intent} if parsed.intent.strip() else {}



register(WORKFLOW_NAME)(ToolGateWorkflow)
