"""模型路由与降级 workflow：**特征注入式**（候选表与门槛由代码写，用户请求由后端写）。

流程与命令审批（`workflows/shell`）同构（`plan → sample → render → ask`），区别在
「谁写文本、谁定标签」：

- **候选目录与筛选门槛是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写一句**用户请求**（真实请求的语气与材料是长尾的，模板拼的脱离现实）；
- 标签由**注入特征**经纯函数算出，模型碰不到。五个题族里只有 `kind`（任务类型）由那句
  请求承载，交给 `judges.TraceJudge` 反解校验——答不回来就说明文本与标签脱节了。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「看图」这个任务类型在训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("modelroute")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import hash_bucket, slice_by_bucket
from jc.judge import Judge
from jc.jsontypes import JsonValue
from jc.knobs import require_pool
from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import TagSet
from .spec import RouteSpec

WORKFLOW_NAME = "modelroute"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。本 workflow 没有禁用对，所以留出某一维时不必像
# shell 那样把另一维收窄（对比 `workflows/shell` 的 OOD 收窄）。
#
# 为什么只留出这一个值：留出别的取值要么把同分布侧弄出**死选项**、要么把某一族压偏，
# 都拿枚举验过（见 `tests/workflows/modelroute/test_pipeline.py` 的两条不变量测试）：
# - 留出 `interactive` → `atlas-lite` 再也选不上（它只在「交互式 + borealis 挂了」下赢）；
# - 留出 `atlas_down` → 第一行的 `blocked` 永远是 `latency` / `budget` / `usable`，
#   `provider_down` 一次都不出现（只有 atlas 挂了才算它）；
# - 留出 `tight` → `blocked` 的 `budget` 同理消失；
# - 留出 `huge` → 窗口需求永远 ≤ 32k，`slack`（窗口有没有两倍富余）在同分布侧塌成
#   73% True；
# - 留出 `math_reason` → 唯一需要 3 档能力的任务没了，`slack` 同样塌成 63% True。
OOD_TASK_KINDS: Final[frozenset[str]] = frozenset({"vision"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260928
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "task_kind": "任务类型",
    "size": "材料规模",
    "latency": "时延要求",
    "budget": "预算档",
    "health": "提供方健康",
}


@dataclass(frozen=True)
class RouteConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间只有 5×3×2×2×3 = 180 组，180 <= 192 所以
      走**全枚举**（这里成对覆盖只会白扔组合）；不带维度限制的全空间 216 组 > 192，
      仍走成对覆盖（`tests/workflows/modelroute/test_features.py` 用它验覆盖器）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 192
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasing_pool", self.phrasing_pool)
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


def _restricted(**overrides: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    dims = dict(features.DIMS)
    dims.update(overrides)
    return dims


def id_dims() -> dict[str, tuple[str, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        task_kind=tuple(v for v in features.TASK_KINDS if v not in OOD_TASK_KINDS),
    )


def ood_dims() -> list[dict[str, tuple[str, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（任务类型）。

    预期行为：本 workflow 的维度之间**没有禁用对**，所以留出某一维时不需要像 shell 那样
    把另一维收窄；返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个
    留出维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(task_kind=tuple(sorted(OOD_TASK_KINDS)))]


def bucket_of(case: TagSet, seed: int = SPLIT_SEED) -> int:
    return hash_bucket(case, seed)


def cases_for(config: RouteConfig) -> list[TagSet]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的两组组合（并集去重）；其余切分先剔除留出取值，
    再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, str], ...], TagSet] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()), config.split, bucket_of, SPLIT_BUCKETS
        )
    return cases[: config.case_limit] if config.case_limit else cases


class ModelRouteWorkflow(BackendHolder):
    """模型路由与降级：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 用户请求由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> RouteConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return RouteConfig(split=split)

    def case_label(self, case: TagSet) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: TagSet) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "task_kind": case.task_kind,
            "size": case.size,
            "latency": case.latency,
            "budget": case.budget,
            "health": case.health,
        }

    def plan_sample(
        self, rng: random.Random, config: RouteConfig, case: TagSet, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool, key="tags")

    def sample(self, rng: random.Random, split: SplitSpec) -> RouteSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return RouteSpec(
            tags=split.params["tags"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: RouteSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一句用户请求，再与代码渲染的候选表 / 门槛行拼起来；后端没注入
        就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，绝不能悄悄
        降级成模板（那会让「请求由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("modelroute workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.tags, loc, spec.render_task_id))
        render.check_draft(draft.data, loc)
        return render.to_state(draft.data, spec.tags, loc)

    def ask(self, spec: RouteSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.tags, locales.load(locale), spec.phrasing)

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。预期行为：未知名字直接报错。"""
        out: list[Judge] = []
        for name in names:
            if name == "hygiene":
                out.append(judges_mod.HygieneJudge(locale))
            elif name == "frame":
                out.append(judges_mod.FrameJudge(locale))
            elif name == "trace":
                out.append(judges_mod.TraceJudge(self.backend, locale))
            else:
                raise KeyError(f"未知判别器：{name!r}（可用：hygiene / frame / trace）")
        return tuple(out)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「请求」）。

        预期行为：只报模型写的那句话；候选模型名是标识符（与语言无关），不报。
        """
        parsed = render.parse_state(sample.get("state"))
        return {"请求": parsed.request} if parsed.request.strip() else {}



register(WORKFLOW_NAME)(ModelRouteWorkflow)
