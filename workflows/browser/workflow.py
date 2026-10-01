"""浏览器 / GUI 操作 workflow：**特征注入式**（元素表与两张判据表由代码写，那句话由后端写）。

流程与工具调用门控（`workflows/toolgate`）同构（`plan → sample → render → ask`），区别在
「判什么」：

- **界面元素表、门控表、意图表是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**用户这一轮说的那一句话**（真实 agent 里用户那句话是长尾的，模板拼的脱离现实）；
- 标签由**注入特征**经纯函数算出，模型碰不到。七个题族里只有意图那一族由那句话承载，
  交给 `judges.TraceJudge` 反解校验（连「指的是第几行、哪个元素」一起反解）——答不回来
  就说明文本与标签脱节了。

与工具调用门控（`workflows/toolgate`）的分工：那个判**一次工具调用该不该放行**；
这里判**一屏界面上下一步该动哪个元素、动什么、要不要拦一下**。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「界面 = 设置页」这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("browser")` 可用）。
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
from .spec import BrowserSpec

WORKFLOW_NAME = "browser"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是 `screen=settings`：`screen` 只决定**填充行的角色顺序**，它不出现在任何题族的
# 选项里、也不是任何判据的条件。都拿枚举逐档验过（见 `tests/workflows/browser/test_pipeline.py`
# 的不变量测试）：
# - 留出 `intent` / `state` / `sensitive` / `decoy` 的任何一档 → 直接杀掉对应那一族的一个
#   答案档（前四维分别是四族题的选项 / 答案）；
# - 留出 `screen=form` 或 `screen=list` → 填充行角色顺序变短，但 `form` 与 `list` 的四种
#   角色都齐（都是四角色的排列），所以还能凑齐；只有 `settings` 那一档**唯一不杀档**
#   （它同样是四角色排列，剔掉目标角色后仍有三种填充角色，标签池不变）。
OOD_SCREENS: Final[frozenset[str]] = frozenset({"settings"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260928
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "intent": "用户这一轮想干什么",
    "state": "目标元素的状态",
    "sensitive": "目标元素是否敏感",
    "decoy": "有没有同角色干扰元素",
    "screen": "这一屏是什么界面",
}


@dataclass(frozen=True)
class BrowserConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 4×4×2×2×2 = 128 组、全空间 192 组，
      256 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 256
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
    return _restricted(screen=tuple(v for v in features.SCREENS if v not in OOD_SCREENS))


def ood_dims() -> list[dict[str, tuple[str, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（界面留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(screen=tuple(sorted(OOD_SCREENS)))]


def bucket_of(case: TagSet, seed: int = SPLIT_SEED) -> int:
    return hash_bucket(case, seed)


def cases_for(config: BrowserConfig) -> list[TagSet]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
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


class BrowserWorkflow(BackendHolder):
    """浏览器 / GUI 操作：正文、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 那句话由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> BrowserConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return BrowserConfig(split=split)

    def case_label(self, case: TagSet) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: TagSet) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "intent": case.intent,
            "state": case.state,
            "sensitive": case.sensitive,
            "decoy": case.decoy,
            "screen": case.screen,
        }

    def plan_sample(
        self, rng: random.Random, config: BrowserConfig, case: TagSet, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool, key="tags")

    def sample(self, rng: random.Random, split: SplitSpec) -> BrowserSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return BrowserSpec(
            tags=split.params["tags"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: BrowserSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一句用户请求，再与代码渲染的元素表 / 门控表 / 意图表拼起来；
        后端没注入就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，
        绝不能悄悄降级成模板（那会让「那句话由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("browser workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.tags, loc, spec.render_task_id))
        render.check_draft(draft.data, spec.tags, loc)
        return render.to_state(draft.data, spec.tags, loc)

    def ask(self, spec: BrowserSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
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
        """输入：契约样本。输出：必须与声明语言一致的正文（「用户请求」）。

        预期行为：只报模型写的那句话。元素表 / 门控表 / 意图表里的名字**一律不报**——它们是
        `features.LABELS` 里的中文固定词表，两种语言共用（见 `locales/en.toml` 的说明），
        报进去会让每一条 en 产物都被判成写错了语言。
        """
        parsed = render.parse_state(sample.get("state"))
        return {"用户请求": parsed.request} if parsed.request.strip() else {}



@register(WORKFLOW_NAME)
def _factory() -> BrowserWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return BrowserWorkflow()
