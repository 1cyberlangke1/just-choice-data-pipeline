"""命令审批 workflow：**特征注入式**（标签先行，模型只写命令文本）。

流程与邮件分诊完全一致（`plan → sample → render → ask`），区别只在「内容从哪来」：
- 待审命令由后端按注入特征写出来（真实分布的命令只有模型写得出来，模板拼的脱离现实）；
- 答案由**注入特征**经纯函数算出，模型碰不到；
- 正确性靠「可反解」：`judges.TraceJudge` 把命令交回后端重新读出目标位置，对不上就丢。

**本 workflow 只把命令当字符串读，绝不执行任何命令。**

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——删表、删容器两个动作与「远端主机」这个位置在训练侧一个都不出现。

导入本模块即自注册（`jc.workflow.load("shell")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final, cast

from jc.cover import hash_bucket, restricted_dims
from jc.judge import BACKEND, BARE, LOCALE
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register, sliced_cases

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import TagSet
from .spec import ShellSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "shell"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
OOD_GOALS: Final[frozenset[str]] = frozenset({"drop_table", "docker_remove"})
OOD_TARGETS: Final[frozenset[str]] = frozenset({"remote"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260928
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "dialect": "方言",
    "goal": "目标动作",
    "target": "目标位置",
    "elevation": "提权",
}


@dataclass(frozen=True)
class ShellConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——本 workflow 的空间有 300 组，这个值实际总是让枚举器
      走**成对覆盖**；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 32
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)


def _restricted(**overrides: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    return restricted_dims(features.DIMS, overrides)


def id_dims() -> dict[str, tuple[str, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        goal=tuple(v for v in features.GOALS if v not in OOD_GOALS),
        target=tuple(v for v in features.TARGETS if v not in OOD_TARGETS),
    )


def ood_dims() -> list[dict[str, tuple[str, ...]]]:
    """输入：无。输出：两组只含 OOD 取值的维度表（动作留出一组、位置留出一组）。

    预期行为：只留出目标那一维，**另一维必须跟着收窄到与它相容的取值**——
    删表只在生产上成立、删容器只在容器上成立，若把目标维换成「只留两个动作」却让
    位置维保留全部五个取值，覆盖器会在 `target=workdir` 上找不到任何一个可取的
    `goal`，直接抛「取值对约束太紧」。远端那一组同理。
    """
    goals_on_remote = tuple(
        goal for goal in features.GOALS if features.allow_pair("goal", goal, "target", "remote")
    )
    return [
        _restricted(goal=tuple(sorted(OOD_GOALS)), target=("container", "production")),
        _restricted(target=tuple(sorted(OOD_TARGETS)), goal=goals_on_remote),
    ]


def bucket_of(case: TagSet, seed: int = SPLIT_SEED) -> int:
    return hash_bucket(case, seed)


def cases_for(config: ShellConfig) -> list[TagSet]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    实现见 `jc.workflow.sliced_cases`（11 家逐字同构，只留那一份）。
    """
    return cast(
        "list[TagSet]",
        sliced_cases(config, features, ood_dims, id_dims, bucket_of, SPLIT_BUCKETS),
    )


class ShellWorkflow(BackendHolder):
    """命令审批：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 命令由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("dialect", "destructive", "elevated", "hygiene", "trace")
    #: 判别器名单表：名字 → （`judges` 里的类名，怎么造）。
    judge_table = {
        "dialect": ("DialectJudge", BARE),
        "destructive": ("DestructiveJudge", BARE),
        "elevated": ("ElevatedJudge", BARE),
        "hygiene": ("HygieneJudge", LOCALE),
        "trace": ("TraceJudge", BACKEND),
    }
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> ShellConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return ShellConfig(split=split)

    def case_label(self, case: TagSet) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: TagSet) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "dialect": case.dialect,
            "goal": case.goal,
            "target": case.target,
            "elevation": case.elevation,
        }

    def plan_sample(
        self, rng: random.Random, config: ShellConfig, case: TagSet, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool, key="tags")

    def sample(self, rng: random.Random, split: SplitSpec) -> ShellSpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return ShellSpec(
            tags=split.params["tags"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: ShellSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一条命令；后端没注入就报错——`needs` 非空时管线一定会注入，
        这里没有后端说明接线错了，绝不能悄悄降级成模板。
        """
        if self.backend is None:
            raise RuntimeError("shell workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.tags, loc, spec.render_task_id))
        render.check_draft(draft.data, loc)
        return render.to_state(draft.data, loc, spec.tags.dialect)

    def ask(self, spec: ShellSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        return questions.build(spec.tags, locales.load(locale), spec.phrasing)

    def reparse(self, state: object, *, case: object = None) -> render.ParsedCommand:
        """输入：渲染后的 state。输出：纯代码能反解的部分（命令那一行）。"""
        return render.parse_state(state)


register(WORKFLOW_NAME)(ShellWorkflow)
