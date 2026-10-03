"""完成声明核验 workflow：**特征注入式**（事实与门槛由代码写，那句声明由后端写）。

流程与模型路由（`workflows/modelroute`）同构（`plan → sample → render → ask`），区别在
「谁写文本、谁定标签」：

- **自报事实与验收门槛是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写一句**完成声明**（真实 agent 的收尾话术是长尾的，模板拼的脱离现实）；
- 标签由**注入特征**经纯函数算出，模型碰不到。六个题族里只有 `kind`（声称做完了什么）
  与 `tone`（口气）由那句声明承载，交给 `judges.TraceJudge` 反解校验——答不回来就说明
  文本与标签脱节了。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「新加了接口」这个声称、与「把话说满」这个口气，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("verify")` 可用）。
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
from .spec import VerifySpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "verify"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是这两个：留出 `claim_kind=api_added` 让「声称加了新接口」这一类收尾话术在训练侧
# 一次都不出现；留出 `tone=absolute` 让「把话说满」的口气整个没见过。
#
# 为什么只留出这两个：留出别的取值会**连带**杀掉别的族的好几档，同分布侧就出现死选项了。
# 都拿枚举逐档验过（见 `tests/workflows/verify/test_pipeline.py` 的不变量测试）：
# - 留出 `all_pass` → `verdict=verified`、`gap` 里的 tighten_scope / commit_the_change /
#   nothing_missing、`confidence=3` 一起消失；
# - 留出 `coverage=covered` → 同样那四档一起消失；
# - 留出 `not_run` → `run_the_checks` 消失（只有「没跑检查」才要「去跑检查」）；
# - 留出 `artifact=committed` → `nothing_missing` / `confidence=3` 消失，而且 noul 那一族
#   的答案在同分布侧恒为 false（均衡闸会当场标它）；
# - 留出 `claim_kind` 或 `tone` 的某个取值，只杀掉它**自己那一族**的那一档——所以 OOD
#   只留这两维。
OOD_CLAIM_KINDS: Final[frozenset[str]] = frozenset({"api_added"})
OOD_TONES: Final[frozenset[str]] = frozenset({"absolute"})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260928
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "claim_kind": "声称做完的事",
    "tone": "口气",
    "checks": "检查结果",
    "coverage": "覆盖程度",
    "artifact": "是否落盘",
}


@dataclass(frozen=True)
class VerifyConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 4×2×4×3×2 = 192 组、剔除禁用对后 160 组，
      256 让枚举器稳稳走**全枚举**（成对覆盖在这里只会白扔组合）；不带维度限制的全空间
      360 组 > 256，仍走成对覆盖（`tests/workflows/verify/test_features.py` 用它验覆盖器）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 256
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1, 2)


_restricted = restrictor(features.DIMS)


def id_dims() -> dict[str, tuple[str, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        claim_kind=tuple(v for v in features.CLAIM_KINDS if v not in OOD_CLAIM_KINDS),
        tone=tuple(v for v in features.TONES if v not in OOD_TONES),
    )


def ood_dims() -> list[dict[str, tuple[str, ...]]]:
    """输入：无。输出：两组只含 OOD 取值的维度表（声称留出一组、口气留出一组）。

    预期行为：本 workflow 的禁用对只发生在 `checks` 与 `coverage` 之间，留出这两维时
    不必像 shell 那样把另一维收窄；返回列表而不是单个表，是为了与其它 workflow 的形状
    一致（将来加第三个留出维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(claim_kind=tuple(sorted(OOD_CLAIM_KINDS))),
        _restricted(tone=tuple(sorted(OOD_TONES))),
    ]


def bucket_of(case: TagSet, seed: int = SPLIT_SEED) -> int:
    return hash_bucket(case, seed)


def cases_for(config: VerifyConfig) -> list[TagSet]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    实现见 `jc.workflow.sliced_cases`（11 家逐字同构，只留那一份）。
    """
    return cast(
        "list[TagSet]",
        sliced_cases(config, features, ood_dims, id_dims, bucket_of, SPLIT_BUCKETS),
    )


class VerifyWorkflow(BackendHolder):
    """完成声明核验：声明、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 那句声明由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> VerifyConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return VerifyConfig(split=split)

    def case_label(self, case: TagSet) -> str:
        """输入：一个 case。输出：一行可读描述。"""
        return " ".join(f"{key}={value}" for key, value in case.as_dict().items())

    def stats_dimensions(self, case: TagSet) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {
            "claim_kind": case.claim_kind,
            "tone": case.tone,
            "checks": case.checks,
            "coverage": case.coverage,
            "artifact": case.artifact,
        }

    def plan_sample(
        self, rng: random.Random, config: VerifyConfig, case: TagSet, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool, key="tags")

    def sample(self, rng: random.Random, split: SplitSpec) -> VerifySpec:
        """输入：独立 RNG、生成计划。输出：真值规格。"""
        return VerifySpec(
            tags=split.params["tags"],
            phrasing=split.params["phrasing"],
            split=split.name,
            seq=split.params["seq"],
        )

    def render(self, spec: VerifySpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要一句完成声明，再与代码渲染的事实行 / 门槛行拼起来；后端没注入
        就报错——`needs` 非空时管线一定会注入，这里没有后端说明接线错了，绝不能悄悄
        降级成模板（那会让「声明由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("verify workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec.tags, loc, spec.render_task_id))
        render.check_draft(draft.data, loc)
        return render.to_state(draft.data, spec.tags, loc)

    def ask(self, spec: VerifySpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.tags, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「声明」）。

        预期行为：只报模型写的那句声明；「自报事实」那一行由代码写，不报。
        """
        parsed = render.parse_state(sample.get("state"))
        return {"声明": parsed.claim} if parsed.claim.strip() else {}



register(WORKFLOW_NAME)(VerifyWorkflow)
