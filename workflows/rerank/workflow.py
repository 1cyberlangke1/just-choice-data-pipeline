"""检索重排 workflow：**特征注入式**（排序口径与候选编号由代码写，查询与候选原文由后端写）。

流程与打分评分（`workflows/rubric`）同构（`plan → sample → render → ask`），区别在「判什么」：

- **查询抬头、关键词抬头、候选抬头、排序口径是结构化事实**，由代码渲染进 state——让模型去
  写只会写歪；
- 后端只写**查询、关键词表与那 5 条候选原文**（真实检索结果里的句子是长尾的，模板拼的脱离
  现实）；
- 每条候选沾上几个关键词由**注入特征经纯函数**算出（`features.plan_for`），模型碰不到；写出
  来的候选必须兑现这份计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` 再查一遍）。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——「关键词个数 = 4」这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("rerank")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_pick, hash_bucket, slice_by_bucket, restrictor
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import RerankCase
from .spec import RerankSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "rerank"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么是「关键词个数 = 4」：它是**只影响渲染、不影响标签**的那一维——名次由覆盖率算，
# 而覆盖率的口径在 2 / 3 / 4 个词下是同一个，所以留出它不会让任何一族的答案在训练侧退化成
# 常量（逐档枚举验过，见 `tests/workflows/rerank/test_pipeline.py`）。
#
# 反过来，留出某个**相关候选条数**会让 `count` 少一格（0~4 缺一个），留出某个**位置段**会让
# `top` 少几个答案（第一名只能落在没被留出的那一段里）。这两类都不能留。
OOD_TERMS: Final[frozenset[int]] = frozenset({4})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260929
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "terms": "关键词个数",
    "relevant": "相关候选条数",
    "best": "最相关候选的位置",
}


@dataclass(frozen=True)
class RerankConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——同分布空间 2×5×3 = 30 组、全空间 3×5×3 = 45 组，
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
    return _restricted(terms=tuple(v for v in features.TERM_COUNTS if v not in OOD_TERMS))


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（关键词个数留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出
    维度时只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(terms=tuple(v for v in features.TERM_COUNTS if v in OOD_TERMS))]


def bucket_of(case: RerankCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _cover_pick(cases: list[RerankCase], limit: int) -> list[RerankCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：相关条数 → 关键词个数 → 位置段），同分时
    取原顺序里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`relevant` 有五个取值，而 `count` 的五个档位一一对应它——纯截断的话，
    10 条的小批量很可能一个 `n0`（一条都不相关）都出不来，均衡闸与审计都会当场报出来。
    覆盖优先把这种「小批量必然缺一档」的坑填掉，代价只是 case 顺序不再等于桶序（切分归属
    仍是桶算的，没变）。
    """
    names = ("relevant", "terms", "best")
    return cover_pick(cases, limit, names, lambda case: case.as_dict())


def cases_for(config: RerankConfig) -> list[RerankCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], RerankCase] = {}
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
class RerankWorkflow(BackendHolder):
    """检索重排：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 查询与候选原文由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> RerankConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return RerankConfig(split=split)

    def stats_dimensions(self, case: RerankCase) -> dict[str, JsonValue]:
        """输入：一个 case。输出：统计维度（一维一栏）。"""
        return {"terms": case.terms, "relevant": case.relevant, "best": case.best}

    def plan_sample(
        self, rng: random.Random, config: RerankConfig, case: RerankCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> RerankSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, RerankCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return RerankSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: RerankSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要查询、关键词表与那 5 条候选原文，先按名次计划校验（不合格抛错，管线
        据此重渲染），再与代码渲染的抬头 / 编号 / 排序口径拼起来；后端没注入就报错——`needs`
        非空时管线一定会注入，这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由
        模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("rerank workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        query = draft.data.get("query")
        terms = draft.data.get("terms")
        candidates = draft.data.get("candidates")
        if not isinstance(query, str) or not isinstance(terms, list) or not isinstance(candidates, list):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.RerankDraft(
            query=query,
            terms=[str(item) for item in terms],
            candidates=[str(item) for item in candidates],
        )
        render.check_draft(parsed, spec.plan, loc)
        return render.to_state(parsed, spec.plan, loc)

    def ask(self, spec: RerankSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（查询 + 每条候选）。

        预期行为：只报模型写的查询与候选原文；抬头标记、编号、排序口径都是代码按语言写的，不报。
        真机实测：en 侧出现过整段写成中文（汉字占比 0.577）、zh 侧出现过整段写成英文（0.081），
        这条判据就是为它们加的。
        """
        parsed = render.parse_state(sample.get("state"))
        texts: dict[str, str] = {}
        if parsed.query:
            texts["查询"] = parsed.query
        for index, candidate in enumerate(parsed.candidates, start=1):
            texts[f"第 {index} 条候选"] = candidate
        return texts
