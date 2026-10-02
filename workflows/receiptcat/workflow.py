"""报销收据归类 workflow：**特征注入式**（三张表由代码写，一条收据行由后端写）。

流程与社交帖子分诊（`workflows/posttriage`）、内容审核（`workflows/guardrail`）同构
（`plan → sample → render → ask`），区别在「判什么」：

- **归类表、单笔上限表、审核档位表是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**一条收据行**（真实收据上的商户名与条目是长尾的，模板拼出来的脱离现实分布）；
- **归哪一类、金额落哪一档由代码按字面算**（`features.categories_hit` + `band_of`），所以标签
  是注入特征经纯函数算出来的（`features.plan_for` + `label_for`），模型碰不到；写出来的收据行
  必须兑现这份计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` / `FrameJudge`
  再各查一遍）。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`decoy=near`（收据行里埋一个「看着像某一类、但不在归类表里」的
  说法，按字面算不算命中）这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("receiptcat")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import hash_bucket, slice_by_bucket, restricted_dims
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import ReceiptCase
from .spec import ReceiptCatSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "receiptcat"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `decoy=near`：它是**只改一处渲染、却改了一族答案**的那一维——`near` 那一族的真值
# 直接就是「有没有诱饵」，而 `cat` / `over` / `tier` 三族一个字符都不受它影响。同分布侧一次都
# 见不到诱饵，模型必须真的按字面判（只认表里的词），而不是靠「这条看着像餐饮」蒙。
#
# 反过来，留出 `category` 会让 `cat` 那一族在同分布侧少一个选项、留出 `band` 会让 `tier` 少一档，
# 均衡闸与审计都会当场报出来。所以留出的是 `decoy`。
OOD_DECOYS: Final[frozenset[str]] = frozenset({features.DECOY_NEAR})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "category": "该归到哪一类",
    "band": "金额落在哪一档",
    "decoy": "要不要埋诱饵",
}


@dataclass(frozen=True)
class ReceiptCatConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 4×3×2 = 24 组，512 让枚举器稳稳走**全枚举**
      （成对覆盖在这里只会白扔组合）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 512
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1)


def _restricted(**overrides: tuple[JsonValue, ...]) -> dict[str, tuple[JsonValue, ...]]:
    """输入：要覆盖的维度。输出：覆盖后的维度表。预期行为：不改动 `features.DIMS`。"""
    return restricted_dims(features.DIMS, overrides)


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        decoy=tuple(value for value in features.DECOYS if value not in OOD_DECOYS)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（诱饵留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(decoy=tuple(value for value in features.DECOYS if value in OOD_DECOYS))
    ]


def bucket_of(case: ReceiptCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _dims_of(case: ReceiptCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多一个「审核档位」）。

    预期行为：`as_dict` 的三个维度之外再加 `tier`——小批量里最容易被忽略的正是它：只按
    `category` / `band` / `decoy` 铺开的话，10 条很可能一条「财务复核」都出不来，`tier`
    那一族的第四个档位取不到，均衡闸与审计都会当场报出来。
    """
    return {
        "category": case.category,
        "band": case.band,
        "decoy": case.decoy,
        "tier": features.tier_of(case.band, case.category),
    }


def _cover_pick(cases: list[ReceiptCase], limit: int) -> list[ReceiptCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：类别 → 金额档位 → 诱饵 → 审核档位），同分时
    取原顺序里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`category` 有 4 个取值、`band` 有 3 个、`tier` 有 4 个，纯截断的话 10 条
    的小批量很可能某一类一条都出不来，`cat` 那一族就少一个选项，均衡闸当场标红。覆盖优先把这种
    「小批量必然缺一档」的坑填掉，代价只是 case 顺序不再等于桶序（切分归属仍是桶算的，没变）。
    """
    names = ("category", "band", "decoy", "tier")
    chosen: list[ReceiptCase] = []
    seen: dict[str, set[str]] = {name: set() for name in names}
    remaining = list(cases)
    while remaining and len(chosen) < limit:
        keys = [
            (
                sum(1 for name in names if _dims_of(case)[name] not in seen[name]),
                -index,
            )
            for index, case in enumerate(remaining)
        ]
        best = max(range(len(remaining)), key=lambda index: keys[index])
        case = remaining.pop(best)
        chosen.append(case)
        for name in names:
            seen[name].add(_dims_of(case)[name])
    return chosen


def cases_for(config: ReceiptCatConfig) -> list[ReceiptCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], ReceiptCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()), config.split, bucket_of, SPLIT_BUCKETS
        )
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class ReceiptCatWorkflow(BackendHolder):
    """报销收据归类：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 收据行由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> ReceiptCatConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return ReceiptCatConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: ReceiptCatConfig, case: ReceiptCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> ReceiptCatSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, ReceiptCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return ReceiptCatSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
        )

    def render(self, spec: ReceiptCatSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那条收据行，先按注入特征校验（不合格抛错，管线据此重渲染），再与代码
        渲染的三张表拼起来；后端没注入就报错——`needs` 非空时管线一定会注入，这里没有后端说明
        接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("receiptcat workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        text = draft.data.get("text")
        if not isinstance(text, str):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.ContentDraft(text=text)
        render.check_draft(parsed, spec.case, loc)
        return render.to_state(parsed, loc)

    def ask(
        self, spec: ReceiptCatSpec, rng: random.Random, locale: str
    ) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.case, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「收据行」）。

        预期行为：只报模型写的那条收据行；三张表是注入的结构化事实（类别名 / 关键词 / 上限
        由语言表写），不报。
        """
        parsed = render.parse_state(sample.get("state"))
        text = parsed.text
        return {"收据行": text} if text and text.strip() else {}



@register(WORKFLOW_NAME)
def _factory() -> ReceiptCatWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return ReceiptCatWorkflow()
