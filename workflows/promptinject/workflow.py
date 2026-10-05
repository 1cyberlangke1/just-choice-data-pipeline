"""提示注入 workflow：**特征注入式**（判据表与阈值表由代码写，不可信内容由后端写）。

流程与内容审核（`workflows/guardrail`）、检索重排（`workflows/rerank`）同构
（`plan → sample → render → ask`），区别在「判什么」：

- **注入判据表、处置阈值表、内容抬头是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**不可信内容**（一段网页正文 / 工具返回值 / 检索片段 / 文件内容读起来是长尾的，模板
  拼出来的脱离现实分布）；
- **命中哪几条由代码按字面算**（`features.PATTERNS` 的四条正则），所以标签是注入特征经纯函数
  算出来的（`features.plan_for` + `label_for`），模型碰不到；写出来的内容必须兑现这份计划
  （`render.check_draft` 在渲染期查、`judges.HygieneJudge` / `FrameJudge` 再各查一遍）。

**与 `guardrail` 的区别**：guardrail 判的是「一条用户发的帖子有没有踩政策条款」（联系方式 /
外链 / @ / 情绪化喊话，判据是**字符形状**）；本 workflow 判的是「Agent 读进来的一段外部内容
有没有在**试图操纵它**」（覆盖指令 / 伪造角色 / 外送数据 / 诱导执行，判据是**操纵短语的形状**）。
两者的判据空间、素材来源与处置语义都不一样，不重复。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`source=file`（素材是一段文件内容）这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("promptinject")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_pick, hash_bucket, slice_by_bucket, restrictor
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import PromptInjectCase
from .spec import PromptInjectSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "promptinject"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `source=file`：它是**只影响渲染、不影响标签**的那一维——标签由 `hits` 算
# （`source` 一个字符都不参与），素材来源只决定这段内容长成什么样（网页正文 / 工具返回值 /
# 文件内容）。同分布侧一次都见不到它，模型必须真的按字面判，而不是靠「这种来源里出现 X 就报警」
# 的先验蒙。
#
# 反过来，留出 `hits` 会让 `count` / `combo` / `action` 三族各少几档（16 组少一半），那些档位在
# 同分布侧就取不到了——均衡闸与审计都会当场报出来。所以留出的是 `source`。
OOD_SOURCES: Final[frozenset[str]] = frozenset({features.SOURCE_FILE})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "hits": "该命中哪几条判据",
    "source": "素材是哪一种来源",
}


@dataclass(frozen=True)
class PromptInjectConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 16×2 = 32 组，512 让枚举器稳稳走**全枚举**
      （成对覆盖在这里只会白扔组合）；
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
    return _restricted(
        source=tuple(value for value in features.SOURCES if value not in OOD_SOURCES)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：一组只含 OOD 取值的维度表（素材来源留出）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(source=tuple(value for value in features.SOURCES if value in OOD_SOURCES))
    ]


def bucket_of(case: PromptInjectCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _dims_of(case: PromptInjectCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多一个「命中条数」）。

    预期行为：`as_dict` 的两个维度之外再加 `count`（命中条数）——小批量里最容易被忽略的正是它：
    只按 `hits` / `source` 铺开的话，10 条会落在「0 条命中 2 条、1 条命中 8 条」上，`count` 与
    `action` 两族的答案分布当场偏到 80%，均衡闸标红。
    """
    return {"hits": case.hits, "source": case.source, "count": str(len(case.tags()))}


def _cover_pick(cases: list[PromptInjectCase], limit: int) -> list[PromptInjectCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：命中组合 → 素材来源 → 命中条数），同分时取原顺序里
    靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`hits` 有 16 个取值、`count` 有 5 个，纯截断的话 10 条的小批量很可能一条
    「4 条全中」都出不来，`combo` 与 `action` 的某些档位也取不到，均衡闸与审计都会当场报出来。
    覆盖优先把这种「小批量必然缺一档」的坑填掉，代价只是 case 顺序不再等于桶序（切分归属仍是桶
    算的，没变）。
    """
    names = ("hits", "source", "count")
    return cover_pick(cases, limit, names, _dims_of)


def cases_for(config: PromptInjectConfig) -> list[PromptInjectCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], PromptInjectCase] = {}
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
class PromptInjectWorkflow(BackendHolder):
    """提示注入：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 不可信内容由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> PromptInjectConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return PromptInjectConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: PromptInjectConfig, case: PromptInjectCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> PromptInjectSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, PromptInjectCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return PromptInjectSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: PromptInjectSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那段不可信内容，先按注入特征校验（不合格抛错，管线据此重渲染），再与代码
        渲染的判据表 / 阈值表 / 抬头拼起来；后端没注入就报错——`needs` 非空时管线一定会注入，
        这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("promptinject workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        text = draft.data.get("text")
        if not isinstance(text, str):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        parsed = render.ContentDraft(text=text)
        render.check_draft(parsed, spec.case)
        return render.to_state(parsed, loc)

    def ask(self, spec: PromptInjectSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「不可信内容」）。

        预期行为：只报模型写的那段不可信内容；`tags` 是注入判据的编号（由语言表写），不报。
        """
        parsed = render.parse_state(sample.get("state"))
        text = parsed.text
        return {"不可信内容": text} if text and text.strip() else {}
