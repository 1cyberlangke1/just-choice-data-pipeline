"""文档包切分 workflow：**特征注入式**（种类表、切分规则与页序抬头由代码写，页面正文由后端写）。

流程与日志行分诊（`workflows/logtriage`）、提示注入（`workflows/promptinject`）同构
（`plan → sample → render → ask`），区别在「判什么」：

- **种类表、切分规则、页序抬头是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**每一页的正文**（申请表、信函、收据、合同读起来都是长尾的，模板拼出来的脱离现实分布）；
- **每一页属于哪一类由代码按字面算**（`features.PATTERNS` 的四条正则），所以标签是注入特征经
  纯函数算出来的（`features.plan_for` + `label_for`），模型碰不到；写出来的正文必须兑现这份计划
  （`render.check_draft` 在渲染期查、`judges.HygieneJudge` / `FrameJudge` 再各查一遍）。

**为什么要这一族**：社区里 `tax-doc-classifier`（★314）拿 Jev 从一页 PDF 的文本里判它属于哪一类
表格与哪一类页面；`docjev`（★307）把一包文档切成一份份文档并给出有序的页区间。共同点是一叠纸
进来、要一个**立刻能落地**的分堆结论（哪几类、几份、从哪切），而不是让人把每页从头读到尾。

**与 `logtriage` 的区别**：`logtriage` 判的是一条日志行该走哪一档处置，材料是一行；本 workflow
判的是**一包多页材料**的构成与切分（逐页归类 → 类集合 → 份数），材料是多页、标签口径是切分规则。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 tag 组合的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`source=photo`（手机拍照）这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("segmentation")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_pick, hash_bucket, slice_by_bucket, restrictor
from jc.judge import BACKEND, LOCALE
from jc.jsontypes import JsonValue
from jc.knobs import require_pool
from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import SegmentationCase
from .spec import SegmentationSpec

WORKFLOW_NAME = "segmentation"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `source=photo`：它是**只影响渲染、不影响标签**的那一维——标签由页面的抬头算
# （`source` 一个字符都不参与），来源只决定这一包读起来像什么（扫描仪的页码戳 / 传真机的抬头 /
# 手机拍照的文件名时间戳）。同分布侧一次都见不到它，模型必须真的按抬头判，而不是靠「这种来源
# 里出现 X 就该归 Y 类」的先验蒙。
#
# 反过来，留出 `classes` 会让 `count` / `combo` / `route` 三族各少几档（14 个子集少一半），那些
# 档位在同分布侧就取不到了——均衡闸与审计都会当场报出来。所以留出的是 `source`。
OOD_SOURCES: Final[frozenset[str]] = frozenset({features.SOURCE_PHOTO})

# 同分布切分的哈希分桶边界（组合留出：桶按 tag 组合算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "classes": "包里出现了哪几类",
    "pages": "这一包几页",
    "source": "这一包从哪一种来源来",
}


@dataclass(frozen=True)
class SegmentationConfig:
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 42×3 = 126 组，512 让枚举器稳稳走**全枚举**
      （成对覆盖在这里只会白扔组合）；
    - `case_limit` >0 时每个切分只取前 N 个组合（小规模试跑用，**会牺牲成对覆盖**）；
    - `phrasing_pool` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = 512
    case_limit: int = 0
    phrasing_pool: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasing_pool", self.phrasing_pool)
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能是负数：{self.case_limit}")
        if self.cover_limit <= 0:
            raise ValueError(f"cover_limit 必须是正数：{self.cover_limit}")


_restricted = restrictor(features.DIMS)


def id_dims() -> dict[str, tuple[JsonValue, ...]]:
    """输入：无。输出：剔除 OOD 取值后的维度表（同分布切分用）。"""
    return _restricted(
        source=tuple(value for value in features.SOURCES if value not in OOD_SOURCES)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含 OOD 取值的维度表（`test_ood` 用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [
        _restricted(source=tuple(value for value in features.SOURCES if value in OOD_SOURCES))
    ]


def bucket_of(case: SegmentationCase) -> int:
    return hash_bucket(case, SPLIT_SEED)


def _dims_of(case: SegmentationCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（比 `as_dict` 多「类数」与「切分档位」）。

    预期行为：`as_dict` 的三个维度之外再加 `count`（类数）与 `route`（切分档位）——小批量里
    最容易被忽略的正是这两个：只按 `classes` / `pages` / `source` 铺开的话，10 条会落在
    「1 类的 2 条、2 类的 4 条、3 类的 1 条」上，`count` 的第 3 档与 `route` 的 `many` 档当场
    缺档，均衡闸标红。

    为什么两个都要：`count` 与 `route` 不是一一对应（同为 2 类，`form+receipt` 是 2 份而
    `receipt+contract` 的 4 页包是 4 份），覆盖了类数并不等于覆盖了档位。
    """
    plan = features.plan_for(case, 0)
    return {
        "classes": case.classes,
        "pages": case.pages,
        "source": case.source,
        "count": str(plan.count),
        "route": plan.route,
    }


def _cover_pick(cases: list[SegmentationCase], limit: int) -> list[SegmentationCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：种类组合 → 来源 → 类数 → 切分档位 → 页数），
    同分时取原顺序里靠前的；名额没用完就按原顺序接着填。

    为什么不是直接截断：`classes` 有 14 个取值、`count` 与 `route` 各 3 个，纯截断的话 10 条的
    小批量很可能一条「3 类」都出不来，`combo` 与 `route` 的某些档位也取不到，均衡闸与审计都会
    当场报出来。覆盖优先把这种「小批量必然缺一档」的坑填掉，代价只是 case 顺序不再等于桶序
    （切分归属仍是桶算的，没变）。
    """
    names = ("classes", "source", "count", "route", "pages")
    return cover_pick(cases, limit, names, _dims_of)


def cases_for(config: SegmentationConfig) -> list[SegmentationCase]:
    """输入：采样参数。输出：该切分的 tag 组合列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], SegmentationCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()), config.split, bucket_of, SPLIT_BUCKETS
        )
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class SegmentationWorkflow(BackendHolder):
    """文档包切分：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 页面正文由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "read")
    #: 判别器名单表：名字 → （`judges` 里的类名，怎么造）。
    judge_table = {
        "hygiene": ("HygieneJudge", LOCALE),
        "frame": ("FrameJudge", LOCALE),
        "read": ("ReadJudge", BACKEND),
    }
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> SegmentationConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return SegmentationConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: SegmentationConfig, case: SegmentationCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> SegmentationSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, SegmentationCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return SegmentationSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: SegmentationSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那一包页面正文，先按注入特征校验（不合格抛错，管线据此重渲染），再与
        代码渲染的种类表 / 切分规则 / 页序抬头拼起来；后端没注入就报错——`needs` 非空时管线一定
        会注入，这里没有后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提
        失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("segmentation workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        raw = draft.data.get("pages")
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError(f"渲染草稿缺 pages 字段（或不是字符串数组）：{draft.data!r}")
        parsed = render.PacketDraft(pages=[str(item) for item in raw])
        render.check_draft(parsed, spec.plan, spec.case.source, loc)
        return render.to_state(parsed, loc)

    def ask(self, spec: SegmentationSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（逐页正文）。

        预期行为：只报模型写的那几页；`classes` 是种类编号（由语言表写），不报。
        """
        parsed = render.parse_state(sample.get("state"))
        text = "\n".join(parsed.pages).strip()
        return {"页面正文": text} if text else {}



@register(WORKFLOW_NAME)
def _factory() -> SegmentationWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return SegmentationWorkflow()
