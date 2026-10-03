"""根因定位 workflow：**特征注入式**（依赖图与规则表由代码写，现场日志由后端写）。

流程与日志行分诊（`workflows/logtriage`）同构（`plan → sample → render → ask`），区别在「判什么」：

- **依赖图、判读规则、事件形状表、现场时刻是结构化事实**，由代码渲染进 state——让模型去写
  只会写歪；
- 后端只写**六个服务各自的现场日志**（超时 / 报错 / 正常三种形状）；
- **谁出错、谁受影响由代码按字面算**（`features.PATTERNS` 的三条正则），所以标签是注入特征
  经纯函数算出来的（`features.plan_for` + `label_for`），模型碰不到；写出来的日志必须兑现这份
  计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` / `FrameJudge` 再各查一遍）。

**为什么要这一族**：社区里 `Stratum`（★51）把「日志异常检测与根因分析」明写成 Jev 的使用场景，
`claude-code-trace`（★370）从会话日志里判代理进度与恢复，`jevlogs` 在昂贵分析之前先做日志分诊。
`logtriage` 判的是**单行**日志该不该进告警通道；本 workflow 判的是**一整条时间线**上
「错误是从哪儿开始往上冒的」——判据空间（依赖图 + 传播规则）与素材形态（多服务时间线）
都不一样，不重复。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 (图, 根因, 探针) 的哈希分桶，同组合不跨切分；
- test_ood：**值留出**——`graph=G4` 这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("causechain")` 可用）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.cover import hash_bucket, slice_by_bucket, restrictor
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import CauseCase
from .spec import CauseSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "causechain"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `graph=G4`：四张图都能把四族的**每一档**取到（根因六个服务、跳数三档、
# 受影响集合的邻近干扰、布尔两边），所以留出一张图不会让任何档位在同分布侧消失——
# 留出验的是「读法不靠拓扑先验」，而不是「某个档位只在留出侧才有」。
#
# 反过来，留出某个 `root` 会让「根因是它」那一档在同分布侧取不到，均衡闸与审计都会当场报出来。
OOD_GRAPHS: Final[frozenset[str]] = frozenset({"G4"})

# 同分布切分的哈希分桶边界（组合留出：桶按 (图, 根因, 探针) 算，同组合永远同切分）。
SPLIT_SEED: Final[int] = 20261001
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

#: 每对 (图, 根因) 最多留几条 case。
#:
#: 探针的可行性随 (图, 根因) 变——根因是入口服务时它没有调用方，「1 跳」「2 跳以上」
#: 「非根因的出错服务」三种探针一个都问不出来，那一组只剩 2 条；而根因在下游的那几组
#: 五种探针全都能出。不匀一下，`root` 那一族的答案分布会按构造偏成「S5 / S6 最多、S1 最少」。
CASES_PER_PAIR: Final[int] = 3

DIMENSION_TITLES: Final[dict[str, str]] = {
    "graph": "依赖图是哪一张",
    "root": "故障注入在哪个服务",
    "probe": "这次问的是谁",
}


@dataclass(frozen=True)
class CauseConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 4×6×5 = 120 组，512 让枚举器稳稳走**全枚举**
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
    return _restricted(graph=tuple(name for name in features.GRAPHS if name not in OOD_GRAPHS))


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含 OOD 取值的维度表（`test_ood` 用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(graph=tuple(name for name in features.GRAPHS if name in OOD_GRAPHS))]


def bucket_of(case: CauseCase) -> int:
    """输入：一个 case。输出：0~99 的哈希桶号。"""
    return hash_bucket(case, SPLIT_SEED)


def _balanced_pick(cases: list[CauseCase], per_pair: int) -> list[CauseCase]:
    """按 (图, 根因) 分组，每组最多留 `per_pair` 条，挑的时候优先补全局用得最少的探针值。

    输入：case 列表、每组上限。输出：挑出来的 case（顺序稳定）。
    预期行为：同一入参必得同一输出；每组内尽量不重复取同一个探针值。

    为什么要这一步：见 `CASES_PER_PAIR`。这一步只动「留哪些」，不动 case 本身的取值，
    所以哈希分桶的结果不受影响（切分仍按 (图, 根因, 探针) 算）。
    """
    groups: dict[tuple[str, str], list[CauseCase]] = {}
    for case in cases:
        groups.setdefault((case.graph, case.root), []).append(case)
    used: dict[str, int] = dict.fromkeys(features.PROBES, 0)
    picked: list[CauseCase] = []
    for key in sorted(groups):
        chosen: list[CauseCase] = []
        remaining = list(groups[key])
        while remaining and len(chosen) < per_pair:
            taken = {case.probe for case in chosen}
            pool = [case for case in remaining if case.probe not in taken] or remaining
            best = min(pool, key=lambda case: (used[case.probe], features.PROBES.index(case.probe)))
            remaining.remove(best)
            chosen.append(best)
            used[best.probe] += 1
        picked.extend(chosen)
    return picked


def _cover_pick(cases: list[CauseCase], limit: int) -> list[CauseCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：图 → 根因 → 探针），同分时取原顺序里靠前的；
    名额没用完就按原顺序接着填。

    为什么不是直接截断：探针有五个取值、根因有六个，纯截断的话 10 条的小批量很可能一条
    「2 跳以上」或「根因不是它」都出不来，`hop` 与 `isroot` 两族的某些档位当场缺档，
    均衡闸与审计都会报出来。
    """
    names = ("graph", "root", "probe")
    chosen: list[CauseCase] = []
    seen: dict[str, set[str]] = {name: set() for name in names}
    remaining = list(cases)
    while remaining and len(chosen) < limit:
        keys = [
            (
                sum(1 for name in names if case.as_dict()[name] not in seen[name]),
                -index,
            )
            for index, case in enumerate(remaining)
        ]
        best = max(range(len(remaining)), key=lambda index: keys[index])
        case = remaining.pop(best)
        chosen.append(case)
        for name in names:
            seen[name].add(str(case.as_dict()[name]))
    return chosen


def cases_for(config: CauseConfig) -> list[CauseCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的图；其余切分先剔除留出取值，**再匀一次**（`_balanced_pick`）、
    最后按哈希桶分。匀在前、分在后，是为了让三个同分布切分各自也拿到一份平的探针分布。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], CauseCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = _balanced_pick(list(seen.values()), CASES_PER_PAIR)
    else:
        balanced = _balanced_pick(
            features.build_cases(config.cover_limit, dims=id_dims()), CASES_PER_PAIR
        )
        cases = slice_by_bucket(balanced, config.split, bucket_of, SPLIT_BUCKETS)
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class CauseChainWorkflow(BackendHolder):
    """根因定位：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs = frozenset({"text"})  # 现场日志由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> CauseConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return CauseConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: CauseConfig, case: CauseCase, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、序号。输出：生成计划（问法从池里抽）。"""
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> CauseSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, CauseCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return CauseSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: CauseSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要六个服务的现场日志，先按注入特征校验（不合格抛错，管线据此重渲染），
        再与代码渲染的依赖图 / 规则表 / 事件形状表 / 时刻拼起来；后端没注入就报错——
        `needs` 非空时管线一定会注入，这里没有后端说明接线错了，绝不能悄悄降级成模板
        （那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("causechain workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        raw_logs = draft.data.get("logs")
        if not isinstance(raw_logs, dict):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        logs: dict[str, str] = {}
        for key, value in raw_logs.items():
            if not isinstance(value, str):
                raise ValueError(f"渲染草稿的 logs[{key!r}] 不是字符串：{value!r}")
            logs[str(key)] = value
        parsed = render.ContentDraft(logs=logs)
        render.check_draft(parsed, spec.plan, loc)
        return render.to_state(parsed, spec.plan, loc)

    def ask(self, spec: CauseSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（六行现场日志）。

        预期行为：只报模型写的那六行日志；依赖图与规则表由语言文件写，不报。
        """
        parsed = render.parse_state(sample.get("state"))
        lang = render.detect_locale(sample.get("state")) or locales.available()[0]
        logs = render.parse_logs(parsed.text, locales.load(lang))
        return {
            f"{service} 的日志": text
            for service, text in logs.items()
            if text and text.strip()
        }



@register(WORKFLOW_NAME)
def _factory() -> CauseChainWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return CauseChainWorkflow()
