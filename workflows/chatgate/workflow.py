"""聊天回复门控 workflow：**特征注入式**（判据表、毛病表、档位表由代码写，一条消息与三条候选由后端写）。

流程与提示注入（`workflows/promptinject`）、日志分诊（`workflows/logtriage`）同构
（`plan → sample → render → ask`），区别在「判什么」：

- **判据表、毛病表、档位表是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**一条对方发来的消息 + 三条我可以发回去的候选**（聊天内容读起来都是长尾的，模板拼出来
  的脱离现实分布）；
- **踩中哪几条、哪条候选干净都由代码按字面算**（`features.PATTERNS` / `FLAW_PATTERNS` 那七条
  正则），所以标签是注入特征经纯函数算出来的（`features.plan_for` + `label_for`），模型碰不到；
  写出来的东西必须兑现这份计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` /
  `FrameJudge` 再各查一遍）。

**为什么要这一族**：社区清单里最大的一支应用是聊天助手（`jev-chat-jarvis` ★1696、
`jev-chat-windows` ★467、`crush-monitor` ★183、`Jev-X-Sentiment-Analysis` ★162、
`yanwai` ★15、`wechat-jev-assistant` ★13），共同形状是「一条消息进来 → 先判要多大分寸 →
再决定三条候选里该发哪条」，而不是让模型直接生成要发出去的那句话。

**与 `mail` / `guardrail` / `logtriage` 的区别**：那几家的判据空间是**内容安全 / 运维**的形状
（发件人身份、外链、政策条款、日志级别词），本 workflow 判的是**人际沟通**的形状（要不要答复、
有没有明确时限、要不要我担义务、涉不涉及隐私），而且多一层**候选回复的取舍**。

切分按 PLAN 3.3 的泛化留出：
- train / val / test_iid：**组合留出**——按 case 的哈希分桶，同 case 不跨切分；
- test_ood：**值留出**——`scene=stranger`（陌生人私信）这一档，训练侧一次都不出现。

导入本模块即自注册（`jc.workflow.load("chatgate")` 可用）。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Final

from jc.cover import restrictor
from jc.jsontypes import JsonValue

from jc.workflow import BackendHolder, LabeledQuestion, SplitSpec, plan_with_phrasing, register

from . import audit as audit_mod, features, judges as judges_mod, locales, questions, render
from .features import ChatCase
from .spec import ChatGateSpec
from jc.knobs import CommonKnobs

WORKFLOW_NAME = "chatgate"
WORKFLOW_VERSION = "0.1.0"

# 只在 test_ood 出现的取值（值留出）。
#
# 为什么留出 `scene=stranger`：它是**只影响渲染、不影响标签**的那一维——标签由 `degree` 与
# `slot` 算（`scene` 一个字符都不参与），场景只决定这条消息长什么样（同事私聊 / 工作群 /
# 陌生人私信）。同分布侧一次都见不到它，模型必须真的按字面判，而不是靠「陌生人发来的就可疑」
# 这类场景先验蒙。
#
# 反过来，留出 `degree` 会让 `count` / `route` 两族各少几档（5 档少一档），那些档位在同分布侧
# 就取不到了——均衡闸与审计都会当场报出来。所以留出的是 `scene`。
OOD_SCENES: Final[frozenset[str]] = frozenset({features.SCENE_STRANGER})

# 同分布切分的哈希分桶边界（组合留出：桶按 case 算，同 case 永远同切分）。
SPLIT_SEED: Final[int] = 20260930
SPLIT_BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "train": (0, 60),
    "val": (60, 80),
    "test_iid": (80, 100),
}

DIMENSION_TITLES: Final[dict[str, str]] = {
    "degree": "这条消息该踩几条判据",
    "scene": "消息长在哪种聊天场景里",
    "slot": "该发的候选落在第几位",
}


@dataclass(frozen=True)
class ChatGateConfig(CommonKnobs):
    """一个切分的采样参数。

    字段：
    - `split` 切分名（决定 case 怎么分桶）；
    - `cover_limit` 全枚举上限——全空间 5×3×3 = 45 组，512 让枚举器稳稳走**全枚举**
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
        scene=tuple(value for value in features.SCENES if value not in OOD_SCENES)
    )


def ood_dims() -> list[dict[str, tuple[JsonValue, ...]]]:
    """输入：无。输出：只含 OOD 取值的维度表（`test_ood` 用）。

    预期行为：返回列表而不是单个表，是为了与其它 workflow 的形状一致（将来加第二个留出维度时
    只改这里，`cases_for` 会自动并集去重）。
    """
    return [_restricted(scene=tuple(value for value in features.SCENES if value in OOD_SCENES))]


def bucket_of(case: ChatCase) -> int:
    """输入：一个 case。输出：它的哈希桶号（0~99）。

    预期行为：只按 case 指纹算，与切分名无关——同一个 case 永远落在同一个桶里，所以「同 case 不跨
    切分」是算出来的，不是靠人排的。
    """
    payload = case.fingerprint()
    return int(hashlib.sha256(f"{SPLIT_SEED}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100


def _slice(cases: list[ChatCase], split: str) -> list[ChatCase]:
    """按桶号排序后按比例切片。

    输入：同分布 case 列表、切分名。输出：该切分的 case（顺序稳定）。
    预期行为：先按 (桶号, 指纹) 排序再切片，比例才是**精确**的。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), case.fingerprint()))
    low, high = SPLIT_BUCKETS[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def _dims_of(case: ChatCase) -> dict[str, str]:
    """输入：一个 case。输出：覆盖挑选用的取值表（三个维度都取字符串形式）。"""
    return {"degree": str(case.degree), "scene": case.scene, "slot": str(case.slot)}


def _cover_pick(cases: list[ChatCase], limit: int) -> list[ChatCase]:
    """按「取值覆盖优先」挑前 `limit` 个 case（小规模试跑用）。

    预期行为：贪心地先补齐没见过的维度取值（顺序：命中条数 → 场景 → 候选位置），同分时取**当前
    被挑得最少**的那个取值所在的行，再同分才取原顺序里靠前的。

    为什么不是直接截断：`degree` 有 5 个取值、`slot` 有 3 个，纯截断的话 10 条的小批量很可能一条
    「4 条全中」或某个候选位置都出不来，`count` / `route` / `reply` 的某些档位也取不到，均衡闸与
    审计都会当场报出来。覆盖优先把这种「小批量必然缺一档」的坑填掉，代价只是 case 顺序不再等于
    桶序（切分归属仍是桶算的，没变）。

    为什么要看「当前被挑得最少」：全空间只有 5×3×3 = 45 组，10 条的小批量里**每个 case 都恰好
    补一个新取值**，光看覆盖数全是平手，同分全按原顺序填的话很容易连着挑中同一档命中条数——
    实测 10 条时 `count` 的 dev 冲到 0.4（噪声地板 0.38），均衡闸判红。加上这一档平手规则之后，
    小批量的条数分布才是摊开的。
    """
    names = ("degree", "scene", "slot")
    chosen: list[ChatCase] = []
    seen: dict[str, set[str]] = {name: set() for name in names}
    freq: dict[str, dict[str, int]] = {name: {} for name in names}
    remaining = list(cases)
    while remaining and len(chosen) < limit:
        keys = [
            (
                sum(1 for name in names if _dims_of(case)[name] not in seen[name]),
                -max(freq[name].get(_dims_of(case)[name], 0) for name in names),
                -index,
            )
            for index, case in enumerate(remaining)
        ]
        best = max(range(len(remaining)), key=lambda index: keys[index])
        case = remaining.pop(best)
        chosen.append(case)
        for name in names:
            seen[name].add(_dims_of(case)[name])
            freq[name][_dims_of(case)[name]] = freq[name].get(_dims_of(case)[name], 0) + 1
    return chosen


def cases_for(config: ChatGateConfig) -> list[ChatCase]:
    """输入：采样参数。输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 收值留出的组合（并集去重）；其余切分先剔除留出取值，再按哈希桶分。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], ChatCase] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases = list(seen.values())
    else:
        cases = _slice(features.build_cases(config.cover_limit, dims=id_dims()), config.split)
    return _cover_pick(cases, config.case_limit) if config.case_limit else cases


class ChatGateWorkflow(BackendHolder):
    """聊天回复门控：特征、计划、生成、质量四段（`jc.workflow.Workflow` 协议）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul"})
    needs = frozenset({"text"})  # 消息与候选由后端写：本 workflow 不是纯构造式
    judges: tuple[str, ...] = ("hygiene", "frame", "trace")
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    cases_for_fn = staticmethod(cases_for)
    dimension_titles_map = DIMENSION_TITLES
    pkg_render = render
    pkg_judges = judges_mod
    pkg_audit = audit_mod
    pkg_locales = locales

    def config_for(self, split: str) -> ChatGateConfig:
        """输入：切分名。输出：该切分的采样参数。预期行为：未知切分直接报错。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分 {split!r}（可用：{', '.join(self.split_names())}）")
        return ChatGateConfig(split=split)

    def plan_sample(
        self, rng: random.Random, config: ChatGateConfig, case: ChatCase, seq: int
    ) -> SplitSpec:
        return plan_with_phrasing(config, rng, case, seq, config.phrasing_pool)

    def sample(self, rng: random.Random, split: SplitSpec) -> ChatGateSpec:
        """输入：独立 RNG（不用）、生成计划。输出：真值规格。"""
        del rng
        case = split.params["case"]
        phrasing = split.params["phrasing"]
        if not isinstance(case, ChatCase) or not isinstance(phrasing, int):
            raise TypeError(f"生成计划里的 case / phrasing 类型不对：{split.params!r}")
        return ChatGateSpec(
            case=case,
            phrasing=phrasing,
            split=split.name,
            seq=int(split.params["seq"]),
            plan=features.plan_for(case, phrasing),
        )

    def render(self, spec: ChatGateSpec, rng: random.Random, locale: str) -> str:
        """输入：规格、RNG、语言。输出：契约允许的 state（字符串）。

        预期行为：向后端要那条消息与三条候选，先按注入特征校验（不合格抛错，管线据此重渲染），
        再与代码渲染的三张表拼起来；后端没注入就报错——`needs` 非空时管线一定会注入，这里没有
        后端说明接线错了，绝不能悄悄降级成模板（那会让「文本由模型写」这个前提失效）。
        """
        del rng
        if self.backend is None:
            raise RuntimeError("chatgate workflow 需要生成后端，但还没被 bind_backend 注入")
        loc = locales.load(locale)
        draft = self.backend.generate(render.build_task(spec, loc))
        text = draft.data.get("text")
        candidates = draft.data.get("candidates")
        if not isinstance(text, str) or not isinstance(candidates, list):
            raise ValueError(f"渲染草稿缺字段：{draft.data!r}")
        bodies = [item for item in candidates if isinstance(item, str)]
        if len(bodies) != len(candidates):
            raise ValueError(f"渲染草稿的候选里混了非字符串：{candidates!r}")
        parsed = render.MessageDraft(text=text, candidates=bodies)
        render.check_draft(parsed, spec.plan, loc)
        return render.to_state(parsed, loc)

    def ask(self, spec: ChatGateSpec, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：规格、RNG、语言。输出：题目 + 金答案（纯代码推导，不碰模型）。"""
        del rng
        return questions.build(spec.plan, locales.load(locale), spec.phrasing)

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：必须与声明语言一致的正文（「对方消息」与「候选回复」）。

        预期行为：三张表是代码写的（编号 + 判据原文，按语言取），不报；只报模型写的那几段正文。
        """
        parsed = render.parse_state(sample.get("state"))
        out: dict[str, str] = {}
        if parsed.message and parsed.message.strip():
            out["对方消息"] = parsed.message
        replies = [body for body in parsed.replies if body.strip()]
        if replies:
            out["候选回复"] = "\n".join(replies)
        return out



@register(WORKFLOW_NAME)
def _factory() -> ChatGateWorkflow:
    """输入：无。输出：workflow 实例。预期行为：导入本模块即完成自注册。"""
    return ChatGateWorkflow()
