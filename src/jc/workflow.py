"""通用抽象：workflow 协议、题目与切分的数据类、注册表。

概念（对应 PLAN 3.2）：
- **workflow 是一等公民**：一段生成逻辑，可以是纯构造式的（`needs` 为空、完全不碰模型），
  也可以需要后端（LLM / JEV / adapter）；题型（choice/noul/score）只决定**输出形状**，
  workflow 决定**答案从哪来**；
- 通用层（本包）**不 import 任何具体 workflow**：workflow 通过 `register` 自注册，
  由 `load()` 按名字动态导入。

输入/输出：本模块只有数据类、协议与注册表，无副作用、不含领域词表。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import importlib
import pkgutil
import random
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from .cover import slice_by_bucket
from .contract import TYPE_CHOICE, TYPE_NOUL, TYPE_SCORE, is_index, option_names
from .judge import COMMON_JUDGES, ROUNDTRIP_JUDGES, Judge, JudgeTable, standard_judges
from .locale_files import LocaleApi

if TYPE_CHECKING:  # 只在类型检查时引入：核心层不装 [llm] extra 也能跑构造式 workflow
    from .backend.types import GenerationBackend


class DraftError(ValueError):
    """后端写出来的草稿没兑现 workflow 的契约（长度 / 标点 / 关键词 / 反解不回来）。

    输入：人话的失败原因。输出：异常（继承 ValueError，历史调用方照旧能 catch）。
    预期行为：类属性 `retryable_for_case = True` 是**唯一**的判据——管线看到它就认为
    「这一条是模型没照契约写」，重渲染一次（采样抖动，换个说法往往就对了）；没有这个
    标记的异常仍然一律当生成器 bug，**绝不重渲染**（重渲染只会把 bug 藏起来）。
    """

    retryable_for_case = True


@dataclass(frozen=True)
class LabeledQuestion:
    """一道题 + 它的金答案（组装前的内部形态）。

    字段：
    - family：题族名，作为题号；同族重复时由组装器加 `_2`、`_3` 后缀；
    - question：契约形态的题目对象 `{"type", "instructions", "criteria"?}`；
    - answer：choice 用**选项名**、score 用**等级下标**、noul 用布尔；
    - shuffle：choice 的选项是否允许打乱（有自然顺序的集合必须为 False）。
    """

    family: str
    question: dict[str, JsonValue]
    answer: str | int | bool
    shuffle: bool = True


@dataclass(frozen=True)
class SplitSpec:
    """一条样本的生成计划（切分名 + workflow 自定义参数）。

    通用层只透传 `params`，不解释它的内容——具体含义由 workflow 自己定。
    """

    name: str
    params: dict[str, JsonValue] = field(default_factory=dict)


def sliced_cases(
    config: object,
    features: ModuleType,
    ood_dims: Callable[[], Sequence[Mapping[str, tuple[str, ...]]]],
    id_dims: Callable[[], Mapping[str, tuple[str, ...]]],
    bucket_of: Callable[[object], int],
    split_buckets: Mapping[str, tuple[int, int]],
) -> list[JsonValue]:
    """输入：采样参数、该包的 `features` 模块（只用来调 `build_cases`）、两张维度表
    （`ood_dims` 留出维度表 / `id_dims` 同分布维度表）、桶函数、桶表。
    输出：该切分的 case 列表（顺序稳定）。

    预期行为：`test_ood` 收「留出维度各自枚举出的组合」的并集，按 case 内容（`as_dict()` 的
    键值对）去重——同一组取值落在两张留出表里只算一条；其余切分先取同分布组合，再按哈希桶
    切片；`config.case_limit` 非 0 时从头截断。

    为什么收在这里：`cases_for` 是**模块级**函数（`StandardMethods` 收不进去），而这一族 11 家
    workflow 的 `cases_for` 归一化后逐字相同，写法不一样的一家都不碰。四张表与桶函数都住在
    各包的 `workflow.py` 里（不在 `features.py`），所以从调用点显式传进来，不靠反射。
    """
    if config.split == "test_ood":
        seen: dict[tuple[tuple[str, JsonValue], ...], JsonValue] = {}
        for dims in ood_dims():
            for case in features.build_cases(config.cover_limit, dims=dims):
                seen.setdefault(tuple(case.as_dict().items()), case)
        cases: list[JsonValue] = list(seen.values())
    else:
        cases = slice_by_bucket(
            features.build_cases(config.cover_limit, dims=id_dims()),
            config.split,
            bucket_of,
            split_buckets,
        )
    return cases[: config.case_limit] if config.case_limit else cases


class StandardMethods:
    """workflow 协议里**逐字同构**的那些方法的默认实现：51 个包各抄一份的那 8 段，只留这一份。

    输入：无。输出：无。
    预期行为：子类在类体里把「这些实现住在哪个对象里」用类属性指出来（见下），本类就把对应的
    协议方法实现好。某个包里写法不一样时（例如 `maze` 的 `build_cases` 自己铺格子、
    `browser` 的 `case_label` 自己拼键值对、`config_for` 各家有各家的切分表），照旧在子类里
    覆盖同名方法——本类只负责「一样的那部分」，不替谁做决定。

    声明项（用得到才写，值都是**自己包里的对象**）：

    - `pkg_locales`：包里的 `locales` 模块 → `available_locales`；
    - `pkg_render`：包里的 `render` 模块（要有 `parse_state`）→ `reparse`；
    - `pkg_audit`：包里的 `audit` 模块（要有 `audit_run`）→ `audit`；
    - `split_names_tuple`：四个切分名 → `split_names`；
    - `dimension_titles_map`：`{维度名: 中文标题}` → `dimension_titles`；
    - `cases_for_fn`：包里的 `cases_for` 函数 → `build_cases`。这一格**必须**写成
      `cases_for_fn = staticmethod(cases_for)`：函数直接当类属性会被绑成方法，`self` 会顶掉
      第一个参数（`config`），调用时当场 `TypeError`；`staticmethod` 把它挡在描述符协议外。

    `case_label` / `stats_dimensions` 不要声明项：它们只依赖 case 自己的 `describe()` /
    `as_dict()`，是协议里唯一两条与「实现住在哪个模块」无关的。
    `make_judges` 在这里：纯构造式那一档（`roundtrip` / `label`）不吃后端，所以它的默认
    实现属于本类；`BackendHolder` 覆盖它，多传一个后端进去。`language_texts` 也在这里：报空表
    是**默认**口径（下面有它的理由），要报正文的那几家在自己的类里覆盖。本类一共实现 10 个
    协议方法。
    """

    pkg_locales: ClassVar[LocaleApi]
    pkg_render: ClassVar[ModuleType]
    pkg_audit: ClassVar[ModuleType]
    pkg_judges: ClassVar[ModuleType]
    split_names_tuple: ClassVar[tuple[str, ...]]
    dimension_titles_map: ClassVar[Mapping[str, str]]
    cases_for_fn: ClassVar[Callable[[object], list[JsonValue]]]
    #: 判别器名单表：名字 → （`pkg_judges` 里的类名，怎么造）。默认是纯构造式那一档。
    judge_table: ClassVar[JudgeTable] = ROUNDTRIP_JUDGES

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：该 workflow 支持的切分名（顺序就是 `split_names_tuple` 的顺序）。"""
        return tuple(self.split_names_tuple)

    def build_cases(self, config: object) -> list[JsonValue]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。预期行为：由包里的 `cases_for` 算。"""
        return self.cases_for_fn(config)

    def case_label(self, case: object) -> str:
        """输入：一个 case。输出：一行可读描述。预期行为：由 case 自己的 `describe()` 给。"""
        return case.describe()

    def stats_dimensions(self, case: object) -> Mapping[str, JsonValue]:
        """输入：一个 case。输出：各维取值（均衡统计用）。预期行为：由 case 自己的 `as_dict()` 给。"""
        return case.as_dict()

    def dimension_titles(self) -> Mapping[str, str]:
        """输入：无。输出：`{维度名: 中文标题}`，用于把统计报告渲染得好读。"""
        return dict(self.dimension_titles_map)

    def reparse(self, state: object, *, case: object = None) -> object | None:
        """输入：渲染后的 state。输出：纯代码能反解的部分，或 None（该格式不可反解）。

        预期行为：交给包里的 `render.parse_state`；`case` 用不上（有的包要，会自己覆盖）。
        """
        del case
        return self.pkg_render.parse_state(state)

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分审计计数, 缺失项列表)。"""
        return self.pkg_audit.audit_run(run_dir, splits, locale)

    def available_locales(self) -> list[str]:
        """输入：无。输出：该 workflow 支持的语言标签列表（`jc locales` 用它）。"""
        return self.pkg_locales.available()

    def language_texts(self, sample: dict[str, JsonValue]) -> Mapping[str, str]:
        """输入：契约样本。输出：空表——本族没有「模型写的自然语言正文」。

        预期行为：报空表的情形有两种，都写在下面，17 家各自的类里原先各抄一份同样的 `return {}`，
        现在只留这一份：

        - **纯构造式**（`needs` 为空）：state 全由代码按语言模板拼，天然与声明语言一致，
          没有「模型写歪」这回事；
        - **模型写的东西与语言无关**：`review` 写的是 diff、`lint` 写的是片段原文（真机 zh 侧
          汉字占比低到 0.099 全是正常产物）、`shell` 写的是命令行、`maze` 写的是 ASCII 迷宫图
          ——这些喂给语言判据只会把好数据当坏数据。

        走 LLM 写**自然语言正文**的那几家必须在自己的类里覆盖本方法。漏了不会静默：
        `tests/workflows/_contract.py` 的 `assert_language_texts_are_declared` 拿**真产物**
        断言「该报正文的必须报出来」，空表当场红。
        """
        del sample
        return {}

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。

        预期行为：名字表在 `judge_table`、类在 `pkg_judges` 模块里，由 `jc.judge.standard_judges`
        统一实例化——未知名字、表里指向不存在的类、吃法写错都在那里报错。纯构造式没有后端，
        所以这一档里不许有 `BACKEND` 吃法（有就会当场报错，`BackendHolder` 才传后端）。
        """
        return standard_judges(names, locale, self.pkg_judges, None, self.judge_table)


class BackendHolder(StandardMethods):
    """需要生成后端的 workflow 的公共底座：记住注入进来的后端 + 判别器实例化。

    输入：无（构造时后端还没到）。输出：无。
    预期行为：`backend` 初值是 `None`，`bind_backend` 只赋值、不做别的。48 个包里需要后端的那
    36 个此前各自抄了一份同样的 `__init__` + `bind_backend`，现在只留这一份。

    它还继承 `StandardMethods`（协议方法的默认实现）：走 LLM 的包一律继承本类，纯构造式的包
    直接继承 `StandardMethods`——它们不需要后端，也就不该多出一个 `backend` 字段。

    它**不是** `Workflow` 协议的一部分：纯构造式的 workflow 不继承它，`needs` 为空时管线也
    不会调 `bind_backend`。继承它只是省掉两段抄写，协议仍是结构化的（鸭子类型），
    所以「谁继承了谁」不影响 `isinstance` 判断，也不影响注册表。
    """

    backend: GenerationBackend | None = None

    #: 包里的 `judges` 模块：判别器类都住在里面（`make_judges` 从这里取）。
    pkg_judges: ClassVar[ModuleType]

    #: 走 LLM 的包默认用全体共有那一档（`hygiene` / `frame` / `trace`）。
    judge_table: ClassVar[JudgeTable] = COMMON_JUDGES

    def bind_backend(self, backend: GenerationBackend) -> None:
        """输入：生成后端。输出：无。预期行为：存下后端，供 `render` 与判别器使用。"""
        self.backend = backend

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。

        预期行为：同 `StandardMethods.make_judges`，但把后端也交下去——`BACKEND` 那一档
        没被注入后端时直接报错（静默降级会让「文本由模型写」这个前提失效）。
        """
        return standard_judges(names, locale, self.pkg_judges, self.backend, self.judge_table)


@runtime_checkable
class Workflow(Protocol):
    """一个 workflow 的完整契约（构造式与模型驱动式共用）。

    声明项：
    - `kinds`：产出哪些题型（用于统计与校验）；
    - `needs`：需要哪些后端能力，**为空表示纯构造式、管线不会初始化任何后端**；
    - `judges`：默认判别器名单（由 `make_judges` 实例化）。

    方法：`plan_sample` 决定怎么造这一条（题族、变体等），`sample/render/ask` 是三步生成，
    `reparse` 供往返校验，`audit` 供产物审计；
    `bind_backend` 只给模型驱动式用（`needs` 非空时由管线注入后端，`render` 阶段调它写文本）。
    """

    name: str
    version: str
    kinds: frozenset[str]
    needs: frozenset[str]
    judges: tuple[str, ...]

    def split_names(self) -> tuple[str, ...]:
        """返回该 workflow 支持的切分名（如 train/val/test_iid/test_ood）。"""

    def config_for(self, split: str) -> object:
        """输入：切分名。输出：该切分的采样参数（workflow 自定义类型）。"""

    def build_cases(self, config: object) -> list[JsonValue]:
        """输入：采样参数。输出：case 矩阵（顺序稳定）。"""

    def case_label(self, case: object) -> str:
        """输入：一个 case。输出：一行可读描述（用于报告与文件名）。"""

    def stats_dimensions(self, case: object) -> Mapping[str, JsonValue]:
        """输入：一个 case。输出：{维度名: 取值}，供统计与均衡报告使用。"""

    def dimension_titles(self) -> Mapping[str, str]:
        """返回 {维度名: 中文标题}，用于把统计报告渲染得好读。"""

    def plan_sample(
        self, rng: random.Random, config: object, case: object, seq: int
    ) -> SplitSpec:
        """输入：RNG、采样参数、case、样本序号。输出：这一条的生成计划。"""

    def sample(self, rng: random.Random, split: SplitSpec) -> object:
        """输入：独立 RNG + 生成计划。输出：可 JSON 序列化的真值规格 spec。"""

    def render(self, spec: object, rng: random.Random, locale: str) -> object:
        """输入：spec、RNG、语言。输出：契约允许的 state。"""

    def ask(self, spec: object, rng: random.Random, locale: str) -> list[LabeledQuestion]:
        """输入：spec、RNG、语言。输出：题目 + 金答案（纯代码推导，禁止调模型）。"""

    def reparse(self, state: object, *, case: object = None) -> object | None:
        """输入：渲染后的 state。输出：反解结果，或 None（该格式不可反解）。"""

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。"""

    def language_texts(self, sample: dict[str, JsonValue]) -> Mapping[str, str]:
        """输入：契约样本。输出：{字段名: 必须与**声明语言**一致的文本}（空 = 没有自然语言正文）。

        预期行为：只报**自然语言正文**的字段——代码、命令行、ASCII 图、编号、选项名一律不报
        （那些不该按语言判，报进去就是把好数据当坏数据）。通用层拿它查「产物文本与声明语言
        是否一致」（`jc.language`），审计逐行调它；**它是纯函数，只读样本，不看调用方给的语言**
        ——语言由产物自己的 manifest 说了算。
        """

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """输入：run 目录、切分、语言。输出：(各切分审计计数, 缺失项列表)。"""

    def bind_backend(self, backend: GenerationBackend) -> None:
        """输入：生成后端。输出：无。

        预期行为：`needs` 非空的 workflow 在这里存下后端（`render` 阶段要用它写文本）；
        纯构造式 workflow（`needs` 为空）**根本不会被调用**——收到后端就说明接线错了，
        实现里应当直接报错，别静默吞掉。
        后端**只出草稿**，编号 / 题号 / `type` 仍然只由组装器产生（PLAN 3.1 不变量①）。
        """

    def available_locales(self) -> list[str]:
        """输入：无。输出：该 workflow 支持的语言标签列表（`jc locales` 用它）。"""


_REGISTRY: dict[str, Callable[[], Workflow]] = {}


def config_for_split(
    split: str,
    ood_config: object,
    id_config: object,
    split_buckets: Mapping[str, tuple[int, int]],
    names: Sequence[str],
) -> object:
    """输入：切分名、OOD 配置、同分布配置、桶表、可用的切分名（报错时用）。
    输出：该切分的采样参数。

    预期行为：`test_ood` 用 OOD 配置；桶表里的切分用同分布配置，**并把 `split` 换成调用方
    要的那一档**——同分布配置是个单例，直接返回它会让 train / val / test_iid 三次都按同一
    档切片（切出来三份一模一样的数据）；其余切分报人话的 `KeyError`。

    为什么收在这里：这一族 12 家的 `config_for` 归一化后逐字相同，四张表住在各包自己的
    `workflow.py` 里，所以从调用点显式传进来，不靠反射。
    """
    if split == "test_ood":
        return ood_config
    if split in split_buckets:
        return replace(id_config, split=split)
    raise KeyError(f"未知切分：{split!r}（可用：{names}）")


def register(name: str) -> Callable[[Callable[[], Workflow]], Callable[[], Workflow]]:
    """注册一个 workflow 工厂。

    输入：workflow 名。
    输出：装饰器。预期行为：同名重复注册直接报错，避免静默覆盖。
    """

    def decorator(factory: Callable[[], Workflow]) -> Callable[[], Workflow]:
        if name in _REGISTRY:
            raise ValueError(f"workflow 已注册：{name}")
        _REGISTRY[name] = factory
        return factory

    return decorator


def registered() -> list[str]:
    """返回已注册的 workflow 名（排序后）。"""
    return sorted(_REGISTRY)


def discover() -> list[str]:
    """导入 `workflows` 包下所有子包（触发自注册），返回已注册的 workflow 名。

    输入：无。输出：排序后的已注册名字。
    预期行为：注册靠「导入即注册」，所以列名单前必须先把各 workflow 导进来；
    单个 workflow 导入失败只写一行警告到 stderr，不连坐其他 workflow，
    也不让通用层静态依赖任何具体 workflow。
    """
    try:
        package = importlib.import_module("workflows")
    except ModuleNotFoundError:
        return registered()
    for module in pkgutil.iter_modules(package.__path__):
        if module.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"workflows.{module.name}")
        except Exception as exc:  # noqa: BLE001 坏掉的 workflow 不该拖垮名单
            print(f"⚠️ 跳过 workflow {module.name}：{exc}", file=sys.stderr)
    return registered()


def load(name: str) -> Workflow:
    """按名字加载并返回 workflow 实例（必要时动态导入 `workflows.<name>`）。

    输入：workflow 名。
    输出：Workflow 实例。
    预期行为：未注册时尝试 `import workflows.<name>` 触发自注册；
    仍找不到就报错并列出已注册的名字。**通用层不静态依赖任何 workflow**。
    """
    if name not in _REGISTRY:
        try:
            importlib.import_module(f"workflows.{name}")
        except ModuleNotFoundError as exc:
            # 报错前先把名单补全：走到这里说明这个名字没被导入过，注册表很可能还是空的
            # （谁都没 import 过任何 workflow），直接读它会报成「（空）」——那等于没报。
            discover()
            known = ", ".join(registered()) or "（空）"
            raise KeyError(f"未找到 workflow {name!r}（已注册：{known}）") from exc
    if name not in _REGISTRY:
        raise KeyError(f"workflow {name!r} 被导入但没有自注册；已注册：{', '.join(registered())}")
    return _REGISTRY[name]()


def family_of(qid: str) -> str:
    """题号 → 题族名：去掉同族后缀（`can_move_2` → `can_move`）。

    输入：题号。输出：题族名。预期行为：题族名本身不含数字。
    """
    return "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")


def show_answers(want: tuple[str | bool | int, ...]) -> str:
    """输入：可接受的答案。输出：给人看的写法（多个候选用 ` / ` 连）。

    预期行为：只做展示、不参与判定——判定比的是元组成员本身（`in`），这里只负责把
    「按 state 重算应该是哪个」写进理由里。
    """
    return " / ".join(str(value) for value in want)


def compare_answers(
    sample: dict[str, JsonValue],
    acceptable: Callable[[str], tuple[str | bool | int, ...]],
) -> list[str]:
    """输入：样本、按题族给可接受答案的函数。输出：问题列表（空 = 全对）。

    预期行为：逐题按**题族**重算再比对——choice 比选项描述原文、noul 比布尔、score 比档位。
    可接受的答案做成元组，是为了让审计一次给出「各语言的选项描述」（判别器只给一个）；
    `acceptable` 抛 `ValueError` / `KeyError` 记成「重算不出答案」，不往下算。

    为什么 choice 比的是**描述原文**而不是选项名：组装后的 `criteria` 是「选项名 → 选项描述」，
    `choice` 是**描述在其中的书写顺序**，而选项名是各语言自己起的——比描述才跨语言说得通。

    为什么 `criteria` 要查一遍类型：构造式 workflow 的样本由各家自己组装，形状不对时这里报
    「选项形状不对」，比抛 `AttributeError` 好（判别器崩了不能把整批带走）。
    """
    problems: list[str] = []
    for qid, question in sample["questions"].items():
        family = family_of(qid)
        try:
            want = acceptable(family)
        except (ValueError, KeyError) as exc:
            problems.append(f"{qid}: 重算不出答案（{exc}）")
            continue
        answer = sample["answers"][qid]
        if question["type"] == "choice":
            criteria = question["criteria"]
            if not isinstance(criteria, dict):
                problems.append(f"{qid}: 选项形状不对")
                continue
            options = [str(text) for text in criteria.values()]
            index = answer["choice"]
            if not isinstance(index, int) or not 0 <= index < len(options):
                problems.append(f"{qid}: 答案编号越界（{index}）")
                continue
            if options[index] not in want:
                problems.append(f"{qid}: 答案指向 {options[index]}，按 state 重算是 {show_answers(want)}")
        elif question["type"] == "noul":
            if answer["noul"] not in want:
                problems.append(f"{qid}: 答案是 {answer['noul']}，按 state 重算是 {show_answers(want)}")
        elif question["type"] == "score":
            if answer["score"] not in want:
                problems.append(f"{qid}: 答案是 {answer['score']}，按 state 重算是 {show_answers(want)}")
        else:
            problems.append(f"{qid}: 未知题型 {question['type']!r}")
    return problems


def answer_value(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> JsonValue:
    """输入：一道题、它的答案。输出：可比的值（对不上给 None）。

    预期行为：choice 给**选项名**（不是编号——编号只由组装器产生，见 `jc.assemble`）、
    score 给**档位下标**、noul 给布尔。编号越界、类型不对（布尔不是编号）、criteria 形状
    不对、题族不认识，一律给 None——那是对不上，不是崩；答案缺字段也不抛。

    为什么 choice 比的是**选项名**而不是选项描述原文：这一层只判「答案指向第几个选项」，
    描述抄没抄对由产物级审计（`jc.contract.recorded_text`）管——两把尺子量的不是一件事。
    """
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == TYPE_CHOICE:
        names = option_names(question)
        index = answer.get("choice")
        if not is_index(index) or not 0 <= index < len(names):
            return None
        return names[index]
    if kind == TYPE_SCORE:
        if not isinstance(criteria, list):
            return None
        index = answer.get("score")
        if not is_index(index) or not 0 <= index < len(criteria):
            return None
        return index
    if kind == TYPE_NOUL:
        value = answer.get("noul")
        return value if isinstance(value, bool) else None
    return None


def compare_expected(
    sample: dict[str, JsonValue],
    expected: dict[str, list[JsonValue]],
    legal: Collection[str],
) -> list[str]:
    """输入：一条样本、按 state 重算出来的期望答案表、合法题族集合。输出：差异列表（空 = 全对）。

    预期行为：按题号走一遍，同族按出现顺序对上第几条；题多了、题少了、答案不对、题族不认识
    都各记一笔——不静默取第一个。「少题」那一趟遍历**期望表**（不是产物）：遍历产物时整族
    缺席会一声不响，漏报比误报危险。questions / answers 不是对象时直接返回一条，不抛。

    为什么题族表由调用方传进来：题族是各 workflow 自己的领域词表（`questions.FAMILIES`），
    通用层不认领域词表。
    """
    problems: list[str] = []
    questions_field = sample.get("questions")
    answers_field = sample.get("answers")
    if not isinstance(questions_field, dict) or not isinstance(answers_field, dict):
        return ["样本的 questions / answers 不是对象"]
    seen: dict[str, int] = {}
    for qid, question in questions_field.items():
        family = family_of(qid)
        if family not in legal:
            problems.append(f"未知题族 {family!r}（题号 {qid}）")
            continue
        index = seen.get(family, 0)
        seen[family] = index + 1
        want = expected.get(family, [])
        if index >= len(want):
            problems.append(f"{qid} 多出来了：文本里只算得出 {len(want)} 道 {family} 题")
            continue
        answer = answers_field.get(qid, {})
        got = answer_value(question, answer if isinstance(answer, dict) else {})
        if got != want[index]:
            problems.append(f"{qid}：答案是 {got!r}，按文本重算应为 {want[index]!r}")
    for family, want in expected.items():
        count = seen.get(family, 0)
        if count < len(want):
            problems.append(f"{family} 少了 {len(want) - count} 道题")
    return problems


class PhrasingConfig(Protocol):
    """带切分名的采样参数（每个 workflow 自己的 Config 都满足）。"""

    split: str


def plan_with_phrasing(
    config: PhrasingConfig,
    rng: random.Random,
    case: object,
    seq: int,
    pool: Sequence[str],
    *,
    key: str = "case",
) -> SplitSpec:
    """输入：采样参数、RNG、case、样本序号、问法池、case 在计划里的字段名。输出：生成计划。

    预期行为：问法编号由 RNG 从池里抽（问法**不是** case 维度），判定计划在 `sample` 阶段按
    「case 指纹 + 问法编号」确定性地算出来。40 个包的 `plan_sample` 只有两处不同：问法池叫
    `phrasing_pool` 还是 `phrasings`，以及那一格叫 `case` 还是 `tags`（11 个包的 case 就是一组
    tag）——两处都参数化，机制只留这一份。
    """
    return SplitSpec(
        name=config.split,
        params={
            key: case,
            "phrasing": rng.choice(pool),
            "seq": seq,
        },
    )
