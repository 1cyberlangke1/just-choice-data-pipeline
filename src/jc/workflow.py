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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .judge import Judge

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
