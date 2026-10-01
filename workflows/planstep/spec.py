"""有界任务推进（planstep）的真值规格：注入特征 + 切分信息 + 由规则算出来的标签。

输入/输出：只有不可变数据类与纯函数；**不含领域词表**（任务名、题面与选项描述全在
`locales/`）。预期行为：四个题族的答案全部由 `features` 那四条规则算出来，零 token。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Final

from jc.jsontypes import JsonValue
from jc.knobs import require_pool

from .features import PlanCase, PlanLabels, label_for

#: 问法池（题面措辞不是判据，换说法不影响答案）。
PHRASINGS: Final[tuple[int, ...]] = (0, 1, 2)

#: 全枚举上限：同分布空间 10 图形状 × 6 个已完成项数 = 60 组，512 让枚举器稳稳走**全枚举**
#: （成对覆盖在这里只会白扔组合）。
COVER_LIMIT: Final[int] = 512


@dataclass(frozen=True)
class PlanConfig:
    """一个切分的采样参数。

    字段：`split` 切分名；`cover_limit` 全枚举上限；`case_limit` >0 时每个切分只取前 N 个
    组合（小规模试跑用，**会牺牲成对覆盖**）；`phrasings` 问法编号池。
    """

    split: str = "train"
    cover_limit: int = COVER_LIMIT
    case_limit: int = 0
    phrasings: tuple[int, ...] = PHRASINGS

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        require_pool("phrasings", self.phrasings)
        if self.cover_limit < 1:
            raise ValueError(f"cover_limit 必须 >= 1：{self.cover_limit}")
        if self.case_limit < 0:
            raise ValueError(f"case_limit 不能为负：{self.case_limit}（0 = 取全部）")


@dataclass(frozen=True)
class PlanSpec:
    """一条任务推进请求的完整真值：case + 由规则算出来的四个答案。

    字段：`case` 注入特征；`split` 切分名；`seq` 样本序号；`phrasing` 问法编号。
    """

    case: PlanCase
    split: str
    seq: int
    phrasing: int

    @property
    def labels(self) -> PlanLabels:
        """输入：无。输出：这一条的全部金标签（还剩几项 / 下一步 / 还要几轮 / 解不解锁）。"""
        return label_for(self.case)

    @property
    def query_id(self) -> str:
        """输入：无。输出：请求号（由素材指纹派生，稳定可复算）。"""
        return query_id_of(self.case)

    @property
    def remaining(self) -> int:
        """输入：无。输出：还剩几项没做完（`remaining` 那一族的答案）。"""
        return self.labels.remaining

    @property
    def next_key(self) -> str:
        """输入：无。输出：下一步那一项的选项名（`t1`~`t5` / `none`）。"""
        return self.labels.next_key

    @property
    def rounds(self) -> int:
        """输入：无。输出：还能推进几轮（`rounds` 那一族的答案）。"""
        return self.labels.rounds

    @property
    def unlocks(self) -> bool:
        """输入：无。输出：做完下一步会不会立刻解锁别的任务。"""
        return self.labels.unlocks


def query_id_of(case: PlanCase) -> str:
    """输入：一个 case。输出：请求号（由素材指纹派生，稳定可复算）。

    预期行为：同一张计划在任何切分、任何语言下都是同一个请求号——「同输入同标签」这条
    一致性校验要靠它认人。
    """
    digest = hashlib.sha256(case.fingerprint().encode("utf-8")).hexdigest()[:6].upper()
    return f"P-{digest}"


def sample_spec(rng: random.Random, case: PlanCase, split: str, seq: int, phrasing: int) -> PlanSpec:
    """输入：RNG（**不用**）、case、切分、序号、问法。输出：真值规格。

    预期行为：不掷骰子——同一 case 必得同一份规格，这样「同素材同题必须同答」才成立。
    """
    del rng  # 规格完全由 case 决定；签名里的 RNG 只为与协议一致
    return PlanSpec(case=case, split=split, seq=seq, phrasing=phrasing)


def spec_to_dict(spec: PlanSpec) -> dict[str, JsonValue]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.as_dict(),
        "split": spec.split,
        "seq": spec.seq,
        "query_id": spec.query_id,
        "remaining": spec.remaining,
        "next": spec.next_key,
        "rounds": spec.rounds,
        "unlocks": spec.unlocks,
    }
