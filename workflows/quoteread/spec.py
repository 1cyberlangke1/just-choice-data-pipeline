"""行情读数判读（quoteread）的真值规格：注入特征 + 切分信息 + 由规则算出来的标签。

输入/输出：只有不可变数据类与纯函数；**不含领域词表**（标的名、单位写法、题面与选项描述
全在 `locales/`）。预期行为：四个题族的答案全部由 `features` 那四条规则算出来，零 token。

**这一族的轴是「单位换算 + 档位」**：`receiptcat` 是**同量纲**比上限（金额对金额），这里
多了一步——读数表里每行的单位可能不同（`%` 或 `bp`），要先把 `-0.45%` 读成 `-45 基准点`，
才谈得上比档位、比警戒线、比谁涨得多。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Final

from jc.knobs import require_pool

from .features import QuoteCase, QuoteLabels, label_for

#: 问法池（题面措辞不是判据，换说法不影响答案）。
PHRASINGS: Final[tuple[int, ...]] = (0, 1, 2)

#: 全枚举上限：同分布空间 8 读数模式 × 4 单位模式 = 32 组，512 让枚举器稳稳走**全枚举**
#: （成对覆盖在这里只会白扔组合）。
COVER_LIMIT: Final[int] = 512


@dataclass(frozen=True)
class QuoteConfig:
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
class QuoteSpec:
    """一条行情读数请求的完整真值：case + 由规则算出来的四个答案。

    字段：`case` 注入特征；`split` 切分名；`seq` 样本序号；`phrasing` 问法编号。
    """

    case: QuoteCase
    split: str
    seq: int
    phrasing: int

    @property
    def labels(self) -> QuoteLabels:
        """输入：无。输出：这一条的全部金标签（档位 / 领涨 / 越线 / 背离）。"""
        return label_for(self.case)

    @property
    def query_id(self) -> str:
        """输入：无。输出：请求号（由素材指纹派生，稳定可复算）。"""
        return query_id_of(self.case)

    @property
    def tier(self) -> int:
        """输入：无。输出：最大绝对读数落在哪一档（`tier` 那一族的答案）。"""
        return self.labels.tier

    @property
    def leader(self) -> str:
        """输入：无。输出：涨幅最大那一行的**标的名键**（`leader` 那一族的答案）。"""
        return self.case.name_ids[self.labels.leader]

    @property
    def alert(self) -> bool:
        """输入：无。输出：有没有读数越过警戒线（`alert` 那一族的答案）。"""
        return self.labels.alert

    @property
    def divergent(self) -> bool:
        """输入：无。输出：有没有一涨一跌（`divergent` 那一族的答案）。"""
        return self.labels.divergent


def query_id_of(case: QuoteCase) -> str:
    """输入：一个 case。输出：请求号（由素材指纹派生，稳定可复算）。

    预期行为：同一份素材在任何切分、任何语言下都是同一个请求号——「同输入同标签」这条
    一致性校验要靠它认人。
    """
    digest = hashlib.sha256(case.fingerprint().encode("utf-8")).hexdigest()[:6].upper()
    return f"Q-{digest}"


def sample_spec(
    rng: random.Random, case: QuoteCase, split: str, seq: int, phrasing: int
) -> QuoteSpec:
    """输入：RNG（**不用**）、case、切分、序号、问法。输出：真值规格。

    预期行为：不掷骰子——同一 case 必得同一份规格，这样「同素材同题必须同答」才成立。
    """
    del rng  # 规格完全由 case 决定；签名里的 RNG 只为与协议一致
    return QuoteSpec(case=case, split=split, seq=seq, phrasing=phrasing)
