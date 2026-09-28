"""函数调用（functions）workflow 的真值规格：一张工具表 + 一个 case 的绑定。

输入/输出：不可变数据类与纯函数；这里只有**键**（工具名 / 参数名 / 取值名），
中文英文的短语全在 `locales/`。
预期行为：答案只从渲染出来的请求文本反推——工具靠意图短语认，参数靠「值短语」认，
所以构造侧与判别侧共用同一套读法（`parse.read_request`）。

覆盖的决策形状（社区用例地图缺口 #4「类型化函数 / 工具派发」）：
- **多槽绑定**：一句自然语言请求 → 一个函数名 + 每个参数（闭集取值）；
- **按所选函数决定问哪些参数**：参数题只出**金工具**的那些参数（两阶段依赖）；
- **可选参数的在场判断**（`stated_*`）：请求压根没提 → 用函数默认值，这一族专门考「没提」。

三类参数：`choice`（闭集里取一个）、`set`（取一个子集，逐成员问）、`flag`（布尔，
出现即 True，取值由「提没提」决定）。必填参数不出 `stated_*` 题——它恒为 True，是常量答案。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import hashlib
import random
from dataclasses import dataclass
from itertools import product
from typing import Final

from jc.knobs import require_pool

# ---------- 参数类别 ----------

KIND_CHOICE: Final[str] = "choice"  # 闭集里取一个
KIND_SET: Final[str] = "set"  # 取一个子集（逐成员问）
KIND_FLAG: Final[str] = "flag"  # 布尔：出现即 True

# ---------- 取值表（键；文案在 locales/） ----------

STOCKS: Final[tuple[str, ...]] = ("NVDA", "AMD", "AAPL", "MSFT", "TSLA")
BENCHMARKS: Final[tuple[str, ...]] = ("SPY", "QQQ")
WINDOWS: Final[tuple[str, ...]] = ("1d", "1w", "1mo", "3mo")
RESOLUTIONS: Final[tuple[str, ...]] = ("1m", "5m", "15m", "1h", "1d")
STYLES: Final[tuple[str, ...]] = ("line", "candles")
DIRECTIONS: Final[tuple[str, ...]] = ("up", "down", "both")


@dataclass(frozen=True)
class Arg:
    """一个参数：名字 + 类别 + 取值表 + 是否必填。

    预期行为：`choice` / `set` 的 `values` 是取值表；`flag` 没有取值表（取值由「提没提」决定）。
    必填参数在每种语气下都必须被提到（由测试钉住）。
    """

    name: str
    kind: str
    values: tuple[str, ...] = ()
    required: bool = False
    #: 短语表（`[val]` / `[opt]`）用的键：`symbols`（集合）与 `symbol`（单个）共用一张股票表，
    #: 所以这里允许点名另一张表；留空表示就用参数自己的名字。
    phrase: str = ""


#: 工具表（取自官方 cookbook 的行情助手，参数取自它的 `Literal` 签名）。
#:
#: 与官方签名的一处**有意偏离**：`window` 在真实 API 里是可选的，这里标成必填。
#: 两个理由：① 请求里总要给一个时间范围，不给就没法画图；② 它同时是留出维度，
#: 必须**每次都出现在文本里**——一旦某档语气不提它，那个 case 与「只有窗口不同」的
#: 邻居就会渲染出同一句话，切分之间立刻撞车（实测踩过）。标成必填还顺带避免了
#: 「恒为 True 的 `stated_window` 题」。
TOOLS: Final[dict[str, tuple[Arg, ...]]] = {
    "plot_price": (
        Arg("symbol", KIND_CHOICE, STOCKS, required=True),
        Arg("window", KIND_CHOICE, WINDOWS, required=True),
        Arg("style", KIND_CHOICE, STYLES),
        Arg("resolution", KIND_CHOICE, RESOLUTIONS),
        Arg("include_volume", KIND_FLAG),
    ),
    "compare_returns": (
        Arg("symbols", KIND_SET, STOCKS, required=True, phrase="symbol"),
        Arg("window", KIND_CHOICE, WINDOWS, required=True),
        Arg("normalize", KIND_FLAG),
        Arg("resolution", KIND_CHOICE, RESOLUTIONS),
    ),
    "rolling_correlation": (
        Arg("symbol", KIND_CHOICE, STOCKS, required=True),
        Arg("benchmark", KIND_CHOICE, BENCHMARKS, required=True),
        Arg("window", KIND_CHOICE, WINDOWS, required=True),
        Arg("resolution", KIND_CHOICE, RESOLUTIONS),
        Arg("smoothing", KIND_FLAG),
    ),
    "top_movers": (
        Arg("window", KIND_CHOICE, WINDOWS, required=True),
        Arg("direction", KIND_CHOICE, DIRECTIONS),
        Arg("include_etfs", KIND_FLAG),
    ),
}
#: 取值表带「两种写法」的短语表键：`[val.<键>.ticker]`（写代码）与 `[val.<键>.name]`（写名字）。
ALIASED_KEYS: Final[frozenset[str]] = frozenset({"symbol", "benchmark"})
TOOL_NAMES: Final[tuple[str, ...]] = tuple(TOOLS)
#: 参数名 → 定义。同名参数在不同工具里必须是同一份定义（由测试钉住）。
ARG_BY_NAME: Final[dict[str, Arg]] = {
    arg.name: arg for args in TOOLS.values() for arg in args
}


def phrase_key(arg: Arg) -> str:
    """输入：参数定义。输出：它在 `[val]` / `[opt]` 里用的短语表键。

    预期行为：`symbols` 与 `symbol` 共用同一张股票短语表——两处写同一批股票，
    分成两张表就会漏改一张（而且反解要同时认两套短语，纯属自找歧义）。
    """
    return arg.phrase or arg.name


def phrase_key_of(name: str) -> str:
    """输入：参数名。输出：它的短语表键。预期行为：参数名不在工具表里直接报错。"""
    return phrase_key(ARG_BY_NAME[name])


def has_alias(tool: str) -> bool:
    """输入：工具名。输出：它的参数里有没有带两种写法的取值表。

    预期行为：`top_movers` 没有标的类参数，两种写法渲染出来一模一样——所以它的 `alias`
    维度只有一个取值（`build_cases` 会跳过另一支），否则同一句话会在两个「不同」的 case
    里各出一份，切分之间就会撞车。留出维度必须**每一维都真的体现在文本里**。
    """
    return any(phrase_key(arg) in ALIASED_KEYS for arg in TOOLS[tool])

# ---------- 语气（决定「哪些参数被提到」） ----------

#: 语气只有两档，而且是**有意**的：每个可选参数恰好被一种语气提到、另一种不提，
#: 于是 `stated_<参数>` 的 True 占比正好是 1/2、`member_*` 的 True 占比也正好是 1/2
#: （两种语气的成员数取 2 与 3，平均 2.5/5）。三档语气做不到这件事——每个参数只能被
#: 1/3 或 2/3 的语气提到，那一族就成了「偏心的抛硬币」，均衡闸在大批产物上会当场标它。
VOICES: Final[tuple[str, ...]] = ("terse", "chatty")
#: 语气 → 该语气下**被提到**的参数（没列出的可选参数一律不出现）。
VOICE_ARGS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "plot_price": {
        "terse": ("symbol", "window"),
        "chatty": ("symbol", "window", "style", "resolution", "include_volume"),
    },
    "compare_returns": {
        "terse": ("symbols", "window"),
        "chatty": ("symbols", "window", "normalize", "resolution"),
    },
    "rolling_correlation": {
        "terse": ("symbol", "benchmark", "window"),
        "chatty": ("symbol", "benchmark", "window", "resolution", "smoothing"),
    },
    "top_movers": {
        "terse": ("window",),
        "chatty": ("window", "direction", "include_etfs"),
    },
}
#: 语气 → `set` 参数列出几个成员（越啰嗦列得越多；不能超过取值表长度）。
VOICE_MEMBERS: Final[dict[str, int]] = {"terse": 2, "chatty": 3}

#: 哪些参数的内容写进 `lead` 的 `{subject}` 槽（其余被提到的参数各有自己的分段模板）。
SUBJECT_ARGS: Final[dict[str, tuple[str, ...]]] = {
    "plot_price": ("symbol",),
    "compare_returns": ("symbols",),
    "rolling_correlation": ("symbol", "benchmark"),
    "top_movers": (),
}

#: 别名的两种写法：直接写代码（`NVDA`）/ 写中文名（`英伟达`）。
ALIASES: Final[tuple[str, ...]] = ("ticker", "name")
#: 问法（句面模板）编号。
PHRASINGS: Final[tuple[int, ...]] = (0, 1)

# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的工具：`top_movers` 整个留出（连它的意图短语与参数一起没见过）。
SEEN_TOOLS: Final[frozenset[str]] = frozenset(
    {"plot_price", "compare_returns", "rolling_correlation"}
)
#: 训练侧见过的窗口：`3mo` 留出。
SEEN_WINDOWS: Final[frozenset[str]] = frozenset({"1d", "1w", "1mo"})
#: 训练侧见过的别名：只见过写代码，没见过写中文名。
SEEN_ALIASES: Final[frozenset[str]] = frozenset({"ticker"})
#: 训练侧见过的问法：第二套句面留出。
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({0})


@dataclass(frozen=True)
class CallCase:
    """一个 case：工具 × 语气 × 窗口取值 × 别名 × 问法。"""

    tool: str
    voice: str
    window: str
    alias: str
    phrasing: int

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：{维度名: 取值}（五个维度全摊平）。"""
        return {
            "tool": self.tool,
            "voice": self.voice,
            "window": self.window,
            "alias": self.alias,
            "phrasing": self.phrasing,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.tool}/{self.voice}/{self.window}/{self.alias}/{self.phrasing}"

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个组合里「训练侧没见过」的维度名（按固定顺序）。

        预期行为：`alias` 只对**真的有别名写法**的工具算数——`top_movers` 压根没有
        第二种写法，它那儿的 `alias` 是同一个取值。
        """
        out: list[str] = []
        if self.tool not in SEEN_TOOLS:
            out.append("tool")
        if self.window not in SEEN_WINDOWS:
            out.append("window")
        if self.alias not in SEEN_ALIASES and has_alias(self.tool):
            out.append("alias")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(out)


def material_of(case: CallCase) -> str:
    """输入：case。输出：素材指纹（确定性；切分分桶与内容种子都由它派生）。"""
    return f"{case.tool}|{case.voice}|{case.window}|{case.alias}|{case.phrasing}"


@dataclass(frozen=True)
class CallConfig:
    """一个切分的采样参数：五张取值表 + 是否只留 OOD。"""

    tools: tuple[str, ...] = TOOL_NAMES
    voices: tuple[str, ...] = VOICES
    windows: tuple[str, ...] = WINDOWS
    aliases: tuple[str, ...] = ALIASES
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("tools", self.tools)
        require_pool("voices", self.voices)
        require_pool("windows", self.windows)
        require_pool("aliases", self.aliases)
        require_pool("phrasings", self.phrasings)


def build_cases(config: CallConfig) -> list[CallCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：五张表取全组合（别名只对真有第二种写法的工具展开）；`ood_only` 时只留
    「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧**按构造就不重叠**。
    """
    out: list[CallCase] = []
    for tool, voice, window, alias, phrasing in product(
        config.tools, config.voices, config.windows, config.aliases, config.phrasings
    ):
        if alias != ALIASES[0] and not has_alias(tool):
            # 这个工具没有别名写法：第二种写法渲染出来跟第一种一模一样，别重复出。
            continue
        case = CallCase(tool, voice, window, alias, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 绑定计划（还没填文案） ----------


@dataclass(frozen=True)
class Binding:
    """一个参数的绑定（也是读法从文本里读出来的那一格）。

    `stated=False` 表示请求里**一个词都没提**这个参数 → 用函数默认值（`stated_*` 题的 False）。
    `members` 只有 `set` 参数用；`flag` 参数的 `value` 在提到时恒为 `"on"`。
    """

    name: str
    kind: str
    stated: bool
    value: str = ""
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class CallPlan:
    """一份请求的绑定计划：工具 + 每个参数 + 别名 + 问法。

    为什么不直接存文本：`sample` 阶段拿不到语言文件（协议里没有 locale），所以这里只定
    「工具 + 每个参数取什么值 + 用哪套句面」，真正的文本在 `render` 阶段填。
    同一份计划在任何语言下都能渲染，题目与答案两边用的是同一份计划。
    """

    tool: str
    args: tuple[Binding, ...]
    alias: str
    phrasing: int

    def binding(self, name: str) -> Binding:
        """输入：参数名。输出：它的绑定。预期行为：参数名不在工具表里直接报错。"""
        for binding in self.args:
            if binding.name == name:
                return binding
        raise KeyError(f"{self.tool} 没有参数 {name!r}")


@dataclass(frozen=True)
class CallSpec:
    """一条样本的真值：case + 绑定计划（文案在渲染阶段才填）。"""

    case: CallCase
    split: str
    seq: int
    plan: CallPlan


def _pick(rng: random.Random, pool: tuple[str, ...]) -> str:
    """输入：RNG、候选表。输出：一个取值。预期行为：确定性（同一个 RNG 状态同一结果）。"""
    return pool[rng.randrange(len(pool))]


def build_plan(case: CallCase) -> CallPlan:
    """输入：case。输出：绑定计划（与语言无关的随机）。

    预期行为：随机种子由**素材指纹**派生，所以同一个 case 在任何切分、任何进程里都得到
    同一份计划（「同素材同题必须同答」这条一致性校验才成立）。窗口取值来自 case 本身
    （它是留出维度），其余可选参数的取值由指纹掷出。
    """
    seed = int(hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    stated = set(VOICE_ARGS[case.tool][case.voice])
    args: list[Binding] = []
    for arg in TOOLS[case.tool]:
        if arg.name not in stated:
            args.append(Binding(arg.name, arg.kind, False))
        elif arg.kind == KIND_SET:
            count = min(VOICE_MEMBERS[case.voice], len(arg.values))
            picked = set(rng.sample(arg.values, count))
            # 按取值表本身的顺序排，而不是按字母排：读法（`parse.stated_values`）也是按
            # 取值表顺序返回的，两边顺序一致，「反解出来的计划 == 金计划」才成立。
            args.append(
                Binding(
                    arg.name,
                    arg.kind,
                    True,
                    members=tuple(value for value in arg.values if value in picked),
                )
            )
        elif arg.kind == KIND_FLAG:
            args.append(Binding(arg.name, arg.kind, True, value="on"))
        else:
            value = case.window if arg.name == "window" else _pick(rng, arg.values)
            args.append(Binding(arg.name, arg.kind, True, value=value))
    return CallPlan(case.tool, tuple(args), case.alias, case.phrasing)


def sample_spec(case: CallCase, split: str, seq: int) -> CallSpec:
    """输入：case、切分、序号。输出：真值规格。

    预期行为：**不掷骰子**——计划完全由 case 指纹决定，这样「同素材同题必须同答」成立。
    """
    return CallSpec(case, split, seq, build_plan(case))
