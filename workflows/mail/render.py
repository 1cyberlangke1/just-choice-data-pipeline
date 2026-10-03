"""邮件 workflow 的文本外壳：把特征清单交给后端写成一封邮件。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- 提示词**只描述注入特征**，不含任何答案、题号或标签；
- 后端回四个可见部分（显示名 / 发件地址 / 主题 / 正文），拼成 state；
- 文本是否真的体现了这些特征，交给 `judges.TraceJudge` 反解一次再比对——
  反解不出来就不猜，直接丢（`docs/可靠合成数据范式.md` §9 的纪律）。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue
from jc.workflow import DraftError

import re
from dataclasses import dataclass
from typing import Final

from jc.backend.types import GenerationTask
from pydantic import BaseModel, Field

from . import features, locales
from .features import TagSet

# 提示词里特征的固定顺序：顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("sender", "lure", "ask", "link", "tone")

# 只认带 scheme 或 www. 的写法。裸域名不认——签名里的公司名会被误伤，
# 而「有没有链接」这道判别器宁可漏检也不能误判（漏检只是少拒几条，误判会拒掉好数据）。
URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

#: 数地址用的「写全的网址」：**只吃 ASCII 网址字符**，遇到中文标点或括号就停。
#: 不能拿 `URL_PATTERN`（`\S+`）来数：它会把紧跟其后的中文一起吞掉
#: （`https://a.test/x（实际指向 …` 变成一个匹配），于是「同一个地址在括号里又写了一遍」
#: 会被数成两个不同的地址——模型很爱这么写（实跑抓到过
#: 「订单查询入口：http://…（链接文字与目标一致，均指向 http://…）」）。
FULL_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'*+,;=%]+",
    re.IGNORECASE,
)

#: 在「写全的网址」之外再算上**裸域名**（`mall.example.com`）。只在清单说这封邮件有链接时
#: 才该用：模型偶尔把「显示的文字」写成不带 scheme 的裸域名（实跑 141 条里 2 条），
#: 那种写法确实把两个地址都写清楚了，不该因此拒掉。没有链接的邮件最后一行是落款或
#: 「官方域名」行，那里的裸域名不算地址——否则每一封机构来信都会被数出一个地址。
ADDRESS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'*+,;=%]+"
    r"|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",
    re.IGNORECASE,
)

#: `From: 显示名 <地址>` 里的地址部分。只认尖括号里那一段：显示名里也可能带域名
#: （「汇信银行 huixinbank.com 客服」），拿整行去找域名会把显示名误当成发件地址。
FROM_ADDRESS_PATTERN: Final[re.Pattern[str]] = re.compile(r"^From:.*?<([^>]*)>", re.MULTILINE)


class MailDraft(BaseModel):
    """后端要填的草稿：一封邮件的五个可见部分。

    字段刻意全是字符串——后端**只写文本**，编号、题号与 `type` 一律由组装器产生
    （PLAN 3.1 不变量①）。

    **落款单独占一个字段**，不许塞进 `body`：实跑里模型会整块漏掉落款，链接与
    「官方域名」行就跟着一起蒸发（LinkJudge 直接判「注入有链接、文本里却没有」）。
    放进 schema 是结构上强制——提示词里写十遍规则，不如让它少一个必填字段。
    """

    from_name: str = Field(
        description="发件人显示名：自己编一个具体的机构名或普通人名，不要用「某某」「示例」这类占位词"
    )
    from_address: str = Field(
        description=(
            "发件地址，形如「用户名@域名」；域名用这一封信里自己编的那个，"
            "不要照抄本字段说明里的任何字"
        )
    )
    subject: str = Field(description="邮件主题")
    body: str = Field(description="邮件正文，3~8 句，可含换行；**不含落款**")
    signature: str = Field(
        description="正文末尾的落款 / 签名区（至少写署名）；官方域名另有专门字段，不要写在这里",
    )
    official_domain: str = Field(
        description=(
            "落款里「官方域名」那一行要写的域名（只写域名本身，不带 http:// 与斜杠）；"
            "发件人是陌生个人、不声称任何机构身份时填空字符串"
        )
    )
    link_line: str = Field(
        description=(
            "链接那一行：清单说有链接时必须写出来（文字与目标不一致时两个地址都写）；"
            "清单说没有链接就填空字符串"
        )
    )


@dataclass(frozen=True)
class ParsedMail:
    """从邮件文本里纯代码反解出来的事实（能反解的部分只有链接）。"""

    urls: tuple[str, ...]

    @property
    def has_url(self) -> bool:
        """输入：无。输出：文本里有没有链接。"""
        return bool(self.urls)


def parse_state(state: object) -> ParsedMail:
    """从 state 里抽出所有链接。

    输入：state（本 workflow 只会是字符串）。
    输出：`ParsedMail`。
    预期行为：非字符串 state 视为没有链接——契约允许 object/array，但本 workflow 不产。
    """
    if not isinstance(state, str):
        return ParsedMail(())
    return ParsedMail(tuple(URL_PATTERN.findall(state)))


def link_line_urls(state: object, *, bare_domains: bool = False) -> tuple[str, ...]:
    """输入：state、是否把裸域名也算地址。输出：**最后一行**里的不同地址（去重、保持顺序）。

    预期行为：「链接那一行」由 `to_state` 追加在末尾，所以最后一行就是链接行——
    这条布局不变量由本模块的测试盯着。去重是因为模型会把同一个地址在括号里再写一遍
    （实跑抓到过「订单查询入口：http://…（链接文字与目标一致，均指向 http://…）」）。
    没有链接的邮件，最后一行是落款或「官方域名」行，这里自然返回空（`bare_domains` 见上）。
    """
    if not isinstance(state, str):
        return ()
    lines = state.splitlines()
    if not lines:
        return ()
    pattern = ADDRESS_PATTERN if bare_domains else FULL_URL_PATTERN
    return tuple(dict.fromkeys(pattern.findall(lines[-1])))


def link_text_of(state: object, *, bare_domains: bool = False) -> str:
    """输入：state、是否把裸域名也算地址。输出：这一族的答案（`no_link` / `matches` / `mismatched`）。

    预期行为：纯代码反解——渲染规则要求「链接文字与真实目标不一致时两个地址都写」，
    所以**数链接行里的不同地址**就够了：0 个 → 没有链接、1 个 → 文字就是目标、
    2 个及以上 → 文字与目标不是同一个地址。2026-09-26 在 130 条真端点产物上核过：
    `matches` 恰好 1 个、`mismatched` 恰好 2 个、`no_link` 0 个，去重后 129/130 一致
    （唯一一条例外是模型把显示地址写成不带 scheme 的裸域名——规则 4 本来就要求网址写全，
    所以那种写法该拒）。
    """
    urls = link_line_urls(state, bare_domains=bare_domains)
    if not urls:
        return "no_link"
    return "matches" if len(urls) == 1 else "mismatched"


def build_prompt(
    tags: TagSet, loc: dict[str, JsonValue], *, link_path: str, link_ip: str
) -> str:
    """组装渲染提示词。

    输入：特征组合、语言字典、这一条的链接路径段与 IP。
    输出：提示词字符串。
    预期行为：只出现特征清单与硬性要求；维度顺序按 `DIM_ORDER` 固定。

    **链接地址按样本给**（范式 §9.25 / §9.60）：提示词原来在 `hint` 里直接给了成品地址
    （`http://203.0.113.47/verify` 那种），真机 `mail-kimi-ood-c` 六条里五条把它原样抄了下来。
    现在 `hint` 只说形状，具体地址由 `link_address_line` 按这一条的样本序号给出——模型有东西
    可写，又不会整批塌成同一个地址。清单说「没有链接」时这一行不出现（那一条要求全篇无网址）。
    """
    lines = [loc["prompt"]["intro"], ""]
    values = tags.as_dict()
    lines += [locales.feature_line(loc, dim, values[dim]) for dim in DIM_ORDER]
    if tags.link != features.LINK_NONE:
        lines.append(
            str(loc["prompt"]["link_address_line"]).format(path=link_path, ip=link_ip)
        )
    lines += [""]
    lines += [
        "- " + rule.format(language=loc["language_name"]) for rule in loc["prompt"]["rules"]
    ]
    return "\n".join(lines)


def build_task(
    tags: TagSet, loc: dict[str, JsonValue], task_id: str, *, link_path: str, link_ip: str
) -> GenerationTask:
    """把一次渲染包成后端任务。

    输入：特征组合、语言字典、任务 id、这一条的链接路径段与 IP。
    输出：`GenerationTask`。
    预期行为：`spec` 里只放**特征**（不是答案）——后端看不到金标签，
    这样「标签由特征决定」这条性质不会被模型污染。
    """
    return GenerationTask(
        task_id=task_id,
        kind="text",  # 不是题型：这一步只产文本，题型的编号与 type 由组装器决定
        domain="mail",
        prompt=build_prompt(tags, loc, link_path=link_path, link_ip=link_ip),
        response_model=MailDraft,
        spec={"tags": tags.as_dict()},
    )


def domain_line_prefix(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：「官方域名」那一行的行首标记（模板里 `{domain}` 之前的部分）。

    预期行为：从 locale 模板推出来，代码里不写死任何语言的字面量——zh 是 `官方域名：`，
    en 是 `official domain: `。
    """
    return loc["prompt"]["official_domain_line"].split("{", 1)[0]


def owned_domain_line_value(line: str, loc: dict[str, JsonValue]) -> str | None:
    """输入：一行文本、语言字典。输出：「官方域名」那一行的取值；不是那一行 → None。

    预期行为：只认**行首标记**，但大小写不敏感——渲染器那行来自 locale 模板（en 是小写
    `official domain: `），模型偶尔写成 `Official domain: `（实跑 `ml-36c-en` 抓到过），
    按字面量比就会漏掉，state 里于是留下两行。比的是前 `len(prefix)` 个字符的 `lower()`，
    这样切出来的取值长度与模板一致（`casefold` 会改个别字符的长度，这里不用）。
    """
    prefix = domain_line_prefix(loc)
    stripped = line.strip()
    if stripped[: len(prefix)].lower() != prefix.lower():
        return None
    return stripped[len(prefix) :].strip()


def strip_owned_domain_lines(text: str, loc: dict[str, JsonValue]) -> str:
    """输入：一段模型写的文本、语言字典。输出：剔掉「官方域名」行之后的文本。

    预期行为：那一行**归渲染器写**（`to_state` 负责追加），模型写进 `body` / `signature`
    的副本一律作废——否则 state 里会出现两行，或者出现一行与注入特征矛盾的域名。
    实测 kimi-k3 六条里四条把这一行又写进了落款（glm 那次 35 条里 2 条），
    所以这不是偶发：schema 里写了「不要写在这里」也没用，只能结构上接管。
    只认行首标记，不做模糊匹配；大小写不敏感（口径见 `owned_domain_line_value`）。
    """
    kept = [line for line in text.splitlines() if owned_domain_line_value(line, loc) is None]
    return "\n".join(kept).strip()


def official_domain_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：「官方域名」那一行里的域名（没有那一行 → 空串）。

    预期行为：只认行首标记那一行、只取标记之后的文本（去空白）。这一行**归渲染器写**，
    而 `check_official_domain` 又强制「声称机构 ⟺ 这一行非空」——所以「这一行在不在」
    就等于「这封信有没有声称机构身份」，`sender_identity` 那一族的真值全靠它。
    """
    if not isinstance(state, str):
        return ""
    for line in state.splitlines():
        value = owned_domain_line_value(line, loc)
        if value is not None:
            return value
    return ""


def sender_domain_of(state: object) -> str:
    """输入：state。输出：`From:` 行里发件地址（解析不出 → 空串）。

    预期行为：只取尖括号里那一段——显示名里也可能带域名，不能被当成发件地址。
    返回的是地址原文（含 `@` 前面那段），归一化交给 `host_of`。
    """
    if not isinstance(state, str):
        return ""
    match = FROM_ADDRESS_PATTERN.search(state)
    return match.group(1).strip() if match else ""


def host_of(text: str) -> str:
    """输入：域名或邮箱地址。输出：小写、去掉 `www.` 的域名。

    预期行为：有 `@` 就取 `@` 后面那段，否则整串当域名；空串返回空串（调用方据此判
    「读不到」）。去掉 `www.` 是因为模型偶尔把官方域名写成 `www.example.test`，
    那与 `example.test` 是同一个站点，不该被算成「冒充」。
    """
    lowered = text.strip().lower()
    if not lowered:
        return ""
    host = lowered.rsplit("@", 1)[-1] if "@" in lowered else lowered
    return host[4:] if host.startswith("www.") else host


def to_state(data: dict[str, JsonValue], loc: dict[str, JsonValue]) -> str:
    """把草稿拼成 state。

    输入：草稿字典（`from_name` / `from_address` / `subject` / `body` / `signature` /
    `official_domain` / `link_line`）、语言字典（「官方域名」那一行的写法）。
    输出：`From: 名字 <地址>` + 主题 + 空行 + 正文 + 空行 + 落款（+ 官方域名行 + 链接行）。
    预期行为：只拼接与去首尾空白，不改写模型写的字——反解校验要靠原文；唯一的例外是
    「官方域名」那一行：它归渲染器写，模型在正文或落款里又写一遍的副本会被剔掉
    （见 `strip_owned_domain_lines`），所以 state 里只会有一行、且一定是注入的那个域名。
    `official_domain` 为空串时不出现那一行。
    """
    body = strip_owned_domain_lines(str(data["body"]).strip(), loc)
    signature = strip_owned_domain_lines(str(data["signature"]).strip(), loc)
    state = (
        f"From: {data['from_name'].strip()} <{data['from_address'].strip()}>\n"
        f"Subject: {data['subject'].strip()}\n\n"
        f"{body}\n\n"
        f"{signature}"
    )
    domain = str(data.get("official_domain", "")).strip()
    if domain:
        state = f"{state}\n{loc['prompt']['official_domain_line'].format(domain=domain)}"
    link_line = data["link_line"].strip()
    return f"{state}\n{link_line}" if link_line else state


def check_official_domain(tags: TagSet, data: dict[str, JsonValue]) -> None:
    """输入：注入特征与草稿字典。输出：无；不合规就抛 `DraftError`。

    预期行为：**声称机构身份**（`known_vendor` / `forged_brand`）时「官方域名」字段必须非空，
    陌生个人必须留空。这条不能只靠提示词：实跑里模型整块漏掉了这一行，
    判别器在邮件里看不到「冒充」的证据，于是把注入的 `forged` 反解成 `matches`
    （smoke-8 / smoke-10a 两次都栽在这里）。
    """
    domain = str(data.get("official_domain", "")).strip()
    claims_org = tags.sender != "unknown_personal"
    if claims_org and not domain:
        raise DraftError("草稿声称了机构身份，却没有写「官方域名」字段——判别器看不出真伪")
    if not claims_org and domain:
        raise DraftError("草稿不声称机构身份（陌生个人），「官方域名」字段必须留空")


def check_signature(data: dict[str, JsonValue]) -> None:
    """输入：草稿字典。输出：无；落款是空的就抛 `DraftError`。

    预期行为：**落款必须非空**——落款整块没了，链接与「官方域名」行会跟着一起蒸发
    （判别器判「注入有链接、文本里却没有」）。这条以前挂在 schema 的 `min_length=1` 上，
    但字符串长度约束会让英伟达那条路把换行压平（见 `tests/core/test_draft_schema.py`），
    所以把关位置从 schema 挪到这里：空落款照样当场拒稿、走重渲染，只是不再依赖 pydantic。
    """
    if not str(data.get("signature", "")).strip():
        raise DraftError("草稿的「落款」是空的——链接与「官方域名」行会跟着一起蒸发")


def check_subject(data: dict[str, JsonValue]) -> None:
    """输入：草稿字典。输出：无；主题是空的就抛 `DraftError`。

    预期行为：**主题必须非空**——真机里模型偶尔把 `subject` 留成空串，渲染出来就是一行
    「Subject:」后面什么都没有（36 条那两批 68 条里 11 条，10 条那批 10 条里 1 条）。
    空主题的信不像真的，判别器也读不出这一封想干什么。与 `check_signature` 同理，把关位置
    放在渲染期而不是 schema 的 `min_length`（字符串长度约束会让英伟达那条路把换行压平）。
    """
    if not str(data.get("subject", "")).strip():
        raise DraftError("草稿的「主题」是空的——渲染出来就是一行「Subject:」后面什么都没有")


#: 占位式的名字 / 域名：模型偷懒时会把**类别词**当成真名抄下来，或者照抄字段说明里的示例。
#: 2026-10-03 全量扫 998 条 mail 产物，62 条命中：
#: `VendorName <support@vendorofficial.com>`、`Vendor Support <support@vendor.com>`、
#: `Acme Corp <info@acmecorp.com>`、`ExampleVendor <support@examplevendor.com>`、
#: `某某公司 <no-reply@companyofficial.com>`、`示例银行客服 <service@example-bank.com>`。
#: 两档判据，避免误伤真名字：
#: - `PLACEHOLDER_STEMS` 是长词根，**按子串**比——`VendorName` 归一化成 `vendorname` 也认；
#: - `PLACEHOLDER_WORDS` 是短词，**按整词**比——`Brand` 按子串比会把 `Brandon Smith`
#:   这种真人名误判成占位词（拒稿要多花一次调用，宁可漏一条也不误伤）。
PLACEHOLDER_STEMS: Final[tuple[str, ...]] = (
    "vendor",
    "company",
    "brandname",
    "sender",
    "recipient",
    "placeholder",
    "corporation",
    "somename",
    "acme",
    "examplebank",
    "examplecorp",
    "examplebrand",
    "examplecompany",
    "某某",
    "示例",
    "占位",
)
PLACEHOLDER_WORDS: Final[frozenset[str]] = frozenset({"brand"})

#: 按词切名字用的形状：只吃 ASCII 字母数字（中文名字由 `PLACEHOLDER_STEMS` 的子串比管）。
WORD_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


def placeholder_in(text: str) -> str | None:
    """输入：一个名字或域名。输出：命中的占位词根；没命中 → None。

    预期行为：先归一化——去掉空白与标点、转小写，中文与字母数字都留着——再按子串比长词根，
    最后按整词比短词（两档的取舍见 `PLACEHOLDER_STEMS` 上面那段）。空串返回 None：
    空名字归 schema 的必填字段管，这里不重复报。
    """
    lowered = text.lower()
    squashed = re.sub(r"[\W_]+", "", lowered)
    for stem in PLACEHOLDER_STEMS:
        if stem in squashed:
            return stem
    for word in WORD_PATTERN.findall(lowered):
        if word in PLACEHOLDER_WORDS:
            return word
    return None


def check_names(data: dict[str, JsonValue]) -> None:
    """输入：草稿字典。输出：无；显示名或域名里带占位词就抛 `DraftError`。

    预期行为：查 `from_name` / `from_address` / `official_domain` 三个字段。
    提示词里写了「自己编一个具体的机构名」，但真机里模型照样把类别词当名字抄下来
    （统计见 `PLACEHOLDER_STEMS` 上面那段）。这种样本的**标签是对的**（金答案由特征算），
    可名字一看就是模板，收进数据集只会教模型「Vendor 是个公司名」——所以当场拒稿走重渲染，
    代价是一次调用。
    """
    for field in ("from_name", "from_address", "official_domain"):
        value = str(data.get(field, ""))
        hit = placeholder_in(value)
        if hit:
            raise DraftError(
                f"草稿的「{field}」里出现了占位词 {hit!r}（{value}）——要自己编一个具体的名字"
            )
