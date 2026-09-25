"""邮件 workflow 的文本外壳测试：草稿 schema 与 state 拼接。

输入：`workflows.mail.render` 的 pydantic 模型与纯函数。输出：断言结果。
预期行为：落款是**必填的独立字段**——漏了当场校验失败，不许静默把链接丢掉；
`to_state` 只拼接与去首尾空白，不改写模型写的字（唯一的例外是「官方域名」那一行，
它归渲染器写）。不发网络请求。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from workflows.mail import features, locales, render


def _loc() -> dict[str, object]:
    """输入：无。输出：中文语言字典。预期行为：`to_state` 要用它拼「官方域名」那一行。"""
    return locales.load("zh")


def _draft(**overrides: str) -> dict[str, str]:
    """输入：要覆盖的字段。输出：一份完整的草稿字典。"""
    base = {
        "from_name": "Example Bank",
        "from_address": "service@example-bank.com",
        "subject": "账户通知",
        "body": "请核对您的账户信息。",
        "signature": "Example Bank",
        "official_domain": "example-bank.com",
        "link_line": "https://example-bank.com/verify",
    }
    return {**base, **overrides}


def test_signature_is_a_required_non_empty_field() -> None:
    """落款漏了或写成空串都要校验失败。

    输入：缺落款 / 空落款两份草稿。输出：无。
    预期行为：实跑里模型会整块漏掉落款，链接与「官方域名」行跟着一起蒸发；
    放进 schema 才能结构上挡住，而不是靠提示词里多写一条规则。
    """
    without = _draft()
    del without["signature"]
    with pytest.raises(ValidationError):
        render.MailDraft(**without)
    with pytest.raises(ValidationError):
        render.MailDraft(**_draft(signature=""))
    assert render.MailDraft(**_draft()).signature


def test_to_state_appends_the_signature_after_the_body() -> None:
    """state = 发件人 + 主题 + 空行 + 正文 + 空行 + 落款 + 链接行，且只去首尾空白。"""
    state = render.to_state(
        _draft(body="  正文。  ", signature="  落款  ", link_line="  https://e.test/a  "),
        _loc(),
    )
    assert state == (
        "From: Example Bank <service@example-bank.com>\n"
        "Subject: 账户通知\n\n"
        "正文。\n\n"
        "落款\n"
        "官方域名：example-bank.com\n"
        "https://e.test/a"
    )


def test_to_state_leaves_out_an_empty_link_line() -> None:
    """清单说没有链接时，链接字段留空，state 里就不许多出那一行。"""
    state = render.to_state(_draft(link_line=""), _loc())
    assert state.endswith("官方域名：example-bank.com")


def test_to_state_leaves_out_an_empty_official_domain() -> None:
    """陌生个人的邮件没有官方域名：字段留空时 state 里不许出现那一行。"""
    state = render.to_state(_draft(official_domain=""), _loc())
    assert "官方域名" not in state


def test_to_state_takes_over_a_domain_line_the_model_wrote_itself() -> None:
    """模型把「官方域名」那一行写进落款时，state 里只能剩渲染器写的那一行。

    输入：落款里自带域名行、正文里也塞了一行的草稿。输出：state。
    预期行为：模型写的副本一律作废——否则 state 里会出现两行（实测 kimi-k3 六条里四条
    这么干，glm 那次 35 条里 2 条），或者出现一行与注入特征矛盾的域名。
    写成**别的**域名同样作废：落盘的那一行永远是 `official_domain` 字段里的那个。
    """
    state = render.to_state(
        _draft(
            body="正文。\n官方域名：evil.test",
            signature="落款\n官方域名：example-bank.com",
        ),
        _loc(),
    )
    assert state.count("官方域名") == 1
    assert state.endswith("官方域名：example-bank.com\nhttps://example-bank.com/verify")
    assert "evil.test" not in state


def test_to_state_keeps_other_lines_of_the_signature() -> None:
    """剔的只是那一行，落款里的其它内容一个字都不许动。"""
    state = render.to_state(
        _draft(signature="诺兰商城客户服务部\n官方域名：example-bank.com\n客服热线：400-000-0000"),
        _loc(),
    )
    assert "诺兰商城客户服务部" in state
    assert "客服热线：400-000-0000" in state
    assert state.count("官方域名") == 1


def test_strip_owned_domain_lines_reads_the_marker_from_the_locale() -> None:
    """行首标记从 locale 模板推出来，所以英文邮件照样剔得掉。"""
    assert render.domain_line_prefix(locales.load("zh")) == "官方域名："
    assert render.domain_line_prefix(locales.load("en")) == "official domain: "
    assert (
        render.strip_owned_domain_lines("Regards\nofficial domain: e.test", locales.load("en"))
        == "Regards"
    )


def test_official_domain_is_required_for_org_claims() -> None:
    """声称机构身份（正规商家 / 冒充品牌）时，官方域名必须填。

    输入：两种机构身份 × 空/非空官方域名。输出：无。
    预期行为：**只有陌生个人才允许留空**——实跑里模型整块漏掉这一行，
    判别器看不到「冒充」的证据，把注入的 `forged` 反解成 `matches`。
    """
    org_cases = ("known_vendor", "forged_brand")
    for sender in org_cases:
        tags = features.TagSet(sender, "none", "none", "none", "formal")
        render.check_official_domain(tags, {"official_domain": "example-bank.com"})
        with pytest.raises(ValueError, match="官方域名"):
            render.check_official_domain(tags, {"official_domain": "  "})


def test_official_domain_must_be_empty_for_plain_person() -> None:
    """陌生个人不声称机构身份：填了官方域名反而是错的（文本会读成机构邮件）。"""
    tags = features.TagSet("unknown_personal", "none", "none", "none", "formal")
    render.check_official_domain(tags, {"official_domain": ""})
    with pytest.raises(ValueError, match="必须留空"):
        render.check_official_domain(tags, {"official_domain": "example-bank.com"})


def test_links_in_the_signature_reach_the_link_judge() -> None:
    """落款里的链接必须能被 `parse_state` 抓到——判别器只看整条 state。"""
    state = render.to_state(
        _draft(link_line="Example Bank https://example-bank.com/verify"), _loc()
    )
    assert render.parse_state(state).urls == ("https://example-bank.com/verify",)
