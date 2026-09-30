"""邮件 workflow 的端到端测试：假后端跑通整条管线，并验证判别器真的会拦错。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import json
from pathlib import Path


import pytest

from jc import jsonio
from jc import pipeline, workflow as workflow_mod
from jc.backend.types import AllProvidersDown, GenerationBackend, TransportError
from workflows.mail import features, judges as judges_mod, locales, workflow as mail_workflow
from workflows.mail.spec import MailSpec

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="mail",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=mail_workflow.MailConfig(split=split, case_limit=CASE_LIMIT),
        backend=backend,
    )


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: GenerationBackend) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、五个判别器都记到通过；产物能过审计。
    """
    result = pipeline.run(make_config(tmp_path, "m1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {
        "link",
        "link_text",
        "sender_identity",
        "hygiene",
        "trace",
    }
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("mail").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    assert report["train"]["link_bad"] == 0


def test_trace_judge_rejects_when_reverse_parse_disagrees(tmp_path: Path, fake_backend: GenerationBackend) -> None:
    """输入：反解时故意说反话的假后端。输出：RunResult。

    预期行为：整批被拒，拒绝码是 E_JUDGE_TRACE——「文本对不上注入特征」绝不允许混进数据集。
    """
    result = pipeline.run(make_config(tmp_path, "m2", fake_backend(trace_lies=True)))
    assert result.accepted == 0
    assert result.stats["reject_codes"] == {"E_JUDGE_TRACE": CASE_LIMIT}
    # 被拒的行必须带上 state：不然「模型写了什么」这条唯一的诊断线索就丢了
    assert all(row["state"] for row in result.rejected)


def test_link_judge_flags_an_extra_url() -> None:
    """输入：注入「没有链接」但文本里塞了网址。输出：Verdict。

    预期行为：判不通过，并把抓到的链接放进 raw 便于排查。
    """
    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "none", features.LINK_NONE, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    verdict = judges_mod.LinkJudge().judge({"state": "见 https://example.test/a"}, spec)
    assert verdict.ok is False
    assert verdict.raw["urls"] == ["https://example.test/a"]


def test_link_judge_passes_when_consistent() -> None:
    """输入：注入「有链接」且文本里确实有链接。输出：Verdict。预期行为：通过。"""
    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "none", "official_honest", "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    assert judges_mod.LinkJudge().judge({"state": "见 https://example.test/a"}, spec).ok is True
    assert judges_mod.LinkJudge().judge({"state": "没有链接"}, spec).ok is False


def test_link_address_reaches_the_render_task() -> None:
    """输入：一条带链接的规格。输出：无。

    预期行为：`MailSpec` 按样本序号派生的路径段与 IP 真的进了渲染任务的提示词（范式 §9.25 /
    §9.60 的渲染期变量：它不进 case 空间、不进 `stats_dimensions`，只改链接地址怎么写）。真机
    `mail-kimi-ood-c` 六条里五条把提示词里的示例 IP 原样抄了下来，所以这条地址必须逐条不同。
    """
    from workflows.mail import render

    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "none", "ip_mismatch", "formal"),
        phrasing=0,
        split="train",
        seq=3,
    )
    assert spec.link_path == features.link_path_at(3)
    assert spec.link_ip == features.link_ip_at(3)
    task = render.build_task(
        spec.tags,
        locales.load("zh"),
        spec.render_task_id,
        link_path=spec.link_path,
        link_ip=spec.link_ip,
    )
    assert f"/{spec.link_path}" in task.prompt
    assert spec.link_ip in task.prompt


@pytest.mark.parametrize(
    ("link", "state"),
    [
        (features.LINK_NONE, "落款\n官方域名：example.test"),
        ("official_honest", "落款\n官方域名：example.test\nhttps://example.test/a"),
        (
            "official_mismatch",
            "落款\n官方域名：example.test\nhttps://other.test/a（实际指向 https://example.test/a）",
        ),
    ],
)
def test_link_text_judge_derives_the_answer_from_the_link_line(link: str, state: str) -> None:
    """零 token：链接行里 0 / 1 / 2 个不同网址 → `no_link` / `matches` / `mismatched`。

    输入：三种链接行写法（与注入的写法一一对应）。输出：Verdict。
    预期行为：数地址就够、不问模型——这一族的答案本来就是**结构事实**
    （渲染规则要求「文字与目标不一致时两个地址都写」，见 `render.link_text_of`）。
    """
    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "attachment", link, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    verdict = judges_mod.LinkTextJudge().judge({"state": state}, spec)
    assert verdict.ok is True, verdict.reason


def test_link_text_judge_rejects_the_wrong_number_of_addresses() -> None:
    """注入「文字就是目标」，链接行却写了两个地址 → 不通过。

    输入：一份 `official_honest` 规格 + 两地址链接行。输出：Verdict。
    预期行为：不通过，并说清注入什么、反解成什么——地址个数就是这一族的真值。
    """
    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "attachment", "official_honest", "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    verdict = judges_mod.LinkTextJudge().judge(
        {"state": "https://other.test/a（实际指向 https://example.test/a）"}, spec
    )
    assert verdict.ok is False
    assert "matches" in verdict.reason and "mismatched" in verdict.reason


def test_link_text_judge_counts_the_same_address_once() -> None:
    """同一个地址在括号里再写一遍只算一个（实跑抓到过这种写法）。"""
    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "attachment", "official_honest", "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    state = "订单查询入口：https://example.test/a（链接文字与目标一致，均指向 https://example.test/a）"
    assert judges_mod.LinkTextJudge().judge({"state": state}, spec).ok is True


def test_link_text_judge_skips_without_spec() -> None:
    """没有 spec 就没有真值：记成跳过（`skipped=True`），不误判。"""
    verdict = judges_mod.LinkTextJudge().judge({"state": "x"})
    assert verdict.ok is True and verdict.skipped is True


def test_link_text_judge_counts_a_bare_domain_when_the_mail_has_a_link() -> None:
    """显示的文字写成裸域名（不带 scheme）也算一个地址——只要清单说这封邮件有链接。

    输入：`official_mismatch` 规格 + 「裸域名（实际指向 https://…）」的链接行。输出：Verdict。
    预期行为：通过（两个地址都写清楚了，不该拒）；同一份文本在**没有链接**的规格下则算
    `no_link`——那种邮件的最后一行是落款或「官方域名」行，那里的裸域名不算地址。
    """
    linked = MailSpec(
        tags=features.TagSet("known_vendor", "none", "attachment", "official_mismatch", "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    state = "查看详情: mall.example.com（实际指向 https://s.example/abc123）"
    assert judges_mod.LinkTextJudge().judge({"state": state}, linked).ok is True
    linkless = MailSpec(
        tags=features.TagSet("known_vendor", "none", "attachment", features.LINK_NONE, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    assert judges_mod.LinkTextJudge().judge({"state": "落款\n官方域名：example.test"}, linkless).ok


@pytest.mark.parametrize(
    ("sender", "state"),
    [
        # 陌生个人：不声称机构身份，渲染器把「官方域名」那一行留空
        ("unknown_personal", "From: 小雅 <xiaoya@mail.test>\nSubject: 在吗\n\n正文\n小雅"),
        # 正规商家：显示名声称的机构，其官方域名与发件地址的域名一致
        (
            "known_vendor",
            "From: 汇信银行客服 <service@huixinbank.com>\nSubject: 账单\n\n正文\n"
            "汇信银行\n官方域名：huixinbank.com",
        ),
        # 冒充：显示名声称的是别人，官方域名与发件地址的域名不一致
        (
            "forged_brand",
            "From: 汇信银行客服 <service@huixin-bank-secure.com>\nSubject: 账单\n\n正文\n"
            "汇信银行\n官方域名：huixinbank.com",
        ),
    ],
)
def test_sender_identity_judge_reads_the_official_domain_line(sender: str, state: str) -> None:
    """零 token：有没有「官方域名」行 + 它与发件地址域名同不同 → 这一族的答案。

    输入：三种写法（与注入的发件人档位一一对应）。输出：Verdict。
    预期行为：只看这两个结构事实、不问模型——`check_official_domain` 已经在装配时
    结构上强制「声称机构 ⟺ 这一行非空」，所以这一族的真值也是**结构事实**，
    交给 LLM 反解只会多一处误拒（同 10.24 的 `link_text`）。
    """
    spec = MailSpec(
        tags=features.TagSet(sender, "none", "none", features.LINK_NONE, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    verdict = judges_mod.SenderIdentityJudge("zh").judge({"state": state}, spec)
    assert verdict.ok is True, verdict.reason


def test_sender_identity_judge_rejects_a_claim_without_the_domain_line() -> None:
    """注入「正规商家」但落款没有「官方域名」行 → 不通过。

    输入：一份 `known_vendor` 规格 + 没有那一行的文本。输出：Verdict。
    预期行为：不通过，理由里说清注入什么、反解成什么——实跑抓到过这种产物
    （`mail-kimi-ood-c` 的 0000026，装配时就该被拦下）。
    """
    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "none", features.LINK_NONE, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    verdict = judges_mod.SenderIdentityJudge("zh").judge(
        {"state": "From: 汇信银行 <service@huixinbank.com>\nSubject: 账单\n\n正文\n汇信银行"}, spec
    )
    assert verdict.ok is False
    assert "matches" in verdict.reason and "no_claim" in verdict.reason


def test_sender_identity_judge_rejects_a_forged_sender_that_looks_official() -> None:
    """注入「冒充」但官方域名与发件地址一致 → 不通过（文本里没有冒充的证据）。

    输入：一份 `forged_brand` 规格 + 两个域名一致的文本。输出：Verdict。
    预期行为：不通过——判别器只认文本里读得出的东西，注入什么就得在文本里看得见。
    """
    spec = MailSpec(
        tags=features.TagSet("forged_brand", "none", "none", features.LINK_NONE, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    verdict = judges_mod.SenderIdentityJudge("zh").judge(
        {
            "state": "From: 汇信银行 <service@huixinbank.com>\nSubject: 账单\n\n正文\n"
            "汇信银行\n官方域名：huixinbank.com"
        },
        spec,
    )
    assert verdict.ok is False


def test_sender_identity_judge_treats_a_www_prefix_as_the_same_domain() -> None:
    """官方域名写成 `www.` 开头也算同一个域名（模型偶尔这么写）。

    输入：`known_vendor` 规格 + 带 `www.` 的官方域名行。输出：Verdict。预期行为：通过。
    """
    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "none", features.LINK_NONE, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    verdict = judges_mod.SenderIdentityJudge("zh").judge(
        {
            "state": "From: 峰选商城 <service@peakretail.com>\nSubject: 账单\n\n正文\n"
            "峰选商城\n官方域名：www.peakretail.com"
        },
        spec,
    )
    assert verdict.ok is True, verdict.reason


def test_sender_identity_judge_skips_without_spec() -> None:
    """没有 spec 就没有真值：记成跳过（`skipped=True`），不误判。"""
    verdict = judges_mod.SenderIdentityJudge("zh").judge({"state": "x"})
    assert verdict.ok is True and verdict.skipped is True


def _attachment_spec() -> MailSpec:
    """输入：无。输出：一份「只要求打开附件」的真值规格。"""
    return MailSpec(
        tags=features.TagSet("known_vendor", "none", "attachment", features.LINK_NONE, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )


def test_trace_judge_flags_smuggled_requests(fake_backend: GenerationBackend) -> None:
    """夹带检测：反解出的「额外索取」里有清单之外的档位就判不通过。

    输入：一份只要求附件的规格 + 一个「还夹带了索取凭据」的假后端。输出：Verdict。
    预期行为：不通过，并在理由里点名夹带的是哪一档——那道单选本来就有两个说得通的答案，
    单选判别器看不出来（实跑抓到过：清单只要附件，正文却写「请按提示输入密码」）。
    """
    judge = judges_mod.TraceJudge(backend=fake_backend(smuggles="credentials"), locale="zh")
    verdict = judge.judge({"state": "x"}, _attachment_spec())
    assert verdict.ok is False
    assert "夹带了清单之外的索取" in verdict.reason


def test_trace_judge_tolerates_the_same_request_repeated(fake_backend: GenerationBackend) -> None:
    """反解把本来就该有的那一档又列一遍不算夹带——宁可少拒几条好数据。"""
    judge = judges_mod.TraceJudge(backend=fake_backend(smuggles="attachment"), locale="zh")
    assert judge.judge({"state": "x"}, _attachment_spec()).ok is True


def test_hygiene_judge_flags_json_leftovers() -> None:
    """卫生判别器查格式残留（JSON 收尾括号、代码围栏、`<br>` 标签），普通括号不许误伤。

    输入：五条 state（干净 / 多一个 `}` / 落款有 `<br>` / 正文里有括号 / 非文本）。输出：Verdict。
    预期行为：实跑抓到过落款末尾多一个 `}`，而四个原子题全答对了——
    这种损坏只能靠格式卫生抓，口径压窄是为了不误拒好数据。
    """
    judge = judges_mod.HygieneJudge("zh")
    assert judge.judge({"state": "From: a <b@c>\n\n正文。\n\n落款"}).ok is True
    dirty = judge.judge({"state": "From: a <b@c>\n\n正文。\n\n落款\n}"})
    assert dirty.ok is False
    assert dirty.raw["leftovers"] == ["}"]
    tagged = judge.judge({"state": "落款<br>官方域名：example.com<br>https://e.test/a"})
    assert tagged.ok is False
    assert tagged.raw["tags"] == ["<br>"]
    assert judge.judge({"state": "括号（这里是正文）不算残留"}).ok is True
    assert judge.judge({"state": {"not": "text"}}).ok is True


def test_hygiene_judge_flags_a_renderer_owned_line_written_twice() -> None:
    """输入：落款里连着两个「官方域名：」的 state（真产物抓到的原文形态）。输出：Verdict。

    预期行为：判不通过并点名重复的行首标记。那一行归渲染器写（`render.to_state` 追加），
    模型又写进 `signature` 就成了重复行——真端点 35 条里中过 2 条
    （`nim-smoke-13` / `13b` 的 `mail-train-0000001`）。
    """
    judge = judges_mod.HygieneJudge("zh")
    assert judge.judge({"state": "落款\n官方域名：example.com\nhttps://e.test/a"}).ok is True
    twice = judge.judge({"state": "落款\n官方域名：example.com\n官方域名：example.com"})
    assert twice.ok is False
    assert twice.raw["duplicated"] == ["官方域名："]
    assert "官方域名：×2" in twice.reason


def test_hygiene_judge_reads_the_marker_from_the_locale() -> None:
    """输入：英文 state 里重复了 `official domain:` 行。输出：Verdict。

    预期行为：判不通过——行首标记从 locale 模板推出来，代码里没写死中文，换个语言照样认得出。
    """
    judge = judges_mod.HygieneJudge("en")
    verdict = judge.judge(
        {"state": "Regards\nofficial domain: example.com\nofficial domain: example.com"}
    )
    assert verdict.ok is False
    assert verdict.raw["duplicated"] == ["official domain: "]


def test_judges_skip_without_spec() -> None:
    """输入：没有 spec。输出：Verdict。

    预期行为：跳过而不是误判——判别器只在拿得到真值时才有发言权，且用正式的
    `skipped` 字段记账（`ok=True` + `skipped=True`），不往 raw 里塞私货。
    """
    link = judges_mod.LinkJudge().judge({"state": "x"})
    assert link.ok is True and link.skipped is True and link.raw == {}
    trace = judges_mod.TraceJudge(backend=None, locale="zh").judge({"state": "x"})
    assert trace.ok is True and trace.skipped is True


def test_trace_judge_without_backend_is_skipped() -> None:
    """输入：有 spec 但没后端。输出：Verdict。

    预期行为：记成「没判成」（`skipped=True`）——没后端是接线问题，不是这条数据不对；
    记成「不通过」会让一整批数据看起来像被质检拒了。
    """
    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "none", features.LINK_NONE, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    verdict = judges_mod.TraceJudge(backend=None, locale="zh").judge({"state": "x"}, spec)
    assert verdict.ok is True and verdict.skipped is True


def test_trace_judge_skips_endpoint_failure_instead_of_rejecting(
    tmp_path: Path, fake_backend: GenerationBackend
) -> None:
    """输入：反解时端点故障（重试与降级都试过）的假后端。输出：RunResult。

    预期行为：渲染本身没问题，所以数据照收；`trace` 记成跳过（`judge-skipped.jsonl`）
    而不是拒绝——端点抖动不是数据质量问题。实跑踩过：假后端没给出草稿时报
    `No choices in OpenAI response`，被写成 `E_JUDGE_TRACE`，看着像模型写歪了。
    """
    result = pipeline.run(
        make_config(tmp_path, "m3", fake_backend(trace_raises=TransportError("网关 504")))
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert result.stats["reject_codes"] == {}
    assert result.stats["judges"]["trace"] == {"ok": 0, "fail": 0, "skipped": CASE_LIMIT}
    assert result.stats["judge_skipped"] == CASE_LIMIT
    skipped = [row for _, row in jsonio.iter_jsonl(result.run_dir / "judge-skipped.jsonl")]
    assert {row["judge"] for row in skipped} == {"trace"}
    assert all("端点故障" in row["reason"] for row in skipped)


def test_trace_judge_reraises_when_every_provider_is_down(fake_backend: GenerationBackend) -> None:
    """输入：反解时后端报「所有 provider 都挂了」。输出：异常冒泡。

    预期行为：整批停下是唯一的停机条件（PLAN 3.1.4），判别器不许把它吞成「跳过」——
    吞了就会产出一整个「没质检」的假数据集。
    """
    judge = judges_mod.TraceJudge(
        backend=fake_backend(trace_raises=AllProvidersDown("全挂")), locale="zh"
    )
    with pytest.raises(AllProvidersDown):
        judge.judge({"state": "x"}, _attachment_spec())


def test_trace_judge_skips_when_draft_misses_fields(fake_backend: GenerationBackend) -> None:
    """输入：后端返回的草稿只有 `requested` 一个字段。输出：Verdict。

    预期行为：缺字段等于「判不了」，记成跳过并点名缺了哪些字段——绝不拿半份草稿去比对。
    """
    judge = judges_mod.TraceJudge(
        backend=fake_backend(trace_data={"requested": "attachment"}), locale="zh"
    )
    verdict = judge.judge({"state": "x"}, _attachment_spec())
    assert verdict.ok is True and verdict.skipped is True
    assert "缺字段" in verdict.reason
    assert "pressure" in verdict.reason


def test_trace_judge_skips_when_extra_requests_is_not_a_list(fake_backend: GenerationBackend) -> None:
    """输入：草稿字段齐全，但 `extra_requests` 写成了字符串。输出：Verdict。

    预期行为：跳过并说清是「不是列表」——字符串会被逐字符迭代，那样夹带检测会静默失灵。
    """
    judge = judges_mod.TraceJudge(
        backend=fake_backend(
            trace_data={
                "requested": "attachment",
                "extra_requests": "none",
                "pressure": "none",
                "sender_identity": "matches",
            }
        ),
        locale="zh",
    )
    verdict = judge.judge({"state": "x"}, _attachment_spec())
    assert verdict.ok is True and verdict.skipped is True
    assert "不是列表" in verdict.reason


def test_id_splits_share_no_tag_combination() -> None:
    """输入：无。输出：无。

    预期行为：train / val / test_iid 的 tag 组合两两不相交——切分单位是组合，
    同一个组合不许同时出现在训练与测试里。
    """
    workflow = workflow_mod.load("mail")
    seen: dict[tuple[tuple[str, str], ...], str] = {}
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases, f"{split} 一个 case 都没有"
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in seen, f"组合同时出现在 {seen[key]} 与 {split}：{key}"
            seen[key] = split


def test_ood_holds_out_values_and_id_splits_never_see_them() -> None:
    """输入：无。输出：无。

    预期行为：`test_ood` 的每个组合都带一个留出取值（手法 curiosity 或短链接），
    而同分布切分里一个都不许出现——这才是「值留出」，不是「更大的数字」。
    """
    workflow = workflow_mod.load("mail")
    ood = workflow.build_cases(workflow.config_for("test_ood"))
    assert ood
    for case in ood:
        assert case.lure in mail_workflow.OOD_LURES or case.link_target in mail_workflow.OOD_LINK_TARGETS
    for split in ("train", "val", "test_iid"):
        for case in workflow.build_cases(workflow.config_for(split)):
            assert case.lure not in mail_workflow.OOD_LURES
            assert case.link_target not in mail_workflow.OOD_LINK_TARGETS


def test_manifest_records_backend(tmp_path: Path, fake_backend: GenerationBackend) -> None:
    """输入：假后端。输出：manifest。

    预期行为：记下后端名字与 needs，绝不记 key——溯源够用，凭据不入库。
    """
    result = pipeline.run(make_config(tmp_path, "m3", fake_backend()))
    manifest = json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == {"name": "fake", "starting_mode": ""}
    assert manifest["needs"] == ["text"]
    assert manifest["workflow"] == "mail"


def test_render_without_backend_raises() -> None:
    """输入：没有注入后端的 workflow。输出：异常。

    预期行为：直接报错，绝不悄悄降级成模板文本（那会让「文本由后端写」这个前提失效）。
    """
    workflow = mail_workflow.MailWorkflow()
    spec = MailSpec(
        tags=features.TagSet("known_vendor", "none", "none", features.LINK_NONE, "formal"),
        phrasing=0,
        split="train",
        seq=0,
    )
    import random

    try:
        workflow.render(spec, random.Random(0), "zh")
    except RuntimeError as exc:
        assert "bind_backend" in str(exc)
    else:  # pragma: no cover —— 走到这里说明没报错，测试就该失败
        raise AssertionError("没有后端时 render 必须报错")


def _row(state: str, questions: dict[str, JsonValue], answers: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """造一条最小样本行（审计只读 state / questions / answers）。"""
    return {"id": "mail-train-0000000", "state": state, "questions": questions, "answers": answers}


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空目录。输出：(报告, 缺失项)。

    预期行为：缺失的切分要明说「产物不存在」，而不是当成「0 条样本、审计通过」。
    """
    report, missing = workflow_mod.load("mail").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def test_audit_catches_link_contradiction_and_unknown_family(tmp_path: Path) -> None:
    """输入：一条自相矛盾 + 带未知题族的产物行。输出：(报告, 缺失项)。

    预期行为：三件事都要报出来——链接矛盾、未知题族、题族缺失。
    """
    row = _row(
        "没有链接的正文",
        {
            "link_text": {
                "type": "choice",
                "instructions": "x",
                "criteria": {"matches": "a", "mismatched": "b"},
            },
            "made_up_family": {"type": "noul", "instructions": "y"},
        },
        {
            "link_text": {"type": "choice", "choice": 0},
            "made_up_family": {"type": "noul", "noul": True},
        },
    )
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    _, missing = workflow_mod.load("mail").audit(tmp_path, ("train",), "zh")
    assert any("link_text 答案与文本链接有无矛盾" in item for item in missing)
    assert any("未知题族" in item for item in missing)
    assert any("题族 requested 一条都没出" in item for item in missing)


def test_audit_passes_a_consistent_artifact(tmp_path: Path) -> None:
    """输入：一条自洽的产物行（八个题族齐全、有链接且答案不是 no_link）。输出：(报告, 缺失项)。

    预期行为：零缺失，`link_ok` 记到 1。
    """
    choices = {
        "requested": {"none": "a", "credentials": "b"},
        "pressure": {"none": "a", "urgency": "b"},
        "link_text": {"matches": "a", "mismatched": "b"},
        "sender_identity": {"matches": "a", "forged": "b"},
        "action": {"deliver": "a", "block": "b"},
    }
    questions: dict[str, JsonValue] = {
        name: {"type": "choice", "instructions": "x", "criteria": criteria}
        for name, criteria in choices.items()
    }
    answers: dict[str, JsonValue] = {
        name: {"type": "choice", "choice": 0} for name in choices
    }
    for name in ("is_phishing", "is_spam"):
        questions[name] = {"type": "noul", "instructions": "x"}
        answers[name] = {"type": "noul", "noul": False}
    questions["severity"] = {"type": "score", "instructions": "x", "criteria": ["a", "b"]}
    answers["severity"] = {"type": "score", "score": 0}
    jsonio.write_jsonl(
        tmp_path / "train.jsonl", [_row("正文见 https://example.test/a", questions, answers)]
    )
    report, missing = workflow_mod.load("mail").audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["link_ok"] == 1


def test_config_for_rejects_unknown_split() -> None:
    """输入：不存在的切分名。输出：KeyError。预期行为：报错并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        mail_workflow.MailWorkflow().config_for("nope")


def test_make_judges_rejects_unknown_name() -> None:
    """输入：不存在的判别器名。输出：KeyError。预期行为：绝不静默少跑一个判别器。"""
    with pytest.raises(KeyError, match="未知判别器"):
        mail_workflow.MailWorkflow().make_judges(("nope",), "zh")


def test_reparse_and_case_label() -> None:
    """输入：一段文本、一个 case。输出：反解结果、可读描述。

    预期行为：`reparse` 只反解纯代码能查的部分（链接）；`case_label` 列出全部维度。
    """
    workflow = mail_workflow.MailWorkflow()
    assert workflow.reparse("见 https://a.test/x").has_url is True
    assert workflow.reparse({"any": "object"}).has_url is False
    label = workflow.case_label(features.build_cases(32)[0])
    for dim in features.DIMS:
        assert f"{dim}=" in label


def test_spec_to_dict_is_serializable() -> None:
    """输入：一个规格。输出：字典。预期行为：能直接 `json.dumps`，且带上拆分后的链接维。"""
    from workflows.mail.spec import spec_to_dict

    case = features.TagSet("forged_brand", "fear", "credentials", "ip_mismatch", "curt")
    spec = MailSpec(tags=case, phrasing=1, split="train", seq=3)
    payload = spec_to_dict(spec)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["link_target"] == "ip"
    assert payload["anchor"] == "mismatch"
    assert spec.render_task_id == "mail-train-0000003-render"
    assert spec.trace_task_id == "mail-train-0000003-trace"


def test_available_locales_lists_shipped_language_files() -> None:
    """输入：无。输出：语言列表。预期行为：与 locales/ 里的 toml 一一对应，且每份都读得出来。"""
    workflow = mail_workflow.MailWorkflow()
    assert workflow.available_locales() == ["en", "zh"]
    for lang in workflow.available_locales():
        assert locales.load(lang)["language_name"]
