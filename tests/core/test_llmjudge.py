"""通用层测试：多视角 LLM 判别器（`jc.llmjudge`，PLAN 6.2）。

**不发任何网络请求**：假后端按剧本返回结论或抛异常。要验的是四件事——
结论怎么转成 Verdict、什么情况算「没判成」（而不是「数据有问题」）、
视角不适用会不会白烧一次调用、提示词里给的是文字还是编号。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue



import pytest

from jc import llmjudge
from jc.backend.types import AllProvidersDown, Draft, TransportError


class FakeBackend:
    """假后端：按剧本给结论或抛异常，并记下每次调用。"""

    name = "fake"
    capabilities = None

    def __init__(
        self, *, data: dict[str, JsonValue] | None = None, exc: BaseException | None = None
    ) -> None:
        """输入：要返回的结论、要抛的异常（二选一）。输出：实例。"""
        self.data = {"verdict": True, "score": 0.9, "reason": "没问题"} if data is None else data
        self.exc = exc
        self.calls: list[JsonValue] = []

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：草稿。预期行为：先记账再按剧本返回或抛异常。"""
        self.calls.append(task)
        if self.exc is not None:
            raise self.exc
        return Draft(
            data=dict(self.data),
            raw_text="{}",
            backend="fake",
            provider="p",
            model="m",
            mode="tools",
        )


def make_sample(*, kinds: tuple[str, ...] = ("choice",), state: str = "素材：甲乙丙") -> dict[str, JsonValue]:
    """输入：要包含哪些题型、素材文本。输出：最小契约样本。"""
    questions: dict[str, JsonValue] = {}
    answers: dict[str, JsonValue] = {}
    if "choice" in kinds:
        questions["pick"] = {
            "type": "choice",
            "instructions": "选哪个？",
            "criteria": {"a": "甲", "b": "乙"},
        }
        answers["pick"] = {"type": "choice", "choice": 0}
    if "score" in kinds:
        questions["rate"] = {
            "type": "score",
            "instructions": "几档？",
            "criteria": ["0：没有", "1：一个"],
        }
        answers["rate"] = {"type": "score", "score": 1}
    if "noul" in kinds:
        questions["flag"] = {
            "type": "noul",
            "instructions": "对吗？",
            "criteria": {"true": "对", "false": "不对"},
        }
        answers["flag"] = {"type": "noul", "noul": False}
    return {"id": "s-1", "state": state, "questions": questions, "answers": answers}


def test_locale_files_cover_every_perspective() -> None:
    """输入：无。输出：无（断言）。预期行为：每种语言都覆盖全部视角、rules 非空。

    漏一个视角的后果是「`--judges llm:xxx` 到了运行期才炸」，而且只在那一种语言下炸。
    """
    assert set(llmjudge.available_locales()) >= {"zh", "en"}
    for locale in llmjudge.available_locales():
        loc = llmjudge.load(locale)
        assert set(loc["perspectives"]) == set(llmjudge.PERSPECTIVES), locale
        for key, view in loc["perspectives"].items():
            assert view["title"].strip(), (locale, key)
            assert view["rules"], (locale, key)
            assert all(rule.strip() for rule in view["rules"]), (locale, key)
        for field in (
            "intro",
            "material_head",
            "question_head",
            "options_label",
            "levels_label",
            "answer_label",
            "perspective_head",
            "rules_head",
            "closing",
        ):
            assert loc[field].strip(), (locale, field)


def test_locales_have_the_same_shape() -> None:
    """输入：无。输出：无（断言）。预期行为：中英两份的视角集合与 rules 条数一致。

    条数不一致说明翻译时漏了一条判据——那种漏检是静默的，只能靠这条测试兜住。
    """
    zh, en = llmjudge.load("zh"), llmjudge.load("en")
    assert set(zh["perspectives"]) == set(en["perspectives"])
    for key in llmjudge.PERSPECTIVES:
        assert len(zh["perspectives"][key]["rules"]) == len(en["perspectives"][key]["rules"]), key


def test_unknown_locale_lists_what_is_available() -> None:
    """输入：不存在的语言。输出：FileNotFoundError。预期行为：报人话并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="zh"):
        llmjudge.load("de")


def test_make_judge_rejects_unknown_perspective() -> None:
    """输入：不存在的视角名。输出：KeyError。预期行为：列出可用视角，不静默忽略。"""
    with pytest.raises(KeyError, match="solvable"):
        llmjudge.make_judge("vibes", "zh", FakeBackend())


def test_judge_name_is_prefixed() -> None:
    """输入：一个判别器。输出：无（断言）。预期行为：名字带 `llm:` 前缀，统计里认得出。"""
    judge = llmjudge.make_judge("answer", "zh", FakeBackend())
    assert judge.name == "llm:answer"


def test_gold_text_uses_words_not_indices() -> None:
    """输入：三种题型的题目与金答案。输出：金答案的文字。

    预期行为：choice 取选项文字、score 取等级文字、noul 取 criteria 里的文案——
    给模型看编号只会让它困惑（编号是组装器的私事）。
    """
    sample = make_sample(kinds=("choice", "score", "noul"))
    questions, answers = sample["questions"], sample["answers"]
    assert llmjudge.gold_text(questions["pick"], answers["pick"]) == "甲"
    assert llmjudge.gold_text(questions["rate"], answers["rate"]) == "1：一个"
    assert llmjudge.gold_text(questions["flag"], answers["flag"]) == "不对"


def test_gold_text_survives_broken_input() -> None:
    """输入：编号越界、题型对不上、字段缺失的样本。输出：`?` 或原始 JSON。

    预期行为：**不抛异常**——提示词组装失败不该把整条样本带走，
    编号越界是契约问题，由别处报错。
    """
    assert llmjudge.gold_text({"criteria": {"a": "甲"}}, {"type": "choice", "choice": 9}) == "?"
    assert llmjudge.gold_text({"criteria": ["0：没有"]}, {"type": "score", "score": -1}) == "?"
    assert llmjudge.gold_text({"criteria": None}, {"type": "noul", "noul": True}) == "True"
    # 认不出的答案形态：原样把 JSON 摊出来，总比给模型一个空字符串强。
    assert llmjudge.gold_text({"criteria": None}, {"type": "weird", "x": 1}) == '{"type": "weird", "x": 1}'


def test_prompt_renders_score_levels_and_noul_options() -> None:
    """输入：一条含 score 与 noul 的样本。输出：提示词。

    预期行为：score 的等级数组按原序列出、noul 的金答案用 criteria 里的文案——
    三种题型各有各的写法，漏一种就少一半信息。
    """
    loc = llmjudge.load("zh")
    prompt = llmjudge.build_prompt(make_sample(kinds=("score", "noul")), loc, "levels")
    assert "等级：0：没有 / 1：一个" in prompt
    assert "金答案：1：一个" in prompt
    assert "金答案：不对" in prompt


def test_prompt_shows_state_options_and_only_this_perspective() -> None:
    """输入：一条样本、`solvable` 视角。输出：提示词。

    预期行为：带上素材、选项文字、金答案文字，以及**只带这一个视角**的判据——
    一次问五个视角的话，模型只会挑最好答的那个答。
    """
    loc = llmjudge.load("zh")
    prompt = llmjudge.build_prompt(make_sample(), loc, "solvable")
    assert "素材：甲乙丙" in prompt
    assert "甲 / 乙" in prompt
    assert "金答案：甲" in prompt
    assert loc["perspectives"]["solvable"]["title"] in prompt
    for other in ("answer", "distractors", "levels", "wording"):
        assert loc["perspectives"][other]["title"] not in prompt


def test_judge_maps_verdict_to_verdict() -> None:
    """输入：假后端给 verdict=false。输出：不通过的 Verdict。

    预期行为：`ok` 就是模型的 verdict，`score` 与 `reason` 原样带出，
    `raw` 里留 provider / model / 把握，便于事后追是哪家模型判的。
    """
    backend = FakeBackend(data={"verdict": False, "score": 0.3, "reason": "两个选项都对"})
    verdict = llmjudge.make_judge("solvable", "zh", backend).judge(make_sample())
    assert verdict.ok is False
    assert verdict.skipped is False
    assert verdict.score == 0.3
    assert verdict.reason == "两个选项都对"
    assert verdict.raw["provider"] == "p" and verdict.raw["model"] == "m"


def test_task_carries_judge_id_and_response_model() -> None:
    """输入：一条样本。输出：无（断言）。预期行为：任务 id 可追溯、结构化输出用 Judgement。

    `task_id` 里带样本 id 与视角：并发跑的时候，日志里能认出是哪条样本的哪只眼睛。
    """
    backend = FakeBackend()
    llmjudge.make_judge("wording", "zh", backend).judge(make_sample())
    task = backend.calls[0]
    assert task.task_id == "s-1-judge-wording"
    assert task.response_model is llmjudge.Judgement
    assert "素材：甲乙丙" in task.prompt


def test_transport_error_is_skipped_not_failed() -> None:
    """输入：后端抛传输错误（重试与降级都试过之后）。输出：skipped 的 Verdict。

    预期行为：**不算数据有问题**，也不算通过——单独记一笔账。
    记成「不通过」就是把端点抖动写成数据质量问题（PLAN 10.10 踩过）。
    """
    backend = FakeBackend(exc=TransportError("504"))
    verdict = llmjudge.make_judge("solvable", "zh", backend).judge(make_sample())
    assert verdict.skipped is True
    assert "端点故障" in verdict.reason
    assert verdict.raw["error"] == "TransportError"


def test_all_providers_down_still_stops_the_batch() -> None:
    """输入：后端抛 `AllProvidersDown`。输出：异常原样冒泡。

    预期行为：整批停下是 PLAN 3.1.4 的唯一停机条件，判别器不能把它吞成 skipped。
    """
    backend = FakeBackend(exc=AllProvidersDown("全挂了"))
    judge = llmjudge.make_judge("solvable", "zh", backend)
    with pytest.raises(AllProvidersDown):
        judge.judge(make_sample())


def test_inapplicable_perspective_skips_without_calling_backend() -> None:
    """输入：没有 score 题的样本 + `levels` 视角。输出：skipped，且**一次调用都没发**。

    预期行为：没有对象可看就别白烧一次调用——这是花钱的判别器，省一次是一次。
    """
    backend = FakeBackend()
    verdict = llmjudge.make_judge("levels", "zh", backend).judge(make_sample(kinds=("choice",)))
    assert verdict.skipped is True
    assert "不适用" in verdict.reason
    assert backend.calls == []


def test_missing_verdict_field_is_skipped() -> None:
    """输入：后端返回了草稿但没有 verdict 字段。输出：skipped。

    预期行为：解析失败是「没判成」，绝不能当成「数据有问题」——那等于用模型的胡言乱语拒数据。
    """
    backend = FakeBackend(data={"score": 0.5, "reason": "嗯"})
    verdict = llmjudge.make_judge("solvable", "zh", backend).judge(make_sample())
    assert verdict.skipped is True
    assert "verdict" in verdict.reason


def test_non_dict_draft_and_non_numeric_score() -> None:
    """输入：`data` 不是字典、`score` 不是数字。输出：跳过 / score 为 None。

    预期行为：两种脏输入都不许炸，也不许变成「数据有问题」。
    """
    broken = Draft(data=None, raw_text="", backend="b", provider="p", model="m", mode="tools")  # type: ignore[arg-type]
    assert llmjudge.to_verdict("solvable", broken).skipped is True
    weird = Draft(
        data={"verdict": True, "score": "高", "reason": "看着行"},
        raw_text="",
        backend="b",
        provider="p",
        model="m",
        mode="tools",
    )
    verdict = llmjudge.to_verdict("solvable", weird)
    assert verdict.ok is True and verdict.score is None


def test_judgement_model_shape() -> None:
    """输入：无。输出：无（断言）。预期行为：结构化输出的形状就是那三个字段。"""
    assert set(llmjudge.Judgement.model_fields) == {"verdict", "score", "reason"}
