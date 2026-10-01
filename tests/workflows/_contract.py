"""两把端到端扫描共用的一条断言：产物必须过**通用格式契约**。

为什么单独拎出来：`test_construct_sweep.py` 与 `test_llm_locale_sweep.py` 都要查同一件事，
两处各写一份迟早漂移（§9.34 那条原则：凡是能被机器判对错的东西，就别指望人抄对）。这里
只放这一条，扫描本身的逻辑各留在各自的文件里。

输入：run 目录、切分名、报错用的标签（形如 `maze/zh/test_ood`）。输出：无（断言）。
预期行为：产物必须过 `jc.validate.validate_file`——题面 / 答案 / 编号 / `type` / `criteria`
的形态与「一行一道题」的口径都由它管，**与 workflow 无关**。workflow 自己的 `audit` 查的是
领域事实（题族齐不齐、答案与文本自不自洽），两者互补：通用契约管「格式 case」，
领域审计管「这道题答得对不对」。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from jc import jsonio, validate
from jc import workflow as workflow_mod

#: 走 LLM 但**没有**「模型写的自然语言正文」的 workflow：模型写的是代码 / 命令行。
#: 它们必须报空表——报了正文就会把「以代码为主的 state」当坏数据（真机 zh 侧汉字占比低到 0.099）。
LLM_WITHOUT_PROSE: Final[dict[str, str]] = {
    "lint": "模型写的是那段源码（state = 一行中文抬头 + 源码）",
    "review": "模型写的是那段代码改动（body 就是 diff 本身）",
    "shell": "模型写的是命令行（PowerShell / bash 的调用本身）",
}

#: 假后端**按声明的语言**造正文的那几家——只有它们能在端到端里把语言闸（`jc.pipeline` 的
#: `language_gate`）打开。其余假后端只写一种语言（多为中文，`autoformat` / `citation` /
#: `rerank` / `rubric` 只写英文），闸一开就会把**夹具的局限**报成拒稿。这份名单是量出来的
#: （口径见 `tmp/probe_lang.py`：逐家按每种声明语言跑一批假后端，再拿 `Workflow.language_texts`
#: 喂 `jc.language.sample_problem`），不是猜的；`test_locale_faithful_fakes_are_real` 盯着它别腐烂。
LOCALE_FAITHFUL_FAKES: Final[dict[str, str]] = {
    "alertrisk": "「告警描述」那一行按语言现造",
    "browser": "「用户请求」那一行按语言现造（构造参数叫 `lang`，不是 `locale`）",
    "depcheck": "「依赖备注」那一行按语言现造",
    "findtriage": "「发现」「修法」两行按语言现造",
    "hirecheck": "「候选人备注」那一行按语言现造",
    "receiptcat": "「收据行」按语言现造",
    "testimpact": "「提交标题」那一行按语言现造",
}

#: 假后端**收**语言参数、却**不按它造文本**的那几家——不能进 `LOCALE_FAITHFUL_FAKES`，否则
#: 语言闸一开，它们的另一种语言那一侧会全红，那是夹具的局限、不是被测代码的 bug。放进这里
#: 是为了让 `test_any_fake_that_takes_a_locale_is_listed` 还能继续当闸用：谁新写了一个
#: 「收语言参数且真按语言造文本」的夹具却忘了进 `LOCALE_FAITHFUL_FAKES`，测试就会红。
LOCALE_PARAM_BUT_UNFAITHFUL: Final[dict[str, str]] = {
    "autoformat": "构造参数收 `locale`，但正文永远写英文（zh 侧实测 10/10 对不上）",
}


def assert_language_texts_are_declared(
    wf: workflow_mod.Workflow, run_dir: Path, split: str, label: str, *, prose: bool
) -> None:
    """输入：workflow、run 目录、切分、标签、这一族该不该有自然语言正文。输出：无（断言）。

    预期行为：该有正文的必须**真报出字段**——空表意味着 `jc audit` 的语言判据对它永远是空转，
    等于「声明了要查」而实际上一次都没查；不该有的必须报空表——报了就会把代码 / 命令行 /
    ASCII 图喂进判据，把好数据当坏数据。两边都不许悄悄滑过去。
    """
    rows = [row for _, row in jsonio.iter_jsonl(run_dir / f"{split}.jsonl")]
    assert rows, f"{label} 的产物是空的，这条尺子量不到东西"
    first = rows[0]
    assert isinstance(first, dict), f"{label} 的产物头一行不是对象：{first!r}"
    declared = wf.language_texts(first)
    if prose:
        assert declared, (
            f"{label} 声明了「没有自然语言正文」——走 LLM 的 workflow 得说清哪几段是模型写的正文，"
            "否则「产物文本与声明语言一致」这条尺子对它永远是空转"
        )
    else:
        assert not declared, (
            f"{label} 报了自然语言正文 {sorted(declared)}——这一族的 state 是代码 / 命令行 / ASCII 图，"
            "按语言判会把好数据当坏数据"
        )


def assert_contract(run_dir: Path, split: str, label: str) -> None:
    """输入：run 目录、切分、标签。输出：无。

    预期行为：该切分的产物一行都不许违契约；报了就把错误码统计与头一条问题一起说出来，
    省得再去翻文件。
    """
    report = validate.validate_file(run_dir / f"{split}.jsonl")
    issues = report.all_issues()
    assert not issues, (
        f"{label} 的产物过不了通用格式契约：{report.code_counts()}；"
        f"头一条：{issues[0] if issues else ''}"
    )


def assert_audit_ignores_the_callers_locale(
    wf: workflow_mod.Workflow, run_dir: Path, split: str, label: str
) -> None:
    """输入：workflow、run 目录、切分、标签。输出：无。

    预期行为：用**每一种**声明语言的口径各审计一遍，结果必须都是零缺失。

    为什么要有：产物按 `locales[seq % len]` 轮转渲染，一次 run 里可能混着两种语言，调用方
    给的语言未必是产物那一种。审计若拿调用方给的那一种去比规范表，另一种语言的产物就会被
    平白报成「表对不上」——`review` / `browser` / `routing` / `rubric` 四家都踩过这个坑
    （全 en 的产物按 zh 口径审计，8 条全中被误判）。这条把「审计与调用方语言无关」变成
    机器可查，新增 workflow 时不用再靠人记得。
    """
    for locale in wf.available_locales():
        _report, missing = wf.audit(run_dir, (split,), locale)
        assert not missing, (
            f"{label} 用 {locale} 的口径审计就报了缺失——审计的判定不许依赖调用方给的语言："
            f"{missing}"
        )


def assert_audit_reads_a_mixed_file(
    wf: workflow_mod.Workflow,
    run_dirs: list[Path],
    split: str,
    label: str,
    tmp_path: Path,
) -> None:
    """输入：workflow、各语言的 run 目录、切分、标签、临时目录。输出：无。

    预期行为：把各语言 run 的 `{split}.jsonl` 拼成一份，再用**每一种**语言的口径各审一遍，
    都必须零缺失。

    为什么要有：一次 `locales=("zh","en")` 的 run 里两种语言混在**同一个文件**里（管线按
    `locales[seq % len]` 轮转），那才是审计真正要扛住的形状。只审单语言产物时，一半的
    「拿调用方语言去认标记」类 bug 看不见——`bench` / `sim` / `skillpick` / `match` /
    `functions` / `maze` 六家的反解就是拼接这一遍才暴露的。
    """
    if len(run_dirs) < 2:
        return
    mixed = tmp_path / f"mixed-{wf.name}-{split}"
    mixed.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for run_dir in run_dirs:
        lines.extend((run_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
    (mixed / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert_audit_ignores_the_callers_locale(wf, mixed, split, f"{label}（混语言）")
