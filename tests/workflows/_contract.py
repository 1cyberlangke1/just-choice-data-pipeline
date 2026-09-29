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

from jc import validate
from jc import workflow as workflow_mod


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
