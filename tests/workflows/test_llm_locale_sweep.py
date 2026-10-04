"""走 LLM 的 workflow：每种语言 × 每个切分都用**假后端**端到端跑一遍（零 token）。

输入：所有 `needs` 非空的 workflow，加上各自 `tests/workflows/<wf>/` 下的
`conftest.FakeBackend` 与 `test_pipeline.make_config`（只借它们的夹具与配置，不改它们）。
输出：无（断言）。
预期行为：每个 (workflow, 语言, 切分) 跑一遍小 case 集合，断言零拒稿、产物非空，
而且**各语言出的条数一致**——case 集合与语言无关，条数不一致就说明某种语言在悄悄丢样本。
审计那一段由 `_contract` 里两把尺管：**每一种语言的口径**审每一份产物都要零缺失，而且把
两种语言的产物拼成一份（一次 `locales=("zh","en")` 的 run 就是这形状）再审一遍也要零缺失。

为什么值得进测试套件：判别器读的是**渲染出来的文本**，而渲染是语言相关的。各 workflow 的
`test_pipeline.py` 里那条端到端 happy path 只跑 zh（`locales=("zh",)`），于是「en 的读法错」
在单测里结构上看不见——范式 §9.32 那 782 条迷宫拒稿、§10.87 那三个硬写 zh 的 shell 判别器，
都是这个盲区漏出来的。这条测试把「每种语言都要端到端跑一遍」变成机器可查，而且不需要 key。

切分为什么要两遍：`test_ood` 换的是 case 集合（留出侧），而「读法错」的第二个战场就在这里——
某个语言在 `train` 侧恰好没被踩到的读法，可能在留出侧的另一种题族上才暴露。两遍一起扫，
才把「渲染 × 读法」的笛卡尔积补齐。

零 token 的代价：假后端按 `task.spec` 现造草稿，从不发网络请求。

为什么不是每家每个切分一条参数化用例：`_sweep_one` 是同一段端到端流程，74 条用例只是把
同一件事报 74 遍。收成一条循环之后，失败信息攒在 `bad` 里一起报（点名到 workflow / 切分），
用例数从 74 降到 1。
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path

from jc import pipeline
from jc import workflow as workflow_mod

from tests.workflows._contract import (
    LOCALE_FAITHFUL_FAKES,
    LOCALE_PARAM_BUT_UNFAITHFUL,
    LLM_WITHOUT_PROSE,
    assert_audit_ignores_the_callers_locale,
    assert_audit_reads_a_mixed_file,
    assert_contract,
    assert_language_texts_are_declared,
    assert_manifest_and_stats,
    assert_no_unresolved_slots,
)


def llm_workflows() -> list[str]:
    """输入：无。输出：`needs` 非空的 workflow 名（排序后）。

    预期行为：先 `discover()` 触发各 workflow 的「导入即注册」，再按 `needs` 过滤；
    纯构造的那些不在这里（它们由 `test_construct_sweep.py` 管）。
    """
    return [name for name in workflow_mod.discover() if workflow_mod.load(name).needs]


#: 需要后端的 workflow 名单（模块导入时算一次）。
LLM_WORKFLOWS: tuple[str, ...] = tuple(llm_workflows())


def test_llm_workflows_are_found() -> None:
    """输入：无。输出：无。预期行为：至少扫到若干走 LLM 的 workflow，否则这把闸名存实亡。"""
    assert len(LLM_WORKFLOWS) >= 10, f"只找到 {LLM_WORKFLOWS}"


def test_the_prose_exemption_list_names_real_llm_workflows() -> None:
    """输入：无。输出：无。

    预期行为：`LLM_WITHOUT_PROSE` 里的名字必须都是**真的**走 LLM 的 workflow——写错名字（或某个
    workflow 后来改成了纯构造）时这条会红，清单就不会腐烂成「谁都不管」的摆设。
    """
    unknown = sorted(set(LLM_WITHOUT_PROSE) - set(LLM_WORKFLOWS))
    assert not unknown, f"豁免清单里的这些名字不是走 LLM 的 workflow：{unknown}"
    assert LLM_WITHOUT_PROSE, "豁免清单空了？那 lint / review 就该报出正文了"


def test_locale_faithful_fakes_are_real() -> None:
    """输入：无。输出：无。

    预期行为：`LOCALE_FAITHFUL_FAKES` 里的名字必须都是**真的**走 LLM 的 workflow。名字写错、
    某家后来改成纯构造时这条会红，语言闸的开合就不会跟着一份腐烂的名单跑。

    名单是**量出来的**，不是按构造签名推的：口径见 `tmp/probe_lang.py`——逐家按每种声明语言跑
    一批假后端，再拿 `Workflow.language_texts` 喂 `jc.language.sample_problem`，两侧都零对不上
    的才算「按语言造文本」。所以这里**不**断言假后端收 `locale`：`findtriage` 就是从
    `task.spec["locale"]` 现取语言、构造函数里根本没这个参数，签名判据会把它误判成不忠实。
    """
    unknown = sorted(set(LOCALE_FAITHFUL_FAKES) - set(LLM_WORKFLOWS))
    assert not unknown, f"名单里的这些名字不是走 LLM 的 workflow：{unknown}"
    assert LOCALE_FAITHFUL_FAKES, "名单空了？那语言闸在端到端里就一次都没开过"


def test_any_fake_that_takes_a_locale_is_listed() -> None:
    """输入：无。输出：无。

    预期行为：假后端只要**收**语言参数（`locale` / `lang` / `language` 任一），它就有能力按语言
    造文本，那就必须出现在 `LOCALE_FAITHFUL_FAKES` 里——否则语言闸在端到端里对它白关着，
    「en 的读法错」这类 bug 在它身上永远测不出来。`browser` 的夹具参数就叫 `lang` 而不是
    `locale`，此前正是这么漏掉的（`tmp/probe_lang.py` 量出来才发现）。

    `findtriage` 那种从 `task.spec["locale"]` 现取语言的查不出来（构造函数里没这个参数），
    只能靠探针量；这条只管能查的那一半，查不出的那半由 `LOCALE_FAITHFUL_FAKES` 的注释写明。

    「收语言参数」只是**有资格**按语言造文本，不等于真按了——`autoformat` 的夹具收 `locale`、
    正文却永远写英文（`tmp/probe_lang.py` 实测 zh 侧 10/10 对不上），这类夹具的局限记在
    `LOCALE_PARAM_BUT_UNFAITHFUL` 里，不算「该进名单却漏了」。
    """
    capable = sorted(
        name
        for name in LLM_WORKFLOWS
        if any(
            key
            in inspect.signature(
                importlib.import_module(f"tests.workflows.{name}.conftest").FakeBackend
            ).parameters
            for key in ("locale", "lang", "language")
        )
    )
    missing = sorted(set(capable) - set(LOCALE_FAITHFUL_FAKES) - set(LOCALE_PARAM_BUT_UNFAITHFUL))
    assert not missing, f"这些假后端收语言参数却没进名单，闸对它们白关着：{missing}"


def config_for(name: str, tmp_path: Path, locale: str, split: str) -> pipeline.RunConfig:
    """借该 workflow 自己的测试配置，只把语言、切分与 run 名换掉。

    输入：workflow 名、临时目录、语言、切分。
    输出：指向 tmp 的运行配置。
    预期行为：用 `inspect.signature` 兼容两种 `make_config` 形状（`backend` 是位置参数还是
    关键字参数），并且**不复制**各 workflow 的 case 上限与切分口径——那些是它们自己的契约，
    改口径时不该漏改这里。切分只做透传，由 workflow 自己决定它意味着哪套 case。
    """
    tests = importlib.import_module(f"tests.workflows.{name}.test_pipeline")
    conftest = importlib.import_module(f"tests.workflows.{name}.conftest")
    # run 名带上切分：两种切分写同一个 run 目录会互相覆盖，审计就查了个寂寞。
    run_id = f"locale-sweep-{name}-{locale}-{split}"
    kwargs: dict[str, object] = {"split": split}
    if "backend" in inspect.signature(tests.make_config).parameters:
        maker = conftest.FakeBackend
        # 假后端替模型写字，而标记类文案（步骤词、提示框关键词、署名前缀）是语言相关的：**收语言
        # 参数**的假后端要按目标语言现造。参数名两家不统一（多数叫 `locale`，`browser` 叫 `lang`），
        # 所以按名字试一遍；都不收，说明它的草稿与语言无关（命令、正文），那就照默认造。
        takes = [
            name
            for name in ("locale", "lang", "language")
            if name in inspect.signature(maker).parameters
        ]
        kwargs["backend"] = maker(**{takes[0]: locale}) if takes else maker()
    cfg = tests.make_config(tmp_path, run_id, **kwargs)
    # 语言闸（`jc.pipeline` 的 `language_gate`）要求「模型写的自然语言正文」与声明的语言一致。
    # 只有**按语言造文本**的假后端才兑现得了这条——其余假后端只写一种语言，闸一开就会把它们的
    # 另一种语言那一侧全判成拒稿，而那是夹具的局限、不是渲染的 bug。谁算「按语言造文本」写在
    # `LOCALE_FAITHFUL_FAKES` 里（量出来的，见 `_contract` 的注释），`test_locale_faithful_fakes_are_real`
    # 盯着那份名单；闸本身另有一条 `tests/core` 的定点测试专门验它拒得掉。
    return dataclasses.replace(
        cfg,
        locales=(locale,),
        run_id=run_id,
        language_gate=name in LOCALE_FAITHFUL_FAKES,
    )


def _sweep_one(name: str, split: str, tmp_path: Path) -> None:
    """输入：workflow 名、切分、临时目录。输出：无（断言）。

    预期行为：给定切分在**每种语言**下都零拒稿、产物非空，而且**各语言出的条数一致**
    ——case 集合与语言无关，条数不一致就说明某种语言在悄悄丢样本（迷宫 en 侧那次就是这个症状）。
    """
    wf = workflow_mod.load(name)
    locales = wf.available_locales()
    assert locales, f"{name} 没声明任何语言"
    counts: dict[str, int] = {}
    run_dirs: list[Path] = []
    for locale in locales:
        cfg = config_for(name, tmp_path, locale, split)
        result = pipeline.run(cfg)
        reasons = [row.get("reason") for row in result.rejected]
        assert not result.rejected, (
            f"{name}/{locale}/{split} 有 {len(result.rejected)} 条被拒——假后端是忠实的，"
            f"拒稿只可能是渲染或判别器的读法有 bug：{reasons[:3]}"
        )
        assert result.accepted > 0, f"{name}/{locale}/{split} 一条都没出"
        # 每个 case 正好出一条样本：少一条就是某个 case 被静默跳过了（各家 `test_pipeline.py`
        # 里各抄一份的那条断言，现在只留这里）。
        expected = len(wf.build_cases(cfg.config_override))
        assert result.accepted == expected, (
            f"{name}/{locale}/{split} 出了 {result.accepted} 条、case 有 {expected} 个——"
            "每个 case 必须正好出一条样本"
        )
        counts[locale] = result.accepted
        run_dirs.append(result.run_dir)
        # 审计报告里带 `families` 表的那几家：每个题族都得真的出过题，一个都不许是 0
        # （题族一条不出等于那道题根本没进数据集，而审计的 `missing` 只看计数与自洽）。
        report, _missing = wf.audit(result.run_dir, (split,), locale)
        families = report.get(split, {}).get("families")
        if isinstance(families, dict):
            assert all(families.values()), f"{name}/{locale}/{split} 有题族一条都没出：{families}"
        assert_audit_ignores_the_callers_locale(wf, result.run_dir, split, f"{name}/{locale}/{split}")
        assert_contract(result.run_dir, split, f"{name}/{locale}/{split}")
        assert_no_unresolved_slots(result.run_dir, split, f"{name}/{locale}/{split}")
        assert_manifest_and_stats(result.run_dir, split, wf, f"{name}/{locale}/{split}", locale)
        assert_language_texts_are_declared(
            wf,
            result.run_dir,
            split,
            f"{name}/{locale}/{split}",
            prose=name not in LLM_WITHOUT_PROSE,
        )
    assert len(set(counts.values())) == 1, f"{name}/{split} 各语言出的条数不一致：{counts}"
    assert_audit_reads_a_mixed_file(wf, run_dirs, split, f"{name}/{split}", tmp_path)


def test_every_locale_runs_end_to_end(tmp_path: Path) -> None:
    """输入：无（名单现场列）。输出：无。

    预期行为：每个 (workflow, 切分) 在**每种语言**下都过 `_sweep_one` 那一整套；问题攒在
    `bad` 里一起报（点名到 workflow / 切分），不因为第一家炸掉就看不见其余。
    """
    bad: list[str] = []
    for name in LLM_WORKFLOWS:
        for split in ("train", "test_ood"):
            try:
                _sweep_one(name, split, tmp_path)
            except AssertionError as error:
                bad.append(f"{name}/{split}：{error}")
    assert not bad, bad
