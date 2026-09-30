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
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path

import pytest

from jc import pipeline
from jc import workflow as workflow_mod

from tests.workflows._contract import (
    LLM_WITHOUT_PROSE,
    assert_audit_ignores_the_callers_locale,
    assert_audit_reads_a_mixed_file,
    assert_contract,
    assert_language_texts_are_declared,
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
    assert LLM_WITHOUT_PROSE, "豁免清单空了？那 lint / review / shell 就该报出正文了"


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
        # 假后端替模型写字，而标记类文案（步骤词、提示框关键词、署名前缀）是语言相关的：
        # 接受 `locale` 的假后端要按目标语言现造，不接受说明它的草稿与语言无关（命令、正文）。
        kwargs["backend"] = (
            maker(locale=locale) if "locale" in inspect.signature(maker).parameters else maker()
        )
    cfg = tests.make_config(tmp_path, run_id, **kwargs)
    return dataclasses.replace(cfg, locales=(locale,), run_id=run_id)


@pytest.mark.parametrize("split", ("train", "test_ood"))
@pytest.mark.parametrize("name", LLM_WORKFLOWS)
def test_every_locale_runs_end_to_end(name: str, split: str, tmp_path: Path) -> None:
    """输入：workflow 名、切分、临时目录。输出：无。

    预期行为：给定切分在**每种语言**下都零拒稿、产物非空，而且**各语言出的条数一致**
    ——case 集合与语言无关，条数不一致就说明某种语言在悄悄丢样本（迷宫 en 侧那次就是这个症状）。
    """
    wf = workflow_mod.load(name)
    locales = wf.available_locales()
    assert locales, f"{name} 没声明任何语言"
    counts: dict[str, int] = {}
    run_dirs: list[Path] = []
    for locale in locales:
        result = pipeline.run(config_for(name, tmp_path, locale, split))
        reasons = [row.get("reason") for row in result.rejected]
        assert not result.rejected, (
            f"{name}/{locale}/{split} 有 {len(result.rejected)} 条被拒——假后端是忠实的，"
            f"拒稿只可能是渲染或判别器的读法有 bug：{reasons[:3]}"
        )
        assert result.accepted > 0, f"{name}/{locale}/{split} 一条都没出"
        counts[locale] = result.accepted
        run_dirs.append(result.run_dir)
        assert_audit_ignores_the_callers_locale(wf, result.run_dir, split, f"{name}/{locale}/{split}")
        assert_contract(result.run_dir, split, f"{name}/{locale}/{split}")
        assert_language_texts_are_declared(
            wf,
            result.run_dir,
            split,
            f"{name}/{locale}/{split}",
            prose=name not in LLM_WITHOUT_PROSE,
        )
    assert len(set(counts.values())) == 1, f"{name}/{split} 各语言出的条数不一致：{counts}"
    assert_audit_reads_a_mixed_file(wf, run_dirs, split, f"{name}/{split}", tmp_path)
