"""纯构造 workflow 的「每种语言 × 每个切分」端到端回归（**零 token**）。

输入：所有 `needs` 为空的 workflow（不碰任何后端，所以不需要 key）。
输出：无（断言）。
预期行为：每个 (workflow, 语言, 切分) 跑一遍到临时目录，断言

- 一条都没被拒——构造式 workflow 的金答案由构造保证，被拒只能是生成或判别器的 bug；
- 审计没有缺失项——**每一种语言的口径**各审一遍都要零缺失，把两种语言的产物拼成一份
  （一次 `locales=("zh","en")` 的 run 就是这形状）再审一遍也要零缺失；产物非空
  （否则这条测试等于没跑）；
- 各语言出的条数一致——case 集合与语言无关，条数不一致就说明某种语言在悄悄丢样本；
- 产物过得了通用格式契约（`jc.validate`，见 `_contract.py`）。

为什么值得进测试套件：`tmp/construct_sweep.py` 那把零 token 扫描就是这么发现迷宫 en 侧
782 条拒稿的（范式 §9.32），但它躺在 `tmp/` 里，进不了门禁。判别器读题面 / 读 state 的
方式都是**语言相关**的，只跑一种语言等于留盲区；这条测试把「每种语言都要跑」变成机器可查。

切分为什么要两遍：`train` 与 `test_ood` 是**两套 case**（留出尺寸 / 格式 / 符号集 / 值），
跑过 train 不蕴含跑过 ood。补这一遍之前，`maze` 的 `test_ood`（8×10 大迷宫、4000 条）
在**任何测试里都没端到端跑过**——`tests/workflows/maze/test_labels.py` 只验了
`config_for("test_ood").size`，那是配置，不是产物。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import pipeline
from jc import workflow as workflow_mod

from tests.workflows._contract import (
    assert_audit_ignores_the_callers_locale,
    assert_audit_reads_a_mixed_file,
    assert_contract,
)


def constructive_names() -> list[str]:
    """输入：无。输出：`needs` 为空的 workflow 名（排序后）。

    预期行为：先 `discover()` 触发各 workflow 的「导入即注册」，再按 `needs` 过滤；
    需要后端的那些不在这里，因为它们跑起来要 key。
    """
    return [name for name in workflow_mod.discover() if not workflow_mod.load(name).needs]


#: 纯构造 workflow 名单（模块导入时算一次）。
CONSTRUCTIVE: tuple[str, ...] = tuple(constructive_names())


def test_constructive_workflows_are_found() -> None:
    """输入：无。输出：无。预期行为：至少扫到若干纯构造 workflow，否则这把闸名存实亡。"""
    assert len(CONSTRUCTIVE) >= 5, f"只找到 {CONSTRUCTIVE}，纯构造 workflow 不该这么少"


@pytest.mark.parametrize("split", ("train", "test_ood"))
@pytest.mark.parametrize("name", CONSTRUCTIVE)
def test_every_locale_runs_without_rejects(name: str, split: str, tmp_path: Path) -> None:
    """输入：workflow 名、切分、临时目录。输出：无。

    预期行为：给定切分在**每个语言**下都零拒稿、零审计缺失、产物非空、过通用格式契约，
    而且**各语言出的条数一致**。
    """
    wf = workflow_mod.load(name)
    locales = wf.available_locales()
    assert locales, f"{name} 没声明任何语言"
    counts: dict[str, int] = {}
    run_dirs: list[Path] = []
    for locale in locales:
        cfg = pipeline.RunConfig(
            workflow=name,
            split=split,
            locales=(locale,),
            run_seed=20260924,
            per_case=1,
            concurrency=1,
            out_root=tmp_path,
            run_id=f"sweep-{name}-{locale}-{split}",
            config_override=wf.config_for(split),
        )
        result = pipeline.run(cfg)
        reasons = [row.get("reason") for row in result.rejected]
        assert not result.rejected, (
            f"{name}/{locale}/{split} 有 {len(result.rejected)} 条被拒（构造式 workflow 不该被拒）："
            f"{reasons[:3]}"
        )
        assert result.accepted > 0, f"{name}/{locale}/{split} 一条都没出"
        counts[locale] = result.accepted
        run_dirs.append(result.run_dir)
        assert_audit_ignores_the_callers_locale(wf, result.run_dir, split, f"{name}/{locale}/{split}")
        assert_contract(result.run_dir, split, f"{name}/{locale}/{split}")
    assert len(set(counts.values())) == 1, f"{name}/{split} 各语言出的条数不一致：{counts}"
    assert_audit_reads_a_mixed_file(wf, run_dirs, split, f"{name}/{split}", tmp_path)
