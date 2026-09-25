"""通用层测试：命令行入口（`python -m jc <命令>`）。

只测**接线**：命令行参数 → `RunConfig` / 打印文本 / 退出码。
真正的生成与审计由 `tests/workflows/` 与 `tests/core/test_pipeline_backend.py` 覆盖，
这里用假 `pipeline.run` 与空目录，保证跑得快、不联网、不写仓库。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from jc import cli, pipeline
from jc.backend.types import BackendCapabilities
from jc.workflow import discover, load


def test_workflows_lists_every_registered_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    """`jc workflows` 列出的名字必须与注册表逐字一致（顺序也一致）。

    输入：无。输出：无。预期行为：退出码 0，且两个已落地的 workflow 都在名单里。
    """
    assert cli.main(["workflows"]) == 0
    printed = capsys.readouterr().out.split()
    assert printed == discover()
    assert {"maze", "mail"} <= set(printed)


def test_locales_prints_the_languages_of_a_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    """`jc locales --workflow maze` 打印该 workflow 支持的语言标签。"""
    assert cli.main(["locales", "--workflow", "maze"]) == 0
    assert set(capsys.readouterr().out.split()) == {"zh", "en"}


def test_cases_prints_the_matrix_size_and_truncates_after_ten(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`jc cases` 先报矩阵规模，超过 10 个只列前 10 个（开工前看覆盖用）。

    输入：workflow 名与每条样本数。输出：无。
    预期行为：规模数字与 `build_cases` 的真实长度一致；第 11 行起是省略提示。
    """
    workflow = load("maze")
    cases = workflow.build_cases(workflow.config_for("train"))
    assert cli.main(["cases", "--workflow", "maze", "--per-case", "2"]) == 0
    out = capsys.readouterr().out
    assert f"{len(cases)} 个 case × 2 条 = {len(cases) * 2} 条样本" in out
    listed = [line for line in out.splitlines() if line.startswith("  ")]
    assert len([line for line in listed if not line.startswith("  …")]) == 10
    assert listed[-1].startswith("  …")


def test_audit_reports_missing_splits_in_plain_language(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """产物缺失时 `jc audit` 必须返回 1 并逐条列出缺什么，绝不静默通过。"""
    assert cli.main(["audit", "--workflow", "maze", "--dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "审计不通过" in out
    assert "train.jsonl 不存在" in out


def test_gen_wires_every_flag_into_the_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`jc gen` 的每个开关都要落到 `RunConfig` 上；不写后端配置时后端是 None。

    输入：一串命令行参数。输出：无。
    预期行为：逐字段比对 `RunConfig`，并确认默认语言是 `zh`、汇总行被打印出来。
    """
    seen: list[pipeline.RunConfig] = []

    def fake_run(config: pipeline.RunConfig) -> pipeline.RunResult:
        seen.append(config)
        return pipeline.RunResult(run_dir=tmp_path, accepted=3)

    monkeypatch.setattr(pipeline, "run", fake_run)
    code = cli.main(
        [
            "gen",
            "--workflow", "maze",
            "--split", "val",
            "--per-case", "3",
            "--seed", "7",
            "--out", str(tmp_path),
            "--run-id", "r9",
            "--balance-tolerance", "0.05",
            "--balance-strict",
        ]
    )
    assert code == 0
    config = seen[0]
    assert (config.workflow, config.split, config.per_case) == ("maze", "val", 3)
    assert config.locales == ("zh",)
    assert config.run_seed == 7
    assert config.out_root == tmp_path
    assert config.run_id == "r9"
    assert config.balance_tolerance == 0.05
    assert config.balance_strict is True
    assert config.backend is None
    out = capsys.readouterr().out
    assert "合格 3 条" in out
    assert str(tmp_path) in out


def test_audit_prints_the_report_and_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """审计通过时逐项打印计数并返回 0（退出码 0 只由「缺失项为空」决定）。

    输入：一个假 workflow（直接返回现成的计数）。输出：无。
    预期行为：`audit` **按产物自己报的键**逐项打印（映射展开成「取值×条数」），
    这样换 workflow 时不用改通用层代码——写死某个 workflow 的键名会让别的 workflow
    一跑审计就 KeyError。
    """
    counts = {
        "samples": 7,
        "formats": Counter({"box": 7}),
        "types": Counter({"choice": 7}),
        "history": 2,
        "option_cell": 1,
        "trail": 3,
        "cut_samples": 0,
        "off_by_one": 4,
        "off_by_one_dev1": 4,
        "next_step": 5,
        "next_step_on_path": 5,
        "families": Counter({"next_step": 7}),
        "unparsed": 2,
    }

    class FakeWorkflow:
        """假 workflow：只实现 `audit` 用到的四个成员。"""

        name = "fake"
        version = "0.0.1"

        def split_names(self) -> tuple[str, ...]:
            """输入：无。输出：一个切分名。"""
            return ("train",)

        def audit(self, run_dir: Path, splits: tuple[str, ...], locale: str) -> tuple[Any, list[str]]:
            """输入：目录 / 切分 / 语言。输出：(计数, 缺失项)。预期行为：永远通过。"""
            assert splits == ("train",)
            return {"train": counts}, []

    monkeypatch.setattr(cli, "load", lambda name: FakeWorkflow())
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "=== train：7 条 ===" in out
    assert "formats：box×7" in out
    assert "off_by_one：4" in out
    assert "unparsed：2" in out
    # 条数只出现在标题行，不再单独打一行（`cut_samples` 这类计数键不受影响）
    assert not any(line.startswith("  samples：") for line in out.splitlines())
    assert "审计通过" in out


def test_gen_prints_reject_codes_and_unbalanced_families(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """有拒绝码与超容差的题族时，`jc gen` 要把它们列出来（不藏着）。"""
    monkeypatch.setattr(
        pipeline,
        "run",
        lambda config: pipeline.RunResult(
            run_dir=tmp_path,
            accepted=1,
            rejected=[{"id": "x", "code": "E_JUDGE_LINK", "reason": "r"}],
            stats={
                "reject_codes": {"E_JUDGE_LINK": 1},
                "balance": [
                    {"key": "choice:a", "total": 9, "max_dev": 0.4, "noise": 0.1, "ok": False}
                ],
            },
        ),
    )
    assert cli.main(["gen", "--workflow", "maze", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "被拒 1 条" in out
    assert "E_JUDGE_LINK" in out
    assert "答案分布超容差的题族（1 个）" in out
    assert "choice:a：n=9 最大偏差=0.400 噪声地板=0.100" in out


def test_gen_accepts_repeated_locales_and_builds_the_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """给了 `--backend-config` 才装配后端；`--locale` 可重复，顺序照抄。

    输入：临时后端配置 + 两个语言。输出：无。
    预期行为：后端装配走真代码路径，但探测被打桩（不联网、不烧配额）。
    """
    seen: list[pipeline.RunConfig] = []
    monkeypatch.setattr(
        pipeline, "run", lambda config: seen.append(config) or pipeline.RunResult(run_dir=tmp_path)
    )
    monkeypatch.setattr(
        "jc.backend.llm.probe_capabilities",
        lambda **kwargs: BackendCapabilities(json_schema=True),
    )
    monkeypatch.setenv("JC_FAKE_KEYS", "k1")
    config_path = tmp_path / "backends.toml"
    config_path.write_text(
        "\n".join(
            [
                "[[provider]]",
                'name = "fake"',
                'base_url = "https://example.test/v1"',
                'model = "m"',
                'keys_env = "JC_FAKE_KEYS"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    code = cli.main(
        [
            "gen",
            "--workflow", "mail",
            "--locale", "zh",
            "--locale", "en",
            "--backend-config", str(config_path),
            "--out", str(tmp_path),
        ]
    )
    assert code == 0
    config = seen[0]
    assert config.locales == ("zh", "en")
    assert config.backend is not None
    assert config.backend.name == "llm"
    assert config.backend.providers == ["fake"]


def test_unknown_workflow_is_reported_in_plain_language() -> None:
    """名字写错时报人话并列出已注册的 workflow，不静默跑空。"""
    with pytest.raises(KeyError, match="未找到 workflow"):
        cli.main(["cases", "--workflow", "nope"])


def test_missing_required_flag_exits_with_usage() -> None:
    """少给必填参数时按 argparse 的约定退出码 2 收场。"""
    with pytest.raises(SystemExit) as info:
        cli.main(["cases"])
    assert info.value.code == 2


def test_load_backend_without_a_config_does_not_import_the_extra() -> None:
    """不给配置就不装配后端（构造式 workflow 不依赖 `[llm]` extra）。"""
    assert cli._load_backend("") is None


def test_parse_judges_accepts_repeats_and_commas() -> None:
    """输入：`--judges` 的几种写法。输出：判别器名单。

    预期行为：可重复、可逗号分隔、去空白丢空项；一个都没给返回 None
    （= 用 workflow 默认判别器），**不是空元组**（那意味着「一个都别跑」）。
    """
    assert cli._parse_judges(None) is None
    assert cli._parse_judges([""]) is None
    assert cli._parse_judges(["", " , "]) is None
    assert cli._parse_judges(["llm:solvable"]) == ("llm:solvable",)
    assert cli._parse_judges(["llm:solvable,llm:answer"]) == ("llm:solvable", "llm:answer")
    assert cli._parse_judges(["link", " llm:wording "]) == ("link", "llm:wording")


def test_gen_passes_judges_to_the_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """输入：`--judges llm:solvable,llm:levels`。输出：无（断言）。

    预期行为：名单原样落到 `RunConfig.judges`；不给时是 None（走 workflow 默认）。
    """
    seen: list[pipeline.RunConfig] = []

    def fake_run(config: pipeline.RunConfig) -> pipeline.RunResult:
        seen.append(config)
        return pipeline.RunResult(run_dir=tmp_path, accepted=0)

    monkeypatch.setattr(pipeline, "run", fake_run)
    assert cli.main(["gen", "--workflow", "maze", "--out", str(tmp_path)]) == 0
    assert cli.main(
        [
            "gen",
            "--workflow", "maze",
            "--out", str(tmp_path),
            "--judges", "llm:solvable,llm:levels",
        ]
    ) == 0
    assert seen[0].judges is None
    assert seen[1].judges == ("llm:solvable", "llm:levels")


def test_gen_reports_skipped_judge_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """输入：统计里带了 `judge_skipped`。输出：一行提示，指向 judge-skipped.jsonl。

    预期行为：跳过的条数必须露出来——否则「一条都没判成」看起来跟「全过」一模一样。
    """

    def fake_run(config: pipeline.RunConfig) -> pipeline.RunResult:
        return pipeline.RunResult(
            run_dir=tmp_path, accepted=2, stats={"judge_skipped": 3, "balance": []}
        )

    monkeypatch.setattr(pipeline, "run", fake_run)
    assert cli.main(["gen", "--workflow", "maze", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "质检跳过 3 条" in out
    assert "judge-skipped.jsonl" in out


def test_gen_reports_consistency_conflicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """输入：统计里带了 `conflicts`。输出：一行警告，指向 conflicts.jsonl。

    预期行为：自相矛盾的训练对必须喊出来；0 处时不吭声（正常路径别刷屏）。
    """

    def fake_run(config: pipeline.RunConfig) -> pipeline.RunResult:
        return pipeline.RunResult(
            run_dir=tmp_path, accepted=2, stats={"conflicts": 4, "balance": []}
        )

    monkeypatch.setattr(pipeline, "run", fake_run)
    assert cli.main(["gen", "--workflow", "maze", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "4 处「同素材同题不同答」" in out
    assert "conflicts.jsonl" in out


def test_gen_says_nothing_when_there_are_no_conflicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """输入：`conflicts` 为 0。输出：没有那行警告。预期行为：正常路径不刷屏。"""

    def fake_run(config: pipeline.RunConfig) -> pipeline.RunResult:
        return pipeline.RunResult(
            run_dir=tmp_path, accepted=2, stats={"conflicts": 0, "balance": []}
        )

    monkeypatch.setattr(pipeline, "run", fake_run)
    assert cli.main(["gen", "--workflow", "maze", "--out", str(tmp_path)]) == 0
    assert "同素材同题不同答" not in capsys.readouterr().out
