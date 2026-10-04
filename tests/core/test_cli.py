"""通用层测试：命令行入口（`python -m jc <命令>`）。

只测**接线**：命令行参数 → `RunConfig` / 打印文本 / 退出码。
真正的生成与审计由 `tests/workflows/` 与 `tests/core/test_pipeline_backend.py` 覆盖，
这里用假 `pipeline.run` 与空目录，保证跑得快、不联网、不写仓库。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


import pytest

from jc import cli, pipeline
from jc.backend.types import BackendCapabilities
from jc.jsontypes import JsonValue
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

        def audit(self, run_dir: Path, splits: tuple[str, ...], locale: str) -> tuple[object, list[str]]:
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


def test_gen_reads_the_key_from_an_env_file_without_touching_the_process_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """给了 `--env-file` 时 key 只从文件读进内存——进程环境里一个变量都不加。

    输入：一份后端配置 + 一份 dotenv 文件（同名的进程环境变量**故意删掉**）。输出：无。
    预期行为：装配出的后端照样有那家 provider（说明 key 真从文件读到了），而且
    `os.environ` 里始终没有那个变量——「key 不落环境变量」那条规矩的机器可查版。
    """
    seen: list[pipeline.RunConfig] = []
    monkeypatch.setattr(
        pipeline, "run", lambda config: seen.append(config) or pipeline.RunResult(run_dir=tmp_path)
    )
    monkeypatch.setattr(
        "jc.backend.llm.probe_capabilities",
        lambda **kwargs: BackendCapabilities(json_schema=True),
    )
    monkeypatch.delenv("JC_ENV_FILE_KEYS", raising=False)
    config_path = tmp_path / "backends.toml"
    config_path.write_text(
        "\n".join(
            [
                "[[provider]]",
                'name = "fake"',
                'base_url = "https://example.test/v1"',
                'model = "m"',
                'keys_env = "JC_ENV_FILE_KEYS"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    keys_path = tmp_path / "keys.env"
    keys_path.write_text('JC_ENV_FILE_KEYS="k1,k2"\n', encoding="utf-8")
    code = cli.main(
        [
            "gen",
            "--workflow",
            "mail",
            "--backend-config",
            str(config_path),
            "--env-file",
            str(keys_path),
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert seen[0].backend is not None
    assert seen[0].backend.providers == ["fake"]
    assert "JC_ENV_FILE_KEYS" not in os.environ


def test_unknown_workflow_is_reported_in_plain_language() -> None:
    """名字写错时报人话并列出**真实**已注册的 workflow，不静默跑空。

    预期行为：名单必须现算（`load` 在报错前补一次 `discover`）——此前注册表空着就直接读，
    报出来是「已注册：（空）」，等于没报。
    """
    with pytest.raises(KeyError, match="未找到 workflow") as info:
        cli.main(["cases", "--workflow", "nope"])
    assert "maze" in str(info.value)
    assert "（空）" not in str(info.value)


def test_module_entry_point_reports_a_wrong_name_without_a_traceback() -> None:
    """输入：`python -m jc cases --workflow nope`。输出：无。

    预期行为：退出码 2、stderr 是一行人话（列出真实注册的 workflow），**不是 traceback**——
    名字写错是用户输入问题，不该把调用栈糊到人脸上。`cli.main` 自己照旧抛 KeyError
    （见上一条），兜的是命令行这一层。
    """
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)]),
        "PYTHONIOENCODING": "utf-8",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "jc", "cases", "--workflow", "nope"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    assert proc.returncode == 2
    assert "未找到 workflow" in proc.stderr
    assert "maze" in proc.stderr
    assert "Traceback" not in proc.stderr


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


def test_audit_fails_on_duplicate_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """产物里有重复 id 时，`jc audit` 必须失败（退出码 1）。

    输入：一个自己报告「通过」的假 workflow + 一份 id 重复的产物。输出：无。
    预期行为：workflow 自己的审计说通过**不算数**——「id 全局唯一」是文件级不变量，
    逐样本校验器看不到，得由通用层拿 `validate.validate_file` 跨行查；问题文本要点名切分与错误码。
    """

    class FakeWorkflow:
        """假 workflow：自己永远报告通过。"""

        name = "fake"
        version = "0.0.1"

        def split_names(self) -> tuple[str, ...]:
            """输入：无。输出：一个切分名。"""
            return ("train",)

        def audit(self, run_dir: Path, splits: tuple[str, ...], locale: str) -> tuple[object, list[str]]:
            """输入：目录 / 切分 / 语言。输出：(计数, 缺失项)。预期行为：永远通过。"""
            return {"train": {"samples": 2}}, []

    # 逐字合法的样本写两遍：这样产物里**唯一**的问题就是 id 重复（别的错误码会污染这条测试）。
    row = (
        '{"id":"fake-train-0000000","state":"素材",'
        '"questions":{"ask":{"type":"noul","instructions":"对吗？"}},'
        '"answers":{"ask":{"type":"noul","noul":true}}}'
    )
    (tmp_path / "train.jsonl").write_text(f"{row}\n{row}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "load", lambda name: FakeWorkflow())
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "审计不通过" in out
    assert "train" in out
    assert "E_ID_DUP" in out
    assert "fake-train-0000000" in out


def test_audit_passes_when_the_product_also_holds_the_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """产物合法时，通用层的契约复核不能误报（退出码 0）。

    输入：一个假 workflow + 一份**逐字合法**的产物。输出：无。
    预期行为：`validate_file` 是复核而不是加严——真产物全过（本项目全语料实测只挑出 4 个
    有重复 id 的 `sweep-*`），所以合法产物照样「审计通过」。
    """

    class FakeWorkflow:
        """假 workflow：自己永远报告通过。"""

        name = "fake"
        version = "0.0.1"

        def split_names(self) -> tuple[str, ...]:
            """输入：无。输出：一个切分名。"""
            return ("train",)

        def audit(self, run_dir: Path, splits: tuple[str, ...], locale: str) -> tuple[object, list[str]]:
            """输入：目录 / 切分 / 语言。输出：(计数, 缺失项)。预期行为：永远通过。"""
            return {"train": {"samples": 1}}, []

    (tmp_path / "train.jsonl").write_text(
        '{"id":"fake-train-0000000","state":"素材",'
        '"questions":{"ask":{"type":"noul","instructions":"对吗？"}},'
        '"answers":{"ask":{"type":"noul","noul":true}}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "load", lambda name: FakeWorkflow())
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 0
    assert "审计通过" in capsys.readouterr().out


class LocaleSpy:
    """假 workflow：记下 `audit` 每次被调用时收到的语言。

    预期行为：只实现审计要用的两个方法——这条测试问的是「CLI 拿哪种语言去读产物」，
    与产物本身无关。
    """

    name = "fake"
    version = "0.0.1"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：一个切分名。"""
        return ("train",)

    def audit(self, run_dir: Path, splits: tuple[str, ...], locale: str) -> tuple[object, list[str]]:
        """输入：目录 / 切分 / 语言。输出：(计数, 缺失项)。预期行为：把语言记下来。"""
        self.seen.append(locale)
        return {split: {"samples": 0} for split in splits}, []

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：空表。

        预期行为：这一族问的是「CLI 拿哪种语言去读产物」，与产物正文无关，所以不声明任何正文
        （声明了就会顺带跑语言判据，把这条测试的因果搅浑）。
        """
        del sample
        return {}


def write_manifest(run_dir: Path, split: str, locales: list[str]) -> None:
    """输入：run 目录、切分、语言列表。输出：无。

    预期行为：写一份最小 manifest——只放审计推断语言时要读的那个键。
    """
    (run_dir / f"manifest-{split}.json").write_text(
        json.dumps({"workflow": "fake", "split": split, "locales": locales}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_audit_infers_the_locale_from_the_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """输入：一份 manifest 记着 `["en"]` 的 run 目录。输出：无。

    预期行为：不给 `--locale` 时，审计用**产物自己记的语言**读它。产物是哪种语言就得用哪种读，
    否则读出来是空串，会报一堆**假的**「答案与原文矛盾」（实测 en 产物按 zh 读，9 条里 4 条假）。
    """
    spy = LocaleSpy()
    write_manifest(tmp_path, "train", ["en"])
    monkeypatch.setattr(cli, "load", lambda name: spy)
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 0
    assert spy.seen == ["en"]


def test_audit_explicit_locale_beats_the_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """输入：manifest 记着 en，但命令行显式给了 zh。输出：无。预期行为：显式的赢。"""
    spy = LocaleSpy()
    write_manifest(tmp_path, "train", ["en"])
    monkeypatch.setattr(cli, "load", lambda name: spy)
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path), "--locale", "zh"]) == 0
    assert spy.seen == ["zh"]


def test_audit_warns_when_the_run_mixes_locales(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """输入：manifest 记着 `["zh", "en"]` 的 run 目录。输出：无。

    预期行为：混语言的产物没法用单一语言口径逐条读，CLI 要**明说**而不是给个看着像通过的结果；
    退回第一种语言继续跑，并在提示里点出「结果可能包含假失败」。
    """
    spy = LocaleSpy()
    write_manifest(tmp_path, "train", ["zh", "en"])
    monkeypatch.setattr(cli, "load", lambda name: spy)
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 0
    assert "混了语言" in capsys.readouterr().out
    assert spy.seen == ["zh"]


def test_audit_falls_back_to_zh_without_a_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """输入：没有 manifest 的老产物目录。输出：无。预期行为：维持老口径（zh），不报错。"""
    spy = LocaleSpy()
    monkeypatch.setattr(cli, "load", lambda name: spy)
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 0
    assert spy.seen == ["zh"]


#: 一份**逐字合法**的产物行：语言那几条测试要的是「唯一的问题是语言」，
#: 别的错误码（缺字段 / id 不合法）会污染因果。
LEGAL_ROW = (
    '{"id":"fake-train-0000000","state":"素材",'
    '"questions":{"ask":{"type":"noul","instructions":"对吗？"}},'
    '"answers":{"ask":{"type":"noul","noul":true}}}'
)


class LanguageSpy:
    """假 workflow：`language_texts` 报一段**写死**的正文，用来单测语言那条尺子的接线。

    预期行为：只实现审计要用的三个方法；`language_texts` 与样本无关——这条测试问的是
    「CLI 会不会拿产物声明的语言去比它」，不是「某个真 workflow 反解对不对」。
    """

    name = "fake"
    version = "0.0.1"

    def __init__(self, texts: dict[str, str]) -> None:
        """输入：要报出去的正文。输出：实例。"""
        self.texts = texts

    def split_names(self) -> tuple[str, ...]:
        """输入：无。输出：一个切分名。"""
        return ("train",)

    def audit(self, run_dir: Path, splits: tuple[str, ...], locale: str) -> tuple[object, list[str]]:
        """输入：目录 / 切分 / 语言。输出：(计数, 缺失项)。预期行为：永远通过。"""
        return {split: {"samples": 1} for split in splits}, []

    def language_texts(self, sample: dict[str, JsonValue]) -> dict[str, str]:
        """输入：契约样本。输出：写死的那段正文。预期行为：不看样本，直接回写死的那份。"""
        del sample
        return dict(self.texts)


def test_audit_fails_when_the_text_does_not_match_the_declared_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """输入：manifest 声明 en，但正文是一整段中文。输出：无。

    预期行为：退出码 1，并把「哪一条、哪个字段、实测比例」说清楚。产物文本必须与它**声明的**
    语言一致——真机上 `rerank` 的 en 侧整段写成中文、`citation` 的 en 侧有一条写成中文文档，
    此前没有任何机制拦得住。
    """
    (tmp_path / "train.jsonl").write_text(LEGAL_ROW + "\n", encoding="utf-8")
    write_manifest(tmp_path, "train", ["en"])
    spy = LanguageSpy({"正文": "这一段写成了中文，而这一份产物声明的是英文。"})
    monkeypatch.setattr(cli, "load", lambda name: spy)
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "文本与声明语言对不上" in out
    assert "正文" in out
    assert "fake-train-0000000" in out


def test_audit_passes_when_the_text_matches_the_declared_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """输入：manifest 声明 zh，正文也是中文。输出：无。预期行为：退出码 0，不误报。"""
    (tmp_path / "train.jsonl").write_text(LEGAL_ROW + "\n", encoding="utf-8")
    write_manifest(tmp_path, "train", ["zh"])
    spy = LanguageSpy({"正文": "这一段是中文，声明也是中文，两边对得上。"})
    monkeypatch.setattr(cli, "load", lambda name: spy)
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 0
    assert "审计通过" in capsys.readouterr().out


def test_audit_passes_when_the_text_matches_any_declared_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """输入：manifest 记着 `["zh", "en"]`，正文是英文。输出：无。

    预期行为：退出码 0——一次 run 可以把两种语言轮转着写进**同一份**产物（`jc.pipeline` 按
    `locales[seq % len]` 轮转），所以判据是「配不配得上**某一种**声明语言」，不是「配不配得上
    第一种」。这条不这么定，混语言 run 会被整份误判。
    """
    (tmp_path / "train.jsonl").write_text(LEGAL_ROW + "\n", encoding="utf-8")
    write_manifest(tmp_path, "train", ["zh", "en"])
    spy = LanguageSpy({"正文": "this paragraph is written in english and the run declares both"})
    monkeypatch.setattr(cli, "load", lambda name: spy)
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 0


def test_audit_reports_a_workflow_that_never_declares_its_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """输入：一个没有 `language_texts` 的假 workflow + 有 manifest 的产物。输出：无。

    预期行为：退出码 1 并**如实说**「这条检查跳过了」——静默跳过等于告诉用户「查过了，没问题」，
    而实际上一次都没查。真 workflow 由 `tests/workflows/test_protocol.py` 兜住。
    """
    (tmp_path / "train.jsonl").write_text(LEGAL_ROW + "\n", encoding="utf-8")
    write_manifest(tmp_path, "train", ["zh"])

    class NoDeclaration:
        """假 workflow：只实现审计要用的两个方法（**故意**不实现 `language_texts`）。"""

        name = "fake"
        version = "0.0.1"

        def split_names(self) -> tuple[str, ...]:
            """输入：无。输出：一个切分名。"""
            return ("train",)

        def audit(
            self, run_dir: Path, splits: tuple[str, ...], locale: str
        ) -> tuple[object, list[str]]:
            """输入：目录 / 切分 / 语言。输出：(计数, 缺失项)。预期行为：永远通过。"""
            return {split: {"samples": 1} for split in splits}, []

    monkeypatch.setattr(cli, "load", lambda name: NoDeclaration())
    assert cli.main(["audit", "--workflow", "fake", "--dir", str(tmp_path)]) == 1
    assert "没有实现" in capsys.readouterr().out
