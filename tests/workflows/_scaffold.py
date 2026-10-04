"""各 workflow 测试包共用的试跑脚手架：一条 `RunConfig` 只在这里拼一次。

为什么单独拎出来：62 个测试文件各抄一份 `make_config`，只有 3 个字段不同（`run_seed` /
`concurrency` / `language_gate`）。改一处口径要改 62 处，漏改的那份会**静默按旧口径跑**——
测试照样绿，量到的却是别的东西。这里只拼标准形状，各家在包装函数里点名 workflow、override
与那几个开关。

输入：临时目录、run_id、workflow 名、覆盖用的采样参数，以及可选的 backend / 切分 / 语言 /
种子 / 每条样本份数 / 并发 / 语言闸开关。
输出：`jc.pipeline.RunConfig`。
预期行为：字段口径统一——`out_root` 一律指调用方给的临时目录、`per_case` 默认 1、
`locales` 默认 `("zh",)`；`run_seed` / `concurrency` / `language_gate` 由调用方显式给，
各家历史上用的取值原样保留（改它们等于换数据，必须是有意的）。
"""

from __future__ import annotations

from pathlib import Path

from jc import pipeline
from jc import jsonio
from jc.jsontypes import JsonValue


def run_config(
    tmp_path: Path,
    run_id: str,
    *,
    workflow: str,
    override: object,
    backend: object = None,
    split: str = "train",
    locales: tuple[str, ...] = ("zh",),
    seed: int = 7,
    per_case: int = 1,
    concurrency: int = 1,
    language_gate: bool = True,
) -> pipeline.RunConfig:
    """输入：临时目录、run_id、workflow 名、采样覆盖参数与几个可选开关。输出：`RunConfig`。

    预期行为：拼一条指向 `tmp_path` 的试跑配置；`backend` 只在 workflow 需要后端时给，
    纯构造式的包传 None（管线本来就不碰它）。**不**替调用方决定种子与并发——那两样是各包
    历史取值，只能显式传。
    """
    return pipeline.RunConfig(
        workflow=workflow,
        split=split,
        locales=locales,
        run_seed=seed,
        per_case=per_case,
        concurrency=concurrency,
        out_root=tmp_path,
        run_id=run_id,
        config_override=override,
        backend=backend,  # type: ignore[arg-type]  # 只有 needs 非空的 workflow 才真给后端
        language_gate=language_gate,
    )


def read_rows(run_dir: Path) -> list[JsonValue]:
    """输入：run 目录。输出：`train.jsonl` 的样本行（`(行号, 行)` 里的行）。

    预期行为：只读、只认 `train.jsonl`——审计类测试要的就是「管线刚写出来的那一份」。
    """
    return [row for _, row in jsonio.iter_jsonl(run_dir / "train.jsonl")]


def write_rows(run_dir: Path, rows: list[JsonValue]) -> None:
    """输入：run 目录、样本行。输出：无。预期行为：原样覆盖 `train.jsonl`（模拟产物被改坏）。"""
    jsonio.write_jsonl(run_dir / "train.jsonl", rows)


def flip_choice(row: JsonValue, qid: str) -> None:
    """输入：样本行、题号。输出：无（就地改）。预期行为：把 choice 的答案编号改成另一个选项。

    预期行为：`(答案 + 1) % 选项数` —— 一定与真答案不同，用来验审计真的会抓「答案与文本不自洽」。
    """
    question = row["questions"][qid]
    answer = row["answers"][qid]
    answer["choice"] = (answer["choice"] + 1) % len(question["criteria"])
