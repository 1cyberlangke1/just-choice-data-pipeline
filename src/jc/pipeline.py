"""管线编排（通用层）：case 矩阵 → 生成 → 校验 → 判别 → 打包 + 统计 + 均衡闸。

通用层不认识任何具体 workflow：所有领域相关的动作都通过 `Workflow` 协议委托出去
（采样参数、case 矩阵、统计维度、判别器、产物审计）。

设计要点：
- 每条样本的 RNG 由 `sha256(run_seed, workflow, 切分, 序号)` 派生，任何一条都能单独重算，
  并行与续跑都不会改变 id 或内容；
- 序号对 case 矩阵长度取模，保证每个 case 都被轮到；
- **产物边跑边追加**，同时把「这个序号完成了」写进 `progress.jsonl`：进程被杀 / 断电后
  重启会跳过已完成的序号，不重跑也不重写（3.1.5 的顺序铁律：先产物、后进度）；
- 产物：`dataset/<run_id>/<split>.jsonl`（合格）、`rejected.jsonl`（被拒）、
  `judge-skipped.jsonl`（没判成）、`progress.jsonl`（续跑用）、
  `stats-<split>.md`（统计 + 均衡）、`manifest-<split>.json`（溯源）。

输入：RunConfig。
输出：RunResult（计数与路径）。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import assemble, consistency, jsonio, judge as judge_mod, validate
from .workflow import SplitSpec, Workflow, family_of, load

# 管线级（非契约级）拒绝码；判别器失败会带自己的名字（E_JUDGE_<NAME>）
E_ASSEMBLE = "E_ASSEMBLE"
E_EMPTY = "E_EMPTY"


@dataclass(frozen=True)
class RunConfig:
    """一次生成的完整配置（写进 manifest，便于复现）。"""

    workflow: str  # 必填：核心层不认识任何具体 workflow，由调用方点名
    split: str = "train"
    locales: tuple[str, ...] = ("zh",)
    run_seed: int = 20260924
    per_case: int = 1
    concurrency: int = 1  # 同时推进多少条样本（每条内部还受后端的并发闸限制）
    out_root: Path = Path("dataset")
    run_id: str = ""
    balance_tolerance: float = 0.03  # 每个取值的占比偏离「均匀」的上限（3 个百分点）
    balance_strict: bool = False  # True 时超容差直接让整批失败
    judges: tuple[str, ...] | None = None  # None = 用 workflow 声明的默认判别器
    #: 一条样本最多渲染几次：判别器拒掉就重渲染（PLAN 6.1 第二道闸）；1 = 不重试。
    render_attempts: int = 2
    config_override: Any = None  # 覆盖 workflow 的默认采样参数（小规模试跑 / 测试用）
    backend: Any = None  # 生成后端（jc.backend.types.GenerationBackend）；needs 为空的 workflow 用不到

    def resolved_run_id(self) -> str:
        """运行目录名；未指定时用时间戳。"""
        return self.run_id or time.strftime("run-%Y%m%d-%H%M%S")


@dataclass
class RunResult:
    """一次运行的结果：目录、合格数、拒绝明细、统计。"""

    run_dir: Path
    accepted: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class RunAborted(RuntimeError):
    """整批中止（PLAN 3.1.4 说的「全部 provider 都不可用 = 整批停下的唯一条件」）。

    字段：`run_dir` 已经落盘的目录（**中间进度保留**）、`accepted` 已合格条数、
    `seq` 卡住的那一条、`reason` 原始说明。抛之前产物已经写完，所以不丢已做的工作。
    """

    def __init__(self, run_dir: Path, accepted: int, seq: int, reason: str) -> None:
        """输入：产物目录、已合格数、卡住的序号、原因。输出：异常实例。"""
        super().__init__(
            f"第 {seq} 条起全部 provider 都不可用，整批中止；"
            f"已合格 {accepted} 条，中间进度留在 {run_dir}：{reason}"
        )
        self.run_dir = run_dir
        self.accepted = accepted
        self.seq = seq
        self.reason = reason


def sample_rng(run_seed: int, workflow: str, split: str, seq: int) -> random.Random:
    """派生某一条样本的 RNG。

    输入：运行种子、workflow 名、切分、序号。
    输出：random.Random。
    预期行为：不用内置 hash()（它对字符串每进程随机化）；同一入参必得同一 RNG。
    """
    digest = hashlib.sha256(f"{run_seed}|{workflow}|{split}|{seq}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def build_judges(
    workflow: Workflow, names: tuple[str, ...], locale: str, backend: Any = None
) -> tuple[judge_mod.Judge, ...]:
    """委托 workflow 构造判别器。

    输入：workflow、判别器名单、语言、可选后端。
    输出：判别器元组。预期行为：`llm:<视角>` 的名字由通用层装配（要后端，所以放在这里，
    各 workflow 不必为它接线）；其余名字交给 workflow，未知名字由 workflow 报错，
    避免「写了名字却没跑」。
    """
    rules, perspectives = judge_mod.partition_names(names)
    judges = list(workflow.make_judges(rules, locale))
    if perspectives:
        if backend is None:
            raise ValueError(
                f"LLM 质检判别器 {list(perspectives)} 需要后端，但 RunConfig.backend 是空的"
                "（PLAN 3.1：后端由调用方装配后注入）"
            )
        from . import llmjudge

        judges += [llmjudge.make_judge(name, locale, backend) for name in perspectives]
    return tuple(judges)


def _mean(values: list[int]) -> float:
    """算术平均；空列表返回 0.0。"""
    return round(sum(values) / len(values), 1) if values else 0.0


def _consistency_line(stats: dict[str, Any]) -> str:
    """输入：统计字典。输出：一致性那一行（统计报告用）。

    预期行为：0 处冲突写成 ✅，非 0 写成 ⚠️ 并指向 `conflicts.jsonl`——
    「同素材同题不同答」是自相矛盾，不该悄悄过去。
    """
    count = stats.get("conflicts", 0)
    if not count:
        return "✅ 没有「同素材同题不同答」"
    return f"⚠️ {count} 处「同素材同题不同答」，见 conflicts.jsonl"


def balance_report(stats: dict[str, Any], tolerance: float) -> list[dict[str, Any]]:
    """逐题族算答案分布与「均匀」的偏差。

    输入：统计字典、容差（0.03 表示 3 个百分点）。
    输出：按偏差降序的列表，每项 {key, total, space, max_dev, noise, ok}。
    预期行为：
    - choice / score：参考值 1/space（space 取该族出现过的最多选项数），
      偏差取各取值占比与参考值的最大绝对差；
    - noul：参考值 True 占 50%，偏差取 |True 占比 − 0.5|；
    - 小样本下固定 ±3pp 统计上做不到：把「3 倍标准误」当噪声地板，
      只有同时超过容差与噪声地板才算真的失衡；
    - 结构性不可达（例如某个档位在当前规模下永远取不到）与题族间耦合
      （两个题族共享同一个自由度）会在报告里照实标出，由 workflow 决定怎么调。
    """
    report: list[dict[str, Any]] = []
    for key, bucket in stats["answers"].items():
        total = sum(bucket.values())
        if total == 0:
            continue
        if key.startswith("noul:"):
            dev = abs(bucket.get("True", 0) / total - 0.5)
            space = 2
            share = 0.5
        else:
            space = max(stats["answer_space"].get(key, 2), 2)
            dev = max(abs(count / total - 1 / space) for count in bucket.values())
            share = 1 / space
        noise = 3 * (share * (1 - share) / total) ** 0.5
        report.append(
            {
                "key": key,
                "total": total,
                "space": space,
                "max_dev": round(dev, 4),
                "noise": round(noise, 4),
                "ok": dev <= tolerance or dev <= noise,
            }
        )
    report.sort(key=lambda item: -item["max_dev"])
    return report


@dataclass
class _CaseOutcome:
    """一条样本的处理结果（并发安全：只返回，不碰共享状态）。"""

    sample: dict[str, Any] | None  # 合格样本；None = 被拒
    rejected: list[dict[str, Any]]  # 这一条产生的拒绝记录
    judges: list[tuple[str, bool]]  # (判别器名, 是否通过)，只用于统计；不含没判成的
    judge_skips: list[dict[str, Any]] = field(default_factory=list)  # 没判成的（端点故障 / 不适用）
    attempts: int = 1  # 实际渲染了几次（>1 说明前面几次被判别器拒了，重渲染过）


def _is_fatal(exc: BaseException) -> bool:
    """输入：异常。输出：是不是「整批该停下」级别的错误。

    预期行为：只认异常自己声明的 `fatal_for_run` 类属性（由后端实现决定），
    通用层不 import 任何后端模块——判据对任何后端实现都一样。
    """
    return bool(getattr(exc, "fatal_for_run", False))


def _retryable(outcome: _CaseOutcome) -> bool:
    """输入：一次尝试的结果。输出：这条样本要不要重渲染。

    预期行为：**只有判别器拒掉**才重渲染（PLAN 6.1 第二道闸「LLM 渲染重渲染，
    仍不一致才进拒绝」）。生成期与契约级失败不重渲染——那两类是生成器 bug，
    重渲染只会把 bug 藏起来（PLAN 6.1 第一道闸：拒绝率 >0 视为生成器 bug，要修）。
    """
    if outcome.sample is not None or not outcome.rejected:
        return False
    return all(row["stage"] == "judge" for row in outcome.rejected)


def _one_case(
    workflow: Workflow,
    judge_pool: dict[str, tuple[judge_mod.Judge, ...]],
    cases: list[Any],
    config: RunConfig,
    config_for_split: Any,
    seq: int,
) -> _CaseOutcome:
    """跑完一条样本的「计划 → 采样 → 渲染 → 出题 → 组装 → 契约校验 → 判别」。

    输入：workflow、按语言分组的判别器、case 矩阵、运行配置、采样配置、序号。
    输出：`_CaseOutcome`（`attempts` 记实际渲染了几次）。
    预期行为：判别器拒掉就**重渲染**（判据见 `_retryable`），最多 `config.render_attempts`
    次。重渲染用的是同一条 spec、同一份 prompt——模型每次采样不同，所以这等价于
    「换个说法再写一遍」，而标签由特征决定、不由文本决定，所以标签一点都不会变。
    函数内不写任何共享状态，所以可以并发调用。
    """
    limit = max(1, config.render_attempts)
    made = 1
    outcome = _attempt_case(workflow, judge_pool, cases, config, config_for_split, seq)
    outcome.attempts = made
    while made < limit and _retryable(outcome):
        outcome = _attempt_case(workflow, judge_pool, cases, config, config_for_split, seq)
        made += 1
        outcome.attempts = made
    return outcome


def _attempt_case(
    workflow: Workflow,
    judge_pool: dict[str, tuple[judge_mod.Judge, ...]],
    cases: list[Any],
    config: RunConfig,
    config_for_split: Any,
    seq: int,
) -> _CaseOutcome:
    """渲染并判一次（`_one_case` 的其中一次尝试）。

    输入：同 `_one_case`。输出：`_CaseOutcome`（`attempts` 由调用方填）。
    预期行为：生成期任何异常都记成拒绝，但**标了 `fatal_for_run` 的异常直接冒泡**
    （全部 provider 都挂了 = 整批停下，继续跑只会产出一整个「全被拒」的假数据集）。
    """
    case = cases[seq % len(cases)]
    sample_id = f"{workflow.name}-{config.split}-{seq:07d}"
    rng = sample_rng(config.run_seed, workflow.name, config.split, seq)
    locale = config.locales[seq % len(config.locales)]
    try:
        split: SplitSpec = workflow.plan_sample(rng, config_for_split, case, seq)
        spec = workflow.sample(rng, split)
        state = workflow.render(spec, rng, locale)
        labeled = workflow.ask(spec, rng, locale)
        if not labeled:
            raise ValueError("没有出到任何题（题族在当前 spec 下全部不成立）")
        sample = assemble.assemble_sample(sample_id, state, labeled, rng)
    except Exception as exc:  # noqa: BLE001 —— 生成期异常一律记为拒绝，不中断整批
        if _is_fatal(exc):
            raise
        return _CaseOutcome(
            sample=None,
            rejected=[{"id": sample_id, "stage": "generate", "code": E_ASSEMBLE, "reason": str(exc)}],
            judges=[],
        )

    issues = validate.validate_sample(sample)
    if issues:
        return _CaseOutcome(
            sample=None,
            rejected=[
                {
                    "id": sample_id,
                    "stage": "validate",
                    "code": issue.code,
                    "path": issue.path,
                    "reason": issue.msg,
                    "state": sample["state"],
                }
                for issue in issues
            ],
            judges=[],
        )

    verdicts = judge_mod.run_judges(sample, spec, judge_pool[locale])
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    failed = False
    for name, verdict in verdicts:
        if verdict.skipped:
            # 没判成（端点故障 / 视角不适用）：既不放行也不拒绝，单独记一笔账。
            # 记成「拒绝」就是把端点抖动写成数据质量问题（实测踩过，见 PLAN 10.10）。
            skips.append({"id": sample_id, "judge": name, "reason": verdict.reason})
            continue
        if verdict.ok:
            continue
        failed = True
        rows.append(
            {
                "id": sample_id,
                "stage": "judge",
                # 名字里的 `:`（`llm:solvable`）换成 `_`：拒绝码要能当文件名 / 键名用。
                "code": f"E_JUDGE_{name.upper().replace(':', '_')}",
                "reason": verdict.reason,
                # 带上 state：判别器拒掉一条时，唯一的诊断线索就是「模型到底写了什么」。
                # 不带的话只能重跑一遍（还不一定复现），等于没法查合成质量。
                "state": sample["state"],
            }
        )
    return _CaseOutcome(
        sample=None if failed else sample,
        rejected=rows,
        judges=[(name, verdict.ok) for name, verdict in verdicts if not verdict.skipped],
        judge_skips=skips,
    )


def _journal_path(run_dir: Path) -> Path:
    """输入：run 目录。输出：进度日志路径（PLAN 3.1.5 的 `progress.jsonl`）。"""
    return run_dir / "progress.jsonl"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """输入：产物路径。输出：已写下的行。

    预期行为：文件不存在给空表；**只有最后一行允许残缺**——追加写 + fsync 的性质是
    「崩溃时最多留半行」，那半行当它没写过；中间某行坏了说明文件真坏了，照旧抛出去。
    """
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                break  # 崩在写一半的尾行，当它没写过
            raise ValueError(f"产物 {path.name} 第 {index + 1} 行坏了：{exc}") from exc
    return out


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """输入：产物路径、要追加的行。输出：无。

    预期行为：一次写完并 fsync——崩溃时最坏留下半行，读回来时当它没写过（3.1.5）。
    空列表什么都不做，所以被拒行 / 跳过行一个都没有时不会凭空造出空文件。
    """
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(jsonio.dumps_line(row))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_journal(path: Path) -> dict[int, dict[str, Any]]:
    """输入：进度日志路径。输出：{序号: 记录}。

    预期行为：残缺的尾行按「没写过」处理（与产物同一条规矩）；同一序号写了多次
    以最后一次为准（重跑过就该以新结果为准）。
    """
    return {int(record["seq"]): record for record in _read_rows(path)}


def _outcome_from_journal(
    record: dict[str, Any],
    samples: list[dict[str, Any]],
    rejecteds: list[dict[str, Any]],
    skips: list[dict[str, Any]],
) -> _CaseOutcome:
    """输入：一条进度记录、三个产物在**本次启动时**读回来的全部行。输出：outcome。

    预期行为：把当时内存里的那一条还原出来——样本按 `row` 取回，被拒行与「没判成」行
    按记录里的区间切片（区间是写进度时定下的，所以取回来的一定是同一条样本的那几行）。
    还原得出来才谈得上「跳过已完成」：统计与 `RunResult` 都走同一条路径，不重跑判别器。
    """
    r_start, r_count = record["rejected"]
    s_start, s_count = record["skipped"]
    row = int(record["row"])
    return _CaseOutcome(
        sample=samples[row - 1] if record["status"] == "ok" else None,
        rejected=rejecteds[r_start : r_start + r_count],
        judges=[(str(name), bool(ok)) for name, ok in record["judges"]],
        judge_skips=skips[s_start : s_start + s_count],
        attempts=int(record.get("attempts", 1)),
    )


@dataclass
class _Artifacts:
    """一次 run 的产物写入口：追加写 + 进度日志（PLAN 3.1.5）。

    字段：三个产物路径与进度日志路径；`rows` 是各产物**已经写下**的行数。
    预期行为：`commit()` 先写产物并 fsync、再写进度——反过来会在崩溃时留下
    「进度说完成、产物没落盘」，而正着来最坏只是重做一条。
    """

    sample: Path
    rejected: Path
    skipped: Path
    journal: Path
    rows: dict[str, int] = field(default_factory=dict)

    def commit(self, outcome: _CaseOutcome, seq: int) -> dict[str, Any]:
        """输入：一条 outcome、序号。输出：写进进度日志的那条记录。

        预期行为：合格样本进 `<split>.jsonl`、被拒行进 `rejected.jsonl`、
        「没判成」行进 `judge-skipped.jsonl`，三者落盘之后才写进度。记录里带上重建统计
        所需的一切（状态、样本行号、判别器结论、被拒行与跳过行的区间）——
        重启时不必重跑判别器，也能把统计与 `RunResult` 对回来。
        """
        r_start = self.rows["rejected"]
        _append_rows(self.rejected, outcome.rejected)
        self.rows["rejected"] = r_start + len(outcome.rejected)
        s_start = self.rows["skipped"]
        _append_rows(self.skipped, outcome.judge_skips)
        self.rows["skipped"] = s_start + len(outcome.judge_skips)
        row = 0
        if outcome.sample is not None:
            row = self.rows["sample"] + 1
            _append_rows(self.sample, [outcome.sample])
            self.rows["sample"] = row
        record: dict[str, Any] = {
            "seq": seq,
            "status": "ok" if outcome.sample is not None else "invalid",
            "row": row,
            "judges": [[name, ok] for name, ok in outcome.judges],
            "rejected": [r_start, len(outcome.rejected)],
            "skipped": [s_start, len(outcome.judge_skips)],
            "attempts": outcome.attempts,
        }
        _append_rows(self.journal, [record])
        return record


def run(config: RunConfig) -> RunResult:
    """执行一次生成。

    输入：RunConfig。
    输出：RunResult（含目录、合格数、拒绝明细与统计）。
    预期行为：逐条走「计划 → 采样 → 渲染 → 出题 → 组装 → 契约校验 → 判别」；
    任何一步失败都写进 rejected.jsonl（带 id、阶段、原因），不静默丢弃。
    `needs` 非空的 workflow 必须由调用方装配好 `config.backend` 再进来：缺后端直接报错，
    绝不静默降级成「没有文本外壳的样本」。
    """
    workflow = load(config.workflow)
    if workflow.needs:
        if config.backend is None:
            raise ValueError(
                f"workflow {workflow.name!r} 声明需要后端 {sorted(workflow.needs)}，"
                "但 RunConfig.backend 是空的（PLAN 3.1：后端由调用方装配后注入）"
            )
        workflow.bind_backend(config.backend)
    split_names = workflow.split_names()
    if config.split not in split_names:
        raise KeyError(f"切分 {config.split!r} 不在 {workflow.name!r} 支持的切分里：{split_names}")
    judge_names = config.judges if config.judges is not None else workflow.judges

    config_for_split = (
        config.config_override if config.config_override is not None else workflow.config_for(config.split)
    )
    cases = workflow.build_cases(config_for_split)
    run_dir = Path(config.out_root) / config.resolved_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    total = len(cases) * config.per_case
    artifacts = _Artifacts(
        sample=run_dir / f"{config.split}.jsonl",
        rejected=run_dir / "rejected.jsonl",
        skipped=run_dir / "judge-skipped.jsonl",
        journal=_journal_path(run_dir),
    )
    # 续跑：启动时先把已写下的产物读回来。统计、`RunResult` 与「哪些序号已完成」
    # 都从它重建，所以重启既不重跑判别器，也不会把同一条写两遍。
    samples = _read_rows(artifacts.sample)
    rejected_all = _read_rows(artifacts.rejected)
    skipped_all = _read_rows(artifacts.skipped)
    artifacts.rows = {
        "sample": len(samples),
        "rejected": len(rejected_all),
        "skipped": len(skipped_all),
    }
    done = _read_journal(artifacts.journal)
    stats: dict[str, Any] = {
        "cases": len(cases),
        "requested": total,
        "families": {},
        "answers": {},
        "answer_space": {},
        "dimensions": {},
        "types": {},
        "locales": {},
        "state_chars": [],
        "questions_per_sample": [],
        "reject_codes": {},
        "judges": {},
        "render_retries": 0,
    }
    judge_pool = {
        locale: build_judges(workflow, judge_names, locale, config.backend)
        for locale in config.locales
    }
    if config.concurrency < 1:
        raise ValueError(f"concurrency 必须 >= 1，得到 {config.concurrency}")
    if config.render_attempts < 1:
        raise ValueError(f"render_attempts 必须 >= 1，得到 {config.render_attempts}")

    # 只跑还没完成的序号（3.1.5）：`done` 里已有的直接跳过，不重花配额。
    todo = [seq for seq in range(total) if seq not in done]
    aborted: dict[str, Any] | None = None
    fresh: dict[int, _CaseOutcome] = {}
    if config.concurrency > 1 and len(todo) > 1:
        # 并发跑；按序号取回，取回一条就落盘一条，所以产物顺序与串行逐字节一致。
        with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            futures = {
                seq: pool.submit(
                    _one_case, workflow, judge_pool, cases, config, config_for_split, seq
                )
                for seq in todo
            }
            for seq in todo:
                try:
                    outcome = futures[seq].result()
                except Exception as exc:  # noqa: BLE001 —— 只有标了 fatal 的会走到这里
                    aborted = {"seq": seq, "reason": str(exc)}
                    for pending in futures.values():
                        pending.cancel()  # 还没起跑的别再烧配额
                    break
                fresh[seq] = outcome
                done[seq] = artifacts.commit(outcome, seq)
    else:
        for seq in todo:
            try:
                outcome = _one_case(workflow, judge_pool, cases, config, config_for_split, seq)
            except Exception as exc:  # noqa: BLE001 —— 只有标了 fatal 的会走到这里
                aborted = {"seq": seq, "reason": str(exc)}
                break
            fresh[seq] = outcome
            done[seq] = artifacts.commit(outcome, seq)

    accepted_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    judge_skips: list[dict[str, Any]] = []
    # 按序号顺序合并「本次新跑的」与「上次已完成的」：续跑出来的统计与一次跑完的完全一致。
    for seq in sorted(done):
        outcome = (
            fresh[seq]
            if seq in fresh
            else _outcome_from_journal(done[seq], samples, rejected_all, skipped_all)
        )
        stats["render_retries"] += outcome.attempts - 1
        case = cases[seq % len(cases)]
        locale = config.locales[seq % len(config.locales)]
        if outcome.sample is not None:
            accepted_rows.append(outcome.sample)
            _accumulate(stats, outcome.sample, case, locale, workflow)
        for row in outcome.rejected:
            rejected.append(row)
            stats["reject_codes"][row["code"]] = stats["reject_codes"].get(row["code"], 0) + 1
        for name, ok in outcome.judges:
            bucket = stats["judges"].setdefault(name, {"ok": 0, "fail": 0, "skipped": 0})
            bucket["ok" if ok else "fail"] += 1
        for row in outcome.judge_skips:
            judge_skips.append(row)
            bucket = stats["judges"].setdefault(row["judge"], {"ok": 0, "fail": 0, "skipped": 0})
            bucket["skipped"] = bucket.get("skipped", 0) + 1
    if aborted is not None:
        stats["aborted"] = dict(aborted)

    # 一致性：素材相同而标签不同的样本是自相矛盾的训练对（照抄 Open-Jev 的校验）。
    conflict_rows = consistency.conflicts(accepted_rows)
    if conflict_rows:
        jsonio.write_jsonl(run_dir / "conflicts.jsonl", conflict_rows)
    stats["accepted"] = len(accepted_rows)
    stats["rejected"] = len(rejected)
    stats["judge_skipped"] = len(judge_skips)
    stats["conflicts"] = len(conflict_rows)
    stats["state_chars_mean"] = _mean(stats.pop("state_chars"))
    stats["questions_per_sample_mean"] = _mean(stats.pop("questions_per_sample"))
    balance = balance_report(stats, config.balance_tolerance)
    stats["balance"] = balance
    unbalanced = [item for item in balance if not item["ok"]]
    if config.balance_strict and unbalanced:
        detail = ", ".join(f"{item['key']}={item['max_dev']}" for item in unbalanced)
        raise RuntimeError(f"答案分布超出容差 {config.balance_tolerance}：{detail}")
    # 按切分命名，避免一个 run 里跑多个切分时互相覆盖
    (run_dir / f"stats-{config.split}.md").write_text(
        render_stats_md(stats, config, workflow), encoding="utf-8"
    )
    (run_dir / f"manifest-{config.split}.json").write_text(
        jsonio.dumps_pretty(manifest(config, stats, workflow, cases)), encoding="utf-8", newline="\n"
    )
    result = RunResult(run_dir=run_dir, accepted=len(accepted_rows), rejected=rejected, stats=stats)
    if aborted is not None:
        # 产物已经落盘（中间进度不丢），但整批没跑完——用异常把这件事告诉调用方，
        # 而不是返回一个「看上去正常、其实全是被拒」的结果。
        raise RunAborted(run_dir, result.accepted, aborted["seq"], aborted["reason"])
    return result


def _accumulate(
    stats: dict[str, Any],
    sample: dict[str, Any],
    case: Any,
    locale: str,
    workflow: Workflow,
) -> None:
    """把一条合格样本计入统计（题族、答案分布、case 维度、state 长度）。

    输入：统计字典、样本、case、语言、workflow（用它的 stats_dimensions 取维度取值）。
    输出：无（原地累加）。预期行为：维度名与取值都转成字符串，便于写进 JSON。
    """
    for qid, question in sample["questions"].items():
        family = family_of(qid)
        stats["families"][family] = stats["families"].get(family, 0) + 1
        stats["types"][question["type"]] = stats["types"].get(question["type"], 0) + 1
        answer = sample["answers"][qid]
        key = f"{question['type']}:{family}"
        bucket = stats["answers"].setdefault(key, {})
        value = str(answer.get("choice", answer.get("score", answer.get("noul"))))
        bucket[value] = bucket.get(value, 0) + 1
        # 该题族的选项/等级空间（取出现过的最大值），均衡检查要用它当参考
        space = len(question["criteria"]) if "criteria" in question else 2
        stats["answer_space"][key] = max(stats["answer_space"].get(key, 0), space)
    for name, value in workflow.stats_dimensions(case).items():
        bucket = stats["dimensions"].setdefault(name, {})
        bucket[str(value)] = bucket.get(str(value), 0) + 1
    stats["locales"][locale] = stats["locales"].get(locale, 0) + 1
    state = sample["state"]
    stats["state_chars"].append(len(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)))
    stats["questions_per_sample"].append(len(sample["questions"]))


def backend_manifest(config: RunConfig) -> dict[str, Any]:
    """把后端记账写进 manifest（溯源用）。

    输入：运行配置。输出：`{name, starting_mode}`；没有后端时返回 `{}`。
    预期行为：只记后端名字与能力探测结论，**绝不记 key**——key 只在 key 池里，
    对外只暴露 sha256 前 8 位（PLAN 3.1.3）。
    """
    backend = config.backend
    if backend is None:
        return {}
    caps = getattr(backend, "capabilities", None)
    # 优先用后端自己报的起始档位：`capabilities` 只是探测结论，探测没开时是空的，
    # 会把实际从 `json_schema` 起步的 run 记成 `md_json`（错账）。
    effective = getattr(backend, "starting_mode", "")
    return {
        "name": getattr(backend, "name", type(backend).__name__),
        "starting_mode": effective or (caps.starting_mode() if caps is not None else ""),
    }


def manifest(
    config: RunConfig, stats: dict[str, Any], workflow: Workflow, cases: list[Any]
) -> dict[str, Any]:
    """生成 manifest（溯源用）：配置、workflow 声明、case 矩阵规模、统计摘要。"""
    return {
        "run_id": config.resolved_run_id(),
        "backend": backend_manifest(config),
        "workflow": workflow.name,
        "workflow_version": workflow.version,
        "kinds": sorted(workflow.kinds),
        "needs": sorted(workflow.needs),
        "split": config.split,
        "run_seed": config.run_seed,
        "per_case": config.per_case,
        "concurrency": config.concurrency,
        "render_attempts": config.render_attempts,
        "locales": list(config.locales),
        "judges": list(config.judges if config.judges is not None else workflow.judges),
        "case_count": len(cases),
        "cases": [workflow.case_label(case) for case in cases],
        "stats": stats,
    }


def render_stats_md(stats: dict[str, Any], config: RunConfig, workflow: Workflow) -> str:
    """把统计渲染成 markdown 报告。

    输入：统计字典、配置、workflow（提供维度中文标题）。
    输出：markdown 文本。预期行为：先总览，再给均衡、题族、答案分布与各 case 维度。
    """
    lines: list[str] = [
        f"# 生成统计（{workflow.name} v{workflow.version} / {config.split}）",
        "",
        f"- 题型：{'、'.join(sorted(workflow.kinds))}；需要后端：{'、'.join(sorted(workflow.needs)) or '无（纯构造式）'}",
        f"- 请求样本数：{stats['requested']}；case 组合数：{stats['cases']}",
        f"- 合格：{stats['accepted']}；被拒：{stats['rejected']}",
        f"- 重渲染：{stats['render_retries']} 次（每条最多 {config.render_attempts} 次）",
        f"- 一致性：{_consistency_line(stats)}",
        f"- 平均 state 长度：{stats['state_chars_mean']} 字符",
        f"- 平均每题数：{stats['questions_per_sample_mean']}",
        "",
        f"## 答案均衡（容差 {config.balance_tolerance * 100:.1f} 个百分点）",
        "",
        "参考值：choice/score 按 1/选项数均匀，noul 按 True 占 50%；"
        "容差与「3 倍标准误」取较大者（小样本下固定 3pp 统计上做不到）。"
        "标 ⚠️ 的族要么属于结构性不可达（某档位在当前规模下取不到），"
        "要么属于题族间耦合（两个题族共享同一个自由度），报告照实标出，不掩盖。",
        "",
    ]
    for item in stats.get("balance", []):
        mark = "✅" if item["ok"] else "⚠️"
        lines.append(
            f"- {mark} `{item['key']}`：n={item['total']}，空间={item['space']}，"
            f"最大偏差={item['max_dev']:.3f}，噪声地板={item['noise']:.3f}"
        )
    lines += ["", "## 题族分布", ""]
    for name, count in sorted(stats["families"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{name}`：{count}")
    lines += ["", "## 答案分布", ""]
    for key, bucket in sorted(stats["answers"].items()):
        items = "、".join(f"{k}×{v}" for k, v in sorted(bucket.items()))
        lines.append(f"- `{key}`：{items}")
    lines += ["", "## 题型分布", ""]
    for name, count in sorted(stats["types"].items()):
        lines.append(f"- {name}：{count}")
    titles = workflow.dimension_titles()
    for name, bucket in stats["dimensions"].items():
        lines += ["", f"## {titles.get(name, name)}", ""]
        for value, count in sorted(bucket.items()):
            lines.append(f"- {value}：{count}")
    if stats["locales"]:
        lines += ["", "## 语言", ""]
        for name, count in sorted(stats["locales"].items()):
            lines.append(f"- {name}：{count}")
    if stats["reject_codes"]:
        lines += ["", "## 拒绝码", ""]
        for name, count in sorted(stats["reject_codes"].items()):
            lines.append(f"- `{name}`：{count}")
    if stats.get("judges"):
        lines += ["", "## 判别器通过率", ""]
        for name, bucket in sorted(stats["judges"].items()):
            total = bucket["ok"] + bucket["fail"]
            rate = bucket["ok"] / total if total else 0.0
            line = f"- `{name}`：{bucket['ok']}/{total} 通过（{rate:.1%}）"
            if bucket.get("skipped"):
                # 没判成的不进通过率分母，但必须写出来——否则「一条都没判」看起来像「全过」。
                line += f"，跳过 {bucket['skipped']} 条（没判成，不是数据问题）"
            lines.append(line)
    lines.append("")
    return "\n".join(lines)
