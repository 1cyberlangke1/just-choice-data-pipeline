"""管线编排（通用层）：case 矩阵 → 生成 → 校验 → 判别 → 打包 + 统计 + 均衡闸。

通用层不认识任何具体 workflow：所有领域相关的动作都通过 `Workflow` 协议委托出去
（采样参数、case 矩阵、统计维度、判别器、产物审计）。

设计要点：
- 每条样本的 RNG 由 `sha256(run_seed, workflow, 切分, 序号)` 派生，任何一条都能单独重算，
  并行与续跑都不会改变 id 或内容；
- 序号对 case 矩阵长度取模，保证每个 case 都被轮到；
- 产物：`dataset/<run_id>/<split>.jsonl`（合格）、`rejected.jsonl`（被拒）、
  `stats-<split>.md`（统计 + 均衡）、`manifest-<split>.json`（溯源）。

输入：RunConfig。
输出：RunResult（计数与路径）。
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import assemble, jsonio, judge as judge_mod, validate
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
    out_root: Path = Path("dataset")
    run_id: str = ""
    balance_tolerance: float = 0.03  # 每个取值的占比偏离「均匀」的上限（3 个百分点）
    balance_strict: bool = False  # True 时超容差直接让整批失败
    judges: tuple[str, ...] | None = None  # None = 用 workflow 声明的默认判别器
    config_override: Any = None  # 覆盖 workflow 的默认采样参数（小规模试跑 / 测试用）

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


def sample_rng(run_seed: int, workflow: str, split: str, seq: int) -> random.Random:
    """派生某一条样本的 RNG。

    输入：运行种子、workflow 名、切分、序号。
    输出：random.Random。
    预期行为：不用内置 hash()（它对字符串每进程随机化）；同一入参必得同一 RNG。
    """
    digest = hashlib.sha256(f"{run_seed}|{workflow}|{split}|{seq}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def build_judges(workflow: Workflow, names: tuple[str, ...], locale: str) -> tuple[judge_mod.Judge, ...]:
    """委托 workflow 构造判别器。

    输入：workflow、判别器名单、语言。
    输出：判别器元组。预期行为：未知名字由 workflow 报错，避免「写了名字却没跑」。
    """
    return workflow.make_judges(names, locale)


def _mean(values: list[int]) -> float:
    """算术平均；空列表返回 0.0。"""
    return round(sum(values) / len(values), 1) if values else 0.0


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


def run(config: RunConfig) -> RunResult:
    """执行一次生成。

    输入：RunConfig。
    输出：RunResult（含目录、合格数、拒绝明细与统计）。
    预期行为：逐条走「计划 → 采样 → 渲染 → 出题 → 组装 → 契约校验 → 判别」；
    任何一步失败都写进 rejected.jsonl（带 id、阶段、原因），不静默丢弃。
    模型驱动的 workflow（`needs` 非空）尚未接入后端，这里直接报错，避免静默降级。
    """
    workflow = load(config.workflow)
    if workflow.needs:
        raise NotImplementedError(
            f"workflow {workflow.name!r} 声明需要后端 {sorted(workflow.needs)}，"
            "但通用层还没有接入生成后端（见 PLAN 6.8）"
        )
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
    accepted_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
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
    }
    judge_pool = {locale: build_judges(workflow, judge_names, locale) for locale in config.locales}

    for seq in range(total):
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
            rejected.append({"id": sample_id, "stage": "generate", "code": E_ASSEMBLE, "reason": str(exc)})
            stats["reject_codes"][E_ASSEMBLE] = stats["reject_codes"].get(E_ASSEMBLE, 0) + 1
            continue

        issues = validate.validate_sample(sample)
        if issues:
            for issue in issues:
                rejected.append(
                    {
                        "id": sample_id,
                        "stage": "validate",
                        "code": issue.code,
                        "path": issue.path,
                        "reason": issue.msg,
                    }
                )
                stats["reject_codes"][issue.code] = stats["reject_codes"].get(issue.code, 0) + 1
            continue

        verdicts = judge_mod.run_judges(sample, spec, judge_pool[locale])
        failed = False
        for name, verdict in verdicts:
            bucket = stats["judges"].setdefault(name, {"ok": 0, "fail": 0})
            bucket["ok" if verdict.ok else "fail"] += 1
            if not verdict.ok:
                failed = True
                code = f"E_JUDGE_{name.upper()}"
                rejected.append(
                    {"id": sample_id, "stage": "judge", "code": code, "reason": verdict.reason}
                )
                stats["reject_codes"][code] = stats["reject_codes"].get(code, 0) + 1
        if failed:
            continue

        accepted_rows.append(sample)
        _accumulate(stats, sample, case, locale, workflow)

    jsonio.write_jsonl(run_dir / f"{config.split}.jsonl", accepted_rows)
    if rejected:
        jsonio.write_jsonl(run_dir / "rejected.jsonl", rejected)
    stats["accepted"] = len(accepted_rows)
    stats["rejected"] = len(rejected)
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
    return RunResult(run_dir=run_dir, accepted=len(accepted_rows), rejected=rejected, stats=stats)


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


def manifest(
    config: RunConfig, stats: dict[str, Any], workflow: Workflow, cases: list[Any]
) -> dict[str, Any]:
    """生成 manifest（溯源用）：配置、workflow 声明、case 矩阵规模、统计摘要。"""
    return {
        "run_id": config.resolved_run_id(),
        "workflow": workflow.name,
        "workflow_version": workflow.version,
        "kinds": sorted(workflow.kinds),
        "needs": sorted(workflow.needs),
        "split": config.split,
        "run_seed": config.run_seed,
        "per_case": config.per_case,
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
            lines.append(f"- `{name}`：{bucket['ok']}/{total} 通过（{rate:.1%}）")
    lines.append("")
    return "\n".join(lines)
