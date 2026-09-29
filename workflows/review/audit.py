"""改动审查的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有四条：

- **清单行**必须解析得出来（文件类型 / 变更类型 / 规模）；
- **改动正文的形状**必须与清单自洽：行数落在 `features.size_line_bounds(规模)` 里、行前缀与
  变更类型对得上（前缀按 `features.diff_mark` 读，剥掉行首空白再认标记）；
- `severity` 必须与产物里记下的 `rule` 和清单里的文件类型自洽（查 `features.SEVERITY_TABLE`）；
- `gate` 必须与产物里记下的 `severity` 和清单里的变更类型自洽（查 `features.GATE_TABLE`）。

`rule` 本身依赖正文承载的两维（有没有凭据字面量 / 有没有调试输出），纯代码读不出来——
那正是它必须由反解判别器在合成时把关的原因。审计写不出强校验时**不硬编一个看起来像校验的
东西**（见范式 §9.13）：老实写明「这一族靠合成期的反解把关」。

输入：run 目录、切分名、语言。输出：各切分计数 + 缺失项列表。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from jc import jsonio
from jc.jsontypes import JsonValue

from . import features, judges as judges_mod, locales, questions, render


#: 本 workflow 认得的题族（产物里出现别的就是串版本了）。
KNOWN_FAMILIES: frozenset[str] = frozenset(questions.FAMILIES)


def _family_of(qid: str) -> str:
    """输入：题号。输出：题族名（去掉 `_2` 这类去重后缀）。"""
    return "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")


def _recorded(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> JsonValue:
    """输入：题面、答案。输出：把编号换回可读形态后的答案值。

    预期行为：choice 的编号是**产物里那个打乱顺序**里的下标（见 `jc.assemble`），
    所以先取回选项名再比；score / noul 原样返回。
    """
    kind = answer.get("type")
    if kind == "choice":
        names = judges_mod.option_names(question)
        index = answer.get("choice")
        if isinstance(index, int) and 0 <= index < len(names):
            return names[index]
        return None
    if kind == "score":
        return answer.get("score")
    return answer.get("noul")


def severity_from(rule: JsonValue, file_kind: str) -> int | None:
    """输入：产物里记下的规则名、清单里的文件类型。输出：该自洽的档位（推不出就 None）。

    预期行为：兜底规则一律 0 档；其余查 `features.SEVERITY_TABLE`——规则名不认识、
    或者文件类型不在该规则的适用范围里，都返回 None（说明产物串版本了，交给上层记一笔）。
    """
    if not isinstance(rule, str) or rule not in features.REVIEW_RULES:
        return None
    if rule == "none":
        return features.NOT_RAISED
    if file_kind not in features.RULE_SCOPES[rule]:
        return None
    return features.SEVERITY_TABLE[(rule, file_kind)]


def gate_from(severity: JsonValue, change_kind: str) -> str | None:
    """输入：产物里记下的严重度、清单里的变更类型。输出：该自洽的动作（推不出就 None）。"""
    if isinstance(severity, bool) or not isinstance(severity, int):
        return None
    if not 0 <= severity < features.SEVERITY_LEVELS:
        return None
    return features.GATE_TABLE[(severity, change_kind)]


def body_shape_problem(lines: tuple[str, ...], manifest: render.ManifestState) -> str:
    """输入：改动正文的行、解析出来的清单。输出：形状问题的说明（没问题就空串）。

    预期行为：查行数是否落在规模区间里、以及前缀与变更类型对不对得上。前缀的读法走
    `features.diff_mark`（剥掉行首空白再认标记），所以「整段缩进两格」与「上下文行没写
    空格前缀」都不算问题。**「修改」只加行不算问题**：真实 diff 里很常见，真机第一批
    为此误拒过三条。
    只回第一条问题，避免同一处改动刷出一串计数。
    """
    low, high = features.size_line_bounds(manifest.size)
    if not low <= len(lines) <= high:
        return f"行数 {len(lines)} 不在 {manifest.size!r} 要求的 {low}~{high} 里"
    marks = {features.diff_mark(line) for line in lines}
    if manifest.change_kind == "added" and "-" in marks:
        return "「新增」却出现了删除行"
    if manifest.change_kind == "deleted" and "+" in marks:
        return "「删除」却出现了新增行"
    if manifest.change_kind == "added" and "+" not in marks:
        return "「新增」却一行 + 都没有"
    if manifest.change_kind == "deleted" and "-" not in marks:
        return "「删除」却一行 - 都没有"
    if marks == {" "}:
        return "一行都没变（既没有 + 也没有 -）"
    return ""


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。"""
    loc = locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段
    canonical_rules = render.rule_row_variants()
    canonical_severity = render.severity_rows(loc)
    canonical_gate = render.gate_rows(loc)
    report: dict[str, JsonValue] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        counts: dict[str, JsonValue] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "manifest_bad": 0,
            "body_shape_bad": 0,
            "rules_bad": 0,
            "severity_table_bad": 0,
            "gate_table_bad": 0,
            "severity_bad": 0,
            "gate_bad": 0,
            "unknown_families": Counter(),
        }
        if not path.exists():
            missing.append(f"{split}: 产物不存在（{path.name}）")
            report[split] = counts
            continue
        for _, row in jsonio.iter_jsonl(path):
            counts["samples"] += 1
            answers = row["answers"]
            recorded: dict[str, JsonValue] = {}
            for qid, question in row["questions"].items():
                family = _family_of(qid)
                counts["families"][family] += 1
                counts["types"][question["type"]] += 1
                if family not in KNOWN_FAMILIES:
                    counts["unknown_families"][family] += 1
                if family in {"secret", "debug", "rule", "severity", "gate"}:
                    recorded[family] = _recorded(question, answers[qid])
            parsed = render.parse_state(row["state"])
            if parsed.manifest is None:
                counts["manifest_bad"] += 1
                continue
            problem = body_shape_problem(parsed.body_lines, parsed.manifest)
            if problem:
                counts["body_shape_bad"] += 1
            if parsed.rules not in canonical_rules:
                counts["rules_bad"] += 1
            if parsed.severity != canonical_severity:
                counts["severity_table_bad"] += 1
            if parsed.gate != canonical_gate:
                counts["gate_table_bad"] += 1
            if recorded.get("severity") != severity_from(
                recorded.get("rule"), parsed.manifest.file_kind
            ):
                counts["severity_bad"] += 1
            if recorded.get("gate") != gate_from(
                recorded.get("severity"), parsed.manifest.change_kind
            ):
                counts["gate_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["manifest_bad"]:
            missing.append(f"{split}: {counts['manifest_bad']} 条的清单行解析不出来")
        if counts["body_shape_bad"]:
            missing.append(f"{split}: {counts['body_shape_bad']} 条的改动正文形状与清单对不上")
        if counts["rules_bad"]:
            missing.append(f"{split}: {counts['rules_bad']} 条的审查规则表与规范表对不上")
        if counts["severity_table_bad"]:
            missing.append(f"{split}: {counts['severity_table_bad']} 条的严重度表与规范表对不上")
        if counts["gate_table_bad"]:
            missing.append(f"{split}: {counts['gate_table_bad']} 条的门控表与规范表对不上")
        if counts["severity_bad"]:
            missing.append(f"{split}: {counts['severity_bad']} 条的严重度与「命中规则 × 文件类型」不自洽")
        if counts["gate_bad"]:
            missing.append(f"{split}: {counts['gate_bad']} 条的门控与「严重度 × 变更类型」不自洽")
        report[split] = counts
    return report, missing
