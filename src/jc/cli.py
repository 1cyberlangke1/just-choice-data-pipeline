"""命令行入口（通用层）：python -m jc <命令>。

命令：
- workflows  列出已注册的 workflow
- locales    列出某个 workflow 支持的语言
- cases      打印某个 workflow 的 case 矩阵规模（不开工生成，先看覆盖）
- gen        按 case 矩阵生成数据（写 dataset/<run_id>/）
- audit      审计已生成的数据集（委托 workflow 实现）
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path


from . import jsonio, language, pipeline, validate
from .jsontypes import JsonValue
from .workflow import Workflow, discover, load


def _load_backend(path: str) -> object:
    """按需装配生成后端。

    输入：后端配置 TOML 的路径（空串表示这个 workflow 不需要后端）。
    输出：`LLMBackend` 实例或 None。
    预期行为：只有显式给了配置才 import `jc.backend`——核心层不装 `[llm]` extra
    也能跑构造式 workflow；key 只从环境变量读，命令行**不收明文 key**。
    """
    if not path:
        return None
    from .backend import build_backend, load_config

    return build_backend(load_config(path), probe=True)


def _declared_locales(run_dir: Path, split: str) -> tuple[str, ...]:
    """输入：run 目录、切分名。输出：这一份产物 manifest 里记的语言（没记 / 读不了给空元组）。

    预期行为：manifest 不存在时给空元组——老产物没有它，不假装知道；解析失败也给空元组，
    因为这是审计的**旁路**判据，不该因为一个坏 manifest 把整次审计炸掉（坏 manifest 本身
    会在别处报出来）。
    """
    manifest = run_dir / f"manifest-{split}.json"
    if not manifest.is_file():
        return ()
    try:
        meta = jsonio.loads(manifest.read_text(encoding="utf-8"))
    except (jsonio.StrictJsonError, OSError):
        return ()
    if not isinstance(meta, dict):
        return ()
    recorded = meta.get("locales")
    if not isinstance(recorded, list):
        return ()
    return tuple(str(item) for item in recorded)


def _locale_for_split(run_dir: Path, split: str, explicit: str | None) -> str:
    """输入：run 目录、切分名、显式指定的语言（可能为 None）。输出：读这个切分该用的语言。

    预期行为：显式给了就用它；没给就读 `manifest-<split>.json` 里**这个 run 自己记的语言**
    ——产物是哪种语言，审计就得用哪种语言读。这不是洁癖：实测 en 产物按默认的 zh 口径读，
    `shell` 的审计在 9 条里报出 4 条**假的**「答案与原文矛盾」（判别器读得对，审计读错了）。
    manifest 记了多种语言时说明这一份产物混了语言，单一语言口径的审计读不了它——如实提醒，
    并退回第一种语言，不假装结果可信。没有 manifest 的老产物维持老口径（zh）。
    """
    if explicit:
        return explicit
    recorded = list(_declared_locales(run_dir, split))
    if len(recorded) > 1:
        print(
            f"提示：{run_dir.name}/{split} 的 manifest 记了多种语言 {recorded}——这一份产物混了语言，"
            "逐条审计只能按单一语言口径读，结果可能包含假失败；要准确请按语言分开生成，"
            "或用 --locale 指定一种（那时只保证那一种语言的条目可信）。"
        )
    return recorded[0] if recorded else "zh"


def _language_problem(
    workflow: Workflow, path: Path, declared: tuple[str, ...], limit: int = 3
) -> str | None:
    """输入：workflow、产物路径、这一份产物声明的语言。输出：一行问题（带样例）；全过给 None。

    预期行为：逐行拿 workflow 自己声明的**自然语言正文**（`Workflow.language_texts`）去比
    声明的语言，报前 `limit` 条样例（id + 行号 + 理由）。**零 token、不碰网络、不改产物**：
    它查的是已经落盘的东西。反解不了的样本（坏 state）跳过——那由契约校验与领域审计负责报，
    在这里再报一遍只是噪声。

    workflow 没实现 `language_texts` 时**如实报出来**，不静默跳过：跳过等于「这条产物查过了，
    没问题」，而实际上它一次都没查。真 workflow 由 `tests/workflows/test_protocol.py` 兜住。
    """
    if not hasattr(workflow, "language_texts"):
        return f"{path.name}: workflow {workflow.name} 没有实现 `Workflow.language_texts`，语言检查跳过了"
    samples: list[str] = []
    total = 0
    try:
        for lineno, row in jsonio.iter_jsonl(path):
            if not isinstance(row, dict):
                continue
            try:
                texts = workflow.language_texts(row)
            except Exception:  # noqa: BLE001 —— 反解不了的样本不在这里报（见 docstring）
                continue
            if not texts:
                continue
            problem = language.sample_problem(texts, declared)
            if problem is None:
                continue
            total += 1
            if len(samples) < limit:
                samples.append(f"{row.get('id', '（无 id）')}（第 {lineno} 行）{problem}")
    except jsonio.StrictJsonError:
        return None  # 解析不了的行已经由契约校验报过
    if not total:
        return None
    return (
        f"{path.name}: {total} 条产物的文本与声明语言对不上（例如 "
        + "；".join(samples)
        + "）"
    )


def _add_workflow_arg(parser: argparse.ArgumentParser) -> None:
    """给子命令加统一的 --workflow 参数。"""
    parser.add_argument("--workflow", required=True, help="workflow 名（见 `jc workflows`）")


def _parse_judges(values: list[str] | None) -> tuple[str, ...] | None:
    """输入：`--judges` 的原始值（可重复、逗号分隔）。输出：判别器名单，或 None。

    预期行为：去空白、丢空项；一个都没给就返回 None（= 用 workflow 的默认判别器），
    而不是空元组——空元组的意思是「一个判别器都别跑」。
    """
    if not values:
        return None
    names = [item.strip() for value in values for item in value.split(",")]
    return tuple(name for name in names if name) or None


def _format_counter(value: object) -> str:
    """输入：审计计数里的一个值。输出：一行可读文本。

    预期行为：映射（Counter / dict）按「取值×条数」展开，其余原样打印。
    **通用层不认识任何具体 workflow**，所以这里只按产物自己报的键打印；
    写死某个 workflow 的键名会让别的 workflow 一跑审计就 KeyError（实测踩过）。
    """
    if isinstance(value, Mapping):
        return "、".join(f"{key}×{count}" for key, count in sorted(value.items())) or "（空）"
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(prog="jc", description="JEV 风格选择题数据合成管线")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="生成数据")
    _add_workflow_arg(gen)
    gen.add_argument("--split", default="train")
    gen.add_argument("--locale", action="append", dest="locales", default=None, help="可重复；默认 zh")
    gen.add_argument("--per-case", type=int, default=1, help="每个 case 生成多少条样本")
    gen.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="同时推进多少条样本；每条内部还受后端并发闸限制（需要后端的 workflow 才看得出差别）",
    )
    gen.add_argument("--seed", type=int, default=20260924)
    gen.add_argument("--out", default="dataset")
    gen.add_argument("--run-id", default="")
    gen.add_argument(
        "--render-attempts",
        type=int,
        default=2,
        help="一条样本最多渲染几次：判别器拒掉就重渲染（PLAN 6.1）；1 = 不重试",
    )
    gen.add_argument("--balance-tolerance", type=float, default=0.03, help="答案占比偏离均匀的上限")
    gen.add_argument("--balance-strict", action="store_true", help="超容差直接让整批失败")
    gen.add_argument(
        "--backend-config", default="", help="后端配置 TOML；workflow 声明 needs 时必填"
    )
    gen.add_argument(
        "--judges",
        action="append",
        default=None,
        help="判别器名单，逗号分隔；LLM 质检用 llm:<视角>（llm:solvable / llm:answer / "
        "llm:distractors / llm:levels / llm:wording）；不给就用 workflow 的默认判别器",
    )

    cases = sub.add_parser("cases", help="打印 case 矩阵")
    _add_workflow_arg(cases)
    cases.add_argument("--split", default="train")
    cases.add_argument("--per-case", type=int, default=1)

    aud = sub.add_parser("audit", help="审计已生成的数据集")
    _add_workflow_arg(aud)
    aud.add_argument("--dir", required=True, help="run 目录，如 dataset/<run_id>")
    aud.add_argument("--split", action="append", dest="splits", default=None)
    aud.add_argument(
        "--locale", default=None, help="读产物用哪种语言；不给就读 manifest 里记的（默认 zh）"
    )

    sub.add_parser("workflows", help="列出已注册的 workflow")

    loc = sub.add_parser("locales", help="列出某个 workflow 支持的语言")
    _add_workflow_arg(loc)
    return parser


def main(argv: list[str] | None = None) -> int:
    """入口函数。输入：argv（None 取 sys.argv）。输出：进程退出码。"""
    args = _build_parser().parse_args(argv)

    if args.command == "workflows":
        for name in discover():
            print(name)
        return 0

    if args.command == "locales":
        for name in load(args.workflow).available_locales():
            print(name)
        return 0

    if args.command == "cases":
        workflow = load(args.workflow)
        config = workflow.config_for(args.split)
        cases = workflow.build_cases(config)
        print(
            f"workflow {workflow.name} v{workflow.version}，切分 {args.split}："
            f"{len(cases)} 个 case × {args.per_case} 条 = {len(cases) * args.per_case} 条样本"
        )
        for case in cases[:10]:
            print("  " + workflow.case_label(case))
        if len(cases) > 10:
            print(f"  …（其余 {len(cases) - 10} 个）")
        return 0

    if args.command == "audit":
        workflow = load(args.workflow)
        splits = tuple(args.splits) if args.splits else workflow.split_names()
        run_dir = Path(args.dir)
        # 逐切分定语言：一个 run 目录里的不同切分可以是用不同语言生成的，按语言分组调用。
        by_locale: dict[str, list[str]] = {}
        for split in splits:
            by_locale.setdefault(_locale_for_split(run_dir, split, args.locale), []).append(split)
        report: dict[str, JsonValue] = {}
        missing: list[str] = []
        for locale, group in by_locale.items():
            part, part_missing = workflow.audit(run_dir, tuple(group), locale)
            report.update(part)
            missing.extend(part_missing)
        for split, counts in report.items():
            print(f"=== {split}：{counts.get('samples', 0)} 条 ===")
            for key, value in counts.items():
                if key != "samples":
                    print(f"  {key}：{_format_counter(value)}")
        # 通用层再拿**契约校验器**把每个产物重新过一遍：`validate.validate_file` 带「已见 id 集合」，
        # 所以能查出逐样本校验器查不到的**跨行**不变量（`E_ID_DUP`）。四个 `sweep-*` 产物各多出
        # 「恰好一个切分 case 数」的重复行，就是这么漏过去的——workflow 自己的审计也只看覆盖。
        problems = list(missing)
        for split in splits:
            path = Path(args.dir) / f"{split}.jsonl"
            if not path.is_file():
                continue  # 「产物不存在」已由 workflow 的审计报过，别报两遍
            checked = validate.validate_file(path)
            issues = checked.all_issues()
            if not issues:
                continue
            counts_text = "、".join(f"{code}×{n}" for code, n in checked.code_counts().items())
            examples = "、".join(
                f"{issue.sample_id or '（无 id）'}（第 {issue.line} 行）" for issue in issues[:3]
            )
            problems.append(f"{split}: 契约校验不通过（{counts_text}；例如 {examples}）")
        # 产物文本必须与它**声明的**语言一致。语言取 `manifest-<split>.json`，**不是**调用方给的
        # `--locale`——「与声明一致」问的是产物自己说了什么。哪些字段算「自然语言正文」由 workflow
        # 自己声明（`Workflow.language_texts`），通用层只提供判据（`jc.language`）。零 token，
        # 也不进渲染路径：它查的是已经落盘的东西（实测 `rerank` 的 en 侧整段写成中文、`citation`
        # 的 en 侧有一条写成中文文档，都是这么溜进去的）。
        for split in splits:
            path = Path(args.dir) / f"{split}.jsonl"
            if not path.is_file():
                continue
            declared = _declared_locales(run_dir, split)
            if not declared:
                continue  # 没声明语言就无从谈「与声明一致」（老产物 / 没有 manifest 的产物）
            problem = _language_problem(workflow, path, declared)
            if problem:
                problems.append(f"{split}: {problem}")
        if problems:
            print("审计不通过：")
            for item in problems:
                print(f"  - {item}")
            return 1
        print("审计通过：要求的 case 都在产物里，契约校验也过。")
        return 0

    config = pipeline.RunConfig(
        workflow=args.workflow,
        split=args.split,
        locales=tuple(args.locales) if args.locales else ("zh",),
        run_seed=args.seed,
        per_case=args.per_case,
        concurrency=args.concurrency,
        out_root=Path(args.out),
        run_id=args.run_id,
        render_attempts=args.render_attempts,
        balance_tolerance=args.balance_tolerance,
        balance_strict=args.balance_strict,
        judges=_parse_judges(args.judges),
        backend=_load_backend(args.backend_config),
    )
    result = pipeline.run(config)
    print(f"合格 {result.accepted} 条；被拒 {len(result.rejected)} 条")
    if result.stats.get("judge_skipped"):
        print(
            f"质检跳过 {result.stats['judge_skipped']} 条（端点故障或视角不适用，不是数据问题）："
            "见 judge-skipped.jsonl"
        )
    if result.stats.get("conflicts"):
        print(
            f"⚠️ {result.stats['conflicts']} 处「同素材同题不同答」（自相矛盾的训练对）："
            "见 conflicts.jsonl"
        )
    print(f"产物目录：{result.run_dir}")
    if result.stats.get("reject_codes"):
        print(f"拒绝码：{result.stats['reject_codes']}")
    unbalanced = [item for item in result.stats.get("balance", []) if not item["ok"]]
    if unbalanced:
        print(f"答案分布超容差的题族（{len(unbalanced)} 个）：")
        for item in unbalanced:
            print(
                f"  {item['key']}：n={item['total']} 最大偏差={item['max_dev']:.3f} "
                f"噪声地板={item['noise']:.3f}"
            )
    unjudged = [item for item in result.stats.get("balance", []) if not item.get("judged", True)]
    if unjudged:
        names = "、".join(item["key"] for item in unjudged)
        print(f"样本量不足、没判的题族（{len(unjudged)} 个）：{names}")
    return 0
