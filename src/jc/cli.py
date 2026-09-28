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


from . import pipeline, validate
from .workflow import discover, load


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
    aud.add_argument("--locale", default="zh")

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
        report, missing = workflow.audit(Path(args.dir), splits, args.locale)
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
    return 0
