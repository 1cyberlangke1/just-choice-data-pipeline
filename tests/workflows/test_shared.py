"""全库通用的协议级不变量：一份测试跑遍所有已注册的 workflow。

输入：无（workflow 名单由 `jc.workflow.discover()` 现场列出来）。
输出：无。
预期行为：把「每个 workflow 都该满足」的通用断言只写一遍——各 workflow 自己的测试包只留
领域特有的断言，通用断言在这里统一钉住，不再 48 个包里各抄一份。

为什么下面这些是循环、不是参数化：它们查的是**同一段通用代码**（基类的 `bind_backend` /
`config_for` / `make_judges` / `audit`，`jc.locale_files` 的 `load` / `pick`），换的只是各家的
数据目录。51 个 workflow 摊成 51 条用例只是把同一件事报 51 遍；收成循环之后失败信息照样
点名是哪个 workflow（问题攒在 `bad` 里一起报，不因为第一家炸掉就看不见其余），用例数从
425 降到 25。带**取值表**的那些（`STATE_MARKS` / `DECLARED`）保持参数化——它们是数据，
不是同一段代码。
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

from jc import workflow as workflow_mod

#: 已注册的 workflow 名单（导入即注册，现场列，不写死）。
WORKFLOWS: list[str] = workflow_mod.discover()


def locales_of(name: str) -> ModuleType:
    """输入：workflow 名。输出：它的 `locales` 模块。

    预期行为：语言包是每个 workflow 都该有的通用件，缺了就直接报错，不静默跳过。
    """
    return importlib.import_module(f"workflows.{name}.locales")


def test_every_workflow_ships_zh_and_en() -> None:
    """输入：无（名单现场列）。输出：无。

    预期行为：每个 workflow 的语言列表恰好是 en / zh，顺序稳定；而且它自己声明的那份与磁盘上
    的那份**是同一个**——两处漂移（声明多一种但没文件、或反过来）在这里当场红灯，不用各家
    测试包各抄一条。
    """
    bad: list[str] = []
    for name in WORKFLOWS:
        available = locales_of(name).available()
        if available != ["en", "zh"]:
            bad.append(f"{name} 的语言列表是 {available}，不是 ['en', 'zh']")
        declared = workflow_mod.load(name).available_locales()
        if declared != available:
            bad.append(f"{name} 声明 {declared} 与磁盘 {available} 不一致")
    assert not bad, bad


def test_pick_wraps_around() -> None:
    """输入：越界的问法编号。输出：无。预期行为：回绕取模，不抛 IndexError。"""
    bad: list[str] = []
    without: list[str] = []
    for name in WORKFLOWS:
        pick = getattr(locales_of(name), "pick", None)
        if pick is None:
            without.append(name)
            continue
        got = [pick(["a", "b"], index) for index in (0, 1, 3, 4)]
        if got != ["a", "b", "b", "a"]:
            bad.append(f"{name} 的 pick 回绕不对：{got}")
    assert not bad, bad
    assert without, f"一家都没有 pick，这条测试成了空转（只有 {len(WORKFLOWS)} 家都没 pick）"


def test_unknown_locale_is_rejected() -> None:
    """输入：不存在的语言。输出：无。

    预期行为：报 FileNotFoundError，而且报错里**真的列出**了可用语言——只写「没有」的报错
    等于让调用方去猜。这条把「列出来」也钉住，各家测试包就不必各抄一份。
    """
    bad: list[str] = []
    for name in WORKFLOWS:
        module = locales_of(name)
        try:
            module.load("xx")
        except FileNotFoundError as error:
            message = str(error)
            if "没有语言文件" not in message:
                bad.append(f"{name} 的报错不是「没有语言文件」：{message}")
            if "可用" not in message:
                bad.append(f"{name} 的报错没写「可用」：{message}")
            for available in module.available():
                if available not in message:
                    bad.append(f"{name} 的报错没列出可用语言 {available}")
        else:
            bad.append(f"{name} 加载不存在的语言居然没报错")
    assert not bad, bad


def test_build_cases_is_deterministic() -> None:
    """输入：无（名单现场列）。输出：无。

    预期行为：同一个切分的 case 列表连算两遍**逐条一致**（顺序也一致）——case 由指纹派生，
    不许掷骰子，否则「同一份配置跑出同一批题」这件事无从谈起。各家测试包不再各抄一份。
    """
    bad: list[str] = []
    for name in WORKFLOWS:
        wf = workflow_mod.load(name)
        for split in wf.split_names():
            config = wf.config_for(split)
            first = [repr(wf.stats_dimensions(case)) for case in wf.build_cases(config)]
            second = [repr(wf.stats_dimensions(case)) for case in wf.build_cases(config)]
            if first != second:
                bad.append(f"{name}/{split} 两次算出的 case 列表不一致")
    assert not bad, bad


#: 有 state 抬头的 workflow -> 它语言包里那几处抬头字段的键（顺序照各家原样）。
#: 这 12 条原本散在 12 个包的 `test_locales.py` 里各抄一份（只有键名不同），
#: 现在只留这一张表；`citation` / `testimpact` 那两家的抬头是**拼出来的**
#: （模板套小节号 / 两两组合），形状不同，留在各自包里。
STATE_MARKS: Final[dict[str, tuple[str, ...]]] = {
    "adblock": ("tag_mark", "rule_mark", "content_mark"),
    "alertrisk": (
        "fact_mark",
        "matrix_mark",
        "rule_mark",
        "content_mark",
    ),
    "causechain": ("tag_mark", "rule_mark", "content_mark"),
    "depcheck": (
        "fact_mark",
        "rule_mark",
        "act_mark",
        "content_mark",
    ),
    "guardrail": ("clause_mark", "rule_mark", "content_mark"),
    "logtriage": ("tag_mark", "rule_mark", "content_mark"),
    "memorygate": ("tag_mark", "rule_mark", "content_mark"),
    "posttriage": ("signal_mark", "route_mark", "content_mark"),
    "promptinject": ("tag_mark", "rule_mark", "content_mark"),
    "rerank": (
        "query_mark",
        "terms_mark",
        "cands_mark",
        "rule_mark",
    ),
    "tracescore": ("tag_mark", "rule_mark", "content_mark"),
    "turntake": ("tag_mark", "rule_mark", "content_mark"),
}


def test_state_marks_are_non_empty_and_pairwise_non_prefix() -> None:
    """输入：一个 workflow 的两种语言。输出：无。

    预期行为：每处抬头都非空、两两不等、两两不互为前缀——`【判定判据】` 与
    `【判定判据附注】` 那种关系会让解析先命中短的那个，整段 state 当场错位。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for name in sorted(STATE_MARKS):
        try:
            for lang in ("en", "zh"):
                state = locales_of(name).load(lang)["state"]
                marks = [str(state[key]) for key in STATE_MARKS[name]]
                assert all(marks), f"{name}/{lang} 有空抬头"
                assert len(set(marks)) == len(marks), f"{name}/{lang} 有重复抬头"
                for left in marks:
                    for right in marks:
                        if left != right:
                            assert not right.startswith(left), (
                                f"{name}/{lang}：{left!r} 是 {right!r} 的前缀"
                            )
        except AssertionError as exc:
            bad.append(f"{(name,)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_audit_on_an_empty_dir_reports_missing_artifacts(tmp_path: Path) -> None:
    """输入：空目录、两个切分。输出：无。

    预期行为：两个切分的样本数都是 0，且各报一条缺失——审计只读产物，缺文件不许静默当
    「没有样本」。
    """
    bad: list[str] = []
    without: list[str] = []
    for name in WORKFLOWS:
        report, missing = workflow_mod.load(name).audit(tmp_path, ("train", "val"), "zh")
        if "train" not in report:
            without.append(name)
            continue
        if report["train"]["samples"] != 0 or report["val"]["samples"] != 0:
            bad.append(
                f"{name} 空目录下的样本数不是 0："
                f"train={report['train']['samples']} val={report['val']['samples']}"
            )
        if len(missing) < 2:
            bad.append(f"{name} 空目录下只报了 {len(missing)} 条缺失")
    assert not bad, bad
    assert without, "按 train/val 分键的一家都没有，这条测试成了空转（maze 自带格式）"


def test_pure_construction_workflows_refuse_a_backend() -> None:
    """输入：一个假后端。输出：无。

    预期行为：`needs` 为空的 workflow 根本不该被绑后端——收到就报错，不静默吞掉；
    需要后端的 workflow 由各自测试包覆盖，这里跳过。
    """
    bad: list[str] = []
    checked = 0
    for name in WORKFLOWS:
        wf = workflow_mod.load(name)
        if wf.needs:
            continue
        checked += 1
        try:
            wf.bind_backend(object())
        except AssertionError as error:
            if "纯构造式" not in str(error):
                bad.append(f"{name} 拒后端的报错没写「纯构造式」：{error}")
        else:
            bad.append(f"{name} 是纯构造式，却收下了后端")
    assert not bad, bad
    assert checked, "纯构造式一家都没有，这条测试成了空转"


#: 审计**不读语言文件**的几家：它们用合并标签表反解，所以「不存在的语言」不会当场炸
#: （见各自 `audit` 的文档）。名单是量出来的（逐家真调过一遍），不是猜的。
AUDIT_IS_LANGUAGE_AGNOSTIC: Final[dict[str, str]] = {
    "hybrid": "反解语言无关（选项个数与日程场数自洽）",
    "invoice": "反解用合并标签表（税额行自洽）",
    "mail": "只查产物本身（题族 / 题型 / 链接自洽）",
}


def test_unknown_split_and_judge_report_available() -> None:
    """输入：不存在的切分 / 判别器。输出：无。

    预期行为：`config_for` 与 `make_judges` 都当场报「未知…」，不静默取默认值——取默认值会把
    采样参数或判别器名单悄悄跑偏，跑到「一整批跑完」才发现。文案里必须带「未知」两个字，
    否则调用方看不出是拼错了还是真的没有。各家测试包不再各抄一份。
    """
    bad: list[str] = []
    for name in WORKFLOWS:
        wf = workflow_mod.load(name)
        try:
            wf.config_for("nope")
        except KeyError as error:
            if "未知切分" not in str(error):
                bad.append(f"{name} 的未知切分报错没写「未知切分」：{error}")
        else:
            bad.append(f"{name} 的 config_for 收下了不存在的切分")
        try:
            wf.make_judges(("nope",), "zh")
        except KeyError as error:
            if "未知判别器" not in str(error):
                bad.append(f"{name} 的未知判别器报错没写「未知判别器」：{error}")
        else:
            bad.append(f"{name} 的 make_judges 收下了不存在的判别器")
    assert not bad, bad


def test_audit_rejects_an_unknown_locale(tmp_path: Path) -> None:
    """输入：空目录、一个不存在的语言。输出：无。

    预期行为：审计在**读产物之前**就把语言认出来，报 FileNotFoundError；语言无关的那几家
    明确跳过（它们的 `audit` 只用合并标签表反解，本来就不读语言文件）。
    """
    bad: list[str] = []
    checked = 0
    for name in WORKFLOWS:
        if name in AUDIT_IS_LANGUAGE_AGNOSTIC:
            continue
        checked += 1
        try:
            workflow_mod.load(name).audit(tmp_path, ("train",), "nope")
        except FileNotFoundError:
            continue
        bad.append(f"{name} 的审计收下了不存在的语言（它不在语言无关名单里）")
    assert not bad, bad
    assert checked, "语言相关的审计一家都没有，这条测试成了空转"


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow、一个现场采出来的 spec。输出：无。

    预期行为：抛 RuntimeError 且报错里点名 `bind_backend`，不静默降级成模板——降级会让
    「模型没调通」看起来像「数据合成成功了」。纯构造 workflow 不需要后端，跳过。
    """
    bad: list[str] = []
    checked = 0
    for name in WORKFLOWS:
        wf = workflow_mod.load(name)
        if not wf.needs:
            continue
        checked += 1
        config = wf.config_for("train")
        case = wf.build_cases(config)[0]
        spec = wf.sample(random.Random(0), wf.plan_sample(random.Random(0), config, case, 0))
        try:
            wf.render(spec, random.Random(0), "zh")
        except RuntimeError as error:
            if "bind_backend" not in str(error):
                bad.append(f"{name} 的没绑后端报错没点名 bind_backend：{error}")
        else:
            bad.append(f"{name} 没绑后端也渲染出来了")
    assert not bad, bad
    assert checked, "需要后端的一家都没有，这条测试成了空转"


#: 各 workflow 声明的契约字段：名字 → （题型 / 后端需求 / 判别器名单）。
#: 这 20 条原本散在 20 个包的 `test_pipeline.py` 里各抄一份（字段一样、只有取值不同），
#: 现在只留这一张表——它是**声明的事实**，不是从实现里读出来的，所以它红就是实现漂了。
DECLARED: Final[dict[str, tuple[frozenset[str], frozenset[str], tuple[str, ...]]]] = {
    "adblock": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "alertrisk": (frozenset({"choice", "noul", "score"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "autoformat": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "roundtrip", "trace")),
    "causechain": (frozenset({"choice", "noul", "score"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "chatgate": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "citation": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "debugstep": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "depcheck": (frozenset({"choice", "noul", "score"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "guardrail": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "logtriage": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "memorygate": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "posttriage": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "promptinject": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "receiptcat": (frozenset({"choice", "noul", "score"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "segmentation": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "read")),
    "sqlreview": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "subqa": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "testimpact": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "tracescore": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
    "turntake": (frozenset({"choice", "noul"}), frozenset({"text"}), ("hygiene", "frame", "trace")),
}


def test_declared_contract_fields_match_the_registry() -> None:
    """输入：上面那张声明表。输出：无。

    预期行为：每个名字都加载得出来、`name` 与注册名一致、题型 / 后端需求 / 判别器名单与表里
    逐字相同——各家测试包不再各抄一份。
    """
    for name, (kinds, needs, judges) in DECLARED.items():
        wf = workflow_mod.load(name)
        assert wf.name == name, name
        assert wf.kinds == kinds, f"{name} 的题型声明变了：{sorted(wf.kinds)} != {sorted(kinds)}"
        assert wf.needs == needs, f"{name} 的后端需求变了：{sorted(wf.needs)} != {sorted(needs)}"
        assert wf.judges == judges, f"{name} 的判别器名单变了：{wf.judges} != {judges}"


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却点名要 `trace` 判别器。输出：无。

    预期行为：抛 RuntimeError 且报错里点名 `trace`——`trace` 反解要调模型，没后端就该当场炸，
    不许静默降级成「没有这一条判别器」（那会让「模型没调通」看起来像「数据合成成功了」）。
    各家测试包不再各抄一份。
    """
    for name in WORKFLOWS:
        wf = workflow_mod.load(name)
        if "trace" not in wf.judges:
            continue
        with pytest.raises(RuntimeError, match="trace"):
            wf.make_judges(("trace",), "zh")


# ---- 留出取值 / 留出素材：同分布侧与 OOD 侧必须互不重叠 --------------------------
#
# 这两条以前在 8 个包里各抄一份（22 份），逐字相同、只差 workflow 名。收在这里之后
# 口径从「8 家」扩到「所有会报 `unseen_dims()` 的包」（现 11 家），而且名单是现场算的：
# 新包只要给 case 加上 `unseen_dims()` 就自动进闸，不用记得再抄一份测试。

#: 报 `unseen_dims()` 的包至少要有这么多家，否则这条闸名存实亡（名单缩水要当场看见）。
UNSEEN_FLOOR: Final[int] = 8

#: 报 `material_of()` 的包至少要有这么多家。
MATERIAL_FLOOR: Final[int] = 6


def unseen_aware() -> list[str]:
    """输入：无。输出：case 会报 `unseen_dims()` 的 workflow 名（排序后）。

    预期行为：拿 `train` 切分的第一条 case 探一下有没有这个方法——不按包名写死名单，
    也不静默跳过；一个都没有的话由调用方的下限断言当场红灯。
    """
    names: list[str] = []
    for name in WORKFLOWS:
        wf = workflow_mod.load(name)
        cases = wf.build_cases(wf.config_for("train"))
        if cases and hasattr(cases[0], "unseen_dims"):
            names.append(name)
    return names


def material_of_for(name: str) -> Callable[[object], str] | None:
    """输入：workflow 名。输出：它包里的 `material_of`（没有就 None）。

    预期行为：素材指纹有的包放 `spec`、有的放 `features`（`autoformat` 就是后者），
    两处都找；都没有就返回 None，由调用方跳过——「这个包没有素材概念」是合法状态。
    """
    for module_name in (f"workflows.{name}.spec", f"workflows.{name}.features"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        found = getattr(module, "material_of", None)
        if found is not None:
            return found
    return None


def test_id_splits_carry_no_unseen_values() -> None:
    """输入：无（名单现场算）。输出：无。

    预期行为：会报 `unseen_dims()` 的包里，`train` / `val` / `test_iid` 三个切分出的 case
    **一个留出取值都没有**——否则同分布侧与 `test_ood` 会重叠，OOD 评测当场失真。
    各家测试包不再各抄一份。
    """
    names = unseen_aware()
    assert len(names) >= UNSEEN_FLOOR, f"只有 {len(names)} 家报了 unseen_dims：{names}"
    for name in names:
        wf = workflow_mod.load(name)
        for split in ("train", "val", "test_iid"):
            cases = wf.build_cases(wf.config_for(split))
            assert cases, f"{name}/{split} 一条 case 都没有"
            leaked = [case for case in cases if case.unseen_dims()]
            assert not leaked, f"{name}/{split} 混进了留出取值：{leaked[0]}"


def test_ood_split_carries_only_unseen_values() -> None:
    """输入：无（名单现场算）。输出：无。

    预期行为：同一批包的 `test_ood` 出的**每一条**都至少含一个留出取值——一条不含就等于把
    同分布样本混进了留出集，指标看着好其实是自己考自己。各家测试包不再各抄一份。
    """
    names = unseen_aware()
    assert len(names) >= UNSEEN_FLOOR, f"只有 {len(names)} 家报了 unseen_dims：{names}"
    for name in names:
        wf = workflow_mod.load(name)
        cases = wf.build_cases(wf.config_for("test_ood"))
        assert cases, f"{name}/test_ood 一条 case 都没有"
        same = [case for case in cases if not case.unseen_dims()]
        assert not same, f"{name}/test_ood 有 {len(same)} 条不含留出取值：{same[0]}"


def test_ood_materials_never_overlap_the_id_splits() -> None:
    """输入：无（名单现场算）。输出：无。

    预期行为：会报 `material_of()` 的包里，OOD 侧的素材指纹与三个同分布切分**完全不相交**——
    值留出与素材留出叠加之后，两侧按构造不重叠。各家测试包不再各抄一份。
    """
    checked: list[str] = []
    for name in WORKFLOWS:
        fingerprint = material_of_for(name)
        if fingerprint is None:
            continue
        checked.append(name)
        wf = workflow_mod.load(name)
        id_materials = {
            fingerprint(case)
            for split in ("train", "val", "test_iid")
            for case in wf.build_cases(wf.config_for(split))
        }
        ood_materials = {
            fingerprint(case) for case in wf.build_cases(wf.config_for("test_ood"))
        }
        overlap = id_materials & ood_materials
        assert ood_materials and not overlap, f"{name} 两侧素材重叠：{sorted(overlap)[:3]}"
    assert len(checked) >= MATERIAL_FLOOR, f"只有 {len(checked)} 家报了 material_of：{checked}"
