"""README 的 workflow 目录表必须与代码一致（防文档腐烂）。

输入：`README.md` 里那张 workflow 表，以及 `jc.workflow.discover()` 的注册表。
输出：无（断言）。
预期行为：表里的名字集合、判别器名单、构造 / 注入标记都要与代码对得上——表是给人看的，
但**名字与名单是机器可查的**，漂了就红灯。

为什么值得有：这个仓库的常态就是「加新 workflow」，而 README 的目录表是最容易忘记同步的地方。
手写表格出错也不是假设——这张表第一次提交时 autoformat 那行的判别器就写错了
（写成 `frame / hygiene / trace`，实际是 `hygiene / roundtrip / trace`），这条测试就是为它加的。

只管**名字与名单**，不管 `case` 那一列的自然语言描述（那是给人读的，机器判不了好坏）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from jc import workflow as workflow_mod

#: README 表里的行形态：``| `名字` | case 描述 | 生成方式 | 判别器名单 |``。
ROW: Final[re.Pattern[str]] = re.compile(r"^\| `([a-z_]+)` \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$")

#: 「生成方式」列的合法取值。
GENERATIVE: Final[frozenset[str]] = frozenset({"构造", "注入", "混合"})

#: README 正文里那句切分承诺：``每个 workflow 都出 `train` / `val` / `test_iid` / `test_ood`
#: 四个切分``。正文不像表格那样有结构，但**这一句有唯一正确答案**（见范式 §9.34），所以照样钉。
SPLIT_CLAIM: Final[re.Pattern[str]] = re.compile(
    r"每个 workflow 都出 ((?:`[a-z_]+`(?: */ *)?)+) 四个切分"
)


def readme_splits() -> tuple[str, ...]:
    """输入：无。输出：README 那句承诺里列出的切分名（找不到就空元组，测试会点名）。"""
    text = Path("README.md").read_text(encoding="utf-8")
    match = SPLIT_CLAIM.search(text)
    return tuple(re.findall(r"`([a-z_]+)`", match.group(1))) if match else ()


def rows() -> dict[str, tuple[str, str, str]]:
    """输入：无。输出：{workflow 名: (case 描述, 生成方式, 判别器名单)}。

    预期行为：只认严格符合 `ROW` 的表行；表头、分隔行、其它表格一律不进结果。
    """
    text = Path("README.md").read_text(encoding="utf-8")
    out: dict[str, tuple[str, str, str]] = {}
    for line in text.splitlines():
        match = ROW.match(line.strip())
        if match:
            out[match.group(1)] = (match.group(2).strip(), match.group(3).strip(), match.group(4).strip())
    return out


def test_readme_table_is_parseable() -> None:
    """输入：无。输出：无。预期行为：至少解析出若干行，否则这条尺子名存实亡。"""
    assert len(rows()) >= 20, f"只解析到 {sorted(rows())}"


def test_readme_lists_exactly_the_registered_workflows() -> None:
    """输入：无。输出：无。预期行为：表里的名字集合与注册表**完全相等**（多一个少一个都红灯）。"""
    listed = set(rows())
    registered = set(workflow_mod.discover())
    assert listed == registered, (
        f"README 表里多出 {sorted(listed - registered)}、少了 {sorted(registered - listed)}"
    )


def test_readme_generation_column_matches_needs() -> None:
    """输入：无。输出：无。

    预期行为：`needs` 为空 ⟺ 表里写「构造」（零 token）；需要后端 ⟺ 写「注入」或「混合」。
    """
    problems: list[str] = []
    for name, (_case, how, _judges) in sorted(rows().items()):
        assert how in GENERATIVE, f"{name} 的生成方式 {how!r} 不在 {sorted(GENERATIVE)} 里"
        needs = bool(workflow_mod.load(name).needs)
        if needs and how == "构造":
            problems.append(f"{name} 需要后端（{sorted(workflow_mod.load(name).needs)}）却写成「构造」")
        if not needs and how != "构造":
            problems.append(f"{name} 是纯构造（needs 为空）却写成「{how}」")
    assert not problems, "README 的生成方式与代码对不上：\n" + "\n".join(problems)


def test_readme_judge_column_matches_code() -> None:
    """输入：无。输出：无。预期行为：判别器名单与 `workflow.judges` 逐个对得上（顺序无关）。"""
    problems: list[str] = []
    for name, (_case, _how, judges) in sorted(rows().items()):
        listed = {item.strip() for item in judges.split("/")}
        actual = set(workflow_mod.load(name).judges)
        if listed != actual:
            problems.append(
                f"{name}: README 写 {sorted(listed)}，代码是 {sorted(actual)}"
            )
    assert not problems, "README 的判别器名单与代码对不上：\n" + "\n".join(problems)


def test_readme_split_claim_matches_code() -> None:
    """输入：无。输出：无。

    预期行为：README 说「每个 workflow 都出 … 四个切分」，那就每个 workflow 都要正好是那四个。
    为什么值得钉：两把端到端扫描都按 `("train", "test_ood")` 参数化，切分名一改它们会当场炸，
    但那时报出来的是一堆参数化用例里的 KeyError；在这里点名更直接。改了切分就同步改 README
    与这句话——**改不动 README 就说明这条承诺本来就不该那么写**。
    """
    claimed = readme_splits()
    assert claimed, "README 里找不到「每个 workflow 都出 … 四个切分」那句承诺（改写就同步改这条尺子）"
    problems = [
        f"{name}: {tuple(workflow_mod.load(name).split_names())}"
        for name in sorted(workflow_mod.discover())
        if tuple(workflow_mod.load(name).split_names()) != claimed
    ]
    assert not problems, f"README 说每个 workflow 都出 {claimed}，实际：\n" + "\n".join(problems)
