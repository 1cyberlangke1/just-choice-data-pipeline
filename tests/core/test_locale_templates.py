"""语言文件的两条跨 workflow 不变量（零 token 尺子）。

为什么值得做成尺子：

1. `questions.locate()`（hybrid / invoice 等）是「拿题面与模板列表做**整串相等**比对，
   返回第一个命中的题族」。两个题族的模板一模一样时，它静默返回先遍历到的那个，
   判别器于是拿**错的期望**去比答案——与范式 §9.32 那次「读错标记格」是同一类静默错误：
   不报错、只是接受率悄悄掉，日志里看起来像合成质量问题。
2. 题族只在一种语言里定义时，另一种语言要么缺题、要么渲染直接 KeyError。这类不一致
   在「测试只跑一种语言」的情况下同样看不见（§9.32）。

输入：`workflows/*/locales/*.toml`。
输出：无（断言）。
预期行为：任何一处撞车或题族缺失都让 `pre-commit` 拦住这次提交。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Final

#: 语言文件里「装题族问法」的表名：各 workflow 沿用这两个名字之一。
FAMILY_TABLES: Final[tuple[str, ...]] = ("fam", "ask")

#: 扫描入口：workflow 的语言目录。
LOCALE_GLOB: Final[str] = "workflows/*/locales/*.toml"


def locale_files() -> list[Path]:
    """输入：无。输出：所有语言文件路径（排序后）。预期行为：只认 `.toml`。"""
    return sorted(Path().glob(LOCALE_GLOB))


def families(path: Path) -> dict[str, list[str]]:
    """读一个语言文件里的题族 → 问法模板列表。

    输入：语言文件路径。
    输出：{「表名.题族名」: [模板, ...]}。
    预期行为：只收 `fam` / `ask` 两个表下带 `q` 列表的条目，其它表（图例、词表、说明）
    一律忽略——它们不是问法，撞车也无所谓。
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for table in FAMILY_TABLES:
        entries = data.get(table)
        if not isinstance(entries, dict):
            continue
        for family, entry in entries.items():
            if isinstance(entry, dict) and isinstance(entry.get("q"), list):
                out[f"{table}.{family}"] = [str(text) for text in entry["q"]]
    return out


def test_locale_files_are_found() -> None:
    """输入：无。输出：无。预期行为：至少扫到若干语言文件，否则这把尺子名存实亡。"""
    assert len(locale_files()) >= 20


def test_no_two_families_share_a_question_template() -> None:
    """同一语言文件里，两个题族的问法模板不能完全相同。

    预期行为：撞车时报出「哪两个题族、哪一句」，因为 `locate()` 会静默取第一个命中。
    同一条模板在同一题族内部重复不算撞车（那只是问法冗余）。
    """
    collisions: list[str] = []
    for path in locale_files():
        owner: dict[str, str] = {}
        for family, templates in families(path).items():
            for text in templates:
                first = owner.setdefault(text, family)
                if first != family:
                    collisions.append(f"{path.as_posix()}: {first} 与 {family} 共用问法 {text!r}")
    assert not collisions, "题族问法撞车（locate 会静默取第一个命中，期望就错了）：\n" + "\n".join(
        collisions
    )


def test_every_workflow_defines_the_same_families_in_all_languages() -> None:
    """同一 workflow 的各语言文件必须定义同一批题族。

    预期行为：某种语言缺题族时报出来——那种语言会缺题，或者渲染时直接 KeyError。
    只查「集合是否一致」，不管模板条数（条数由各 workflow 自己的语言契约测试管）。
    """
    grouped: dict[str, list[Path]] = {}
    for path in locale_files():
        grouped.setdefault(path.parent.parent.as_posix(), []).append(path)
    problems: list[str] = []
    for workflow_dir, paths in sorted(grouped.items()):
        if len(paths) < 2:
            continue
        names = {path.stem: set(families(path)) for path in paths}
        base_name = sorted(names)[0]
        base = names[base_name]
        for name in sorted(names):
            if name == base_name:
                continue
            missing = sorted(base - names[name])
            extra = sorted(names[name] - base)
            if missing or extra:
                problems.append(f"{workflow_dir}: {name} 缺 {missing}、多 {extra}")
    assert not problems, "各语言题族不一致（另一种语言会缺题或渲染 KeyError）：\n" + "\n".join(problems)
