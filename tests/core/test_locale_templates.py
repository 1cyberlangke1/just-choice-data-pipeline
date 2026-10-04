"""语言文件的四条跨 workflow 不变量（零 token 尺子）。

为什么值得做成尺子：

1. `questions.locate()`（hybrid / invoice 等）是「拿题面与模板列表做**整串相等**比对，
   返回第一个命中的题族」。两个题族的模板一模一样时，它静默返回先遍历到的那个，
   判别器于是拿**错的期望**去比答案——与范式 §9.32 那次「读错标记格」是同一类静默错误：
   不报错、只是接受率悄悄掉，日志里看起来像合成质量问题。
2. 题族只在一种语言里定义时，另一种语言要么缺题、要么渲染直接 KeyError。这类不一致
   在「测试只跑一种语言」的情况下同样看不见（§9.32）。
3. 提示词里没有明说的语言约束时，模型会照着它自己的先验写正文——`rerank` 的 zh 侧真机
   产物里 3/3、4/6、1/4 把查询与候选写成了英文（汉字占比 0.081~0.120），en 侧反过来有
   2/9 写成中文。提示词通篇是目标语言**不足以**让模型不换语言；而这一条一旦漏掉，产物
   照样过契约校验、照样过判别器，只在事后审计里才现形。
4. 题族之外的键（图例 / 词表 / 提示词 / 维度标题…）只在一种语言里定义时，另一种语言跑到
   那个分支才会 KeyError。第 2 条那把尺子只看 `fam` / `ask` 两张表，够不到它们——2026-10-04
   拿 `tmp/probe_locale_parity.py` 把 52 个语言目录的**全键集**对了一遍，当时一处不差；
   这一条就是把它从「碰巧对」变成「钉住」。

输入：`workflows/*/locales/*.toml`。
输出：无（断言）。
预期行为：任何一处撞车或题族缺失都让 `pre-commit` 拦住这次提交。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Final

from jc import workflow as workflow_mod

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


def _key_paths(data: dict[str, object], prefix: str = "") -> set[str]:
    """输入：解析出来的 TOML 字典。输出：摊平后的点分键路径集合。

    预期行为：只展开**表**（`[a.b]`），不展开数组——数组里的表（`[[x]]`）按它自己的键名
    算一格就够了，这里比的是「两边有没有这一格」，不是内容。
    """
    keys: set[str] = set()
    for key, value in data.items():
        path = f"{prefix}{key}"
        keys.add(path)
        if isinstance(value, dict):
            keys |= _key_paths(value, prefix=f"{path}.")
    return keys


def test_every_workflow_defines_the_same_keys_in_all_languages() -> None:
    """同一 workflow 的各语言文件必须有**同一套键**（不只是题族）。

    输入：所有语言文件。输出：无（断言）。
    预期行为：摊平成点分键路径再比。图例 / 词表 / 提示词 / 维度标题这些表里少一个键，
    另一种语言跑到那个分支就是 KeyError（或者悄悄回落到默认值），而「只跑一种语言」的
    测试看不见——与上面那条题族尺子是同一个盲区，只是深一层。
    """
    grouped: dict[str, list[Path]] = {}
    for path in locale_files():
        grouped.setdefault(path.parent.parent.as_posix(), []).append(path)
    problems: list[str] = []
    for workflow_dir, paths in sorted(grouped.items()):
        if len(paths) < 2:
            continue
        shapes = {
            path.stem: _key_paths(tomllib.loads(path.read_text(encoding="utf-8")))
            for path in paths
        }
        base_name = sorted(shapes)[0]
        base = shapes[base_name]
        for name in sorted(shapes):
            if name == base_name:
                continue
            missing = sorted(base - shapes[name])
            extra = sorted(shapes[name] - base)
            if missing or extra:
                problems.append(f"{workflow_dir}: {name} 缺 {missing}、多 {extra}")
    assert not problems, "各语言键集不一致（另一种语言会 KeyError）：\n" + "\n".join(problems)


def _strings(node: object) -> list[str]:
    """输入：TOML 解析出来的任意节点。输出：它下面所有字符串（递归展开表与数组）。

    预期行为：必须递归——12 家 workflow 的语言约束写在 `prompt.rules` 这个**字符串数组**
    里，只看表的一层会漏掉它们，尺子就会把对的判成错的。
    """
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        found: list[str] = []
        for value in node.values():
            found.extend(_strings(value))
        return found
    if isinstance(node, list):
        found: list[str] = []
        for value in node:
            found.extend(_strings(value))
        return found
    return []


def test_every_llm_workflow_pins_the_language_in_its_prompt() -> None:
    """走 LLM 的 workflow，每种语言的提示词里都必须有一句明说的语言约束。

    输入：`jc.workflow` 注册表 + `workflows/*/locales/*.toml`。输出：无（断言）。
    预期行为：`needs` 非空（= 文本由模型写）的 workflow，每种语言都必须有
    - 顶层 `language_name`（那句约束要填的语言名，也是「与声明语言一致」这条判据在人话里
      的说法）；
    - 在**带 `intro` 的那张提示词表**里，至少一个字符串含 `{language}` 占位符。

    为什么查的是「带 `intro` 的表」而不是写死 `[prompt]`：`autoformat` 的提示词表叫
    `[doc]`，各家自己起名；`intro` 是开场白，是唯一在 17 家里都存在的锚点。
    为什么纯构造式（`needs` 为空）豁免：它们根本不发提示词，没有「模型会换语言」这回事。
    """
    workflow_mod.discover()
    problems: list[str] = []
    checked = 0
    for name in workflow_mod.registered():
        workflow = workflow_mod.load(name)
        if not workflow.needs:
            continue
        for locale in workflow.available_locales():
            label = f"{name}/{locale}"
            checked += 1
            path = Path("workflows") / name / "locales" / f"{locale}.toml"
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            language_name = data.get("language_name")
            if not isinstance(language_name, str) or not language_name.strip():
                problems.append(f"{label}: 顶层缺 `language_name`，提示词里没有可填的语言名")
                continue
            tables = [
                key
                for key, value in data.items()
                if isinstance(value, dict) and "intro" in value
            ]
            if not tables:
                problems.append(f"{label}: 找不到带 `intro` 的提示词表")
                continue
            if not any(
                "{language}" in text
                for table in tables
                for text in _strings(data[table])
            ):
                problems.append(
                    f"{label}: 提示词里没有一句 `{{language}}` 约束——模型会照它自己的先验换语言"
                )
    assert checked >= 30, f"这把尺子只量到 {checked} 份语言文件，名存实亡"
    assert not problems, "走 LLM 的 workflow 没把语言钉死在提示词里：\n" + "\n".join(problems)
