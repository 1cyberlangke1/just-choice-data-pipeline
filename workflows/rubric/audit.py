"""打分与评分的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有五条：

- **材料清单行**必须解析得出来（篇幅 / 形式 / 数据），且与材料正文的形状自洽（句数落在
  `features.length_bounds` 里、分条与行文对得上、有数与无数据对得上）；
- **评分单**必须解析得出来，且判据名与行序、权重列、要求列逐格等于 `features` 里的固定表；
  观测那一列，正文承载的两条必须是「按材料数」的标记，其余三条必须等于
  `OBS_TABLE[维度][清单里的取值]`；
- **档位表 / 处置表**必须逐行等于规范表；
- 拿重算出来的观测走 `features.score_from_obs`，得到的**逐条达标 / 总分 / 档位 / 处置 /
  最严重的一条**必须与产物里记下的答案逐项一致；
- `top` 这一族的**出不出**也要对得上：重算说「唯一」就必须出，说「并列或全达标」就不许出。

唯一纯代码读不出来的是「那两维观测与材料**语义上**是否一致」——机械口径能数出 `TODO`
这种写法，但材料要是把 `TODO` 写在引用的示例里，读起来就是另一回事。那正是
`judges.TraceJudge` 在合成时把关的事，审计这里老实写明「这一条靠合成期的反解把关」
（范式 §9.13：写不出强校验时不要硬编一个看起来像校验的东西）。

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

#: 逐条达标那五道题的题号（与评分单行序一一对应）。
MET_QIDS: tuple[str, ...] = tuple(f"met_{index}" for index in range(1, len(features.CRITERIA) + 1))


def _family_of(qid: str) -> str:
    """输入：题号。输出：题族名（去掉 `_1` 这类序号后缀）。"""
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


def body_shape_problem(body: str, manifest: render.ManifestState) -> str:
    """输入：材料正文、清单行。输出：第一处形状问题（没问题给空串）。

    预期行为：只回第一条问题，避免同一段材料刷出一串计数。口径与 `render.check_draft`
    一致（下界照 `LENGTH_SENTENCES` 不放、上界乘 `LENGTH_SENTENCES_SLACK`）。
    """
    sentences = features.count_sentences(body)
    low, high = features.length_bounds(manifest.length)
    if not low <= sentences <= high:
        return f"句数 {sentences} 不在 {manifest.length!r} 要求的 {low}~{high} 里"
    bullets = features.bullet_lines(render.split_body_lines(body))
    if manifest.form == "bulleted" and not bullets:
        return "「分条」却一行 `- ` 开头的都没有"
    if manifest.form == "prose" and bullets:
        return "「行文」却出现了 `- ` 开头的分条行"
    numbered = features.has_number(body)
    if manifest.data == "numbers" and not numbered:
        return "「有数据」却一个阿拉伯数字都没有"
    if manifest.data == "none" and numbered:
        return "「无数据」却出现了阿拉伯数字"
    return ""


def score_table_problem(
    rows: tuple[tuple[str, str, str, str], ...],
    manifest: render.ManifestState,
    marks: frozenset[str],
) -> str:
    """输入：解析出来的评分单、清单行、「按材料数」标记的集合（`render.count_marks()`）。
    输出：第一处问题（没问题给空串）。

    预期行为：逐行核对判据名与顺序、权重与要求两列、观测那一格的形态——正文承载的两条
    必须是**任一语言**的「按材料数」标记（观测只能从材料里数出来；一次 run 里语言按样本
    轮转，拿调用方给的那一种去认会把另一种语言的产物误判成「评分单对不上」），其余三条
    必须等于 `OBS_TABLE[维度][清单里的取值]`。权重与要求取的是 `features` 里的固定表，
    所以「渲染把权重写歪了」也在这里暴露。
    """
    if len(rows) != len(features.CRITERIA):
        return f"评分单有 {len(rows)} 行，应为 {len(features.CRITERIA)} 行"
    wanted_mark = "/".join(sorted(marks))
    for expected, cells in zip(features.CRITERIA, rows, strict=True):
        name, weight, need, obs_cell = cells
        if name != expected:
            return f"评分单里有判据 {name!r}，应为 {expected!r}"
        if weight != str(features.WEIGHT_OF[name]):
            return f"{name} 的权重是 {weight!r}，应为 {features.WEIGHT_OF[name]}"
        if need != str(features.NEED_OF[name]):
            return f"{name} 的要求是 {need!r}，应为 {features.NEED_OF[name]}"
        dim = features.OBS_DIM[name]
        if dim in features.BODY_DIMS:
            if obs_cell not in marks:
                return f"{name} 的观测该印「{wanted_mark}」（只能从材料里数），实际是 {obs_cell!r}"
            continue
        want = str(features.OBS_TABLE[dim][getattr(manifest, dim)])
        if obs_cell != want:
            return f"{name} 的观测是 {obs_cell!r}，按清单应为 {want}"
    return ""


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。"""
    locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段（标记按「任一语言」收）
    marks = render.count_marks()
    canonical_band = render.band_rows()
    canonical_gate = render.gate_rows()
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
            "score_table_bad": 0,
            "band_table_bad": 0,
            "gate_table_bad": 0,
            "score_bad": 0,
            "met_bad": 0,
            "band_bad": 0,
            "gate_bad": 0,
            "top_bad": 0,
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
                recorded[qid] = _recorded(question, answers[qid])
            parsed = render.parse_state(row["state"])
            if parsed.manifest is None:
                counts["manifest_bad"] += 1
                continue
            if body_shape_problem(parsed.body, parsed.manifest):
                counts["body_shape_bad"] += 1
            if score_table_problem(parsed.score, parsed.manifest, marks):
                counts["score_table_bad"] += 1
            if parsed.band != canonical_band:
                counts["band_table_bad"] += 1
            if parsed.gate != canonical_gate:
                counts["gate_table_bad"] += 1
            obs = render.obs_from_score(parsed.score, parsed.body, marks)
            if obs is None:
                counts["score_bad"] += 1
                continue
            expected = features.score_from_obs(obs)
            for index, qid in enumerate(MET_QIDS, start=1):
                if recorded.get(qid) is not expected.met_at(index):
                    counts["met_bad"] += 1
            if recorded.get("band") != features.BANDS.index(expected.band):
                counts["band_bad"] += 1
            if recorded.get("gate") != expected.gate:
                counts["gate_bad"] += 1
            if recorded.get("top") != expected.top:
                counts["top_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("清单行解析不出来", "manifest_bad", "材料清单那一行解不出三格"),
            ("材料形状与清单对不上", "body_shape_bad", "句数 / 分条 / 数字与清单打架"),
            ("评分单与固定表对不上", "score_table_bad", "判据名、权重、要求或观测形态不对"),
            ("档位表与规范表对不上", "band_table_bad", "档位表不是规范形态"),
            ("处置表与规范表对不上", "gate_table_bad", "处置表不是规范形态"),
            ("评分单重算不出观测", "score_bad", "观测那一格既不是数字也不是标记"),
            ("逐条达标与评分单不自洽", "met_bad", "met_1~met_5 与重算的达标对不上"),
            ("档位与总分不自洽", "band_bad", "band 与重算的档位对不上"),
            ("处置与档位红线不自洽", "gate_bad", "gate 与重算的处置对不上"),
            ("最严重的一条不自洽", "top_bad", "top 的出不出或取值与重算对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
