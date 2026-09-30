"""报销收据归类的产物审计：答案必须与**从 state 里重推出来的结论**逐条自洽。

输入：run 目录、切分名、语言。输出：(各切分计数, 缺失项列表)。
预期行为：与 `jc.validate` 的分工是——通用契约管「格式 case」（题面 / 答案 / 编号 / `type`），
本模块管**领域事实**：题族齐不齐、答案与文本自不自洽。

审计**不碰模型**：它走 `features.observed_labels` 那条纯代码路径（关键词表 + 金额比较），
与构造侧的 `label_for` 是两条独立的路——两者对不上就说明渲染没兑现计划，当场就能看见。

`locale` 只用来**当场验一下这个语言存不存在**（别拖到解析阶段才报）；规范文案则按**每条
state 自己的语言**取（`render.detect_locale`）——一次 run 里混了 zh / en 时，用调用方给的
那一种去比会平白报一堆「文案对不上」。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from jc.contract import option_names
from jc import jsonio
from jc.jsontypes import JsonValue
from jc.workflow import family_of

from . import features, locales, questions, render

#: 本 workflow 认得的题族（审计按它查「题族一条都没出」）。
KNOWN_FAMILIES: Final[frozenset[str]] = frozenset(questions.FAMILIES)


def _recorded(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> JsonValue:
    """输入：题面、答案。输出：把编号换回**选项描述**（choice）或原样取值（noul / score）。

    预期行为：choice 的编号是**产物里那个打乱顺序**里的下标（见 `jc.assemble`），所以先取回
    选项描述再比——这样「选项文案抄错」也会被审计抓住，而只比选项名是抓不住的。score 的编号
    就是档位下标，直接拿来比。
    """
    kind = answer.get("type")
    if kind == "choice":
        names = option_names(question)
        index = answer.get("choice")
        criteria = question.get("criteria")
        if isinstance(index, int) and 0 <= index < len(names) and isinstance(criteria, dict):
            return criteria.get(names[index])
        return None
    if kind == "score":
        return answer.get("score")
    return answer.get("noul")


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

    预期行为：逐行重推三张表与收据行，再把四族的答案与重推结论比一遍；题族缺一档、或任何
    一族出现不自洽，都记进缺失项。
    """
    locales.load(locale)
    report: dict[str, JsonValue] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        counts: dict[str, JsonValue] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "frame_bad": 0,
            "content_bad": 0,
            "cat_bad": 0,
            "over_bad": 0,
            "tier_bad": 0,
            "near_bad": 0,
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
                family = family_of(qid)
                counts["families"][family] += 1
                counts["types"][question["type"]] += 1
                if family not in KNOWN_FAMILIES:
                    counts["unknown_families"][family] += 1
                recorded[qid] = _recorded(question, answers[qid])
            state = row["state"]
            parsed = render.parse_state(state)
            if (
                parsed.categories != features.CATEGORY_IDS
                or parsed.caps != features.CATEGORY_IDS
                or parsed.tiers != len(features.TIERS)
            ):
                counts["frame_bad"] += 1
                continue
            if parsed.text is None or not parsed.text.strip():
                counts["content_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(state) or locale)
            observed = features.observed_labels(parsed.text, state_loc)
            if observed is None:
                counts["content_bad"] += 1
                continue
            if recorded.get("cat") != questions.category_text(state_loc, observed.category):
                counts["cat_bad"] += 1
            if recorded.get("over") is not observed.over:
                counts["over_bad"] += 1
            if recorded.get("tier") != questions.tier_index(observed.tier):
                counts["tier_bad"] += 1
            if recorded.get("near") is not observed.near:
                counts["near_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("三张表或收据行的骨架不合规", "frame_bad", "归类表 / 上限表不是 C1~C4，或档位表不是四行"),
            ("收据行认不出来", "content_bad", "按关键词表算不出唯一的类别或唯一的金额"),
            ("归类答案与收据行不自洽", "cat_bad", "cat 那一族的答案与按文本重推的对不上"),
            ("超限答案与收据行不自洽", "over_bad", "over 那一族的答案与按文本重推的对不上"),
            ("审核档位与收据行不自洽", "tier_bad", "tier 那一族的答案与按文本重推的对不上"),
            ("诱饵答案与收据行不自洽", "near_bad", "near 那一族的答案与按文本重推的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
