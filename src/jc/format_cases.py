"""契约格式 case 库的构造器（通用层，与具体 workflow 无关）。

职责：把「契约允许什么、拒绝什么」变成两份可交付的 case 内容：
- `valid/`：合法样本，覆盖 state 形态 × 题型 × 描述形态 × 数量与编号边界；
- `invalid/`：每个错误码一个文件，**文件名即期望**（PLAN 1.6）。

输入/输出：只生产内容（样本对象 / 原始字节），不读磁盘；落盘走 `write_all()`。
命令行入口：`python -m jc.format_cases <目标目录>`（默认 `cases/format`）。

预期行为：输出确定性（同版本逐字节一致）；整库 id 唯一；invalid 的每一行都只触发
文件名那个错误码——由 `tests/core/test_format_cases.py` 逐文件验收。
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from . import contract, cover, jsonio

_TASK = "fmt"
_SPLIT = "train"
_DEFAULT_ROOT = "cases/format"


# ---------- 组装 ----------
def _row(
    seq: int,
    state: Any,
    questions: dict[str, Any],
    answers: dict[str, Any],
    *,
    split: str = _SPLIT,
) -> dict[str, Any]:
    """组装一行样本。

    输入：序号、state、题目字典、答案字典、可选切分。输出：契约形态的样本。
    预期行为：id 用 `{任务}-{切分}-{序号:07d}`（切分可覆盖，矩阵要覆盖四种切分）；
    只写契约允许的四个顶层字段。
    """
    return {
        "id": f"{_TASK}-{split}-{seq:07d}",
        "state": state,
        "questions": questions,
        "answers": answers,
    }


def _single(
    seq: int,
    state: Any,
    question: dict[str, Any],
    answer: dict[str, Any],
) -> dict[str, Any]:
    """组装单题样本（正例里绝大多数是单题）。

    输入：序号、state、一道题、它的答案。输出：样本。
    预期行为：题号统一用 `ask`，同一份文件里行与行才好对比。
    """
    return _row(seq, state, {"ask": question}, {"ask": answer})


def _choice(
    options: dict[str, Any],
    *,
    instructions: Any = "where next?",
    answer: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """造一道 choice 题与它的答案。

    输入：criteria 选项表、可选 instructions、答案编号。输出：(题目, 答案)。
    预期行为：答案编号默认 0，负例里要越界时由调用方自己改，避免额外错误码。
    """
    question = {
        "type": contract.TYPE_CHOICE,
        "instructions": instructions,
        "criteria": options,
    }
    return question, {"type": contract.TYPE_CHOICE, "choice": answer}


def _noul(
    *,
    instructions: Any = "yes or no?",
    criteria: dict[str, Any] | None = None,
    answer: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """造一道 noul 题与它的答案。

    输入：可选 instructions / criteria / 答案布尔。输出：(题目, 答案)。
    预期行为：criteria 为 None 时整块省略（契约允许 noul 没有 criteria）。
    """
    question: dict[str, Any] = {"type": contract.TYPE_NOUL, "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    return question, {"type": contract.TYPE_NOUL, "noul": answer}


def _score(
    levels: Any,
    *,
    instructions: Any = "how much?",
    answer: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """造一道 score 题与它的答案。

    输入：等级列表（负例里故意传别的形态）、可选 instructions、答案下标。输出：(题目, 答案)。
    预期行为：等级描述原样写进 criteria，不做任何规范化。
    """
    question = {
        "type": contract.TYPE_SCORE,
        "instructions": instructions,
        "criteria": levels,
    }
    return question, {"type": contract.TYPE_SCORE, "score": answer}


# ---------- 正例矩阵（维度枚举） ----------
# schema 维度：组合数 3×3×3×3×3×2×2×4×3 = 11664 > 全枚举上限，
# 所以走成对覆盖（PLAN 3.3）——任意两维的任意取值组合都保证出现一次。
SCHEMA_DIMS: Final[dict[str, tuple[str, ...]]] = {
    "state_kind": ("string", "object", "array"),
    "question_type": contract.QUESTION_TYPES,
    "instructions_kind": ("string", "object", "array"),
    "desc_kind": ("string", "object", "array"),
    "criteria_shape": ("min", "regular", "max"),
    "answer_pos": ("first", "last"),
    "text_script": ("ascii", "cjk"),
    "split": contract.SPLITS,
    "question_count": ("single", "mixed", "same_family"),
}

_CHOICE_SIZES: Final[dict[str, int]] = {
    "min": contract.CHOICE_MIN_OPTIONS,
    "regular": 3,
    "max": contract.CHOICE_MAX_OPTIONS,
}
_SCORE_SIZES: Final[dict[str, int]] = {
    "min": contract.SCORE_MIN_LEVELS,
    "regular": 3,
    "max": contract.SCORE_MAX_LEVELS,
}


def matrix_combos() -> list[dict[str, Any]]:
    """返回正例矩阵用的维度组合（成对覆盖，确定性）。

    输入：无。输出：组合列表。预期行为：等价于 `cover.cover_dims(SCHEMA_DIMS)`。
    """
    return cover.cover_dims(SCHEMA_DIMS)


def _text(script: str, ascii_text: str, cjk_text: str) -> str:
    """按 text_script 取文案（ascii / cjk 两套）。"""
    return cjk_text if script == "cjk" else ascii_text


def _state_for(kind: str, script: str) -> Any:
    """按 state_kind 造 state（string / object / array 三种都非空）。"""
    if kind == "string":
        return _text(script, "a small state", "起点在左上角，出口在右下角。")
    if kind == "object":
        return {"rows": ["#S.#", "..#E"], "note": _text(script, "as is", "原样")}
    return [_text(script, "line one", "第一行"), _text(script, "line two", "第二行")]


def _desc_for(kind: str, script: str) -> Any:
    """按 desc_kind 造描述值（只出 string / object / array，三种题型都合法）。"""
    if kind == "string":
        return _text(script, "go north", "向北走")
    if kind == "object":
        return {"dx": 1, "dy": 0, "hint": _text(script, "one step", "一步")}
    return [_text(script, "north", "北"), _text(script, "up", "上")]


def _option_names(count: int, script: str) -> list[str]:
    """造 count 个合法选项名。

    输入：个数、文案语言。输出：名字列表。
    预期行为：非空、首尾无空白、不是纯整数字符串、转小写后不重复。
    """
    head = ["上", "下", "左", "右"] if script == "cjk" else ["up", "down", "left", "right"]
    rest = [
        f"选项{index:03d}" if script == "cjk" else f"opt{index:03d}"
        for index in range(len(head), count)
    ]
    return head[:count] + rest


def _level_descriptions(desc: Any, count: int) -> list[Any]:
    """把一份描述值复制成 count 份**互不相同**的等级描述（形态保持不变）。

    输入：描述值、等级数。输出：等级列表。
    预期行为：string 加后缀、object 加键、array 加元素；等级语义不重叠，读起来像真的。
    """
    out: list[Any] = []
    for index in range(count):
        if isinstance(desc, str):
            out.append(f"{desc} #{index + 1}")
        elif isinstance(desc, dict):
            out.append({**desc, "level": index + 1})
        else:
            out.append([*desc, index + 1])
    return out


def _matrix_question(dims: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """按维度造主问题，并给出它的取值个数。

    输入：维度取值。输出：(题目, 取值个数)；取值个数 = choice 选项数 / score 等级数 / noul 记 2。
    预期行为：instructions 三种形态、描述值三种形态、数量三档（noul 的档位映射为
    「省略 criteria / 只给 true / true+false 都给」）。
    """
    qtype = dims["question_type"]
    instructions: Any = _text(dims["text_script"], "which one?", "选哪个？")
    if dims["instructions_kind"] == "object":
        instructions = {"ask": instructions}
    elif dims["instructions_kind"] == "array":
        instructions = [instructions]
    desc = _desc_for(dims["desc_kind"], dims["text_script"])

    if qtype == contract.TYPE_CHOICE:
        size = _CHOICE_SIZES[dims["criteria_shape"]]
        names = _option_names(size, dims["text_script"])
        question = {
            "type": qtype,
            "instructions": instructions,
            "criteria": {name: desc for name in names},
        }
        return question, size
    if qtype == contract.TYPE_SCORE:
        size = _SCORE_SIZES[dims["criteria_shape"]]
        question = {
            "type": qtype,
            "instructions": instructions,
            "criteria": _level_descriptions(desc, size),
        }
        return question, size

    question = {"type": qtype, "instructions": instructions}
    shape = dims["criteria_shape"]
    if shape == "regular":
        question["criteria"] = {"true": desc}
    elif shape == "max":
        question["criteria"] = {"true": desc, "false": desc}
    return question, 2


def _matrix_answer(qtype: str, size: int, pos: str) -> dict[str, Any]:
    """按题型与 answer_pos 造答案（first = 首，last = 末）。"""
    if qtype == contract.TYPE_NOUL:
        return {"type": qtype, "noul": pos == "last"}
    field = "choice" if qtype == contract.TYPE_CHOICE else "score"
    return {"type": qtype, field: size - 1 if pos == "last" else 0}


def _matrix_row(seq: int, dims: Mapping[str, Any]) -> dict[str, Any]:
    """按一组维度取值造一行合法样本。

    输入：序号、维度取值。输出：样本。
    预期行为：question_count 决定单题 / 混排 / 同族重复；混排时额外挂两道默认形态的题，
    只为覆盖「一个 state 挂多题」这个形态。
    """
    question, size = _matrix_question(dims)
    answer = _matrix_answer(dims["question_type"], size, dims["answer_pos"])
    if dims["question_count"] == "same_family":
        questions = {"ask": question, "ask_2": question, "ask_3": question}
        answers = {"ask": answer, "ask_2": answer, "ask_3": answer}
    elif dims["question_count"] == "mixed":
        extra_q, extra_a = _noul()
        much_q, much_a = _score(["low", "high"])
        questions = {"ask": question, "yes_no": extra_q, "how_much": much_q}
        answers = {"ask": answer, "yes_no": extra_a, "how_much": much_a}
    else:
        questions, answers = {"ask": question}, {"ask": answer}
    state = _state_for(dims["state_kind"], dims["text_script"])
    return _row(seq, state, questions, answers, split=dims["split"])


# ---------- 正例 ----------
def valid_files() -> dict[str, list[dict[str, Any]]]:
    """返回 {相对路径: 行列表}（键按构造顺序，写出时保持同样顺序）。

    输入：无。输出：正例文件内容。
    预期行为：确定性；id 整库唯一；覆盖 state 形态 × 题型 × 描述形态 × 数量与编号边界。
    """
    seq = itertools.count(1)
    files: dict[str, list[dict[str, Any]]] = {}

    # 正例矩阵：维度枚举器给的成对覆盖组合（系统化的那一份，覆盖全部 schema 维度）。
    files["valid/pairwise-matrix.jsonl"] = [
        _matrix_row(next(seq), combo) for combo in matrix_combos()
    ]

    # 描述值形态：string / object / array / null 四种都要出现。
    files["valid/desc-kinds.jsonl"] = [
        _single(
            next(seq),
            "desc kinds",
            *_choice(
                {
                    "up": "go north",
                    "right": {"dx": 1, "dy": 0},
                    "down": ["south", "下行"],
                    "left": None,
                }
            ),
        ),
        _single(next(seq), "noul desc", *_noul(criteria={"true": "看起来是", "false": None})),
        _single(next(seq), "score desc", *_score(["低", {"hint": "偏低"}, ["偏低", "偏低"]])),
    ]

    # instructions 的三种合法形态。
    files["valid/instructions-forms.jsonl"] = [
        _single(next(seq), "string", *_choice({"a": None, "b": None}, instructions="which one?")),
        _single(
            next(seq),
            "object",
            *_noul(instructions={"ask": "是不是", "hint": "只看下一格"}),
        ),
        _single(next(seq), "array", *_score(["low", "high"], instructions=["第一步", "第二步"])),
    ]

    # noul：criteria 可省略，也可只给一个 key，值允许 null。
    files["valid/noul-variants.jsonl"] = [
        _single(next(seq), "criteria omitted", *_noul()),
        _single(next(seq), "both keys", *_noul(criteria={"true": "是", "false": "否"})),
        _single(next(seq), "null values", *_noul(criteria={"true": None, "false": None}, answer=False)),
    ]

    # score：等级数边界 2 / 10，答案落在两端。
    files["valid/score-bounds.jsonl"] = [
        _single(next(seq), "two levels, lowest", *_score(["低", "高"], answer=0)),
        _single(next(seq), "two levels, highest", *_score(["低", "高"], answer=1)),
        _single(next(seq), "ten levels, highest", *_score([f"第 {i} 档" for i in range(1, 11)], answer=9)),
    ]

    # choice：选项数边界 2 / 255，答案落在两端。
    files["valid/choice-bounds.jsonl"] = [
        _single(next(seq), "two options, first", *_choice({"up": None, "down": None}, answer=0)),
        _single(next(seq), "two options, last", *_choice({"up": None, "down": None}, answer=1)),
        _single(
            next(seq),
            "255 options, last",
            *_choice({f"opt{i:03d}": None for i in range(255)}, answer=254),
        ),
    ]

    # 一个 state 挂多题（混排），以及同族题号加 `_2` / `_3`。
    pick_q, pick_a = _choice({"up": None, "down": None})
    yes_q, yes_a = _noul()
    much_q, much_a = _score(["低", "高"])
    files["valid/multi-question.jsonl"] = [
        _row(
            next(seq),
            "mixed types",
            {"pick": pick_q, "yes_no": yes_q, "how_much": much_q},
            {"pick": pick_a, "yes_no": yes_a, "how_much": much_a},
        ),
        _row(
            next(seq),
            "repeated family",
            {"ask": pick_q, "ask_2": pick_q, "ask_3": pick_q},
            {"ask": pick_a, "ask_2": pick_a, "ask_3": pick_a},
        ),
    ]

    # 非 ASCII：state / instructions / 选项名 / 等级描述都允许任意语言。
    files["valid/unicode.jsonl"] = [
        _single(
            next(seq),
            "中文 state：起点在左上角，出口在右下角。",
            *_choice(
                {"上": "向北走", "下": None, "左": "向西走", "右": None},
                instructions="下一步往哪走？",
            ),
        ),
        _single(
            next(seq),
            {"备注": "对象形态的 state 也可以", "格子": ["S", "E"]},
            *_score(["很差", "一般", "很好"], instructions="可读性如何？", answer=2),
        ),
    ]

    return files


# ---------- 负例 ----------
def invalid_files() -> dict[str, bytes]:
    """返回 {相对路径: 文件原始字节}，**文件名即期望错误码**。

    输入：无。输出：负例文件内容。
    预期行为：每份文件只触发文件名那个错误码；E_BOM / E_ID_DUP 各带一行合法前提
    （BOM 属于整份文件；重复 id 需要先有一行），其余文件每一行都必须被拒。
    """
    out: dict[str, bytes] = {}

    def put(code: str, rows: list[dict[str, Any]]) -> None:
        """把若干样本行写成该错误码的负例文件。"""
        text = "".join(jsonio.dumps_line(row) + "\n" for row in rows)
        out[f"invalid/{code}.jsonl"] = text.encode("utf-8")

    def put_raw(code: str, text: str) -> None:
        """把原始文本写成该错误码的负例文件（解析期就要坏掉的行用这个）。"""
        out[f"invalid/{code}.jsonl"] = text.encode("utf-8")

    def base(seq: int = 1) -> dict[str, Any]:
        """一份合法样本，各负例都在它上面只改一处。

        输入：行内序号。输出：合法样本。
        预期行为：同一文件里放多行时必须给不同序号，否则会额外触发 E_ID_DUP。
        """
        question, answer = _choice({"up": None, "down": None})
        return _single(seq, "state text", question, answer)

    # 文件级：带 UTF-8 BOM（剥掉 BOM 后那一行本身合法）。
    out[f"invalid/{contract.E_BOM}.jsonl"] = b"\xef\xbb\xbf" + (
        jsonio.dumps_line(base()) + "\n"
    ).encode("utf-8")

    # 行级解析失败：少一个右花括号。
    put_raw(contract.E_JSON_PARSE, '{"id":"fmt-train-0000001","state":"state text",\n')

    # 重复 key：解析期直接拒（若静默保留最后一个，选项数与编号会一起错位）。
    put_raw(
        contract.E_JSON_DUP_KEY,
        '{"id":"fmt-train-0000001","id":"fmt-train-0000002","state":"state text",'
        '"questions":{"ask":{"type":"choice","instructions":"go?","criteria":{"up":null,"down":null}}},'
        '"answers":{"ask":{"type":"choice","choice":0}}}\n',
    )

    # NaN / Infinity 不是合法 JSON。
    put_raw(
        contract.E_JSON_NAN,
        '{"id":"fmt-train-0000001","state":NaN,'
        '"questions":{"ask":{"type":"choice","instructions":"go?","criteria":{"up":null,"down":null}}},'
        '"answers":{"ask":{"type":"choice","choice":0}}}\n',
    )

    # 每行必须是 JSON 对象。
    put_raw(contract.E_ROW_NOT_OBJECT, '[1,2,3]\n"just a string"\n')

    # 顶层缺字段 / 题目缺 instructions / choice 缺 criteria：都归 E_FIELD_MISSING。
    no_answers = base(1)
    no_answers.pop("answers")
    no_instr = base(2)
    no_instr["questions"]["ask"].pop("instructions")
    no_criteria = base(3)
    no_criteria["questions"]["ask"].pop("criteria")
    put(contract.E_FIELD_MISSING, [no_answers, no_instr, no_criteria])

    # 顶层多出字段（能抓到 meta / answer 这类拼写错误）。
    extra_top = base()
    extra_top["meta"] = {}
    put(contract.E_FIELD_UNKNOWN, [extra_top])

    # id 不符合 {任务}-{切分}-{序号:07d}。
    bad_id = base()
    bad_id["id"] = "bad-id"
    put(contract.E_ID_INVALID, [bad_id])

    # id 重复：第 1 行是前提（合法），第 2 行才报 E_ID_DUP。
    put(contract.E_ID_DUP, [base(), base()])

    # state 空 / 类型不对。
    blank_state = base(1)
    blank_state["state"] = "   "
    num_state = base(2)
    num_state["state"] = 5
    put(contract.E_STATE_INVALID, [blank_state, num_state])

    # questions 空（answers 同步清空，免得再报一条 keyset 不符）。
    empty_q = base()
    empty_q["questions"] = {}
    empty_q["answers"] = {}
    put(contract.E_QUESTIONS_EMPTY, [empty_q])

    # type 不是 choice / noul / score。
    bad_type = base()
    bad_type["questions"]["ask"]["type"] = "pick"
    put(contract.E_Q_TYPE, [bad_type])

    # 题目多出字段。
    extra_q = base()
    extra_q["questions"]["ask"]["note"] = "x"
    put(contract.E_Q_FIELD_UNKNOWN, [extra_q])

    # instructions 空 / 类型不对。
    blank_instr = base(1)
    blank_instr["questions"]["ask"]["instructions"] = ""
    num_instr = base(2)
    num_instr["questions"]["ask"]["instructions"] = 5
    put(contract.E_INSTRUCTIONS_INVALID, [blank_instr, num_instr])

    # 描述值类型不对：choice 选项 / score 等级 / noul 条件各一处。
    desc_choice = base(1)
    desc_choice["questions"]["ask"]["criteria"] = {"up": 1, "down": None}
    desc_score_q, desc_score_a = _score(["ok", 3])
    desc_noul_q, desc_noul_a = _noul(criteria={"true": 1})
    put(
        contract.E_DESC_TYPE,
        [
            desc_choice,
            _single(2, "state text", desc_score_q, desc_score_a),
            _single(3, "state text", desc_noul_q, desc_noul_a),
        ],
    )

    # choice 的 criteria 不是对象。
    choice_type = base()
    choice_type["questions"]["ask"]["criteria"] = ["up", "down"]
    put(contract.E_CHOICE_CRITERIA_TYPE, [choice_type])

    # 选项数不足 2。
    few_options = base()
    few_options["questions"]["ask"]["criteria"] = {"up": None}
    put(contract.E_CHOICE_OPTION_COUNT, [few_options])

    # 选项名首尾带空白 / 是纯整数字符串。
    padded_name = base(1)
    padded_name["questions"]["ask"]["criteria"] = {" up": None, "down": None}
    int_name = base(2)
    int_name["questions"]["ask"]["criteria"] = {"12": None, "down": None}
    put(contract.E_CHOICE_OPTION_NAME, [padded_name, int_name])

    # 选项名去空白转小写后重复。
    dup_name = base()
    dup_name["questions"]["ask"]["criteria"] = {"up": None, "UP": None}
    put(contract.E_CHOICE_OPTION_DUP, [dup_name])

    # noul 的 criteria 只允许 true / false。
    noul_key_q, noul_key_a = _noul(criteria={"yes": "y"})
    put(contract.E_NOUL_CRITERIA_KEY, [_single(1, "state text", noul_key_q, noul_key_a)])

    # score 的 criteria 不是数组。
    score_obj_q, score_obj_a = _score({"a": 1, "b": 2})
    put(contract.E_SCORE_CRITERIA_TYPE, [_single(1, "state text", score_obj_q, score_obj_a)])

    # 等级数不足 2。
    one_level_q, one_level_a = _score(["only"])
    put(contract.E_SCORE_LEVEL_COUNT, [_single(1, "state text", one_level_q, one_level_a)])

    # score 等级不允许 null（等级描述就是等级定义）。
    null_level_q, null_level_a = _score(["low", None])
    put(contract.E_SCORE_LEVEL_NULL, [_single(1, "state text", null_level_q, null_level_a)])

    # answers 少题号 / 多题号。
    missing_key = base(1)
    missing_key["answers"] = {}
    extra_key = base(2)
    extra_key["answers"]["ghost"] = {"type": contract.TYPE_CHOICE, "choice": 0}
    put(contract.E_ANSWERS_KEYSET, [missing_key, extra_key])

    # 答案多出字段。
    extra_field = base()
    extra_field["answers"]["ask"]["extra"] = 1
    put(contract.E_ANSWER_SHAPE, [extra_field])

    # 答案 type 与题目不一致。
    mismatch = base()
    mismatch["answers"]["ask"]["type"] = contract.TYPE_NOUL
    put(contract.E_ANSWER_TYPE_MISMATCH, [mismatch])

    # 编号不是整数：浮点 / 布尔。
    float_idx = base(1)
    float_idx["answers"]["ask"]["choice"] = 0.0
    bool_idx = base(2)
    bool_idx["answers"]["ask"]["choice"] = True
    put(contract.E_INDEX_NOT_INT, [float_idx, bool_idx])

    # 编号越界（两个选项，答案给 2）。
    out_of_range = base()
    out_of_range["answers"]["ask"]["choice"] = 2
    put(contract.E_INDEX_RANGE, [out_of_range])

    # noul 答案不是 JSON 布尔。
    noul_plain_q, _ = _noul()
    put(
        contract.E_NOUL_NOT_BOOL,
        [_single(1, "state text", noul_plain_q, {"type": contract.TYPE_NOUL, "noul": 1})],
    )

    return out


# ---------- 落盘 ----------
def write_all(root: str | Path = _DEFAULT_ROOT) -> list[Path]:
    """把整库写到 root 下（valid/ 与 invalid/），返回写出的文件路径。

    输入：目标根目录（通常就是仓库的 cases/format）。输出：写出的路径列表。
    预期行为：自动建目录、覆盖同名文件、统一 LF；不动目录里其他文件。
    """
    base = Path(root)
    written: list[Path] = []
    for rel, rows in valid_files().items():
        path = base / rel
        jsonio.write_jsonl(path, rows)
        written.append(path)
    for rel, raw in invalid_files().items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        written.append(path)
    return written


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_ROOT
    for written_path in write_all(target):
        print(written_path.as_posix())
