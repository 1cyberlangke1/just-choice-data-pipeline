"""迷宫 workflow 的判别器：构造式数据用纯规则判别即可。

- `RoundtripJudge`：从渲染文本反解迷宫，与真值 spec 比对
  （抓差一错位、S/E 画错位置）；
- `LabelJudge`：反解迷宫后**独立重算**每道题的金答案，与数据里的答案比对
  （抓答案算错、编号串位）。

两者都不调模型——答案由构造保证，判别器只需查「渲染与组装有没有出错」。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from collections import deque
from dataclasses import dataclass


from jc.judge import Verdict

from . import grid, locales
from . import spec as maze_spec
from .workflow import ParsedMaze, parse_state, roundtrip_diff


@dataclass(frozen=True)
class RoundtripJudge:
    """结构判别器：渲染文本反解后必须与真值 spec 完全一致。"""

    name: str = "roundtrip"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        if not isinstance(spec, maze_spec.MazeSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        parsed = parse_state(sample["state"], spec.render)
        if parsed is None:
            return Verdict(False, reason="该格式无法反解", raw={"fmt": spec.render.fmt})
        diff = roundtrip_diff(spec, parsed)
        return Verdict(not diff, reason="; ".join(diff), raw={"diff": diff})


def _dist_to_set(
    mask: tuple[int, ...], rows: int, cols: int, sources: set[int]
) -> dict[int, int]:
    """多源 BFS：每格到 sources 的最短距离（判别器自带的独立实现）。"""
    dist = {src: 0 for src in sources}
    queue = deque(sources)
    while queue:
        cur = queue.popleft()
        for _, nb in grid.open_neighbors(mask, cur, rows, cols):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


@dataclass(frozen=True)
class LabelJudge:
    """答案判别器：反解迷宫后独立重算金答案，逐题比对。

    字段：locale 语言（决定选项文案），max_problems 报告里最多列几条问题。
    """

    locale: str = "zh"
    max_problems: int = 5
    name: str = "label"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        loc = locales.load(self.locale)
        if not isinstance(spec, maze_spec.MazeSpec):
            return Verdict(True, reason="没有 spec，跳过答案判别", raw={"skipped": True})
        parsed = parse_state(sample["state"], spec.render)
        if parsed is None:
            return Verdict(False, reason="该格式无法反解", raw={"fmt": spec.render.fmt})
        problems = self._check(sample, parsed, loc)
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )

    def _check(self, sample: dict[str, JsonValue], parsed: ParsedMaze, loc: dict[str, JsonValue]) -> list[str]:
        """逐题独立复算，返回不一致说明列表。"""
        problems: list[str] = []
        mask, rows, cols = parsed.mask, parsed.rows, parsed.cols
        start, exit_cell = parsed.start, parsed.exit
        if start is None or exit_cell is None:
            return ["反解结果缺少起点或出口"]
        path = grid.shortest_path(mask, rows, cols, start, exit_cell)
        dist_exit = grid.distances(mask, rows, cols, exit_cell)
        current = parsed.current if parsed.current is not None else start
        path_set = set(path or [])

        for qid, question in sample["questions"].items():
            family = "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")
            answer = sample["answers"][qid]
            instructions = question["instructions"]
            text = instructions["question"] if isinstance(instructions, dict) else instructions
            expected = self._expected(
                family,
                question,
                text,
                parsed,
                loc,
                path,
                path_set,
                dist_exit,
                current,
                start,
                exit_cell,
            )
            if expected is None:
                continue
            actual = self._actual(question, answer)
            if actual != expected:
                problems.append(f"{qid}({family}): 数据里是 {actual!r}，独立复算是 {expected!r}")
        return problems

    @staticmethod
    def _actual(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> object:
        """把答案取出成可比较的形态：choice → 选项文案；score/noul → 原值。"""
        if question["type"] == "choice":
            return list(question["criteria"])[answer["choice"]]
        if question["type"] == "score":
            return answer["score"]
        return answer["noul"]

    def _expected(
        self,
        family: str,
        question: dict[str, JsonValue],
        text: str,
        parsed: ParsedMaze,
        loc: dict[str, JsonValue],
        path: list[int] | None,
        path_set: set[int],
        dist_exit: list[int],
        current: int,
        start: int,
        exit_cell: int,
    ) -> object | None:
        """独立复算某题的期望答案；无法判定时返回 None（跳过）。"""
        mask, rows, cols = parsed.mask, parsed.rows, parsed.cols
        letters = [ch for ch in text if ch in "ABCD"]
        letter = letters[0] if letters else None
        dir_idx = next((i for i, w in enumerate(loc["moves"].values()) if w in text), None)
        if family in ("next_step", "off_by_one"):
            if dist_exit[current] < 0:
                return None
            options = list(question["criteria"])
            best = [
                d
                for d, nb in grid.open_neighbors(mask, current, rows, cols)
                if dist_exit[nb] == dist_exit[current] - 1
            ]
            if len(best) != 1:
                return None
            word = list(loc["moves"].values())[best[0]]
            hit = [o for o in options if o.startswith(word)]
            return hit[0] if len(hit) == 1 else None
        if family == "path_via":
            on_path = [l for l, c in parsed.marks.items() if c in path_set]
            if len(on_path) > 1:
                return None
            return on_path[0] if on_path else loc["fam"]["path_via"]["none"]
        if family == "step_result":
            if letter is None or dir_idx is None:
                return None
            vocab = loc["fam"]["step_result"]["options"]
            cell = parsed.marks[letter]
            nxt = grid.neighbor(cell, dir_idx, rows, cols)
            if nxt is None:
                return vocab[4]
            if not grid.has_passage(mask, cell, dir_idx, rows, cols):
                return vocab[0]
            if nxt == start:
                return vocab[2]
            if nxt == exit_cell:
                return vocab[3]
            return vocab[1]
        if family == "can_move":
            if letter is None or dir_idx is None:
                return None
            return grid.has_passage(mask, parsed.marks[letter], dir_idx, rows, cols)
        if family == "dead_end":
            if letter is None:
                return None
            return len(grid.open_neighbors(mask, parsed.marks[letter], rows, cols)) == 1
        if family == "on_path":
            if letter is None:
                return None
            return parsed.marks[letter] in path_set
        if family == "reachable":
            return dist_exit[start] >= 0
        if family == "on_optimal":
            return current in path_set
        if family == "deviation":
            if not path_set:
                return None
            return min(_dist_to_set(mask, rows, cols, path_set)[current], 3)
        if family == "path_length":
            if path is None:
                return None
            steps = len(path) - 1
            return next(
                idx
                for idx, (low, high) in enumerate(maze_spec.PATH_LENGTH_BUCKETS)
                if low <= steps <= high
            )
        if family == "exits":
            if letter is None:
                return None
            count = len(grid.open_neighbors(mask, parsed.marks[letter], rows, cols))
            return 0 if count <= 1 else (1 if count <= 2 else 2)
        return None
