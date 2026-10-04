"""留出侧的题族不许塌：整族消失 / 塌成常量，两条都在这里钉住（零 token）。

为什么要有：范式 §9.79 第六条。留出某一维时，**依赖它的那一族会在留出侧塌**，而同分布侧的
分布看起来完全正常——`elementpick` 那次只有真机 `--balance-strict` 才照出来（单元测试因为
小配置的标签池凑巧把偏差压到噪声地板下而全绿）；`clipcut` 那次是 `dup_any` 在留出侧 8 条
全 `True`（留出侧只有一条主题序列，而它自带重复）。两次都是「只看同分布侧看不出来」。

口径：**走组装器**（`jc.assemble`），与 `jc.pipeline` 的答案统计是同一把尺（choice 记的是
打乱后的编号，不是选项名）。`ask` 按协议是纯代码推导，所以注入式 workflow 也能零 token 扫；
state 用占位串——组装不看它。

两条硬要求：
1. **整族消失**：同分布侧出过的题族，留出侧不许一条都不出；
2. **塌成常量**：同分布侧有两个以上答案的题族，留出侧也不许只剩一个答案（样本量太小的不算，
   见 `MIN_N`）。

两条收在一条测试里、循环扫名单：`buckets_of` 每个 workflow 要跑一遍全切分，原来两条参数化
用例各跑一遍等于把同一份扫描做两遍（98 条用例、两倍时间）；现在每个 workflow 只扫一次，
两条要求一起判，问题攒在 `bad` 里一起报。
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Final

from jc import assemble, workflow as workflow_mod
from jc.workflow import family_of

#: 每个切分最多扫多少个 case（大 case 矩阵只取前 N 个，够看出结构性问题）。
CASE_CAP: Final[int] = 400
#: 样本量小于它的族不做「常量」判断（小样本的常量多半是抽样）。
MIN_N: Final[int] = 8
#: 扫不动的 workflow：它们的 `ask` 依赖**渲染出来的文本**（文本由后端写），没有后端就没法出题。
NEEDS_RENDER: Final[frozenset[str]] = frozenset({"autoformat", "hybrid"})
#: 允许在留出侧塌成常量的 workflow：**留出维与题族一一对应**时，留出那个取值必然让该族在留出侧
#: 只剩一个答案。`receiptcat` 留出的是 `decoy=near`，而 `near` 那一族问的正是「有没有诱饵」——
#: 换任何一维当留出维都一样（`category` ↔ `cat`、`band` ↔ `tier`）。这种塌陷不是渲染 bug，
#: 而是「拿这一维当留出维」的必然结果；它仍然测试「留出侧不许答成另一侧」。
CONSTANT_ON_OOD_OK: Final[frozenset[str]] = frozenset({"receiptcat"})

#: 注册表**在导入期快照**，不在用例体里现场 `discover()`：注册表是全局可变的，
#: `tests/core/test_workflow.py::test_register_rejects_duplicate` 会往里塞一个
#: `dummy-for-test`（它只有 `name`，没有 `available_locales`）。现场 discover() 会让这个假
#: workflow 混进扫描名单，跑全套时当场 AttributeError——单跑这个文件却看不见（是顺序问题）。
DISCOVERED: Final[tuple[str, ...]] = tuple(workflow_mod.discover())

#: 能零 token 扫的 workflow 名单（从上面的快照里减去 `NEEDS_RENDER`）。
SCANNABLE: Final[tuple[str, ...]] = tuple(n for n in DISCOVERED if n not in NEEDS_RENDER)


def buckets_of(name: str) -> dict[str, dict[str, Counter[str]]]:
    """输入：workflow 名。输出：{切分: {`题型:题族`: 答案编号计数}}。"""
    wf = workflow_mod.load(name)
    locale = wf.available_locales()[0]
    rng = random.Random(20260930)
    out: dict[str, dict[str, Counter[str]]] = {}
    for split in wf.split_names():
        config = wf.config_for(split)
        stats: dict[str, Counter[str]] = {}
        for seq, case in enumerate(wf.build_cases(config)[:CASE_CAP]):
            spec = wf.sample(rng, wf.plan_sample(rng, config, case, seq))
            sample = assemble.assemble_sample("probe", "", wf.ask(spec, rng, locale), rng)
            for qid, question in sample["questions"].items():
                answer = sample["answers"][qid]
                value = str(answer.get("choice", answer.get("score", answer.get("noul"))))
                stats.setdefault(f"{question['type']}:{family_of(qid)}", Counter())[value] += 1
        out[split] = stats
    return out


def test_the_scan_covers_the_catalogue() -> None:
    """输入：无。输出：无。

    预期行为：扫到的 workflow 数与注册表只差 `NEEDS_RENDER` 那几个——否则一把「全库闸门」
    会悄悄退化成「只看几家」。
    """
    assert set(SCANNABLE) | NEEDS_RENDER == set(DISCOVERED)
    assert len(SCANNABLE) >= 40


def test_the_held_out_side_neither_loses_nor_flattens_a_family() -> None:
    """输入：无（名单现场列）。输出：无。

    预期行为（两条一起判，每个 workflow 只扫一遍）：

    1. 同分布侧出过的题族，留出侧一条都不许少——留出侧少一族，意味着那一族在「没见过的取值」
       上根本没有题，留出集就白留了；
    2. 同分布侧出现过两个以上答案、且留出侧样本量够大的族，留出侧也不许只剩一个答案——那一族
       在留出侧就一条信息都不带了（`clipcut/dup_any` 就是这么塌的）。`CONSTANT_ON_OOD_OK` 里
       那几家跳过第二条（留出维与题族一一对应，塌成常量是设计后果，见文件头注释）。

    失败信息攒在 `bad` 里一起报，不因为第一家炸掉就看不见其余。
    """
    bad: list[str] = []
    for name in SCANNABLE:
        table = buckets_of(name)
        id_splits = [split for split in table if split != "test_ood"]
        if not id_splits:
            bad.append(f"{name} 没有同分布切分")
            continue
        id_families = set().union(*(set(table[split]) for split in id_splits))
        gone = sorted(id_families - set(table["test_ood"]))
        if gone:
            bad.append(f"{name} 的 test_ood 侧少了这些题族：{gone}")
        if name in CONSTANT_ON_OOD_OK:
            continue
        collapsed: list[str] = []
        for key, bucket in sorted(table["test_ood"].items()):
            if len(bucket) > 1 or sum(bucket.values()) < MIN_N:
                continue
            id_values: set[str] = set()
            for split in id_splits:
                id_values |= set(table[split].get(key, {}))
            if len(id_values) > 1:
                collapsed.append(
                    f"{key}（留出侧 {sum(bucket.values())} 条全是 {sorted(bucket)[0]}）"
                )
        if collapsed:
            bad.append(f"{name} 的留出侧塌成常量：{collapsed}")
    assert not bad, bad
