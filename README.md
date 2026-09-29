# just-choice-data-pipeline

JEV 风格选择题的**合成数据管线**：一个通用层（`src/jc/`）+ 一组各自独立的 workflow
（`workflows/`）。产物是 JSONL，一条记录一道题，答案存**编号 / 布尔 / 整数**——不做概率、
不做置信度。

## 三原语与存储口径

| 题型 | 答案是什么 | 存法 |
|---|---|---|
| `choice` | 答案是**选项** | `choice` = 选项下标，**从 0 开始** |
| `noul` | 答案是 true / false | `noul` |
| `score` | 答案是整数评分（如 1~5 档） | `score` |

选项可以有描述，也可以没有（可选）。题号写在 `questions` 的 key 上（题族名 + 同族后缀），
`state` 是渲染出来的题面上下文，同组样本共享。上游 Open-Jev 存的是与选项对齐的概率向量，
本仓库刻意改存编号——下游要概率自己算，管线不替它猜。

## 目录结构

```
src/jc/          通用层：管线、契约校验、判别器接口、后端（instructor）、覆盖器、审计
workflows/<名>/  一个 case 一个 workflow：features / spec / render / questions / judges / audit
tests/           全库测试（含两条零 token 的端到端扫描）
docs/            调研与范式笔记
```

分层规矩：`src/jc/` **不认识任何具体 workflow**（不出现领域词表、不 import `workflows`），
workflow 靠「导入即注册」挂上来（`jc workflows` 列名单）。加一个新 workflow 不用改通用层一行。

## workflow 目录（24 个）

「构造」= 答案由代码算出来，**零 token**；「注入」= 结构与判据由代码写、**文本由后端写**。

| workflow | 覆盖的 case | 生成 | 判别器 |
|---|---|---|---|
| `autoformat` | 结构恢复（结构由代码定、句子由模型写） | 注入 | hygiene / roundtrip / trace |
| `bench` | 评测与基准（一屏记录与全部判断） | 构造 | label / roundtrip |
| `browser` | 浏览器 / GUI 操作 | 注入 | frame / hygiene / trace |
| `citation` | 引文核查（文档骨架由代码写、短文正文由模型写） | 注入 | frame / hygiene / trace |
| `compact` | 会话上下文压缩（处置表与位置行由代码写） | 注入 | frame / hygiene / trace |
| `functions` | 函数调用（工具表与全部答案） | 构造 | label / roundtrip |
| `guardrail` | 内容审核（条款表与阈值表由代码写、待审正文由模型写） | 注入 | frame / hygiene / trace |
| `hybrid` | 会议日程（构造骨架 + 模型补表述） | 混合 | answers / skeleton / text |
| `invoice` | 单据字段抽取（答案从素材里抽出来） | 构造 | label / roundtrip |
| `lint` | 行级规则检查（规则表由代码写、片段原文由模型写） | 注入 | frame / hygiene / trace |
| `mail` | 邮件安全分诊（标签先行，模型只写文本外壳） | 注入 | hygiene / link / link_text / sender_identity / trace |
| `match` | 记录匹配（两条记录、八个答案） | 构造 | label / roundtrip |
| `maze` | 迷宫（导航、连通性、局部结构） | 构造 | label / roundtrip |
| `modelroute` | 模型路由与降级（候选表与门槛由代码写） | 注入 | frame / hygiene / trace |
| `rerank` | 检索重排（排序口径与候选编号由代码写） | 注入 | frame / hygiene / trace |
| `review` | 改动审查（清单行与三张表由代码写） | 注入 | frame / hygiene / trace |
| `routing` | 工单分诊（属性行与三张表由代码写） | 注入 | frame / hygiene / trace |
| `rubric` | 打分与评分（材料清单与三张表由代码写） | 注入 | frame / hygiene / trace |
| `search` | 检索结果判读（规则表与来源行由代码写） | 注入 | frame / hygiene / trace |
| `shell` | 命令审批（bash / PowerShell 两种方言） | 注入 | destructive / dialect / elevated / hygiene / trace |
| `sim` | 游戏与模拟控制（一屏局面与全部判断） | 构造 | label / roundtrip |
| `skillpick` | 技能选择（一屏选择单与全部判断） | 构造 | label / roundtrip |
| `toolgate` | 工具调用门控（策略表与状态行由代码写） | 注入 | frame / hygiene / trace |
| `verify` | 完成声明核验（事实与门槛由代码写） | 注入 | frame / hygiene / trace |

七个纯构造 workflow（`bench` / `functions` / `invoice` / `match` / `maze` / `sim` / `skillpick`）
声明 `needs` 为空——管线不会初始化任何后端，**全程不调模型**。

每个 workflow 都出 `train` / `val` / `test_iid` / `test_ood` 四个切分，语言有 `zh` 与 `en`。
`test_ood` 是留出侧（留出某个取值或整个维度，取决于 workflow）。

## 怎么跑

```powershell
# 通用层与纯构造 workflow 只用标准库；装 LLM 后端要带 [llm]
.venv\Scripts\python.exe -m pip install -e ".[llm,dev]"

# 列 workflow / 语言
python -m jc workflows
python -m jc locales --workflow maze

# 看 case 矩阵（不生成）
python -m jc cases --workflow maze --split test_ood

# 零 token 生成（纯构造 workflow）
python -m jc gen --workflow maze --split train --locale zh --out dataset --run-id maze-train

# 审计已生成的产物（跨行 id 唯一性、答案与文本自洽、题族齐全…）
python -m jc audit --workflow maze --dir dataset/maze-train --split train
```

需要后端的 workflow 多一个 `--backend-config`：指向一份 TOML，声明各 provider 与
`keys_env`（**从哪个变量名读 key**）。CLI 会从进程环境读那个变量；本仓库的小批量脚本走
Python API，把 key 读进内存后 `build_backend(config, env=...)` 注入——**key 不落环境变量、
不进仓库、不进日志**。配置文件里也支持多 key 轮转、限速、熔断、以及多家 provider 的降级链。

`--render-attempts`（默认 2）是「判别器拒掉就重渲染」的次数；`--judges` 可挂 LLM 质检视角
（`llm:solvable` / `llm:answer` / `llm:distractors` / `llm:levels` / `llm:wording`）。
`--balance-tolerance`（默认 0.03）是「某个取值的占比偏离均匀」的上限，超了只报警并写进
统计；要它直接让整批失败就加 `--balance-strict`。

## 零 token 的验证闸（不需要任何 key）

```powershell
python -m pytest -q          # 全库
```

其中两条是**全语言 × 全切分端到端**扫描，专门钉「只测一种语言」与「只测 `train`」这两个盲区：

- `tests/workflows/test_construct_sweep.py`：每个纯构造 workflow × 每种语言 ×（`train` /
  `test_ood`）各跑一遍，断言零拒稿、零审计缺失、产物非空、**各语言出的条数一致**、过通用契约；
- `tests/workflows/test_llm_locale_sweep.py`：每个走 LLM 的 workflow × 每种语言 ×（`train` /
  `test_ood`），用各自的**假后端**（按 `task.spec` 现造草稿，不发网络）跑端到端，断言同上。

两条轴都要覆盖，是因为它们互相独立：`test_ood` 是**另一套 case**（留出取值 / 尺寸 / 格式），
跑过 `train` 不蕴含跑过它；而渲染出来的文本又是**语言相关**的，只跑一种语言等于留盲区。
两把扫描共用 `tests/workflows/_contract.py` 里三条断言：产物必须过 `jc.validate`；用
**每一种语言的口径**审计每一份产物都要零缺失；把两种语言的产物拼成一份（一次
`locales=("zh","en")` 的 run 就是这形状）再审一遍也要零缺失——**审计的判定不许依赖调用方
给的语言**。workflow 自己的 `audit` 查领域事实（题族齐不齐、答案与文本自不自洽），
通用契约查格式（题面 / 答案 / 编号 / `type` / `criteria` 形态、一行一道题、id 唯一），
两者互补。

另有三把**毫秒级**的全库尺子（都在 `tests/core/`）：不许显式 `Any`、生产代码不许
`locales.load("字面量")`（语言必须从调用方取）、语言文件里两个题族不许共用同一句问法。
它们已接进**本机**的 `.git/hooks/pre-commit`——`.git/hooks/` 不入库，换机器要自己接一次。

## 设计纪律（代码结构保证的不变量）

- **编号 / 题号 / `type` 只由组装器产生**，后端只出草稿——模型不能决定答案存成什么形状；
- **判别器拒稿 → 重渲染；端点故障 → 记跳过**，不算数据问题（两者在产物里分开记账）；
- **断点续跑**：产物已有行、进度日志却为空时直接报人话停下，绝不静默从 0 重跑并追加；
- **同分布侧与 `test_ood` 侧**的取值留出由测试锁死，改取值表时会当场红。

## 文档

- `docs/可靠合成数据范式.md`——本项目最厚的笔记：可靠性范式、踩过的坑与教训（按主题编号）；
- `docs/JEV生态与开源复现调研.md`——JEV 是什么、大家拿它干什么、开源复现全景、他们怎么合成；
- `docs/黑盒API合成数据范式调研.md`——用闭源 API 合成数据时的范式与坑。
