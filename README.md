# just-choice-data-pipeline

批量合成「选择题」数据集：每道题给一段情景、一组选项和一个正确答案，答案只存**序号 / 对错 /
档位**，不存概率、不存置信度。

数据由 51 个互相独立的场景生成器（下面叫 workflow）产出，分两种做法：

- **纯代码生成**（14 个）——题目、选项、答案全部由代码算出来，**不调用任何模型**，跑一次不花钱；
- **模型写文本**（37 个）——题目结构、选项、判定标准由代码定死，只把那段情景描述交给大模型写。
  答案永远由代码决定，模型只负责把话说自然。

## 一条数据长什么样

三种题型（`choice` 多选一 / `noul` 判断对错 / `score` 打分）：

| 题型 | 问什么 | 答案是什么 | 答案字段 |
|---|---|---|---|
| 多选一 | 从若干选项里选一个 | 选中的那个选项 | `choice`：选项序号，**从 0 开始** |
| 判断对错 | 这句话对不对 | 是 / 否 | `noul`：`true` / `false` |
| 打分 | 打一个档位（例如 1~5 档） | 一个档位 | `score`：档位序号，**从 0 开始** |

选项可以带一句说明，也可以不带（写 `null`）。上游 Open-Jev 存的是每个选项的概率，本仓库刻意
改存序号——下游要概率自己算。

`*.jsonl` 里一行是一条记录，一条记录带**一整组题**，同组题共用同一段情景：

```json
{
  "id": "routing-train-0000000",
  "state": "【工单正文】请问在 DemoApp 中如何正确使用 XYZ 功能……\n【工单属性】（产品域 | 客户等级 | 合规标记）\n- analytics | enterprise | security\n…",
  "questions": {
    "issue": {
      "type": "choice",
      "instructions": "这条工单属于哪一类问题？",
      "criteria": {
        "bug": "缺陷：行为与文档写的不一样",
        "outage": "故障：某条线整体不可用",
        "howto": "咨询：问正确用法",
        "data_loss": "数据丢失：记录少了 / 对不上",
        "feature": "需求：想要一种新做法"
      }
    }
  },
  "answers": { "issue": { "type": "choice", "choice": 2 } }
}
```

上面这条的答案是 `choice: 2`，按 `criteria` 的书写顺序数是第三个选项 `howto`——不是按字母序，
也不是按你自己重排的顺序。

| 字段 | 含义 |
|---|---|
| `id` | 唯一编号，形如 `routing-train-0000000` |
| `state` | 情景描述（题面正文），同组题共用 |
| `questions` | 题号 → 题目：`type` 题型、`instructions` 问法、`criteria` 选项表 |
| `answers` | 题号 → 答案 |

一次生成产出一个目录：

```
<run-id>/
  train.jsonl          训练用
  val.jsonl            验证用
  test_iid.jsonl       同分布测试用
  test_ood.jsonl       留出测试用（换掉某个取值或整个维度）
  manifest-*.json      这一份产物的声明与统计
  calls-*.jsonl        每条样本由哪个模型、哪个 key 写的
  stats-*.md           统计的人读版
  progress-*.jsonl     断点续跑的进度
  rejected.jsonl       质检没过、被丢掉的草稿
  judge-skipped.jsonl  端点故障、没判成的样本
```

`rejected.jsonl` 与 `judge-skipped.jsonl` 里的东西**不会**出现在 `train.jsonl` 里。

## 目录结构

```
src/jc/          通用管线：格式规范、校验、质检规则接口、模型后端、命令行
workflows/<名>/  每个场景一个：题目结构、情景渲染、质检规则、领域校验
tests/           全库测试
docs/            对外文档
```

`src/jc/` 里不出现任何具体场景的词汇，也不 import `workflows/`；加一个新 workflow 不用改通用层。

## 51 个 workflow

「生成方式」三种取值：**构造** = 题面和答案全由代码算出来，不调模型；**注入** = 题目结构由代码
定死、只有情景文本交给模型写；**混合** = 两者都用。「质检规则」列是 `workflows/<名>/judges.py`
里那些检查器的名字。

| workflow | 造什么题 | 生成方式 | 质检规则 |
|---|---|---|---|
| `adblock` | 判断 HTML 片段里哪些元素是广告 | 注入 | frame / hygiene / trace |
| `alertrisk` | 给一条安全警报定风险等级、决定怎么处置 | 注入 | frame / hygiene / trace |
| `autoformat` | 从格式混乱的文本里恢复结构（标题 / 列表 / 代码块） | 注入 | hygiene / roundtrip / trace |
| `bench` | 读一屏评测记录，回答指标与结论 | 构造 | label / roundtrip |
| `browser` | 浏览器 / GUI 操作：这一步该点哪里 | 注入 | frame / hygiene / trace |
| `cachereuse` | 两条操作轨迹逐步比对，判断能复用多长的前缀 | 构造 | roundtrip / label |
| `causechain` | 从六个服务的日志里定位根因 | 注入 | hygiene / frame / trace |
| `chatgate` | 判断一条聊天回复能不能发出去 | 注入 | frame / hygiene / trace |
| `citation` | 核查引文与正文对不对得上 | 注入 | frame / hygiene / trace |
| `clipcut` | 按规则筛片段、去重复（先到先得） | 构造 | roundtrip / label |
| `compact` | 压缩会话上下文：哪些内容保留、哪些丢掉 | 注入 | frame / hygiene / trace |
| `debugstep` | 调试会话里下一步该做什么动作 | 注入 | frame / hygiene / trace |
| `depcheck` | 依赖与许可证风险门控 | 注入 | frame / hygiene / trace |
| `elementpick` | 在一屏界面元素里定位该操作哪一个 | 构造 | label / roundtrip |
| `findtriage` | 判断某条改动是不是这次问题引入的 | 注入 | frame / hygiene / trace |
| `functions` | 按工具表决定调用哪个函数、传什么参数 | 构造 | label / roundtrip |
| `guardrail` | 内容审核：这段正文违不违规 | 注入 | frame / hygiene / trace |
| `hirecheck` | 招聘初筛：这份候选人材料过不过门禁 | 注入 | frame / hygiene / trace |
| `hybrid` | 会议日程：骨架由代码算，表述由模型补 | 混合 | answers / skeleton / text |
| `invoice` | 从单据素材里抽出字段 | 构造 | label / roundtrip |
| `lint` | 按规则表逐行检查代码片段 | 注入 | frame / hygiene / trace |
| `logtriage` | 日志行分诊：这行是噪声还是信号 | 注入 | frame / hygiene / trace |
| `mail` | 邮件安全分诊（标签由代码定，文本由模型写） | 注入 | hygiene / link / link_text / sender_identity / trace |
| `match` | 两条记录之间做八项匹配判断 | 构造 | label / roundtrip |
| `maze` | 迷宫：导航、连通性、局部结构 | 构造 | label / roundtrip |
| `memorygate` | 判断一条记忆该不该写入 | 注入 | frame / hygiene / trace |
| `modelroute` | 选哪个模型、要不要降级 | 注入 | frame / hygiene / trace |
| `planstep` | 有界任务推进：下一步能做什么 | 构造 | roundtrip / label |
| `promptinject` | 检测不可信内容里的提示注入 | 注入 | frame / hygiene / trace |
| `posttriage` | 社交帖子分诊 | 注入 | frame / hygiene / trace |
| `quoteread` | 行情读数判读：先换算单位再比档 | 构造 | roundtrip / label |
| `receiptcat` | 报销收据归类与审核 | 注入 | hygiene / frame / trace |
| `rerank` | 把检索结果重排 | 注入 | frame / hygiene / trace |
| `review` | 审查一次代码改动 | 注入 | frame / hygiene / trace |
| `routing` | 工单分诊：这条工单属于哪类问题 | 注入 | frame / hygiene / trace |
| `rubric` | 按评分标准给材料打分 | 注入 | frame / hygiene / trace |
| `search` | 判断检索结果可不可信 | 注入 | frame / hygiene / trace |
| `segmentation` | 把一包文档按页序切分 | 注入 | hygiene / frame / read |
| `sentiment` | 从文本里判情绪（符号、转折、落差全算得出来） | 构造 | roundtrip / label |
| `shell` | 命令审批（bash 与 PowerShell 两种） | 注入 | destructive / dialect / elevated / hygiene / trace |
| `siftrank` | 把四批排名聚合成一条相关性顺序 | 构造 | roundtrip / label |
| `sim` | 游戏 / 模拟局面控制 | 构造 | label / roundtrip |
| `skillpick` | 从技能清单里挑技能 | 构造 | label / roundtrip |
| `sourceselect` | 选哪些来源可信、要不要补检索 | 注入 | frame / hygiene / trace |
| `sqlreview` | 评审一次数据库变更 | 注入 | frame / hygiene / trace |
| `subqa` | 双语字幕对质检 | 注入 | frame / hygiene / trace |
| `testimpact` | 一次改动该跑哪些测试 | 注入 | hygiene / frame / trace |
| `toolgate` | 工具调用门控：这次调用允不允许 | 注入 | frame / hygiene / trace |
| `tracescore` | 给一段代理执行轨迹自评 | 注入 | hygiene / frame / trace |
| `turntake` | 判断这轮对话该不该结束 | 注入 | frame / hygiene / trace |
| `verify` | 核验一句「已经做完了」的声明 | 注入 | frame / hygiene / trace |

十四个纯构造 workflow（`bench` / `cachereuse` / `clipcut` / `elementpick` / `functions` / `invoice` /
`match` / `maze` / `planstep` / `quoteread` / `sentiment` / `siftrank` / `sim` / `skillpick`）
不初始化任何模型后端，全程不调模型。

每个 workflow 都出 `train` / `val` / `test_iid` / `test_ood` 四个切分，语言有 `zh` 与 `en`。
`test_ood` 是留出侧：换掉某个取值或整个维度，用来看模型有没有真泛化。

## 怎么跑

```powershell
# 通用管线与 14 个纯代码 workflow 只用标准库；要跑走模型的 37 个就装 [llm]
.venv\Scripts\python.exe -m pip install -e ".[llm,dev]"

python -m jc workflows                                  # 列出全部 workflow
python -m jc locales --workflow maze                    # 列出支持的语言
python -m jc cases --workflow maze --split test_ood     # 看会出哪些题（只打印，不生成）

# 不调模型地生成一份数据
python -m jc gen --workflow maze --split train --locale zh --out dataset --run-id maze-train

# 校验已经生成的产物
python -m jc audit --workflow maze --dir dataset/maze-train --split train
```

走模型的 workflow 多一个 `--backend-config`，指向一份 TOML，里面写各家的端点和「从哪个环境变量
读 key」。CLI 从进程环境读那个变量；仓库里的小批量脚本走 Python API，把 key 读进内存后注入——
**key 不落环境变量、不进仓库、不进日志**。TOML 的形状见仓库根的
[backends.example.toml](backends.example.toml)：里面只有环境变量名、没有任何 key，所以可以入库。

```powershell
copy backends.example.toml tmp\backends.toml   # 改成你要连的端点（tmp/ 不入库）
python -m jc gen --workflow shell --backend-config tmp/backends.toml --env-file tmp/keys.env --run-id my-run
```

`tmp/keys.env` 是 dotenv 风格（`EXAMPLE_API_KEY=key1,key2`，多 key 用逗号分隔），不入库；不给
`--env-file` 就读进程环境。配置里支持多 key 轮转、限速、失败熔断、多家 provider 的降级链；链上
某一家没有 key 会被自动跳过，所以撤下凭证不用改配置文件。

几个常用开关：`--render-attempts`（默认 2）是质检不过时重写几次；`--judges` 可以再挂 LLM 质检
（`llm:solvable` / `llm:answer` / `llm:distractors` / `llm:levels` / `llm:wording`）；
`--balance-tolerance`（默认 0.03）是「某个答案的占比偏离均匀」的上限，超了只写进统计，要它直接
让整批失败就加 `--balance-strict`。

## 不花钱就能跑的检查

```powershell
python -m pytest -q
```

除了常规单测，还有几条值得知道：

- 两条**全语言 × 全划分**的端到端扫描：每个 workflow × `zh`/`en` ×（`train` / `test_ood`）各跑
  一遍，断言没有质检失败、没有校验缺失、产物非空、两种语言出的条数一致。走模型的那条用假后端，
  不发网络请求。
- 产物文本必须与它声明的语言一致。判断标准：英文侧看汉字占比（≤ 0.30），中文侧看汉字与拉丁词
  的比例（≥ 0.20）。
- 生成时也当场拦：写完一条样本立刻查一次语言，不一致就重写；重写次数用完仍不一致才不入库。
- 走模型的 workflow，每种语言的提示词里必须有一句明确的语言约束（`全部用{language}写` /
  `Write everything in {language}`）——通篇用目标语言只是上下文，不给约束力，模型照样会换语言。
- 四条毫秒级的全库检查：不许写 `Any` 及其等价物、生产代码不许把语言写成字面量、语言文件里两个
  题目分组不许共用同一句问法、提示词里必须有那句语言约束。它们挂在本机的
  `.git/hooks/pre-commit` 上（`.git/hooks/` 不入库，换机器要自己接一次）。

## 设计上的几条硬规矩

- **编号、题号、题型只由组装器产生**：模型只写草稿，不能决定答案存成什么形状；
- **质检不过就重写，端点故障就跳过**：两者在产物里分开记录，都不算数据质量问题；
- **断点续跑**：产物已有内容而进度记录为空时直接停下报错，绝不静默从头重跑并追加；
- **留出侧的取值由测试锁死**：改取值表会当场红灯。

## 文档

- [docs/数据格式.md](docs/数据格式.md)——数据格式规范：字段、三种题型、序号口径、怎么校验
- [docs/工作流清单.md](docs/工作流清单.md)——51 个 workflow 逐行（脚本从代码实时统计）
- [docs/JEV生态与开源复现调研.md](docs/JEV生态与开源复现调研.md)——JEV 是什么、有哪些开源复现
- [docs/JEV场景覆盖地图.md](docs/JEV场景覆盖地图.md)——社区用例与本仓库覆盖情况对照
- [docs/黑盒API合成数据范式调研.md](docs/黑盒API合成数据范式调研.md)——用闭源 API 批量合成数据的经验

维护者另有本机笔记（逐批合成记录、质量体检、答案分布、人工抽检等），放在 `docs_local/`，
**不入库**，不随仓库分发。

## 许可证

MIT，见 [LICENSE](LICENSE)。
