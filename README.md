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

## 产物格式

一条记录就是 `train.jsonl` 里的一行（**一份记录带整组题**，同组样本共享 `state`）：

| 字段 | 是什么 |
|---|---|
| `id` | `{workflow}-{切分}-{序号:07d}`，一份产物里唯一 |
| `state` | 题面上下文（渲染出来的那一屏；模型写的部分也在里面） |
| `questions` | 题号 → 题目对象，题号就是题族名（同族多题时带后缀） |
| `questions.<题号>.type` | `choice` / `noul` / `score` |
| `questions.<题号>.instructions` | 问法：string，或按语言 / 问法轮转的 object / array |
| `questions.<题号>.criteria` | **选项表**——`choice` 是「选项名 → 描述」的对象；`score` 是等级描述数组；`noul` 可省略，写了只允许 `true` / `false` 两个键 |
| `answers.<题号>` | `{"type": "choice", "choice": 编号}` / `{"type": "noul", "noul": true}` / `{"type": "score", "score": 3}` |

两条容易踩的口径：

- **描述可以缺**：`choice` 的某个选项写成 `null` 就是「这个选项没有描述」（`maze` 的格子选项
  全是 `null`）；`score` 的等级描述**不允许** `null`（等级描述就是等级定义）。
- **编号按这一条的 `criteria` 现读**：`answers.choice` 是选项在 `criteria` 里的**书写顺序**
  下标（从 0 开始）。有的 workflow 会按样本打乱选项顺序，所以别跨条复用下标、别按自己重排的
  顺序去数。`score` 的答案同样是等级下标，从 0 开始。

一个真实例子（`dataset/rt-36c-zh/train.jsonl` 第 1 行，`state` 截断）：

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

一份产物目录里还有几份账：`manifest-<切分>.json`（这一份产物的声明与统计）、
`calls-<切分>.jsonl`（逐条来源：provider / 模型 / key 指纹 / 重试次数 / token）、
`stats-<切分>.md`（统计的人读版）、`progress-<切分>.jsonl`（断点续跑的进度日志）。
`rejected.jsonl` 是被判别器拒掉的草稿、`judge-skipped.jsonl` 是端点故障导致的「没判成」，
两者**都不进 `train.jsonl`**。

## 目录结构

```
src/jc/          通用层：管线、契约校验、判别器接口、后端（instructor）、覆盖器、审计
workflows/<名>/  一个 case 一个 workflow：features / spec / render / questions / judges / audit
tests/           全库测试（含两条零 token 的端到端扫描）
docs/            对外文档：数据格式、工作流目录、调研
docs_local/      维护者笔记：质量体检、均衡账、多样性、人工抽检（**不入库**，随生成重算）
```

分层规矩：`src/jc/` **不认识任何具体 workflow**（不出现领域词表、不 import `workflows`），
workflow 靠「导入即注册」挂上来（`jc workflows` 列名单）。加一个新 workflow 不用改通用层一行。

## workflow 目录（51 个）

「构造」= 答案由代码算出来，**零 token**；「注入」= 结构与判据由代码写、**文本由后端写**。

| workflow | 覆盖的 case | 生成 | 判别器 |
|---|---|---|---|
| `adblock` | 广告元素判定（判据表与档位表由代码写、HTML 片段由模型写） | 注入 | frame / hygiene / trace |
| `alertrisk` | 安全警报风险定级（因子表、定级矩阵与处置档位表由代码写、一屏警报记录由模型写） | 注入 | frame / hygiene / trace |
| `autoformat` | 结构恢复（结构由代码定、句子由模型写） | 注入 | hygiene / roundtrip / trace |
| `bench` | 评测与基准（一屏记录与全部判断） | 构造 | label / roundtrip |
| `browser` | 浏览器 / GUI 操作 | 注入 | frame / hygiene / trace |
| `cachereuse` | 缓存复用判定（两条操作轨迹逐步对齐，可复用长度是前缀） | 构造 | roundtrip / label |
| `causechain` | 根因定位（依赖图、判读规则与事件形状表由代码写、六个服务的现场日志由模型写） | 注入 | hygiene / frame / trace |
| `chatgate` | 聊天回复门控（判据表、毛病表、档位表由代码写、一条消息与三条候选由模型写） | 注入 | frame / hygiene / trace |
| `citation` | 引文核查（文档骨架由代码写、短文正文由模型写） | 注入 | frame / hygiene / trace |
| `clipcut` | 片段筛选与去冗余（片段表与高亮标记由代码写，保留规则先到先得） | 构造 | roundtrip / label |
| `compact` | 会话上下文压缩（处置表与位置行由代码写） | 注入 | frame / hygiene / trace |
| `debugstep` | 调试器下一步动作（判据表与处置档位表由代码写、一屏调试会话由模型写） | 注入 | frame / hygiene / trace |
| `depcheck` | 依赖与许可证风险门控（标记表、定级规则与处置档位表由代码写、一屏依赖记录由模型写） | 注入 | frame / hygiene / trace |
| `elementpick` | 界面元素定位（一屏元素清单与全部判断，动作落在哪个元素上） | 构造 | label / roundtrip |
| `findtriage` | 改动引入性判定（改动记录与判定口径由代码写、那条发现与修法由模型写） | 注入 | frame / hygiene / trace |
| `functions` | 函数调用（工具表与全部答案） | 构造 | label / roundtrip |
| `guardrail` | 内容审核（条款表与阈值表由代码写、待审正文由模型写） | 注入 | frame / hygiene / trace |
| `hirecheck` | 招聘初筛门禁（判据表、定级规则与处置档位表由代码写、一屏候选人材料由模型写） | 注入 | frame / hygiene / trace |
| `hybrid` | 会议日程（构造骨架 + 模型补表述） | 混合 | answers / skeleton / text |
| `invoice` | 单据字段抽取（答案从素材里抽出来） | 构造 | label / roundtrip |
| `lint` | 行级规则检查（规则表由代码写、片段原文由模型写） | 注入 | frame / hygiene / trace |
| `logtriage` | 日志行分诊（噪声判据表与处置档位表由代码写、日志行由模型写） | 注入 | frame / hygiene / trace |
| `mail` | 邮件安全分诊（标签先行，模型只写文本外壳） | 注入 | hygiene / link / link_text / sender_identity / trace |
| `match` | 记录匹配（两条记录、八个答案） | 构造 | label / roundtrip |
| `maze` | 迷宫（导航、连通性、局部结构） | 构造 | label / roundtrip |
| `memorygate` | 记忆写入门控（入库判据表与处置档位表由代码写、候选记忆由模型写） | 注入 | frame / hygiene / trace |
| `modelroute` | 模型路由与降级（候选表与门槛由代码写） | 注入 | frame / hygiene / trace |
| `planstep` | 有界任务推进（任务表与四条推进规则由代码写，下一步取决于前置是否全部完成） | 构造 | roundtrip / label |
| `promptinject` | 提示注入检测（判据表与阈值表由代码写、不可信内容由模型写） | 注入 | frame / hygiene / trace |
| `posttriage` | 社交帖子分诊（信号判据表与处置阈值表由代码写、一条帖子由模型写） | 注入 | frame / hygiene / trace |
| `quoteread` | 行情读数判读（读数表、单位换算表与档位表由代码写，先换算成基准点再比档） | 构造 | roundtrip / label |
| `receiptcat` | 报销收据归类（归类表、单笔上限表与审核档位表由代码写、一条收据行由模型写） | 注入 | hygiene / frame / trace |
| `rerank` | 检索重排（排序口径与候选编号由代码写） | 注入 | frame / hygiene / trace |
| `review` | 改动审查（清单行与三张表由代码写） | 注入 | frame / hygiene / trace |
| `routing` | 工单分诊（属性行与三张表由代码写） | 注入 | frame / hygiene / trace |
| `rubric` | 打分与评分（材料清单与三张表由代码写） | 注入 | frame / hygiene / trace |
| `search` | 检索结果判读（规则表与来源行由代码写） | 注入 | frame / hygiene / trace |
| `segmentation` | 文档包切分（种类表、切分规则与页序抬头由代码写、逐页正文由模型写） | 注入 | hygiene / frame / read |
| `sentiment` | 情感与情绪（情绪词表与四条判读规则由代码写，符号、转折位置、落差与回升全算得出来） | 构造 | roundtrip / label |
| `shell` | 命令审批（bash / PowerShell 两种方言） | 注入 | destructive / dialect / elevated / hygiene / trace |
| `siftrank` | 多批次排名聚合（四个批次的排名聚合成一条相关性顺序，肘部切出相关条目） | 构造 | roundtrip / label |
| `sim` | 游戏与模拟控制（一屏局面与全部判断） | 构造 | label / roundtrip |
| `skillpick` | 技能选择（一屏选择单与全部判断） | 构造 | label / roundtrip |
| `sourceselect` | 来源选择（候选表与选源口径由代码写、那句查询由模型写） | 注入 | frame / hygiene / trace |
| `sqlreview` | 数据库变更评审（判据表与处置档位表由代码写、一屏变更记录由模型写） | 注入 | frame / hygiene / trace |
| `subqa` | 字幕对质检（判据表与处置档位表由代码写、一条双语字幕对由模型写） | 注入 | frame / hygiene / trace |
| `testimpact` | 测试影响选择（覆盖表与跑法档位表由代码写、一次改动由模型写） | 注入 | hygiene / frame / trace |
| `toolgate` | 工具调用门控（策略表与状态行由代码写） | 注入 | frame / hygiene / trace |
| `tracescore` | 代理轨迹自评（行为信号表与处置档位表由代码写、一段执行轨迹由模型写） | 注入 | hygiene / frame / trace |
| `turntake` | 回合结束判定（未完判据表与处置档位表由代码写、用户输入由模型写） | 注入 | frame / hygiene / trace |
| `verify` | 完成声明核验（事实与门槛由代码写） | 注入 | frame / hygiene / trace |

十四个纯构造 workflow（`bench` / `cachereuse` / `clipcut` / `elementpick` / `functions` / `invoice` / `match` / `maze` / `planstep` / `quoteread` / `sentiment` / `siftrank` / `sim` / `skillpick`）
声明 `needs` 为空——管线不会初始化任何后端，**全程不调模型**。

每个 workflow 都出 `train` / `val` / `test_iid` / `test_ood` 四个切分，语言有 `zh` 与 `en`。
`test_ood` 是留出侧（留出某个取值或整个维度，取决于 workflow）。

## 怎么跑

```powershell
# 通用层与纯构造 workflow（14 家）只用标准库：`pip install -e .` 就能跑 `jc workflows` /
# `jc gen` / `jc audit`（走 LLM 的那 37 家会被按「缺可选依赖 [llm]」跳过并给出装法）。
.venv\Scripts\python.exe -m pip install -e ".[llm,dev]"

# 列 workflow / 语言
python -m jc workflows
python -m jc locales --workflow maze

# 看 case 矩阵（不生成）
python -m jc cases --workflow maze --split test_ood

# 零 token 生成（纯构造 workflow）
python -m jc gen --workflow maze --split train --locale zh --out dataset --run-id maze-train

# 审计已生成的产物（跨行 id 唯一性、答案与文本自洽、题族齐全、文本与声明语言一致…）
python -m jc audit --workflow maze --dir dataset/maze-train --split train
```

需要后端的 workflow 多一个 `--backend-config`：指向一份 TOML，声明各 provider 与
`keys_env`（**从哪个变量名读 key**）。CLI 会从进程环境读那个变量；本仓库的小批量脚本走
Python API，把 key 读进内存后 `build_backend(config, env=...)` 注入——**key 不落环境变量、
不进仓库、不进日志**。配置文件里也支持多 key 轮转、限速、熔断、以及多家 provider 的降级链。

**这份 TOML 的形状见仓库根的 `backends.example.toml`**：每一格都带注释，里面只有环境变量名、
没有任何 key，所以它本身可以入库。最短的一条路：

```powershell
copy backends.example.toml tmp\backends.toml     # 改成你要连的端点（tmp/ 本来就不入库）
python -m jc gen --workflow shell --backend-config tmp/backends.toml --env-file tmp/keys.env --run-id my-run
```

`tmp/keys.env` 是 dotenv 风格（`EXAMPLE_API_KEY=key1,key2`，多 key 用逗号分隔），**不入库**；
不给 `--env-file` 时改读进程环境。链上某一家没有 key 会被**自动跳过**——撤下凭证不用改配置文件。
`backends.example.toml` 本身由 `tests/core/backend/test_config.py` 盯着：示例里的键必须是
当前 schema 认的键，改了 schema 忘了改示例，那条测试就红。

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
两把扫描共用 `tests/workflows/_contract.py` 里四条断言：产物必须过 `jc.validate`；用
**每一种语言的口径**审计每一份产物都要零缺失；把两种语言的产物拼成一份（一次
`locales=("zh","en")` 的 run 就是这形状）再审一遍也要零缺失——**审计的判定不许依赖调用方
给的语言**；走 LLM 的 workflow 必须**真报出**它的自然语言正文（空表 = 语言判据对它永远空转），
纯构造的必须报空表（报了就会把迷宫图 / 命令行当坏数据）。workflow 自己的 `audit` 查领域事实
（题族齐不齐、答案与文本自不自洽），通用契约查格式（题面 / 答案 / 编号 / `type` / `criteria`
形态、一行一道题、id 唯一），两者互补。

`jc audit` 还查一条**通用**的：产物文本必须与它**声明的**语言一致（`manifest-<切分>.json`
记的语言，不是调用方给的 `--locale`）。哪些字段算「自然语言正文」由各 workflow 自己声明
（`Workflow.language_texts`：代码 / 命令行 / ASCII 图一律不报），通用层只提供判据
（`jc.language`：en 侧按**字母**比（汉字占比 ≤ 0.30），zh 侧按**词**比
（汉字 / (汉字 + 拉丁词) ≥ 0.20）——zh 侧曾按字母比，会把「一句中文夹一串标识符」误杀，
见 `docs_local/可靠合成数据范式.md` §9.41 与 §9.98）。零 token，查的是已经落盘的产物——真机上 `rerank` 的 en 侧整段写成中文、`citation`
的 en 侧有一条写成中文文档，此前就是这么溜进去的。

**生成期也堵**：`jc.pipeline` 的 `RunConfig.language_gate`（默认开）在组装完一条样本之后，
拿同一个判据再查一遍**这条**样本声明的语言；不一致就抛 `DraftError`，走「判别器拒稿 → 重渲染」
那条路，重渲染用完仍不一致才不入库。区别在时机：审计是**事后**兜底（产物已经落盘），闸是
**当场**拦（写不进去）。关掉它只有一个正当理由：夹具（假后端）本身不按语言造文本——那是测试的
局限，不是产物的豁免，所以只有端到端测试里那些假后端会关（名单与口径见
`tests/workflows/_contract.py` 的 `LOCALE_FAITHFUL_FAKES`）。

**兜底之外还要在源头堵**：走 LLM 的 workflow，每种语言的提示词里都必须有一句明说的语言
约束（`全部用{language}写` / `Write everything in {language}`）——提示词通篇是目标语言只是
**上下文**，不给约束力，模型照样换语言（`rerank` 的 zh 侧曾 3/3、4/6、1/4 写成英文）。
加上这句之后，同一条 workflow、同一批后端的真机复核里两侧的汉字占比从「0.000 ~ 0.573」变成
干净的 0.0 / 1.0。细节见范式 §9.42。

另有四把**毫秒级**的全库尺子（都在 `tests/core/`）：不许 `Any` 及其等价物（显式 `Any`、
裸泛型 `x: dict`、`Callable[..., X]`、没写元素类型的空容器、关掉标注检查的 `noqa`、
生产代码直接 `json.loads`）、生产代码不许 `locales.load("字面量")`（语言必须从调用方取）、
语言文件里两个题族不许共用同一句问法、走 LLM 的 workflow 每种语言的提示词里都得有那句
`{language}` 约束。它们已接进**本机**的 `.git/hooks/pre-commit`——`.git/hooks/` 不入库，
换机器要自己接一次。

## 设计纪律（代码结构保证的不变量）

- **编号 / 题号 / `type` 只由组装器产生**，后端只出草稿——模型不能决定答案存成什么形状；
- **判别器拒稿 → 重渲染；端点故障 → 记跳过**，不算数据问题（两者在产物里分开记账）；
- **断点续跑**：产物已有行、进度日志却为空时直接报人话停下，绝不静默从 0 重跑并追加；
- **同分布侧与 `test_ood` 侧**的取值留出由测试锁死，改取值表时会当场红。

## 文档

**对外**（`docs/`，给仓库外面的人看）：

- `docs/README.md`——文档索引；
- `docs/数据格式.md`——JSONL 契约：顶层字段、三种题型、编号口径、怎么校验；
- `docs/工作流清单.md`——51 家逐行：覆盖的 case、要不要后端、题型、题族、规模（现场算）；
- `docs/JEV生态与开源复现调研.md`——JEV 是什么、大家拿它干什么、开源复现全景、他们怎么合成；
- `docs/JEV场景覆盖地图.md`——把社区清单（965 条）按场景簇归堆，逐簇对到本仓库的 workflow，
  标「覆盖 / 部分 / 没覆盖」，缺口逐条给结论（含「不做」的理由）；
- `docs/黑盒API合成数据范式调研.md`——用闭源 API 合成数据时的范式与坑。

**维护者**（`docs_local/`，**不入库**——跟 `tmp/` 一样只在维护者本机；数字现场算、每次生成都会
重算，不适合当对外文档读）：

- `docs_local/项目现状.md`——一页说完：规模、结构、已验证什么、**未做什么**（不装成做完了）；
- `docs_local/可靠合成数据范式.md`——最厚的笔记：可靠性范式、踩过的坑与教训（按主题编号）；
- `docs_local/真机小批量记录.md`——每一批真机合成的账：收 / 拒、拒绝码、判别器、模型、key 指纹、
  逐条归属（哪条由哪个模型写的）；
- `docs_local/合成质量体检.md`——逐份产物的六项检查（契约 / 审计缺 / 冲突 / 语言红 / 复读 / 残留），
  硬闸不过的逐条给成因；
- `docs_local/均衡报告全量.md`——答案分布偏离均匀的账，含基准率说明与「结构上判不了的」清单；
- `docs_local/多样性度量.md`——批内文本多样性（distinct-n / 最坏重合）；
- `docs_local/人工抽检记录.md`——**人读**过的样本逐条判断（读了哪些、发现什么毛病，如实记）；
- `docs_local/合成质量抽查.md`——每家第一条真实样本的原文（「数据长什么样」看原文，不看形容词）；
- `docs_local/重复代码清单.md`——逐字同构的函数抄本，收掉了哪些、还剩哪些（现场算，不手写）。

生成 `docs_local/` 的脚本在 `tmp/`（不入库），每个文档头部都写了怎么重跑。
