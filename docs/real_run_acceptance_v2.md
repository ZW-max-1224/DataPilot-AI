# DataPilot AI 真实运行验收报告 v2

**验收日期：**2026-10-10
**项目目录：**`C:\Users\35002\Documents\Codex\2026-10-04\DataPilot-AI`
**验收范围：**Phase 1–10 加上"Phase 10 后加固"之后的真实运行状态：离线测试、真实 LLM 端到端 Agent 运行、结构化结论、Agent 图表，以及仍未验证的 RAG / MySQL / LangSmith。
**与 v1 的关系：**`docs/real_run_acceptance.md` 是加固之前的快照（当时 149 tests、LLM 未配置、Agent 返回 demo mode）。本文件记录加固之后的真实状态，并延续 v1 的原则：**只把真正跑过的东西写成通过**。

---

## 一、总结

| 项目 | 状态 | 说明 |
|---|---|---|
| 离线测试 | **通过** | 全部通过，0 failed / 0 errors（含新增的 77+ 个回归测试） |
| 真实 LLM 单次调用 | **通过** | 配置真实 Key 后由 `LLMService` 实际调用并返回 |
| **真实 Agent 端到端** | **通过** | 6 个代表性中文提问全部成功，全部返回结构化结论 |
| Agent 图表链路 | **通过** | 模型选择图表参数，服务端 Matplotlib 绘图，图表随结果返回 |
| **Embedding → RAG 端到端** | **通过** | 配置 SiliconFlow `BAAI/bge-m3`（1024 维）后，`scripts/check_rag.py` 与 Agent 级验证全部通过，详见第五节 |
| MySQL 读写 | **未验证** | `MYSQL_USER` 为空时持久化整体关闭 |
| LangSmith Trace | **未验证** | 默认关闭，未产生真实 Trace |
| 代码修改 | 有 | 本次加固修改见 [code-review-and-next-steps.md](code-review-and-next-steps.md) 第八、九、十节 |

**没有使用 Mock 冒充真实服务。** 未验证的项目在上表中如实标注为"未验证"，UI 与文档都不会把它们显示成已通过。

---

## 二、真实 Agent 端到端验收

**验收方式：**用项目自带的 `AnalystAgent` + `LLMService`，对 `data/sample_sales.csv`（1,208 行 × 8 列）提问，统计成功率、结构化结论覆盖率和图表返回情况。

**验收问题（覆盖不同工具路径）：**

| 标签 | 提问 |
|---|---|
| trend | 分析一下最近几个月的销售趋势。 |
| why-drop | 为什么 8 月销售额下降？ |
| top10 | 找出销售额最高的 10 个产品。 |
| anomaly | 有没有异常数据？ |
| category | 哪个产品类别利润最高？ |
| chart | 画一张各月销售额的折线图。 |

**实际观测结果：**

| 提问 | 成功 | 决策轮数 | 调用工具 | 结构化结论 | 结论条数 | 建议条数 | 证据降级 | 返回图表 |
|---|---|---|---|---|---|---|---|---|
| trend | ✅ | 2 | calculate_time_series | ✅ | 5 | 3 | 0 | 0 |
| why-drop | ✅ | 4 | calculate_time_series, calculate_correlation | ✅ | 5 | 3 | 0 | 0 |
| top10 | ✅ | 2 | find_top_n | ✅ | 5 | 3 | 0 | 0 |
| anomaly | ✅ | 7 | detect_outliers, get_dataset_overview, analyze_missing_values | ✅ | 5 | 3 | 0 | 0 |
| category | ✅ | 2 | calculate_group_statistics | ✅ | 5 | 3 | 0 | 0 |
| chart | ✅ | 3 | calculate_time_series, create_chart | ✅ | 5 | 3 | 0 | **1** |

**成功率：6/6。** 每条结论都带有 `fact` / `interpretation` / `hypothesis` 标签，例如"8月销售额 36,514.67，较7月 104,126.96 减少约 67,612，为全年最低月"被标为 `fact` 并附证据工具；"可能是大额订单而非录入错误"被标为 `hypothesis`。**降级数 0** 说明没有出现"标为事实但缺少工具证据"的情况。

**加固前后的对比（同一台机器、同一 endpoint）：**

| 阶段 | 真实提问成功率 |
|---|---|
| 加固前 | **1/10** |
| 第一轮（参数通道 + JSON 修复 + 错误回灌 + 重试 + 部分结果） | 3/10 → 10/10 |
| 加入结构化结论后（第一版严格解析） | 4/6（回归，原因见下） |
| 修正解析策略后（最终） | **6/6** |

**关于 4/6 那次回归：**把 `claim_type` 写成严格枚举、要求严格的 `statement` 字段后，模型返回的等价变体（`claim_type: "conclusion"`、字段名用 `claim`）会让整个决策校验失败。修法是重新分配严格性——安全关键字段保持严格，展示字段做归一化解析——并补上"非空输入却解析出 0 条结论就告警"的日志。这个插曲本身是有价值的一课：**严格性要用在正确的地方，宽容解析必须配可观测性。**

---

## 三、诚实记录：已知的环境限制

1. **LLM endpoint 存在间歇性连接失败。** 在不同时间窗口用 8 次连续直接调用采样，曾观察到 5 次 `APIConnectionError`；另一轮采样中连接正常。因此"6/6"是这台机器、这个 endpoint、当次网络条件下的结果，**不等于永不失败**。
2. **失败时的行为已改变。** 现在任何终止路径（LLM 失败 / 工具连续失败 / 到达 8 轮上限 / JSON 无法修复）都会返回**已经实际计算出的证据**，并在 UI 标注"本次分析未完整结束"。加固前这些证据会被直接丢弃。
3. **`.env` 中的模型名需要你自行核对。** 本次验收使用的是 `.env` 中已配置的 endpoint 与模型名，它们不在 DeepSeek 官方公开的模型列表中，但实测可返回结果。演示前请到你自己的服务商控制台确认。

---

## 四、仍未验证的项目与验证方法

### 4.1 Embedding → RAG（**已验证通过**，方法留档）

配置 SiliconFlow `BAAI/bge-m3`（1024 维）后完成验证。**验证方法（可复现）：**

```powershell
cd "C:\Users\35002\Documents\Codex\2026-10-04\DataPilot-AI"

# 1) 先用自检脚本确认服务商真的可用（5 项：端点/维度/批量/确定性/语义有效性）
.\.venv\Scripts\python.exe scripts\check_embedding.py
#    实测：5/5 通过，近义 0.9515 vs 无关 0.4198

# 2) 再验证检索本身（索引 → 命中排名 → 排序区分度）
.\.venv\Scripts\python.exe scripts\check_rag.py
```

**实测结果（`check_rag.py`）：**索引 11 个切片（analysis_methods 4 + chart_selection 4 + metrics_glossary 3）；3 个相关问题全部**排名第 1** 命中（利润率→metrics_glossary 0.7728、图表选择→chart_selection 0.6709、相关性与因果→analysis_methods 0.7333）；相关问题最高分 **0.7728** vs 无关问题 **0.3109**。

**Agent 级实测（真实 LLM）：**纯知识问答 1/1 成功、知识+Pandas 组合 1/1 成功、知识库无关问题 1/1 成功，**编造来源 0 条**。

**为什么第 5 项（语义有效性）最重要：**如果端点返回随机或占位向量，前 4 项检查都会"通过"，但检索结果毫无意义。这条检查用来防止把"看起来配好了的假向量"写进知识库。

**关于"无相似度阈值"：**项目刻意不设未经校准的阈值，因此无关问题仍会返回低分切片（0.3109），`status` 是 `ok` 而非 `no_results`。`check_rag.py` 因此不把"无关问题必须返回 no_results"当通过条件——那只会逼着代码去编造阈值。判断答案是否可信要看 Sources 与切片内容。

**这次验证顺带发现并修复了一个真实缺陷：**知识库不可用时（`knowledge_service=None`），运行曾被标记为成功、`search_knowledge` 被记为已调用、步骤摘要称"获取结构化分析结果"，而模型其实是用自己的通用知识回答的——用户会误以为那是知识库证据。现在新增 `AgentResponse.retrieval_status`（`not_attempted`/`ok`/`no_results`/`unavailable`）、状态感知的步骤摘要、Prompt 明确要求、以及 UI 警告条。修复后真实复测，模型会主动声明"本次知识库检索不可用……以下内容不来自项目知识库，请勿当作知识库证据引用"。详见 [code-review-and-next-steps.md](code-review-and-next-steps.md) 第十节。

### 4.2 MySQL

留空 `MYSQL_USER` 时持久化关闭。验证方式：配置 `MYSQL_USER`/`MYSQL_PASSWORD` 后运行

```powershell
.\.venv\Scripts\python.exe -m app.database.init
```

再在页面加载数据集并提问一次，确认 `datasets`、`dataset_columns`、`analysis_runs` 三张表有记录。注意：**完整 CSV/XLSX 不会进入 MySQL**，只保存元数据与结构化画像。

### 4.3 LangSmith

把 `LANGSMITH_TRACING=true` 与 `LANGSMITH_API_KEY` 写入 `.env`，跑一次真实提问，然后在 `datapilot-ai` 项目里查看运行记录。Trace 只包含脱敏摘要（问题哈希与长度、行列数、工具名、参数结构类型、耗时），**不包含 Prompt/Completion 正文、文件名、字段名或 DataFrame 内容**。

---

## 五、本次验收执行的命令

```powershell
cd "C:\Users\35002\Documents\Codex\2026-10-04\DataPilot-AI"

# 全量离线测试（不联网、不需要 API Key）—— 当前为 244 passed
.\.venv\Scripts\python.exe -m pytest -q

# 静态检查
.\.venv\Scripts\python.exe -m ruff check .      # All checks passed
.\.venv\Scripts\python.exe -m mypy              # 33 errors（非阻断基线）

# 服务自检
.\.venv\Scripts\python.exe scripts\check_embedding.py
.\.venv\Scripts\python.exe scripts\check_rag.py

# 真实 Agent 验收（使用 .env 中真实配置，对 data/sample_sales.csv 提 6 个问题）
# 探针脚本位于项目之外，验收后已删除；等价做法是在页面 Agent 区域逐个提问。
```

**关于测试命令的一个坑（已在项目里修掉）：**`pyproject.toml` 一度设置了 `addopts = "-q"`，这会让文档里的 `pytest -q` 变成 `-qq`，从而**不打印 "N passed" 汇总行**，看起来像命令没结果。该配置已移除，现在 `pytest -q` 正常输出汇总行。如果你要机器可读的结果，可以用 JUnit XML：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=report.xml
# 然后读取 report.xml 的 tests / failures / errors 属性
```

另外，本项目的本地 Qdrant 客户端在解释器退出时会打印一条无害的 `QdrantClient.__del__` 警告（`sys.meta_path is None`），退出码仍是 0，也不影响任何测试结果。

**另一个已修掉的测试坑：**`tests/test_streamlit_ui.py` 原先读取开发者真实的 `.env`。配置 Embedding 后，知识库区才开始渲染上传框，于是 `app.file_uploader[0]` 从"数据集上传框"变成了"知识库上传框"，导致该测试**本地失败、CI 通过**。现在加了 `autouse` fixture 把 LLM/Embedding/MySQL/LangSmith 的环境变量置空（`os.environ` 优先级高于 `.env`），并改为按 key 定位组件；另有一条测试**显式断言隔离成立**，而不是只保证恰好通过。
