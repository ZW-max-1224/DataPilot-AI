# 配置指南

[返回项目首页](../README.md) · [架构设计](architecture.md)

先复制根目录的 [.env.example](../.env.example) 为 `.env`。模板是环境变量的完整清单；只填写要启用的服务，然后重启应用。不要提交真实密钥。

## 按需求启用

| 使用方式 | 必要配置 |
| --- | --- |
| 数据画像、手动分析和图表 | 无需 API Key |
| 自然语言数据分析 | Chat LLM |
| 分析方法与业务口径检索 | Chat LLM + 独立 Embedding；Qdrant 使用本地目录 |
| 保存元数据与运行摘要 | 可选 MySQL |
| 查看远程运行摘要 | 可选 LangSmith |

## Chat LLM

```dotenv
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
LLM_TIMEOUT=30
LLM_MAX_TOKENS=1024
LLM_MAX_RETRIES=2
```

`LLM_BASE_URL` 填服务商提供的 OpenAI-compatible API 根地址，例如以 `/v1` 结尾的地址。未填写 Key 时显示 Demo Mode，AI 分析不可用；Key 存在但地址或模型不完整时，界面提示修正配置。

`LLM_MAX_TOKENS` 是单次 Prompt 调用的默认上限；Agent 决策使用独立的输出预算。SDK 按 `LLM_MAX_RETRIES` 重试超时、限流与服务端错误，设为 `0` 可关闭 SDK 重试。

## Embedding 与 Qdrant

Chat 和 Embedding 独立配置；不能假设 Chat endpoint 支持向量接口。

```dotenv
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_TIMEOUT=20
EMBEDDING_DIMENSIONS=1024
QDRANT_PATH=data/qdrant
QDRANT_COLLECTION=datapilot_knowledge
```

模板默认 `EMBEDDING_PROVIDER=disabled`。维度必须与服务商实际输出一致；更换维度时，使用新的 collection 或路径并重建索引。Qdrant local 的同一目录只供单进程使用。

服务可用后，在页面的知识库区域索引内置文档，或上传获准使用的 Markdown / TXT。当前不支持把 PDF、DOCX、CSV、Excel 作为知识文档上传。

**Windows / PowerShell 自检：**

```powershell
# 离线验证自检脚本自身逻辑
.\.venv\Scripts\python.exe scripts\check_embedding.py --self-test

# 使用 .env 中的配置验证真实服务
.\.venv\Scripts\python.exe scripts\check_embedding.py
.\.venv\Scripts\python.exe scripts\check_rag.py
```

Embedding 自检覆盖端点、维度、批量顺序、确定性和语义有效性。RAG 自检覆盖索引、预期文档排名及相关与无关问题的分数区分。两条真实自检会调用已配置的外部服务；不会把“无关问题必须零召回”当作通过条件。

## MySQL（可选）

```dotenv
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=datapilot
MYSQL_USER=
MYSQL_PASSWORD=
```

`MYSQL_USER` 留空时关闭持久化。配置后，应用会幂等创建缺失的表，也可手动初始化：

```powershell
.\.venv\Scripts\python.exe -m app.database.init
```

MySQL 保存元数据与运行记录，不保存完整原始表格；用户问题会保存，启用前应确认适合的数据策略。连接或写入失败时，应用降级为内存分析。当前尚未完成真实 MySQL 读写验收。

## LangSmith（可选）

```dotenv
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=datapilot-ai
LANGSMITH_ENDPOINT=
```

启用时将 `LANGSMITH_TRACING` 改为 `true` 并填写 Key；默认 endpoint 可留空。兼容旧的 `LANGCHAIN_*` 变量，新名称优先。

Trace 覆盖 Agent、决策、LLM、工具和知识检索层级，只传输脱敏摘要。故障不会中断分析，界面不会显示 Key。当前尚未完成真实 Trace 验收，详见 [存储与隐私](architecture.md#存储与隐私)。

## 常见情况

| 现象 | 检查方向 |
| --- | --- |
| 页面正常，但 AI 分析不可用 | 检查 Chat 的 Key、Base URL、模型名；未配置时是正常的 Demo Mode |
| Embedding unavailable | 检查是否启用独立 Embedding，并运行服务自检 |
| 更换模型后无法打开向量集合 | 核对向量维度，为新维度使用新集合或路径 |
| Qdrant local 目录被锁定 | 确认没有另一个应用进程使用同一目录 |
| 模型连接失败但仍看到部分结果 | 这是已计算证据的降级返回；未完成部分会被标注 |
