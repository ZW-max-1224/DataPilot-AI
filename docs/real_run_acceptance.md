# DataPilot AI 真实运行验收报告

**验收日期：**2026-10-10（Asia/Shanghai）

**项目目录：**`C:\Users\35002\Documents\Codex\2026-10-04\DataPilot-AI`

**验收范围：**Phase 1–10 当前实现的测试、Streamlit 页面、Demo 数据、CSV/XLSX Loader、Profiler、白名单 Pandas Tool，以及 LLM、RAG、MySQL、LangSmith 的真实运行条件。

## 总结

- **测试：**149 passed，0 failed，0 errors。
- **编译、依赖一致性：**通过。
- **Streamlit：**使用项目默认端口 8501 启动；health、页面和 host-config 请求均返回 HTTP 200。验收后仅停止本次启动的 Streamlit 会话，端口已无监听进程。
- **数据与白名单工具：**真实读取项目 Demo CSV，实际运行 `find_top_n` 并验证结构化结果。
- **真实 LLM：**blocked。项目没有 `.env`，运行环境也没有有效 LLM 配置；Agent 实际返回 `demo_mode`，没有调用模型或工具。
- **真实 Embedding → RAG：**blocked。Embedding 未配置；虽然本地 Qdrant 客户端可用，但没有执行真实 Embedding、文档索引及端到端检索。
- **MySQL：**未配置，未尝试连接或执行 SQL。
- **LangSmith：**disabled，未创建 Trace。
- **代码修改：**没有业务代码修改；本报告是本次唯一新增文件。未提交 Git。

## 项目与环境检查

| 检查 | 实际结果 |
|---|---|
| 工作目录 | `C:\Users\35002\Documents\Codex\2026-10-04\DataPilot-AI` 存在 |
| Python | `.venv` Python 3.11.4；系统 Python 3.14.7。验收命令使用项目 `.venv` |
| 虚拟环境 | `.venv\Scripts\python.exe` 存在并可执行 |
| `.env` | 不存在；未读取或输出任何密钥内容 |
| Git | `master`，No commits yet；项目文件当前整体显示为未跟踪。未暂存、覆盖或删除任何文件 |
| Streamlit 端口 | 启动前 8501、8502、8000 均无监听者；使用默认 8501 |

## 执行命令与结果

以下命令均在项目根目录执行，Python 命令显式使用虚拟环境解释器：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 149 passed in 48.22s；0 failed，0 errors

.\.venv\Scripts\python.exe -m compileall app tests
# exit code 0

.\.venv\Scripts\python.exe -m pip check
# No broken requirements found.

git diff --check
# exit code 0，无输出

.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py --server.headless true --server.port 8501
# 本次启动；验收完成后对该终端会话发送 Ctrl+C
```

HTTP 验证：

| URL | 结果 |
|---|---|
| `http://127.0.0.1:8501/_stcore/health` | HTTP 200，响应 `ok` |
| `http://127.0.0.1:8501/` | HTTP 200，`text/html`，Streamlit 页面 Shell 存在 |
| `http://127.0.0.1:8501/_stcore/host-config` | HTTP 200 |

进程停止后检查 8501，未发现监听进程。测试套件还包含 Streamlit `AppTest`，覆盖 Evaluation 页面在依赖缺失时如实显示状态。

**Git 检查限制：**当前分支没有任何提交，项目文件均显示为未跟踪文件，因此 `git diff --check` 不会扫描这些文件。另对项目文本文件执行了尾随空格/Tab 扫描，结果 `clean`。没有用 Git 暂存或清理用户文件。

## Demo 数据、Loader、Profiler 与 Tool

通过项目实际实现执行的验证：

- 读取 `data/sample_sales.csv`：1,208 行、8 列。
- Dataset Profiler：识别 1,208 行，质量分 92。
- CSV Loader：实际读取项目 Demo CSV 成功。
- XLSX Loader：在内存生成一个两行工作簿，再由项目 `load_excel` 读取成功；没有写入项目数据文件。
- 白名单 Tool：通过 `invoke_analysis_tool` 实际调用 `find_top_n(column="sales", n=1, group_by="product")`，返回结构化结果 `Laptop Pro`、`504431.55`。
- Agent：以当前真实 `LLMSettings` 调用 `AnalystAgent.run`，结果为 `demo_mode`、`success=false`、0 个已调用工具。没有用 Mock 结果冒充真实 Agent/LLM 成功。

## 外部服务状态

| 服务 | 验收状态 | 证据 / 未完成原因 |
|---|---|---|
| LLM API | **BLOCKED** | `LLMSettings.is_configured=false` 且没有 `.env`。缺少有效的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`，因此没有发出 API 请求或真实 Tool Calling |
| Embedding API | **BLOCKED** | `EmbeddingSettings.is_configured=false`。缺少启用配置及有效的 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`；未调用 Embedding Provider |
| 本地 Qdrant | **PASS（本地客户端健康）** | 真实本地客户端 `get_collections` 可用；pytest 的 Qdrant Store 测试通过。因 Embedding 未配置，不能据此声称 RAG 端到端通过 |
| RAG 端到端 | **BLOCKED** | 需要先配置并真实调用 Embedding Provider，再索引知识文档并检索验证；本次没有写入或伪造索引结果 |
| MySQL | **BLOCKED / NOT CONFIGURED** | `MYSQL_USER` 未配置，数据库没有连接，也未执行 `SELECT 1` |
| LangSmith | **DISABLED** | `LangSmithSettings.status=disabled`；没有提交或伪造 Trace |

启用真实验证需要由项目维护者在本地 `.env` 配置：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_DIMENSIONS=1024

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=datapilot
MYSQL_USER=
MYSQL_PASSWORD=

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=datapilot-ai
```

不要将真实凭据写进 Git。Embedding Provider 返回的维度必须与配置和 Qdrant collection 维度一致。

## 失败、受阻项与代码改动

- **测试失败：**无。
- **测试错误：**无。
- **真实运行受阻：**LLM、Embedding、MySQL 未配置；LangSmith 关闭。详见服务状态表。没有为通过验收而使用 Mock 代替这些服务。
- **代码问题：**本次验收未发现需要最小修复的明确代码问题，因此没有修改应用代码，也没有降低或删除测试断言。
- **本次新增：**`docs/real_run_acceptance.md`。
- **本次修改：**无业务代码、配置、测试或 `.env` 修改。
- **Git 提交：**未执行。

## 验收结论

Phase 1–10 的离线测试、编译、依赖检查、Streamlit 本地启动、Demo 数据处理、CSV/XLSX Loader、Profiler 和白名单 Pandas Tool 验证通过。真实 LLM Tool Calling、Embedding→RAG、MySQL 查询及 LangSmith Trace 尚未通过验收；需配置对应服务后重新执行这些真实集成检查。
