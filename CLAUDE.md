# CLAUDE.md — DataPilot AI

给 Claude Code 的项目说明。读完这里再动手，可以少走很多弯路。

## 这个项目是什么

**AI 数据分析 Agent**：用户用自然语言提问，LLM 只能从**静态白名单**里挑选已注册的分析工具（Pandas 计算 / 查知识库），不能执行任意代码。最终回答必须标注每条结论是 `fact` / `interpretation` / `hypothesis`，且标为 `fact` 的结论**必须有真实运行过的工具作为证据**（代码强制，不是靠 prompt）。

当前状态：Phase 1–10 已实现，UI 是 Streamlit。**没有 FastAPI，不要新建。**

## 开发命令（Windows PowerShell，用项目自带的 .venv）

```powershell
# 安装依赖（首次）
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt

# 跑应用
.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py

# 测试（必须全绿，当前基线 311 passed）
.\.venv\Scripts\python.exe -m pytest -q

# 代码检查（必须通过）
.\.venv\Scripts\python.exe -m ruff check .

# 依赖一致性 + 类型检查（mypy 是非阻断的咨询项，见下文）
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m mypy

# 初始化 MySQL 表结构（可选，仅在使用数据库时）
.\.venv\Scripts\python.exe -m app.database.init
```

改完代码，**至少要跑 `pytest -q` 和 `ruff check .`，两个都得过**。

## 目录结构

```text
app/
├── models/       # Pydantic 契约：Profiler、DatasetContext、分析结果、Agent/评估模型
├── services/     # 文件读取、数据画像、LLM、Embedding、知识库、Trace、持久化
├── knowledge/    # Markdown/TXT Loader + 标题/段落感知切块
├── vectorstore/  # Qdrant adapter（本地持久化模式）
├── database/     # SQLAlchemy schema / engine / 幂等初始化
├── repositories/ # 数据集、分析运行、知识文档的仓储层
├── tools/        # 白名单工具：Pandas 分析、可视化、search_knowledge
├── agent/        # AnalystAgent（工具选择循环、结构化结论、可靠性加固）
├── evaluation/   # Phase 10 评估：数据集、runner、确定性 metrics、judge
└── ui/           # Streamlit 单文件应用（app/ui/streamlit_app.py，约 50KB）
data/             # 合成 Demo CSV、内置知识文档、本地忽略目录
docs/             # 算法/画像/工具说明、代码审查报告、真实运行验收记录
i18n/             # en-US.json / zh-CN.json
scripts/          # check_embedding.py / check_rag.py / check_failure_modes.py
tests/            # 全离线单元测试
```

## 硬性约束（改代码前务必确认没破坏）

1. **工具白名单是唯一执行边界。** `app/tools/registry.py` 是唯一的 name → function 映射。禁止动态导入、`exec`/`eval`、任意 Python/SQL/Shell、按模型给的名字导入模块。新增工具必须：注册进 registry + 加 Pydantic 请求/结果模型 + 加测试。

2. **测试必须保持离线。** 不允许真实网络、真实 LLM Key、真实 MySQL、远程 Qdrant。LLM 用 Mock，Qdrant 用真实本地客户端 + `tmp_path`。CI 会跑 `pytest -q`。

3. **`fact` 结论必须有字段级工具证据。** 必须引用本轮成功调用的 `call_id`、JSON Pointer 结果字段、匹配值和单位；程序核对后生成事实文本。缺失或不匹配的引用自动降级为 `hypothesis`，仅写工具名不再足够。总结、解释、建议与原始叙述未经校验。详见 `docs/fact-evidence.md`。不要绕过这个逻辑，也不要让 UI 把模型自身知识说成"知识库证据"。

4. **i18n 不允许 UI 硬编码中文。** 新增界面文案必须同时加到 `i18n/en-US.json` 和 `i18n/zh-CN.json`，**键集合与顺序必须完全一致**（`tests/test_i18n.py` 会静态检查）。语言选择器本身例外。

5. **隐私边界：**
   - 完整 CSV / DataFrame 只留在 Streamlit 会话内存，不进 MySQL、不进 Qdrant、不进 LangSmith。
   - MySQL 只存元数据与结构化画像，不存原始单元格、Prompt/Completion、hidden CoT。
   - LangSmith 只发脱敏摘要（哈希、长度、计数、工具名），不发正文。
   - **不要读取、打印或提交 `.env` 里的任何 Key。** UI 也只显示配置状态，不显示 Key。

6. **依赖只写在 `requirements.txt`（运行时）和 `requirements-dev.txt`（开发工具）。** `pyproject.toml` 的 `[project]` 故意不声明依赖，不要往里加。

## 代码风格

- Python 3.11+。注释和 docstring 用**英文**；README 和 `docs/` 用**中文**。
- ruff：`select = ["E4", "E7", "E9", "F", "W"]`，`line-length = 120`。
- **不要顺手开启 E501 或 I001**。仓库里有 52 处超长行、19 处未排序 import，是历史遗留，属于单独的重构任务，混进功能改动里会让 diff 无法 review。
- `app/ui/streamlit_app.py` 的 `E402` 已按文件豁免（sys.path bootstrap 必须在 first-party import 之前），不要"修复"它。
- mypy 有 **33 个历史类型错误**，CI 里是非阻断的咨询项。不要为了消错而大改；新代码尽量带类型，但别制造新的错误。
- 提交信息用 **Conventional Commits**（英文，如 `fix: ...`、`feat: ...`、`docs: ...`、`test: ...`、`chore: ...`）。

## 已知未验证 / 容易踩的坑

- **MySQL 读写、LangSmith Trace 都还没真实验证**，只有离线 Mock。相关改动要诚实标注验证状态，不要声称"已验证"。
- **Embedding 必须独立配置**，不要假设当前 Chat endpoint 支持 embedding。换维度后旧 Qdrant collection 不可用，需要换 `QDRANT_COLLECTION` 或 `QDRANT_PATH` 重建。
- **Qdrant local 是单进程的**，不要把同一个本地目录同时给多个进程打开。
- LLM endpoint 有间歇性连接失败。失败时 Agent 会返回已算出的证据并标注未完成部分——**这是预期行为，不要改成伪造结论**。
- 知识库检索**故意不设相似度阈值**，无关问题也会返回低分切片。评估里不要把它当"必须 no_results"。
- `tests/test_streamlit_ui.py` 用 Streamlit AppTest；改 UI 时注意会话状态会在切换数据集时被清理。

## Windows 与工具调用注意（实测踩过）

- **在 Bash 工具里必须用正斜杠绝对路径。** `.\.venv\Scripts\python.exe` 会被 shell 吃掉反斜杠，变成 `..venvScriptspython.exe`（exit 127）。正确写法：
  `"C:/Users/35002/Documents/Codex/2026-10-04/DataPilot-AI/.venv/Scripts/python.exe" -m pytest -q`
- **不要用 `echo xxx` 轮询后台子 agent。** 那样只会白烧 token 和好几轮决策。要么继续做手上的事等它回来，要么自己先把同一件事查了，回来再交叉验证。
- 提交前先 `git status --short`，确认没漏掉新文件、也没把 `.env` 带进去。
- 本机 `downloads.claude.ai` 不通，`claude update` / 自动更新会失败（`✘ Auto-update failed`）。这是环境限制，**不影响使用**，忽略即可；需要升级时走 `npm i -g @anthropic-ai/claude-code@latest`，升级后要重建 `claude.exe` 硬链接（见交付目录 README）。

## 更多文档

- `README.md` — 全项目总览、每个 Phase 的设计与安全边界（最详尽的入口）
- `docs/analysis-tools.md` — 工具输入/输出、边界条件、测试覆盖
- `docs/dataset-profiler.md` — 数据画像与质量评分规则
- `docs/code-review-and-next-steps.md` — 审查与加固记录、待办方向（92KB，按需局部阅读）
- `docs/real_run_acceptance_v2.md` — 当前真实运行验收记录
- `scripts/check_embedding.py` / `scripts/check_rag.py` — 配置服务商前先跑这两个自检
- `.env.example` — 所有环境变量及含义
