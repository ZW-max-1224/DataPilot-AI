<div align="center">

# DataPilot AI

### 用自然语言分析数据，让结论有据可查。

上传表格 · 受控计算 · 字段级证据 · 中英双语

[![CI](https://github.com/ZW-max-1224/DataPilot-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/ZW-max-1224/DataPilot-AI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981.svg)](LICENSE)

[核心能力](#核心能力) · [快速开始](#快速开始) · [验证状态](#验证状态) · [项目文档](docs/README.md)

</div>

---

DataPilot AI 是一个 AI 数据分析工作台。导入 CSV / Excel，查看数据质量，再用自然语言探索趋势、排名与异常。**模型选择工具，Pandas 完成计算，程序核对事实引用。**

![DataPilot AI：自然语言提问与结构化分析结果](docs/images/demo.png)

<sub>真实运行截图，摄于 2026-10-10。当前版本已增加字段级引用核验与证据展开查看。</sub>

## 核心能力

| 能力 | 你可以做什么 |
| --- | --- |
| **数据画像** | 查看字段类型、缺失值、重复记录和数据质量评分 |
| **分析与图表** | 完成 Top N、分组统计、时间序列、异常检测与四类图表 |
| **自然语言分析** | 提问“哪个类别利润最高？”或“最近几个月销售趋势如何？” |
| **可追溯的事实** | 查看事实对应的工具调用、结果字段、值和单位 |
| **知识辅助** | 检索分析方法与业务口径，并展示实际检索来源 |

无需 API Key 即可体验数据画像和手动分析；自然语言分析需要配置 LLM，知识检索另需配置 Embedding。

<details>
<summary>查看工具执行过程</summary>

![工具执行过程：参数、检索来源与重复调用复用](docs/images/demo-2.png)

此图来自同一次历史演示。当前界面还会显示调用编号及字段级证据。

</details>

## 快速开始

需要 **Python 3.11+**。

```bash
git clone https://github.com/ZW-max-1224/DataPilot-AI.git
cd DataPilot-AI
python -m venv .venv
```

**Windows / PowerShell**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py
```

<details>
<summary>macOS / Linux</summary>

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app/ui/streamlit_app.py
```

</details>

打开终端输出的网址，在侧栏选择 **使用 Demo 数据**，或上传自己的 CSV / Excel。

**启用 AI 分析：**复制 [.env.example](.env.example) 为 `.env`，填写 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`，然后重启应用。其余服务按需开启，详见 [配置指南](docs/configuration.md)。

## 结论如何核验

```mermaid
flowchart LR
    A[导入表格] --> B[自然语言提问]
    B --> C[白名单工具计算]
    C --> D[字段级引用核验]
    D --> E[展示事实与解释]
```

- **工具受控：**模型只能调用已注册工具，不能执行任意 Python、SQL 或 Shell。
- **引用可核对：**事实必须对应本轮成功调用的具体结果字段，值或单位不匹配就降级为假设。
- **展示有区分：**事实摘录由程序生成；总结、解释、建议和模型原始叙述未经程序校验。

核验保证字段引用与工具结果一致，不保证数据源本身正确。设计细节见 [事实证据校验](docs/fact-evidence.md)。

## 验证状态

| 范围 | 当前记录 |
| --- | --- |
| **离线检查** | 311 项测试通过，Ruff 与依赖检查通过 · 2026-10-22 |
| **字段级事实引用** | 真实模型单场景冒烟通过 · [记录](docs/fact-evidence.md#本次验证记录2026-10-22) |
| **Agent / RAG** | 历史端到端验收通过 · [2026-10-10 记录](docs/real_run_acceptance_v2.md) |
| **仍待完善** | mypy 33 个历史类型错误；MySQL 与 LangSmith 尚未真实验收 |

当前是适合本地运行与演示的 MVP；历史验收和单次冒烟不代表完整效果基准。

## 技术与文档

**交互** Streamlit · **计算** Pandas / Matplotlib · **契约** Pydantic · **模型** OpenAI-compatible API<br>
**可选集成** Qdrant / Embedding · MySQL / SQLAlchemy · LangSmith

| 想了解什么 | 从这里开始 |
| --- | --- |
| 配置模型、知识库与可选服务 | [配置指南](docs/configuration.md) |
| Agent、工具与数据流如何协作 | [架构设计](docs/architecture.md) |
| 评估指标、运行方法与局限 | [评估说明](docs/evaluation.md) |
| 算法、验收和历史记录 | [文档导航](docs/README.md) |

<details>
<summary>开发与检查</summary>

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m mypy
```

自动化测试不依赖真实模型、远程数据库或 API Key。mypy 目前作为非阻断检查。

</details>

[MIT License](LICENSE)
