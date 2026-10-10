# DataPilot AI 代码审查报告与下一步处理指南

**审查日期：**2026-10-10
**审查对象：**`C:\Users\35002\Documents\Codex\2026-10-04\DataPilot-AI`
**审查方式：**静态阅读全部源码 + 在本机用项目自带 `.venv` 运行真实测试与真实 LLM 端到端验证（不是只看代码猜测）
**状态说明：**第 1–7 节是审查报告（**审查当时没有修改任何业务代码**）。审查完成后，按用户要求"先全部修好"执行了修复，修复内容与验证证据见 **第八节**。审查与修复过程中创建的临时探针脚本已全部删除。

---

## 一、总体结论（先看这里）

**这个项目不是"没写完"，而是"功能已经堆到 Phase 10，但最核心的那条链路——真实 Agent 循环——实际是坏的"。**

用一句话概括现状：

> 基础设施（配置、数据库、向量库、测试、安全边界）的工程质量**明显高于**项目当前阶段应有的水平；但作为项目卖点的 **Agent 真实运行**，在我本机实测中是 **1/10 成功**，而且失败方式非常致命：**它会丢掉已经算出来的所有分析证据**。

具体证据：

| 检查项 | 结果 | 说明 |
|---|---|---|
| `pytest -q` | **149 passed, 0 failed** | 测试本身是健康的、真实的 |
| 真实 LLM 单次调用 | **成功** | `deepseek-flash` + `https://api.deepseek.com` 可用 |
| 真实 Agent 端到端 | **1/10 成功** | 10 次真实提问里只有 1 次拿到完整回答 |
| Git 仓库 | **0 个提交** | `master` 分支 `No commits yet`，149 个文件全是未跟踪状态 |
| `.env` | 已配置真实 Key（35 字符） | 已被 `.gitignore` 忽略，未进入版本库（安全） |
| MySQL | `disabled` | `MYSQL_USER` 为空，未连接 |
| Embedding | `disabled` | 未配置，所以 RAG 无法真实运行 |
| Qdrant 本地库 | **空**（`{"collections": {}}`） | 知识库从未真正索引过任何文档 |
| LangSmith | `disabled` | 无 Trace |

**所以最重要的一句话建议是：先停下"加新功能"，回头把 Phase 6 的 Agent 循环修好。**

项目原文第 41 条写着"每完成一个 Phase：必须运行代码、运行测试、修复错误"。Phase 5–10 确实是按这个流程做的，但 **Phase 6 的 Agent 只在 Mock 下验证过，从未在真实模型下验收**。Mock 永远不会返回带 ```json 代码块的输出，也不会中途断网——所以 149 个测试全绿，却掩盖了真实运行必坏的事实。

这是一个非常有价值的教训，而且**在面试里讲出来是加分项**：

> "我的 149 个单元测试全通过，但我用真实 LLM 做端到端验收时发现成功率只有 1/10。原因是 Mock 测试覆盖不到两类真实故障：模型把 JSON 包进 Markdown 代码块、以及网络瞬时中断。这让我意识到 'Mock 全绿' 不等于 '系统可用'，之后我给 Agent 补了 JSON 修复重试和错误反馈循环。"

这比"我的项目 149 个测试通过"有说服力得多。

---

## 二、问题清单（按修复优先级排序）

优先级定义：
- **P0（阻断）**：不修就无法演示、无法写进简历，或会造成错误结论。
- **P1（重要）**：影响项目可信度与面试说服力，但不阻断运行。
- **P2（打磨）**：工程规范性，用于让 GitHub 仓库看起来专业。

---

## P0-1【阻断】模型的工具参数会直接让整个 Agent 崩溃

**位置：**`app/agent/analyst_agent.py:163-169`、`app/tools/registry.py:125-136`

```python
# analyst_agent.py:163
execution = invoke_analysis_tool(
    context,
    name,
    knowledge_service=self._knowledge_service,
    trusted_knowledge_base_id=request.knowledge_base_id,
    **(action.tool_input or {}),        # ← 问题在这里
)
```

**我实测复现的结果：**

```
tool_input={'name': 'x'}    -> ESCAPED as TypeError: invoke_analysis_tool() got multiple values for argument 'name'
tool_input={'dataset': None} -> ESCAPED as TypeError: invoke_analysis_tool() got multiple values for argument 'dataset'
完整 Agent 运行（tool_input={'name':'oops'}）-> 异常逃出 AnalystAgent.run，未被任何 except 捕获
```

**这是什么问题（初学者解释）：**

`**dict` 是 Python 的"字典展开成关键字参数"语法。`invoke_analysis_tool` 函数自己的参数列表里已经有 `dataset`、`name`、`knowledge_service`、`trusted_knowledge_base_id` 这些名字。如果模型返回的 `tool_input` 里恰好也有一个叫 `name` 的键，Python 会在**函数体执行之前**就报错 `got multiple values for argument 'name'`。

关键点：这个错误发生在 Python 解释器"绑定参数"的阶段，**比函数内部的第一行 `try:` 还早**。所以 `registry.py` 里的 `except TypeError` 根本拦不到它，`analyst_agent.py` 里的 `try` 也拦不到它（那些 try 只包住了 LLM 调用和工具名查找）。

**为什么严重：** 这不是"参数写错返回友好错误"，而是**整个 Agent 崩掉**。UI 只捕获 `ValidationError`（`streamlit_app.py:804`），所以 Streamlit 页面会直接显示 Python Traceback——同时违反了项目原文第 30 条"不能直接显示 Traceback"。

模型会不会真的返回 `name` 这个键？会。它看到工具 schema 和用户问题时，完全可能自己发明一个 `name` 字段（比如想表达"按产品名"）。这在 Prompt Injection 场景下更危险：恶意输入可以故意让模型输出这种键。

额外风险：`knowledge_service` 和 `trusted_knowledge_base_id` 是**可信的系统内部参数**（后者决定访问哪个知识库），却和模型数据混在同一个 `**kwargs` 命名空间里。现在是崩溃，重构时一次疏忽就会变成"模型可以自己指定知识库 ID"——这是租户越权。

**修复方向：**

```python
# registry.py：模型参数作为一个整体传入，不再展开
def invoke_analysis_tool(dataset, name, *, parameters=None,
                         knowledge_service=None, trusted_knowledge_base_id="datapilot-default"):
    ...
    validated = spec.input_model.model_validate(parameters or {})
```
```python
# analyst_agent.py：不再 ** 展开；并补一层兜底
try:
    execution = invoke_analysis_tool(context, name, parameters=action.tool_input or {}, ...)
except Exception:
    return self._failure("tool_invocation_error", "工具调用失败，已安全终止。", steps, tools_used, decision_count)
```

**面试怎么讲：**"我不会把模型返回的字典直接用 `**` 展开成函数的关键字参数。因为模型可能提供与函数自身参数同名的键，导致 Python 在参数绑定阶段抛出 TypeError——这个阶段早于任何 try/except。正确做法是把模型参数作为**一个数据对象**传入，验证后再使用，这样可信参数（比如租户 ID）和数据参数在命名空间上就是隔离的。"

---

## P0-2【阻断】模型输出只要带 Markdown 代码块，整轮分析就作废

**位置：**`app/agent/analyst_agent.py:115-119`

```python
try:
    action = AgentAction.model_validate_json(response.content)
except (ValidationError, ValueError):
    decision_span.end({"mode": response.mode, "action": "invalid_output"})
    return self._failure("invalid_model_output", "模型返回格式无效，无法安全执行分析。", ...)
```

**我实测捕获到的模型原始输出（不是我编造的，是抓到的真实响应）：**

```
第一次调用: {"action": "tool_call", "tool_name": "calculate_time_series", ...}          ← 正常
第二次调用: {"action": "tool_call", ...}\n```json\n{"action": "tool_call", ...}\n```    ← 带代码块，解析失败

另一次运行第 5 次调用:
```json
{"action": "tool_call", "tool_name": "calculate_time_series", ...}
```                                                                                      ← 解析失败
```

结果：`success=False, errors=['invalid_model_output']`，**前面已经算出来的时间序列、分组统计、异常检测结果全部丢弃**。

**这是什么问题：**

大模型有很强的"用 Markdown 格式回答"的习惯。即使 Prompt 说了"Return exactly one JSON object"，它仍会时不时加上 ```json 围栏。`pydantic` 的 `model_validate_json` 要求输入是**纯 JSON 字符串**，多一个反引号就失败。

更关键的是**失败后的处理方式**：直接 `return self._failure(...)` 结束整轮。模型本来可以在下一轮修正（它甚至已经准备好了一个合法的 JSON 决策），但代码没给它机会。

**修复方向（三步，按性价比排序）：**

1. **开 JSON 模式**：`chat.completions.create(..., response_format={"type": "json_object"})`。这是 OpenAI 兼容 API 的标准参数，能从源头大幅降低围栏概率。**注意：需要先确认 `deepseek-flash` 支持该参数，不支持会报错，所以要做兼容处理。**
2. **解析前清洗**：去掉 ```json / ``` 围栏、去首尾空白、截取第一个 `{` 到最后一个 `}`。
3. **失败后修复重试**：把校验错误信息追加到下一轮 Prompt，要求"上次输出无效，原因：xxx，请只输出 JSON"，最多重试 2 次（计入 8 次决策预算）。2 次都失败才降级返回。

**面试怎么讲：**"结构化输出不能只靠 Prompt 承诺。我做了三层防御：开启 API 的 JSON mode、解析前清洗 Markdown 围栏、解析失败把错误反馈给模型重试。因为即使 temperature=0，模型仍会偶发加代码块。只靠一次 `model_validate_json` 的成功与否来决定整个任务成败，是把系统稳定性赌在模型的格式遵守度上。"

---

## P0-3【阻断】工具报错就让整轮结束，Agent 没有"自我纠正"能力

**位置：**`app/agent/analyst_agent.py:186-190`

```python
if not execution.success:
    error = execution.error
    code = error.code if error else "tool_execution_error"
    message = error.message if error else "工具执行失败。"
    return self._failure(code, message, steps, tools_used, decision_count)
```

**问题：**Agent 有 8 次决策预算，但一次工具参数错误就烧掉全部。最常见的场景：模型把列名猜错了（`find_top_n(column="revenue")`，真实列名是 `sales`）——这是 LLM 最高频的错误之一。此时正确行为是：把"没有 revenue 这一列，可用列是：date, product, ..., sales"作为**Observation** 喂回模型，让它自己改成 `sales` 重试。

而现在它是直接结束，用户看到"工具参数不匹配"。

**注意对比：**项目在 RAG 上做对了——`registry.py:157` 注释写着"A retrieval status is an observation, not a tool execution crash"。同样的思路没有应用到 Pandas 工具的失败上。

**这正好命中项目原文第 13 条的核心要求：**

> Reason → Tool → Observation → Reason → Tool → Answer

现在缺失的是"Observation 是错误时也要回到 Reason"这一条路径。所以严格说，**当前实现还不是一个完整的 Agent，而是一个"对错误零容忍的工具调用脚本"**。

**修复方向：**

```python
if not execution.success:
    code = execution.error.code if execution.error else "tool_execution_error"
    message = execution.error.message if execution.error else "工具执行失败。"
    history.append({"tool": name, "error": {"code": code, "message": message}})
    steps.append(AgentStep(action_type="error", tool_name=name, summary=...))  # 用上那个闲置的 "error" 类型
    # 同一个工具连续失败 2 次才终止，避免死循环
    continue
```

**顺便修：**`agent_models.py:37` 的 `action_type: Literal["tool_call", "final", "error"]` 里 `"error"` 目前从未被使用——这个字段是为此预留的，正好接上。

**面试怎么讲：**"Agent 的核心不是'调用工具'，而是'失败后还能继续'。我把工具错误当作可观察的观察值反馈给模型，并加了同一工具最多连续失败两次的熔断，防止死循环。这样模型猜错列名时能自己修正，而不是让用户看到失败。"

---

## P0-4【阻断】LLM 瞬时故障会丢掉全部分析证据

**位置：**`app/services/llm_service.py:86-92`、`app/agent/analyst_agent.py:103-110`

```python
client = self._client_factory(
    api_key=..., base_url=..., timeout=..., max_retries=0,   # ← 显式关闭了 SDK 的自动重试
)
```
```python
except LLMServiceError as exc:
    logger.warning("Agent LLM request failed (%s)", type(exc).__name__)
    decision_span.end({"mode": "error", "error": "llm_request_failed"})
    return self._failure("llm_request_failed", "模型调用失败，请检查配置或稍后重试。", steps, tools_used, decision_count)
```

**我实测的证据（8 次连续直接调用，间隔 1 秒）：**

```
direct1: APIConnectionError  -> "Could not connect to the configured LLM provider."
direct2-5: HTTP 200 但内容为空 -> "The LLM provider returned an empty response."
direct6-7: APIConnectionError
direct8: 空响应
```

后续用 `max_tokens=256/1024` 重测，空响应消失，但**连接错误仍占 5/8**。说明两件事：

1. **连接不稳定是真实存在的**（本机到 `api.deepseek.com` 的间歇性 `APIConnectionError`）。这是**外部环境问题，不是你的代码 bug**。
2. **但代码让这个外部问题变成了致命问题**：
   - `max_retries=0` 主动关掉了 OpenAI SDK 默认的自动重试。瞬时网络抖动本可以被 SDK 自愈。
   - 项目自己没有重试/退避。
   - 更严重：**一旦第 N 轮 LLM 失败，前面 N-1 轮辛苦算出的工具结果全部丢弃**，只返回一句"模型调用失败"。

**这条直接违反项目原文第 15 条：**

> 如果达到最大次数：停止分析。**返回当前已经获得的信息。**

现在的实现是"返回一句错误信息"，而不是"返回已经获得的信息"。

**修复方向：**
1. 把 `max_retries=0` 改成 `max_retries=2`（SDK 自带指数退避），或自己实现重试。
2. Agent 层对 `llm_request_failed` 也做 1–2 次重试（同一轮决策内）。
3. **无论因何结束（LLM 失败 / 工具失败 / 到达 8 轮上限），都要能生成一个"基于现有证据的部分结论"**——这才是 Agent 产品该有的行为。

**关于错误信息可诊断性：**`llm_service.py:106-107` 把 `APIError` 一律转成 `"The LLM provider returned an API error."`，丢掉了 HTTP 状态码。**429 限流和 401 鉴权失败在用户眼里长得一模一样**，无法自助排查。

**面试怎么讲：**"我把外部依赖的失败分成两类：可重试的（超时、连接中断、429）和不可重试的（401、参数错误）。可重试的交给 SDK 的指数退避加项目层重试；同时我保证**任何**终止路径都能返回已收集的分析证据，而不是让用户拿到一个空壳错误。因为对数据分析 Agent 来说，3 个已验证的发现 + 一句'后续步骤未完成'，比一句'调用失败'有价值得多。"

**需要你确认：**你的 `.env` 是 `LLM_BASE_URL=https://api.deepseek.com` + `LLM_MODEL=deepseek-flash`。`deepseek-flash` 不是 DeepSeek 官方 API 公开的模型名（官方是 `deepseek-chat` / `deepseek-reasoner`），虽然我实测它**能**返回结果，但这个组合看起来很像是本机环境注入的配置而不是你从服务商控制台拿到的。**请到你的服务商控制台核对模型名，并把真实配置写进 `.env`。** 这直接影响面试演示的稳定性。

---

## P0-5【阻断】项目原文最核心的"结构化结论 + 事实/推测分离"完全没有实现

**位置：**`app/agent/agent_models.py:13-16`、`app/agent/analyst_agent.py:129-137`、`app/ui/streamlit_app.py:808`

```python
final_answer: str | None = Field(default=None, max_length=8_000)
```
```python
(st.success if agent_response.success else st.warning)(agent_response.final_answer)
```

**我的验证：**在整个仓库（`app/` + `i18n/` + `docs/`）里搜索 `Executive Summary`、`Key Findings`、`Evidence`、`Interpretation`、`Hypothesis`、`Recommendations` —— **0 处匹配**。

**这意味着：**
- `final_answer` 是一个**自由字符串**，没有任何 schema 约束。
- Prompt 里只提了一句"label interpretations or hypotheses clearly"，从未告诉模型有这六个段落。
- 模型可以只回一句 `"Sales dropped because of the price change."`（**0 次工具调用**）就被判定为 `success=True`。
- UI 把它塞进 `st.success()` 当纯文本显示，没有分段渲染，也没有任何"这是推测"的视觉区分。
- **Agent 自己画的图也不会显示**（见 P0-6）。

而项目原文第 16、17 条明确要求：Executive Summary / Key Findings / Evidence / Visualization / Recommendations，并且"**不能让 LLM 把猜测说成事实**"。

**这是项目与"AI Agent 实习简历"之间最大的落差**：目前它是"一个能调工具的聊天框"，不是"一个输出可信分析报告的分析师"。

**修复方向：**

```python
class Finding(BaseModel):
    statement: str = Field(max_length=500)
    evidence_tool: str | None = None
    claim_type: Literal["fact", "interpretation", "hypothesis"]   # 事实/解释/假设

class FinalAnswer(BaseModel):
    executive_summary: str
    key_findings: list[Finding]                     # 每条必须标注 claim_type
    recommendations: list[str] = []
```
- 把 `action="final"` 的分支改成返回结构化对象（或让模型返回该对象的 JSON）。
- **校验规则**：如果 `key_findings` 里有 `claim_type="fact"` 但没有对应的 `evidence_tool` 记录在 `tools_used` 里 → 判定无效，要求模型修正。这就是"事实必须有证据"的技术实现。
- UI 按段落渲染，`fact` / `interpretation` / `hypothesis` 用不同样式（比如 fact 用 ✅、hypothesis 用 ⚠️ 并注明"需进一步验证"）。

**面试怎么讲：**"我要求模型的最终输出不是自由文本，而是一个 Pydantic 模型，每条结论都带 `claim_type` 和 `evidence_tool`。系统会校验：标为 fact 的结论必须有对应的工具调用记录支撑，否则拒绝并要求重写。这样就把'不要让模型把猜测说成事实'从 Prompt 里的一句祈使句，变成了代码层面的强制约束。"

---

## P0-6【阻断】Agent 画的图被丢掉，同时每次调用泄漏一个 Matplotlib figure

**位置：**`app/tools/registry.py:117-122`、`app/tools/visualization_tools.py:47,126-133`

```python
def _json_result(value: Any) -> dict[str, Any]:
    if isinstance(value, ChartArtifact):
        return value.result.model_dump(mode="json")     # ← Figure 对象被丢弃
```

**我实测复现：**

```
open figures before=[] after=[1, 2, 3] leaked=3
returned data keys: ['chart_type', 'group_by', 'plotted_rows', 'title', 'warnings', 'x', 'y']
```

即：连续调用 3 次 `create_chart`，matplotlib 全局注册表里就残留 3 个 figure（永不释放），而返回给 Agent 的数据里**只有图表元信息，没有图**。

**两个后果：**
1. **用户在 Agent 模式下永远看不到图**——项目原文第 16 条要求结果里有 Visualization，第 39 条要求"生成图表"是可见的一步。手动 Playground 能画图，Agent 路径画不了（UI 的 Agent 区域只打印 `step.summary` 一行文字）。
2. **内存泄漏**：Streamlit 会话是长驻的，每次 Agent 调用泄漏一个 figure。演示几次、问十几个问题之后，内存持续增长。

**修复方向：**
- 让 `ToolExecutionResult` 增加一个**不参与 JSON 序列化**的 `artifact` 字段承载 `ChartArtifact`；
- Agent 步骤里保存 artifact，UI 用 `st.pyplot(fig)` 渲染，并在 `finally` 里 `plt.close(fig)`；
- JSON-only 路径下，在 `_json_result` 里直接 `plt.close(figure)`；
- 补一个回归测试：`assert plt.get_fignums() == before`。

**面试怎么讲：**"图是二进制资源，不能塞进 JSON 上下文给模型，但也不能因此丢掉。我用一条'数据走 JSON、资源走对象引用'的双通道设计：元信息进模型上下文，Figure 对象由界面层持有并在渲染后主动释放。同时我加了 figure 泄漏的回归测试，因为 Streamlit 会话很长，图形对象泄漏会累积。"

---

## P0-7【阻断】Git 仓库 0 个提交——这个项目目前**不能被别人看到**

**实测：**
```
fatal: your current branch 'master' does not have any commits yet
git status --short → 全部文件都是 ?? (untracked)
```

项目原文 Phase 1 第 10 条是"初始化 Git"。现在 `git init` 做了，但**一次都没提交**。

**这意味着：**
- GitHub 上没有这个项目，简历里没法放链接——而这是项目的最终目的。
- 149 个文件、约 7800 行代码、10 个 Phase 的工作，一旦误删就全部丢失（没有任何版本历史保护）。
- 没有 diff 历史，也就无法向面试官展示"你如何一步步演进架构"。

**好消息：**`.gitignore` 配置是**正确**的。我验证过 `.env` 已被忽略、`data/qdrant/` 被忽略、`__pycache__`/`.venv` 被忽略。

**下一步（务必按顺序）：**

```powershell
cd "C:\Users\35002\Documents\Codex\2026-10-04\DataPilot-AI"

# 1) 先确认 .env 不会被提交（这一步绝不能跳过）
git status --short --ignored | Select-String "\.env"
#    期望看到:  !! .env     (两个感叹号 = 已忽略)

# 2) 暂存除 .env 外的所有内容
git add -A

# 3) 再次确认暂存区里没有 .env 和任何密钥
git diff --cached --name-only | Select-String "\.env$"
#    期望: 没有任何输出（.env.example 出现是正常的）

# 4) 首次提交
git commit -m "chore: initial commit of DataPilot AI phase 1-10 implementation"
```

**安全提醒：**你的 `.env` 里有一个真实可用的 LLM API Key。请务必确认它**从未**被提交、粘贴到聊天记录、或写进 README/文档。如果曾经发生过，请立即到服务商控制台**吊销并重新生成**该 Key。

---

## P1-1【重要】整数"年份"列会变成 1970 年，并作为"事实"呈现给用户

**位置：**`app/tools/analysis_tools.py:167-173`

```python
parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
```

**我实测复现：**
```python
calculate_time_series(
    pd.DataFrame({"year": [2021, 2022, 2023], "sales": [10.0, 20.0, 30.0]}),
    date_column="year", metric="sales", frequency="year")
# 结果: rows=[{'period': '1970', 'sales_sum': 60.0}]，且 success=True
```

**原因：**`pd.to_datetime` 遇到整数列时，默认把整数解释为"**纳秒时间戳**"（即 1970-01-01 起算的纳秒数）。2021 纳秒 ≈ 1970-01-01，所以三年数据被合并成了一个 "1970" 分组。80% 的合法率检查也拦不住，因为解析"成功"了。

**为什么危险：**这是**静默的错误**。系统不会报错，而是自信地输出 `period: 1970`、`trend: insufficient_data`。用户（或面试官）看到的是一个看似正常的表格，但完全错误。对数据分析产品来说，**静默的错误结论比报错更严重**。

**修复方向：**解析前先判断 dtype：
- 若是 `datetime64` → 直接使用；
- 若是字符串/object → 尝试解析；
- 若是数值 → **明确拒绝**，提示 "Column 'year' is numeric; numeric date columns are ambiguous — cast to datetime or a date string"；或者显式支持 s/ms/us/ns 时间戳并写在文档里。

顺带修 `utc=True`：它会把 `2025-01-01 00:30 +08:00` 归到 2024 年 12 月。除非明确要 UTC 口径，否则应保留原始时区。

**面试怎么讲：**"我发现 `pd.to_datetime` 对整数列有隐式行为——它按纳秒时间戳解析，导致 2021 被静默变成 1970。这类 bug 不报错但会输出错误结论，比崩溃更危险。所以我对日期列先做 dtype 校验：数值列直接拒绝并提示用户显式转换。数据工具的正确性不能依赖 pandas 的隐式类型推断。"

---

## P1-2【重要】同一个"分组求和"问题，两个工具给出不同答案

**位置：**`app/tools/analysis_tools.py:121` vs `:156`

**我实测复现（数据：`g=["a","a","b"], v=[None, None, 5.0]`）：**
```
calculate_group_statistics: [{'g': 'b', 'v_sum': 5.0}, {'g': 'a', 'v_sum': 0.0}]
find_top_n(group_by):       [{'g': 'b', 'v': 5.0}, {'g': 'a', 'v': None}]
```

同一个问题，一个说 a 组是 **0.0**，一个说 a 组是 **None**。

**为什么是问题：**a 组的两条记录值都是缺失（NaN）。"缺失"和"零"在业务上完全不同——`0.0` 会被解读为"a 组一笔都没卖出去"，而事实是"a 组没有有效数据"。**对分析结论的影响是实质性的。**

**修复方向：**统一聚合语义。`calculate_group_statistics` 用了 `.sum(min_count=1)`（缺失返回 NaN），`find_top_n` 用了普通 `.sum()`（缺失当 0）。抽一个共用的聚合函数，或者在请求模型里暴露 `na_policy` 参数让调用方显式选择。

**面试怎么讲：**"我发现两个看似等价的聚合工具对全 NaN 分组的处理不一致。NaN 语义是个容易忽略的坑：sum 默认把 NaN 当 0，但业务上'无数据'和'零'必须区分。我统一了聚合层并显式化缺失值策略。"

---

## P1-3【重要】上下文预算失效：模型可能在"没有任何证据"的情况下被要求下结论

**位置：**`app/agent/analyst_agent.py:21-22, 304-307`

```python
MAX_TOOL_RESULT_CHARS = 3_000
MAX_HISTORY_CHARS = 12_000
...
while len(prompt) > MAX_HISTORY_CHARS and prompt_data["tool_history"]:
    prompt_data["tool_history"].pop(0)      # 只删历史，其他部分不动
```

**我实测：40 列数据集 + 4000 字符提问 → 在 `tool_history` 已经被清空到 0 条的情况下，Prompt 仍然是 12,381 字符，超过 12,000 的"上限"。**

原因：Prompt 里还有 `available_tools`（约 5,182 字符，包含全部 10 个工具的完整 JSON Schema）、`decision_schema`（约 678 字符）、最多 40 列的 schema、以及最多 4,000 字符的用户提问——这些都不受 `MAX_HISTORY_CHARS` 约束。

**后果：**`MAX_HISTORY_CHARS` 只是"愿望"而非"硬限制"。在最坏情况下（宽表 + 长问题），历史被清空后仍然超限，而**历史里装的是工具证据**。也就是说：代码为了腾出空间，**优先牺牲了模型的证据**，然后要求模型"基于证据回答问题"。结果就是模型只能重复调用工具直到 8 轮用尽，或者凭空编答案。

**修复方向：**
- 不要把证据**删掉**，而要**压缩**：每个工具结果保留标量结论 + 行数/分组数，丢掉的只是长表格预热。
- 为最后的 N 条结果**预留固定槽位**，永不删除。
- 真正需要硬约束（因为 prompt token 直接决定成本）：写一个断言 + 单元测试，用"最坏情况 schema + 最长提问"验证 `len(prompt) <= budget`。

**面试怎么讲：**"上下文管理不是'超了就删'。我的第一版按长度裁剪 tool history，后来测试发现宽表场景下 base prompt 本身就超预算，裁剪会把证据全删光。正确做法是分层压缩：结论和标量保留，明细表格摘要化，并给最近的关键证据预留不可驱逐的槽位。"

---

## P1-4【重要】"中英文双语"只做了一半，而且界面里的数字和阶段标错了

**位置：**`app/ui/streamlit_app.py:546-760`（知识库区、Evaluation 区）、`i18n/*.json:103`

我验证过的具体问题：

| 问题 | 证据 | 影响 |
|---|---|---|
| 知识库 + Evaluation 两大区块约 120 行文案写死在 Python 里，完全没走 i18n | 例如 `streamlit_app.py:547` `st.header("Knowledge Base / 知识库")`——**一个字符串里同时含两种语言**，语法上就不可能只显示一种 | 切到 English 仍看到中文，README 宣称的双语不成立 |
| 阶段徽章显示 "Phase 6 · Agent" | `i18n/zh-CN.json:103` 与 `i18n/en-US.json:103` 均为 `"current_phase": "Phase 6 · Agent"` | 与 README 的 "Phase 10" 矛盾，面试官一看就发现 |
| Evaluation 说明写"14 个 Case"，实际有 15 个 | UI 硬编码 14；我实测 `len(cases) == 15`；README 也写 15 | 数字对不上 |
| 有 30+ 个 i18n 键定义了却从未被引用 | 例如 `demo_button`、`agent_dataset_required`（而 `streamlit_app.py:772-774` 用内联字面量实现了同样功能） | 说明 i18n 是"先设计后实现"，实现没跟上 |
| 异常检测结果的表头会把内部字段名当标题显示 | `streamlit_app.py:167-172` 用 `translations.get(name, name)` 遍历 `model_dump()`，于是 `q1`、`lower_bound`、`tool_name`（值还是 `detect_outliers`）直接显示在中文界面上 | 界面漏出内部实现细节 |

**修复方向：**把这两块文案提取进两个 JSON；把 `len(evaluation_cases)` 动态渲染而不是写死 14；改掉阶段徽章；对未使用的键要么接上要么删除；加一个测试断言两个语言文件的键集合完全相同。

**补一个防回归测试（很有说服力）：**用正则扫描，若 `st.*()` 调用的字符串字面量里含 CJK 汉字则测试失败。这一条测试就能永久防止 i18n 再次漏做。

**面试怎么讲：**"i18n 不只是翻译文件，还要有机制保证不退化。我加了一个静态检查：UI 代码里的字符串字面量不允许出现中文，否则测试失败。这样新增页面时不可能再漏翻译。"

---

## P1-5【重要】错误的文件上传会丢失真正有用的提示

**位置：**`app/ui/streamlit_app.py:313-317`

```python
except DatasetLoadError:
    for key in (...):
        st.session_state.pop(key, None)
    _clear_analysis_state()
    st.error(t["error"])            # ← 具体的错误原因被丢掉了
```

**这里最讽刺的地方是：`dataset_service.py` 的报错信息写得非常好。**它精确区分了：

- `"The uploaded file is empty."`
- `"The file is larger than the 20 MiB upload limit."`
- `"Could not decode this CSV. Save it as UTF-8 or GB18030 and retry."`（**给了用户可执行的动作**）
- `"Duplicate column names were found. Rename them and upload again."`
- `"Could not read this Excel file. Check that it is a valid .xls or .xlsx workbook."`

而 UI 把这一整套诊断信息统一替换成了 `t["error"]`（"Could not load this file"）。用户拿不到任何下一步动作。

**修复方向：**`except DatasetLoadError as exc: st.error(f"{t['error']} {exc}")`，更彻底的做法是按错误类别映射 i18n 键。

**次要项（同类）：**`streamlit_app.py:485-492` 的"缺失值分析"和"重复值分析"两个按钮**没有 try/except**，而紧邻的 496-518 行的异常值/相关性按钮都有。目前 `analysis_results` 在数据集加载时一定会被初始化，所以**实际不可达**（我验证过这条路径），但这是不一致的防御，属于隐患而非现存 bug。

---

## P1-6【重要】文档与现状不一致（会误导面试官和你自己）

| 文件 | 问题 |
|---|---|
| `docs/real_run_acceptance.md:28` | 写着"`.env` 不存在"，但你现在 `.env` 已存在，且**LLM 已验证可用**。这份验收报告已过期 |
| `docs/real_run_acceptance.md:80` | 写着 LLM "BLOCKED"，现在应该更新为"已验证真实调用" |
| `README.md:7,297-298` | "Phase 9（当前）"和"Phase 10（当前）"**同时标为当前**，自相矛盾 |
| `README.md` 全文 | 除标题外几乎全中文。GitHub 面向国际读者，建议提供英文版或双语 |
| `requirements.txt` | 没有 `langchain`（项目最终选择不用，README 有说明，OK），但也没有任何开发工具：`ruff`、`mypy`、`pytest-cov` 都没有 |

**顺带纠正一个常见误解：**`docs/real_run_acceptance.md` 里"未验证真实 Trace / 未验证真实 Embedding / RAG BLOCKED"的写法**非常诚实，是加分项**。我实测确认：`data/qdrant/meta.json` 内容为 `{"collections": {}}`——**知识库从未真正索引过任何文档**。文档如实说明了这一点，没有伪造。请保持这种诚实，只需要在真实验证后更新它。

---

## P1-7【重要】不支持分号/制表符分隔的 CSV——欧洲和 Excel 导出的文件会被静默当成单列

**位置：**`app/services/dataset_service.py:53-77`

```python
header = next((row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)), [])
_validate_header(header)
df = pd.read_csv(io.StringIO(text), on_bad_lines="error")     # ← 默认 sep=","
```

**我实测复现：**

```
分号分隔 a;b;c      -> shape=(2, 1)  columns=['a;b;c']
制表符分隔 a\tb\tc  -> shape=(1, 1)  columns=['a\tb\tc']
竖线分隔 a|b|c      -> shape=(1, 1)  columns=['a|b|c']
分号文件重复表头 A;A -> shape=(1, 1)  columns=['A;A']   ← 连重复表头都没检测出来
```

**为什么严重：**`csv.reader` 和 `pd.read_csv` 都默认用逗号。分号分隔的 CSV 在欧洲非常常见（Excel 在德语/法语区域的默认导出格式就是分号），中文环境导出也常有制表符。结果是：**整个文件被读成一个名为 `a;b;c` 的文本列**，然后"分析"这个列。

关键在于**它不报错**。用户上传后看到的是"Rows: 2, Columns: 1"，画像正常、质量分正常，只是所有分析都是错的。这比拒绝文件糟得多。而且 `_validate_header` 用的是同一个错误的逗号解析结果，所以"重复列名"这种校验也一起失效了。

**修复方向：**用 `csv.Sniffer` 嗅探分隔符，或按 `[",", ";", "\t", "|"]` 依次尝试并选择列数最多的结果；嗅探结果要同时传给 `_validate_header` 和 `pd.read_csv`。如果嗅探不确定，就明确询问或报错，**绝不静默单列**。

**面试怎么讲：**"我遇到过一个静默错误：分号分隔的 CSV 被 pandas 当成单列，整份数据被误读但界面完全正常。我加了分隔符嗅探，并让表头校验和实际解析共用同一个分隔符。这让我意识到数据入口的容错必须'显式失败'，不能'静默降级'。"

---

## P1-8【重要】知识文档重新索引时不会删除旧切片，检索会一直返回已删除的旧内容

**位置：**`app/services/knowledge_service.py:44-78`

```python
def index_document(self, document: KnowledgeDocument) -> KnowledgeIndexResult:
    ...
    try:
        ...
        self.vector_store.upsert_chunks(batch, vectors)     # ← 只 upsert，没有先 delete
        ...
    except EmbeddingServiceError as exc:
        self._delete_partial(document.document_id)          # 只有失败路径才清理
```

**问题：**`_delete_partial` 只在**异常路径**被调用。正常重新索引时，新切片是"追加/覆盖"，**旧切片不会被删除**。

**具体后果（审查过程中用真实本地 Qdrant 复现）：**同一份文档先索引 12 个切片，修正内容后重新索引成 1 个切片——检索仍返回 4 条命中，**全部是旧切片**（`doc-1:6..9`），没有一条是当前内容。

**为什么严重：**这是"知识库说的事实与文档实际内容不符"。面试演示 RAG 时，如果你先索引一版、改了再索引一版，回答会引用**已经不存在的内容**——而且系统会把它作为"来源"展示出来，看起来还很可信。这正好打击项目最想体现的"有据可依"。

**修复方向：**在 `index_document` 成功写入前（或写入开始时）**无条件**调用 `vector_store.delete_document(document.document_id)`，再 upsert 新切片。要处理"删除成功但 embedding 失败"的中间态——此时文档状态应标记为 `indexing`/失败，而不是保持旧的 `indexed`。

**需要补的测试：**索引 N 个切片 → 重新索引成 M 个切片（M < N）→ 检索结果必须只含新切片。

---

## P1-9【重要】`order_id` 这类主键列被误判为"未知类型"，并因此扣掉质量分

**位置：**`app/services/profiler.py:66-73, 105-110, 183`

分类正则用了 `\bid\b`，而 `\b` 是"单词边界"。下划线 `_` 在正则里算**单词字符**，所以 `\bid\b` **无法匹配 `order_id`**（`id` 前后是 `_` 和结尾，`_` 侧不构成边界）。

**我实测复现：**

```python
profile_dataframe(pd.DataFrame({
    "order_id": [f"ORD{i}" for i in range(50)],
    "sales":    [float(i) for i in range(50)],
    "region":   ["East", "West"] * 25,
}))

# semantic types: {'order_id': 'unknown', 'sales': 'numeric', 'region': 'categorical'}
# quality_score: 85
# quality_issues: [('many_unknown_types', 15)]     ← 一个完全干净的数据集被扣 15 分
```

**为什么重要：**一个没有任何缺失、没有任何重复、类型完全正常的干净数据集，质量分只有 85，理由是"字段类型识别失败"——而失败的原因只是它有一个叫 `order_id` 的主键列。**质量评分是给用户看的第一印象指标**，现在是错的。

**修复方向：**把正则从 `\bid\b` 改成同时匹配下划线/驼峰形式，例如 `(^|_)id($|_)` 或 `(?<![a-z])id(?![a-z])`；并补测试覆盖 `order_id`、`user_id`、`productID`、`id` 这些常见形态。顺带把人类可读的类型识别收敛成一张显式的关键词表，而不是散落在正则里。

**面试怎么讲：**"我的质量评分曾经把一张完全干净的表打成 85 分，原因是 `\bid\b` 匹配不到 `order_id`——下划线属于正则的单词字符，所以没有单词边界。这个 bug 让我明白：给用户看的评分指标必须用真实数据集做回归测试，否则指标本身就是噪音。"

---

## P1-10【重要】行数/列数上限在 pandas 读完整份文件之后才检查

**位置：**`app/services/dataset_service.py:43-46, 70, 91`

**实测：**一个 20.7 MiB 的 CSV（未超过 20 MiB 字节上限，所以通过了大小校验）会让 pandas 先构造出 **1,400,000 × 2** 的 DataFrame、峰值内存约 **138 MB**，然后 `_validate_frame` 才因为超过 100,000 行而报错。

**问题：**代码里完全没有 `nrows=` 或 `chunksize=`。所有上限都是"事后检查"（post-hoc），也就是**用户已经消耗了内存之后才拒绝**。

**为什么重要：**上限存在的意义就是防止资源耗尽。现在它只防止了"把超大数据留在内存里"，没有防止"读进来时就把内存打满"。在 Streamlit 单进程里，一次恶意/误操作的大文件上传可以拖垮整个应用（甚至影响其他用户会话）。

**修复方向：**先按字节数粗筛（已有），再用 `pd.read_csv(..., nrows=MAX_ROWS + 1)` 读取——多读 1 行用来判断"是否超限"；列数在 `nrows` 之后检查成本很低。Excel 同理可先读表头 + `nrows`。

---

## P1-11【重要】Evaluation 的 pass rate 会"越崩越高"：崩溃的用例被移出分母

**位置：**`app/evaluation/runner.py:137-140, 215-232`

```python
except Exception as exc:
    logger.warning("Evaluation case %s unavailable (%s)", case.case_id, type(exc).__name__)
    return self._not_run(case, "unavailable", ...)      # 意外异常 -> unavailable
...
attempted = passed + failed                              # unavailable 不在分母里
pass_rate = passed / attempted if attempted else None
```

**问题：**任何**意外异常**（也就是真正的代码 bug、Agent 崩溃、工具崩溃）都会被归为 `unavailable`，而 `unavailable` **既不算 passed 也不算 failed，直接退出分母**。

**后果：**如果某个用例让 Agent 崩溃，它不会拉低 pass rate——它**消失**了。极端情况：15 个用例里 13 个崩溃，剩下 2 个通过，显示的 pass rate 就是 **100%**。指标会随着系统变差而变好看。

**为什么这个特别值得警惕：**这属于"指标被自己欺骗"，比功能 bug 更隐蔽。而项目原文第 34 条明确要求"不要伪造 AI"——这个指标设计在无意中做到了"伪造成绩"。

**说明一下公平之处：**`unavailable` 的语义本来是对的——LLM/Embedding 没配置时确实不应该算失败（那是环境问题，不是代码问题）。UI 也如实标注了 "Pass Rate (attempted cases only)"。**问题在于把两类完全不同的原因混进了同一个状态**：
- 环境缺失（合理）：LLM 没配置 → 不能算失败
- 代码崩溃（不合理）：抛了未预期的异常 → **必须算失败**

**修复方向：**拆开这两个状态。环境缺失 → `skipped`/`unavailable`（不进分母）；未预期异常 → `failed`（进分母），并把异常类型记进 `failure_reason`。同时报告里应同时给出 `pass_rate(attempted)` 和 `pass_rate(all)` 两个数字，避免分母被任意操纵。

**面试怎么讲：**"我审查自己的评估指标时发现一个陷阱：未预期异常被归为 'unavailable'，而 unavailable 不计入分母，所以系统崩得越多、pass rate 越好看。我把'环境缺失'和'代码崩溃'拆成两个状态——前者跳过，后者必须计入失败。做评估时，指标的分子分母定义比指标本身更重要。"

---

## P2-0【重要，运维】Qdrant 本地库永不关闭，改 Embedding 维度后必须重启应用

**位置：**`app/vectorstore/qdrant_store.py:38-42, 173`、`app/ui/streamlit_app.py:51-56, 203-207`

- `_shared_qdrant_store` 用 `@st.cache_resource` 缓存，**key 包含 `dimensions`**；
- `QdrantVectorStore.close()` 定义了（`:173`），但**全项目从未调用**。

**后果（审查中实际复现）：**同一个本地路径上开第二个客户端会失败：`VectorStoreError: Qdrant local storage is unavailable`。所以你在 Streamlit 里改了 `EMBEDDING_DIMENSIONS` 并刷新页面，缓存 key 变了、需要新建客户端，但旧客户端还占着本地锁 → 打不开 → 知识库直接不可用，**只能重启应用**。而且维度不匹配时没有任何"重建索引"的修复路径。

**修复方向：**在 `@st.cache_resource` 上加 `on_error`/显式失效逻辑，或在维度变化时先 `close()` 旧实例；对维度不匹配给出明确的操作指引（换 collection 名或删除本地目录后重建）。

**这条在本地单机 Qdrant 场景下属于"必须知道"的运维知识**，README 已经提到了"本地目录不应由多个进程同时打开"，建议把"改维度需重启/重建"也写清楚。

---

## P1-12【重要】前导零被静默销毁——邮编/商品编码变成数字

**位置：**`app/services/dataset_service.py:70, 91`

```python
df = pd.read_csv(io.StringIO(text), on_bad_lines="error")   # 没有 dtype / converters
```

**我实测复现：**
```python
load_csv(b"zip,city\n01234,Beijing\n05678,Shanghai\n", "z.csv")
# dtypes: {'zip': 'int64', 'city': 'str'}
# values: [1234, 5678]          ← 前导零消失，且类型变成整数
```

**为什么严重：**邮编、工号、SKU、订单号这类"看起来像数字的字符串"会被 pandas 自动转成 int。`01234` 变成 `1234` 之后：
- 数据本身已经错了（不可逆，除非重新读文件）；
- 画像会把它报告为 `numeric` 列，所以用户**看不出**发生了类型转换；
- 后续任何按该列分组/关联的分析都在错误的值上进行。

同一问题也影响 Excel（openpyxl 读到的字符串 `'01234'` 会被 pandas 转成 int64）。

**修复方向：**至少提供"把标识列按文本读取"的开关；更好的做法是对匹配 `zip|postal|code|sku|id` 的列用 `dtype=str`，并在画像里显式记录本次读入使用的类型推断规则。

---

## P1-13【重要】Evaluation 的"成绩"站不住：真值来自被测代码，指标可以被刷高

这是**本次审查中第二重要的问题域**（仅次于 Agent 崩溃）。因为它直接关系到你在简历上写"我做了 Agent Evaluation"时的可信度。我逐条验证：

### (a) 真值是用被测代码自己生成的——它只能检测"变化"，不能检测"错误"

**位置：**`data/evaluation/datapilot_mvp_v1.json`（如第 9 行的 `"expected": 0.963202`）

审查中我用 `data/sample_sales.csv` 重算确认：`calculate_correlation(...).correlation == 0.963202`、`total_missing == 11`、`outlier_count == 110`。也就是说 **JSON 里的期望值就是把当前实现跑一遍填进去的**。

**为什么这是根本问题：**如果 `calculate_correlation` 的实现是错的（比如算成了 Spearman 或漏了 NaN 处理），只要这个错误是稳定的，Evaluation 就会**依然全部通过**。这种测试是"变更检测器"（change detector），不是"判据"（oracle）。而 README 里"确定性 Ground Truth"的措辞会让读者以为存在独立的正确答案来源。

**修复方向：**至少加一个**手工计算**的小数据集夹具（5–10 行，答案在数据集文件里用注释写明推导过程），并在 JSON 里记录 `"ground_truth_source": "hand-computed"`。这一个动作就能把"我不知道我在测什么"变成"我有独立判据"。

### (b) 只要没有 `expected_facts`，"有据可依"就恒为真

**位置：**`app/evaluation/metrics.py:81-88`

```python
if expectation.expected_facts or expectation.required_sources:
    facts_present = all(_norm(fact) in _norm(answer_text) for fact in expectation.expected_facts)
    metrics["grounded_answer"] = bool(facts_present and citations_valid is not False)
```

**关键点：**Python 里 `all([])` 等于 `True`。所以当一个用例只填了 `required_sources` 而没填 `expected_facts` 时，`grounded_answer` 就退化成"引用列表校验通过了吗"——**回答内容完全不参与判断**。审查确认数据集中有 3 个用例正好是这个形态（`datapilot_mvp_v1.json` 第 11–13 行）。一个回答 "I do not know." 只要引用对得上，`grounded_answer` 就是 `True`。

**修复方向：**`expected_facts` 为空时返回 `None`（不可测量），而不是 `True`。

### (c) 工具选择指标奖励"乱调工具"

**位置：**`app/evaluation/metrics.py:10-15`

```python
required = set(expected)
if not required: return None
return len(required.intersection(actual)) / len(required)     # 只看召回，不看多余调用
```

**我实测复现：**
```python
tool_selection_score(["find_top_n"], ["find_top_n"] + [f"junk{i}" for i in range(20)]) == 1.0
```
调用 1 个正确工具 + 20 个无关工具，仍得满分 1.0。**这个指标在鼓励 Agent 乱调工具**，而"不乱调无关工具"恰恰是 Agent 质量的核心。修复：同时报告 precision（或对多余调用扣分）。

### (d) "accuracy" 其实是一个通过率，调宽容差就能刷高

**位置：**`app/evaluation/metrics.py:79-80`

```python
metrics["numeric_accuracy"] = matched if ground_truth.comparator == "numeric_tolerance" else None
```
`numeric_accuracy` 是"使用数值容差比较的用例的通过率"，**不是误差量级**。容差 `1.0` 和 `1e-5` 在指标上完全等价，所以把容差调宽就能让这个数字变好看。修复：改名为 `numeric_tolerance_pass_rate`，或同时报告平均相对误差。

### (e) 事实匹配用子串，中文场景误判率高

`metrics.py:84` 用 `_norm(fact) in _norm(answer_text)` 做子串匹配。英文下 `"profit"` 会命中 `"profitability is unclear"`；中文下 `"毛利"` 会命中 `"毛利率尚未计算"`。修复：按词边界匹配，或明确标注这是弱代理指标。

### (f) 两个用例名不副实

- `agent_unknown_question`：断言的只是 `expect_success=true`，**不检查"不该调工具"**，所以任何成功回答都通过。
- `agent_invalid_configuration`：只带了一个 `metadata.scenario` 标签，而 `metadata` 字段在 `app/evaluation/runner.py` 里**从未被读取**（审查确认 grep 无匹配）。也就是说一个名为"无效配置"的用例完全没有测无效配置。

### (g) 两个测试函数重名，第一个永远不会执行

**位置：**`tests/test_evaluation_runner.py:129` 与 `:160`

**我实测复现：**
```
total defs: 8 | unique: 7 | duplicated: {'test_optional_judge_is_opt_in_and_never_overrides_deterministic_failure': 2}
```
Python 只保留后一个定义，所以第 129 行的测试**从未运行过**（`pytest --collect-only` 只列出 7 个）。注意：**总测试数 149 里已经少算了它**。这本来应该是"Judge 与确定性结果不一致"的第二个场景。

### (h) chunking 的"overlap"测试断言了错误的常量，且从没验证过重叠

**位置：**`tests/test_knowledge_chunking.py:42-49`

**我实测确认：**
```python
chunks = chunk_document(..., target_tokens=40, overlap_tokens=8, max_tokens=60)
assert all(approximate_token_count(chunk.text) <= MAX_TOKENS for chunk in chunks)   # MAX_TOKENS == 700
```
传入 `max_tokens=60`，却拿模块默认值 **700** 去断言——这个断言是**空的**（即使 chunk 超限 10 倍也会通过）。而且尽管测试名叫"with_overlap"，**没有任何一行断言相邻 chunk 真的重叠**。同文件其他测试都用了自己的字面量，所以这是不一致而非风格选择。

**面试怎么讲（这一整块是最好的题材）：**"我审查自己的评估框架时发现三个问题：真值是用被测代码生成的（只能检测变化而非错误）、`all([])` 让空事实列表的'有据可依'恒为真、工具选择只看召回不看多余调用。我补了手工计算的判据夹具、区分'不可测量'与'通过'、并增加工具 precision。这让我理解到：**评估框架本身的正确性需要被独立审查**，否则它只会系统性地确认自己的假设。"

---

## P1-14【重要】一次写入失败会让持久化在整个进程内永久失效

**位置：**`app/services/persistence_service.py:53-56`（`82-85` 同样）

```python
except Exception as exc:
    self._status = "unavailable"        # ← 任何异常都判定为"数据库不可用"
    logger.warning("Dataset metadata persistence failed (%s)", type(exc).__name__)
    return False
```

**审查中实测确认：**一次外键 `IntegrityError`（为不存在的数据集写分析记录）之后，`status` 变为 `unavailable`，此后即便是**完全合法**的 `persist_dataset` 也永远返回 `False`。

**问题：**把"单条记录写失败"（数据问题，可恢复）和"连接断开"（基础设施问题）混为一谈。单条脏数据就永久关掉了整个会话的持久化，而且 UI 上只显示一个笼统的状态警告。

**修复方向：**区分异常层级——`IntegrityError`/`DataError` 等记录级错误只记录并跳过；`OperationalError`/连接类错误才标记 `unavailable`，并在标记前重试一次 `check_connection()`。另外流式写入短问答场景应考虑加一个"重新尝试连接"的入口。

---

## P1-15【重要】环境变量填错值没有警告，API 错误码丢失

### (a) 非法的数值配置被静默替换为默认值

**位置：**`app/config/settings.py:13-26`

审查实测：`LLM_TIMEOUT="abc"` → `30.0`、`LLM_MAX_TOKENS="-5"` → `1024`、`EMBEDDING_DIMENSIONS="0"` → `1024`，而 `configuration_problem` 返回 `None`、`is_configured` 返回 `True`。**UI 会显示绿色的"已配置"状态**，用户永远不知道自己打错了。缺少边界：`max_tokens <= 8192`、`1 <= dimensions <= 8192`、`1 <= port <= 65535`。

**修复方向：**让 `from_env` 收集 `warnings` 并纳入 `configuration_problem`，UI 显示"`LLM_TIMEOUT='abc'` 不是正数，已使用 30"。

### (b) 401（密钥错）和 429（限流）在用户眼里完全一样

**位置：**`app/services/llm_service.py:106-107`

```python
except APIError as exc:
    raise LLMServiceError("The LLM provider returned an API error.") from exc
```
丢弃 body 是对的（避免泄密），但**连状态码一起丢掉**就让用户无法自助排查。修复：把 `getattr(exc, "status_code", None)` 放进安全消息里（**绝不放 body**），并让非 openai 的异常走独立的内部错误类型，不要伪装成"服务商错误"。

---

## P2：值得顺手清理的项（不影响运行）

- **`QdrantVectorStore.is_available` 会说假话**：它只调 `get_collections()`，所以**维度不匹配的 collection 也会被报告为"可用"**，真正的错误延迟到 upsert 时才以笼统的 `knowledge_base_unavailable` 暴露。
- **`EvaluationRunner._configured_dependencies` 把 qdrant 硬编码为 `True`**（`runner.py:47-56`）：注释说"由 UI 验证"，但默认值会让不传 `dependency_status` 的调用方（脚本、未来的 CLI、忘了传的测试）**在不存在的向量库上"运行" RAG 用例**。同文件 `QdrantSettings` 被 import 但从未使用——这正是"探针没写完"的指纹。
- **`Excel` 只读第一个 sheet，且第一个 sheet 为空时会拒绝一个有效工作簿**：而且为探测表头读了一遍、正式读又读一遍（大文件双倍开销）。
- **没有数据库迁移路径**：`create_all` 幂等创建，但**永远不会修改已存在的表**。将来加字段需要手工建表或引入 Alembic，建议至少在 README 写明。
- **`app/services/profiler.py` 把中文提示硬编码进数据模型**（如 `message=f"缺失单元格占 {missing_rate:.2f}%，高于 20%。"`），而 `i18n/*.json` 里**存在但从未被使用**的 `quality_desc_*` 模板恰好就是这些文案。这既破坏了 i18n，也让展示逻辑漏进了分析层。
- **`settings.py` 里同一段 `.env` 读取逻辑重复了 5 遍**，`_valid_base_url` 完全重复了 2 遍（96-110 与 170-184）。已经能看到漂移：`QdrantSettings` 完全没有 `is_configured`/`configuration_problem`。
- **`aggregate_metrics` 有死逻辑**：三个违规率在 103-105 的 `names` 映射里已经算过，125-131 行又用相同方式算了一遍覆盖掉。
- **`dataset_columns.missing_rate` 存的是百分比**（`dataset_repository.py:42-50` 把 `missing_percentage` 存进名为 `missing_rate` 的列）。75% 缺失会存成 `75.0`，读表的人会理解成 7500%。
- **`.env.example` 只给了旧的 `LANGCHAIN_*` 名称**，而代码优先读 `LANGSMITH_*`；另外 `LANGSMITH_TRACING=`（空值）会**覆盖并关掉** `LANGCHAIN_TRACING_V2=true`，因为这一行用的是 `.get(key, default)` 而不是 `or`。
- **缺少 P1-7 至 P1-13 对应的回归测试**：审查确认 `tests/` 里搜不到任何 `delimiter|sep=|semicolon` 相关内容——**两个阻断级 bug 对测试套件完全不可见**，而现有 `test_quality_score_is_deterministic_and_explained` 断言"干净数据 = 100 分"，恰好看不出 `order_id` 的问题。

## P2-工具链：工程规范（让 GitHub 仓库显得专业）

这些不影响运行，但是面试官打开仓库后会注意到的信号：

1. **没有 CI**：没有 `.github/workflows/`。建议加一个最简 GitHub Actions：安装依赖 → `pytest -q` → `ruff check`。这是"我懂工程规范"的最强信号之一。
2. **没有 `pyproject.toml`**：项目没有打包配置、没有工具配置。建议把 ruff/pytest/mypy 配置集中在这里。
3. **没有 LICENSE**：GitHub 仓库没有许可证。
4. **没有代码检查工具**：`ruff`（格式化+lint）、`mypy`（类型检查）都没装。项目类型注解写得相当好，加上 mypy 能直接体现出来。
5. **没有测试覆盖率**：149 个测试但不知道覆盖多少代码。建议 `pytest-cov`，先看基线再定目标（不要追数字，Coverage 只是找盲区的工具）。
6. **空的遗留目录**：`app/core/__init__.py` 是原始计划里 `app/core/config.py` 的残留（实际配置在 `app/config/`），`app/utils/__init__.py` 也是空的且从未使用。要么删除，要么说明。
7. **`app/tools/__init__.py` 是 0 字节**，而 `app/agent/__init__.py` 有 4 行显式导出——包接口风格不一致。
8. **`analysis_tools.py` 的私有函数被跨模块导入**：`visualization_tools.py:12-14` 导入了 `_numeric_column`、`_require_frame`、`_validate` 等下划线私有函数。这说明应该把公共校验逻辑抽到一个共享模块（比如 `app/tools/_guards.py`），而不是让兄弟模块访问私有 API。
9. **`app/tools/registry.py:83`**：`search_knowledge` 注册的函数是一个 `lambda **_: None` 空操作，实际逻辑走的是 `registry.py:143-159` 的特判分支。这个 lambda 永远不会被正确调用，属于误导性死代码。
10. **`MAX_ITERATIONS = 8` 定义了两遍**：`analyst_agent.py:20` 和 `agent_models.py:57` 的 `le=8` 硬编码。改一处忘另一处会导致 `AgentResponse` 构造抛 ValidationError。应该抽成单一常量。

---

## 三、明确"做得好的地方"（这些要在面试里主动讲）

审查不只是找缺点。以下是我实测确认**质量明显高于该阶段**的部分，请务必在面试中主动展示：

1. **安全边界是真的严**：全仓库搜索确认**没有** `exec`、`eval`、`pickle`、`subprocess`、`os.system`、动态导入。工具分派是 `MappingProxyType` 上的精确字典查找（`registry.py:92-97`），模型无法通过工具名触达任意 Python。文件读取全部走字节流 + 内存，从不打开用户指定的路径。
2. **密钥管理规范**：所有密钥用 Pydantic `SecretStr`，`LLMService`/`PersistenceService` 从不打印密钥。我验证过 UI 只显示"配置状态"不显示值。
3. **数据库层写得专业**：`database/core.py` 有 `pool_pre_ping`、`connect_timeout`、SQLite 内存库的 `StaticPool` + `check_same_thread=False`、session 上下文管理器完整实现 commit/rollback/close、事务失败自动回滚、`create_all` 幂等不删数据。`PersistenceService` 的降级设计正确——数据库不可用绝不阻塞内存分析。
4. **可选依赖的降级处理正确**：MySQL / Qdrant / LangSmith / Embedding 全部可缺失，缺失时明确显示状态而**不伪造结果**。我实测 UI 在 Demo Mode 下确实返回固定文案（`llm_service.py:54-61`），没有假装 AI 在运行。
5. **Qdrant 与数据库连接在 Streamlit 里正确缓存**：`@st.cache_resource`（`streamlit_app.py:51`）+ session_state 复用 PersistenceService（`:188-196`），并在配置变更时 `close()` 旧连接。这是很多人会写错的地方（每次 rerun 新建连接导致泄漏），这里做对了。
6. **数值健壮性细节到位**：inf 被转成 NaN（`analysis_tools.py:86`）、非有限值转 `None`、除零全部有保护、`n` 的上界由 Pydantic 约束（`ge=1, le=100`）、`n` 超过行数时如实返回实际数量。
7. **相关性/因果的措辞谨慎**：全仓库搜索确认没有任何地方声称因果关系，且 `CorrelationResult` 自带 `"Correlation does not imply causation."` 说明。这是数据素养的体现。
8. **上面第 1、3、4、5 条以及 149 个测试，都是可以直接放进简历的。**

---

## 四、下一步处理计划（建议严格按此顺序）

**核心原则：先修真实链路，再谈新功能。**在你的 Agent 真实成功率从 1/10 提到 ≥8/10 之前，**不要碰 RAG 增强、不要碰 FastAPI、不要碰 Docker**。理由：那些都是在"能跑的核心"上加东西；而你现在还没证明核心能跑。

### 第 1 步：保住工作成果（30 分钟，今天必须做）

- [ ] 备份项目目录（复制一份到别处）
- [ ] 按 P0-7 的 4 条命令完成首次 Git 提交，**并确认 `.env` 未被提交**
- [ ] 核对 `.env` 里的模型名是否与服务商控制台一致
- [ ] `docs/real_run_acceptance.md` 更新为当前真实状态（`.env` 已存在、LLM 已验证）

### 第 2 步：修 Agent 崩溃路径（半天）

按顺序修，每修一个就补一个**真实场景**的测试：

- [ ] **P0-1** 参数命名空间：`invoke_analysis_tool` 改为收 `parameters` 字典 + 调用点兜底
  - 测试：`tool_input={'name': 'x'}` 必须返回结构化失败，而不是抛异常
- [ ] **P0-2** JSON 修复：清洗围栏 + 解析失败重试 2 次（视情况开 JSON mode）
  - 测试：把 ```json 围栏的响应喂进去，必须能被正确处理
- [ ] **P0-3** 工具错误反馈：错误进 `history` 后 `continue`，同工具连错 2 次熔断
  - 测试：先猜错列名、再改对，必须能成功完成
- [ ] **P0-4** 重试 + 部分结果返回：`max_retries=2`，任何终止路径都返回已有证据
  - 测试：Mock 一个"第 2 轮抛 LLMServiceError"的 LLM，断言仍返回第 1 轮的发现

### 第 3 步：补上项目的"灵魂"（1–2 天）

- [ ] **P0-5** 结构化结论：`FinalAnswer` + `ClaimType`（fact/interpretation/hypothesis）+ "fact 必须有 tool 证据"的校验 + UI 分段渲染
- [ ] **P0-6** Agent 图表：artifact 双通道 + `plt.close` + figure 泄漏回归测试
- [ ] **P1-1** 数值日期列修复（这是会输出错误结论的真 bug）
- [ ] **P1-2** 统一 NaN 聚合语义
- [ ] **P1-3** 上下文分层压缩
- [ ] **P1-7** CSV 分隔符嗅探（否则欧洲/Excel 导出的文件全部静默误读）
- [ ] **P1-9** 修 `\bid\b` 正则 + 质量分回归测试（干净数据不该被扣分）

### 第 4 步：修数据入口、评估可信度与持久化（可与第 3 步并行，1 天）

- [ ] **P1-8** 重新索引前删除旧切片 + "重索引后检索不得返回旧内容"的测试
- [ ] **P1-10** 用 `nrows=` 做前置行数限制，避免上限变成"事后检查"
- [ ] **P1-11** 拆开 Evaluation 的 `unavailable`（环境缺失）与 `failed`（代码崩溃），崩溃必须进分母
- [ ] **P1-12** 标识列按文本读取（前导零不可丢失）
- [ ] **P1-13** 评估可信度：手工计算判据夹具、空事实返回"不可测量"、工具 precision、修重名测试与空断言
- [ ] **P1-14** 区分记录级与连接级持久化失败
- [ ] **P1-15** 非法环境变量给警告；API 错误保留状态码
- [ ] **P2-0** 维度变更时的 Qdrant 客户端关闭/重建路径
- [ ] **P2** 清理清单（profiler 中文硬编码、settings 重复代码、missing_rate 单位、`.env.example` LangSmith 名称等）

### 第 5 步：真实端到端验收（必须留下证据）
写一个 `docs/real_run_acceptance_v2.md`，包含：

- [ ] 用 8–10 个有代表性的真实问题跑 Agent，**记录成功率**（诚实记录，包括失败）
- [ ] 配置一个真实 Embedding Provider（`EMBEDDING_PROVIDER=openai_compatible`），**真正索引**内置知识文档，然后验证一次真实 RAG 检索
- [ ] 配置 MySQL，真实跑一次 `SELECT 1` 和一条元数据写入
- [ ] 配置 LangSmith，真实产生一次 Trace 并截图
- [ ] 把上面的结果如实写进 README（成功就写成功，没验证就写没验证）

### 第 6 步：工程规范化（1 天）

- [ ] `pyproject.toml` + `ruff` + `mypy` + `pytest-cov` 基线
- [ ] `.github/workflows/ci.yml`（install → pytest → ruff）
- [ ] 补 P1-4 的 i18n 静态检查测试
- [ ] 修文档矛盾（Phase 徽章、14/15、README 的"当前"）
- [ ] 加 LICENSE、清理 `app/core/`、`app/utils/` 等空目录
- [ ] 更新 README：加上真实运行截图、真实成功率、架构图

---

## 五、需要你先回答的两个问题

这两个问题会直接影响第 5 步能否完成：

1. **`LLM_MODEL=deepseek-flash` 是你从服务商控制台拿到的模型名吗？**（官方 DeepSeek API 公开的是 `deepseek-chat` / `deepseek-reasoner`；我实测 `deepseek-flash` 能返回结果，但需要你确认来源。另外我实测本机到该 endpoint 有**间歇性连接失败**，这一点需要你判断是网络环境问题还是 endpoint 配置问题。）
2. **你能拿到一个支持 Embedding 的 API 吗？**（`BAAI/bge-m3` 或 OpenAI `text-embedding-3-small` 都行。）如果拿不到，RAG 就永远只能停留在"代码写好了但没验证过"的状态——而这恰恰是项目原文第 34 条最在意的"不要伪造 AI"。

---

## 六、面试叙事建议（把这次的发现变成优势）

你现在的处境其实**非常适合讲一个高质量的技术故事**：

> "这个项目我按 10 个 Phase 迭代。前面 6 个阶段 Mock 测试全绿，我以为 Agent 已经完成了。直到我用真实 LLM 做端到端验收，发现成功率只有 1/10。我做了故障分析，定位到四类问题：①模型偶尔把 JSON 包进 Markdown 代码块导致解析失败；②工具参数错误后没有反馈给模型，Agent 无法自我纠正；③外部 API 瞬时故障直接终止任务并丢弃全部分析证据；④模型返回的工具参数可能与函数参数同名，导致 Python 参数绑定阶段抛异常，整个流程崩溃。针对这四类问题，我分别实现了输出清洗与修复重试、错误观察回灌与熔断、指数退避与部分结果降级、以及把模型参数作为独立数据对象传入。修完后真实成功率从 1/10 提升到 X/10。"

**第二个可以讲的故事（评估框架的自我审查）：**

> "我在检查自己的 Evaluation 框架时发现，`grounded_answer` 指标在用例没填 `expected_facts` 时会恒为真——因为 Python 的 `all([])` 是 True。我还发现工具选择指标只看召回不看多余调用，所以调用 1 个正确工具加 20 个无关工具仍然得满分。更重要的是，我发现真值是跑被测代码生成的，这是变更检测器而不是判据。我补了手工计算的夹具、区分'不可测量'与'通过'、并增加了工具 precision 指标。"

这两个故事包含：**真实指标、故障归因、分层修复、指标本身的可信度审查**——这是面试官最想听到的东西，远远超过"我用了 LangChain 做了个 Agent"。

---

## 七、本能总结：这个项目的真实状态

| 维度 | 评价 |
|---|---|
| 架构设计 | 优秀。分层清晰（models/services/tools/agent/ui/repositories），依赖方向正确 |
| 安全边界 | 优秀。白名单工具、无 exec/eval、SecretStr、路径不暴露；唯一缺口是 P0-1 的参数命名空间 |
| 测试 | 数量良好（149 个真实断言），但**全部是 Mock**，且有两个阻断级 bug 完全未被覆盖，还有 1 个重名测试从未执行 |
| 文档 | 良好且**诚实**（明确标注未验证项），但有过期内容与自相矛盾 |
| 基础设施 | 优秀。DB 降级与事务、缓存生命周期、可选依赖处理都对 |
| **核心 Agent 可靠性** | **不合格**。真实成功率 1/10，且任何失败都丢弃已收集的证据 |
| 数据入口正确性 | 不合格。分号 CSV 静默变单列、前导零被销毁、数值日期列静默变 1970 |
| **评估可信度** | **不合格**。真值来自被测代码、空事实判定恒真、工具指标鼓励乱调工具、崩溃用例被移出分母 |
| 工程规范 | 待补。无 CI、无 lint、无 pyproject、无 LICENSE、**Git 0 提交** |

**一句话：地基和装修都做得比预期好，但主承重墙（Agent 真实可靠性）和地基验收标准（数据入口与评估可信度）都有结构性问题。按 P0 → P1 的顺序修完，这个项目就真的能拿去面试了。**

---

# 八、修复记录（审查后执行，2026-10-10）

**范围：**P0-1 ～ P0-6 全部阻断项 + P1-1～P1-3、P1-7～P1-12、P1-14 + P2-0 + M17。
**验证方式：**全量 `pytest` + 真实 LLM 端到端运行（同一批问题、同一环境）。
**关键指标：**

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 全量测试 | 149 passed | **226 passed**（+77 个回归测试） |
| 真实 Agent 成功率 | **1/10** | **6/6**（另一轮 10/10；中间一次诚实记录为 4/6，原因见 8.3） |
| 结构化结论覆盖 | **0/10**（代码里根本没有） | **6/6**（每个 5 条结论 + 3 条建议） |
| 事实/推测分离 | 只写在 Prompt 里，无强制 | 代码强制，0 条错误降级 |
| Agent 图表 | 丢弃 + 每次泄漏一个 figure | 6/6 中 1 次成功返回图表，无泄漏 |

> **诚实声明：**修复后的 6/6 是在**同一台机器、同一个 endpoint** 上取得的。该 endpoint 存在**间歇性连接失败**（修复前一轮 8 次调用中有 5 次 `APIConnectionError`），重试机制降低了但它不是零概率。所以"6/6"不等于"永不失败"；失败时现在的行为是**返回已算出的证据**，而不是丢一句错误。

## 8.1 Agent 核心（我直接修改）

| 编号 | 修复内容 | 关键文件 |
|---|---|---|
| **P0-1** | `invoke_analysis_tool` 新增 `parameters=` 安全通道；模型参数不再 `**` 展开，命名冲突不可能在参数绑定阶段抛异常；调用点再加一层 `except` 兜底。旧的关键字调用方式保留，因此测试与脚本无需改动。 | `app/tools/registry.py`、`app/agent/analyst_agent.py` |
| **P0-2** | 新增 `_json_candidates()`：清洗 ```json 围栏、剥离前后散文、提取成对 `{}`（最后一个优先，专治"模型把同一对象重复两次"）。解析失败时把校验错误回灌并重试 2 次；若 `finish_reason=length`，修复提示会明确说"上次被截断了，请更简洁"。 | `app/agent/analyst_agent.py` |
| **P0-3** | `invalid_tool_request` / `tool_execution_error` 现在作为 **Observation** 回灌给模型并 `continue`（同一工具连错 2 次才熔断）；新增**重复调用复用**：参数完全相同的工具调用直接复用上次结果，不再浪费决策预算。 | `app/agent/analyst_agent.py` |
| **P0-4** | 恢复 SDK 传输重试（新增 `LLM_MAX_RETRIES`，默认 2，原先硬编码 `max_retries=0`）；空响应（HTTP 200 但内容为空）单独重试；连接类失败额外 1 次决策级重试；**任何终止路径都通过 `_partial_evidence()` 返回已实际算出的证据**。 | `app/config/settings.py`、`app/services/llm_service.py`、`app/agent/analyst_agent.py` |
| **P0-5** | 新增 `FinalAnswer` / `Finding` / `ClaimType`；`AgentAction.analysis` 为宽容入站模型；`_validate_structured()` **机械强制**"标为 fact 的结论必须有真实运行过的 `evidence_tool`，否则自动降级为 hypothesis 并计数"；UI 分段渲染 + ✅/🔎/⚠️ 徽章 + 降级提示；i18n 新增 11 个键（中英同步）。 | `app/agent/agent_models.py`、`app/agent/analyst_agent.py`、`app/ui/streamlit_app.py`、`i18n/*.json` |
| **P0-6** | `invoke_analysis_tool(..., artifacts=[...])` 把 `ChartArtifact` 交给调用方渲染，JSON 上下文只留元数据；无人渲染的路径立即 `release()` 关闭 figure；Agent 通过 `AgentResponse.charts`（私有属性，不参与序列化）把图交给 UI 渲染后释放。 | `app/tools/registry.py`、`app/tools/visualization_tools.py`、`app/agent/agent_models.py`、`app/ui/streamlit_app.py` |
| **P1-3** | `MAX_PROMPT_CHARS = 16_000` 成为真正的硬约束，并有**分级降级顺序**：先淘汰最旧证据 → 再缩 schema 列数 → 再降级工具 schema → 最后才截断问题。原实现只裁历史，宽表+长问题会在历史已空的情况下仍然超限。 | `app/agent/analyst_agent.py` |
| 额外 | 决策预算上限提升到 2000 tokens（原 1200 会让结构化长答案被截断成非法 JSON）；Prompt 告知"还剩几次决策"并在最后一次强制要求 `action=final`；`MAX_ITERATIONS` 收敛为单一常量；`run()` 最外层兜底，异常不再变成 Traceback。 | 同上 |

**8.2 数据入口与分析工具正确性**

其中 P1-7 / P1-12 / P1-10 / P1-9 由并行子任务完成（均含 fail-before 验证）；P1-1 / P1-2 由我直接完成。

| 编号 | 修复内容 | 证据 |
|---|---|---|
| **P1-7** | 白名单分隔符嗅探（`, ; \t \|`），按众数行宽 + 一致性打分；表头校验与正式解析共用同一分隔符，`A;A` 重复表头重新被拒绝 | 4 种分隔符各一个测试；重复表头测试 |
| **P1-12** | 匹配标识列名（id/code/sku/uuid/zip/postal/phone/account，含 `postalcode`、`productId`）时按文本读取，前导零不再丢失；`sales`/`quantity`/`profit` 仍为数值 | 11 个参数化用例 + 普通数值列守卫 |
| **P1-10** | `nrows=MAX_ROWS+1` 前置限制（CSV 与 Excel），峰值内存实测 223.7 MiB → 193.4 MiB，构造行数 1.4M → ≤100,001 | 断言 `read_csv` 收到 `nrows` |
| **P1-9** | profiler 新增 `identifier` 语义类型，正则从 `\bid\b` 修正为 `(^\|_)(id\|code\|sku\|uuid\|key)s?(_\|$)`；唯一值率 ≥0.98 也归为 identifier，且不再计入 unknown 扣分 | 干净数据集恢复 100 分；自由文本仍判 unknown（防指标失效的守卫） |
| **P1-1** | 数值列不再被当作日期：`_parse_datetime` 遇到 numeric dtype 直接给出可执行提示（原来 `pd.to_datetime` 把整数当纳秒时间戳，`[2021,2022,2023]` 静默变成 `1970` 且 `success=True`）；同时移除 `utc=True`，避免跨时区数据被移出正确月份 | `test_numeric_date_column_is_rejected_instead_of_becoming_1970`、`test_time_series_periods_keep_the_original_calendar_day` |
| **P1-2** | 统一"分组求和"的缺失值语义：`calculate_group_statistics` 对 `sum` 使用 `min_count=1`，全缺失分组返回 `None` 而不是 `0.0`，与 `find_top_n` 的同一问题结果一致 | `test_the_two_group_sum_tools_agree_about_all_missing_groups` |

**8.3 知识库与持久化（并行子任务完成）+ UI 生命周期（我完成）**

| 编号 | 修复内容 |
|---|---|
| **P1-8** | 重索引前**先删除该文档的旧切片**（原来只在异常路径清理，导致删除的内容仍可被检索到——实测 12 切片缩成 1 切片后仍返回 4 条陈旧命中）；删除成功但 embedding 失败时文档不置为 indexed |
| **P1-14** | 区分记录级失败（`IntegrityError`/`DataError` 等 → 记日志、保持 connected）与连接级失败（`OperationalError`/`InterfaceError` → 标记 unavailable）。原来一次外键错误就永久关闭整个进程的持久化 |
| **P2-0** | 维度不匹配给出独立错误码 `embedding_dimension_mismatch` + `rebuild_collection()`；UI 现在显示可操作提示并提供"重建知识库索引"按钮。同时把 Qdrant 客户端改为**会话内复用 + 配置变更时 `close()` 旧实例**——原来 `@st.cache_resource` 以维度为 key，改维度会尝试在同一路径上开第二个客户端，本地锁冲突导致知识库直接不可用 |
| **M17** | `dataset_columns.missing_rate` 改存 0–1 比例（原先把 75% 存成 75.0），不改列名以避免没有迁移路径的表结构变更 |

**8.4 修复过程中发现并修掉的两个"自造"问题（值得学习）**

1. **P0-5 第一版把成功率从 10/10 打回 4/6**：把 `claim_type` 写成严格 `Literal`、`key_findings` 要求严格的 `statement` 字段后，模型返回的轻微变体（`claim_type: "conclusion"`、字段名用 `claim`）直接导致整个决策校验失败。**修法：安全关键字段保持严格，展示字段宽容解析**（把 `claim_type` 归一化、`key_findings[].claim` 等别名解析、`recommendations` 接受对象形式）。这正是"严格性要用在正确的地方"。
2. **结论被静默丢弃**：宽容解析第一版用了 `extra="ignore"` + 默认空字符串，结果模型把结论放在 `claim` 键下、建议放在 `{"text": ...}` 里，全部被静默丢掉（`findings=0`）。**这是先用真实运行取证、而不是猜出来的**：探针抓到了模型原始输出里的真实键名，据此加了别名表，并加了"非空输入却产出 0 条结论就告警"的日志。

**8.5 仍未修复（按建议顺序）**

| 优先级 | 项 | 说明 |
|---|---|---|
| P1 | **P1-4 i18n 剩余部分** | 知识库区与 Evaluation 区约 120 行文案仍硬编码；`current_phase` 仍写 "Phase 6"；Evaluation 说明仍写"14 个 Case"（实际 15）。建议顺带加"UI 字面量不得含中文"的静态检查测试 |
| P1 | **P1-5 上传错误信息** | `streamlit_app.py` 仍然用笼统的 `t["error"]` 覆盖 `DatasetLoadError` 的具体原因（空文件/超限/编码/重复列名）；缺失值/重复值两个按钮仍无 try/except（当前路径不可达，但应统一） |
| P1 | **P1-11 / P1-13 评估可信度** | 未预期异常仍归为 `unavailable` 而移出分母（越崩分越高）；Ground Truth 仍由被测代码生成（变更检测器而非判据）；`all([])` 使空事实列表的 `grounded_answer` 恒真；工具选择指标仍奖励乱调；两个重名测试函数（第一个永不执行）；chunking 测试断言了模块默认 700 而非传入的 60 |
| P1 | **P1-6 文档** | README 仍有"Phase 9（当前）"与"Phase 10（当前）"并存；`docs/real_run_acceptance.md` 仍写着"`.env` 不存在 / LLM BLOCKED"，与现状不符 |
| P2 | 工程工具链 | 无 CI、无 `pyproject.toml`、无 ruff/mypy/coverage、无 LICENSE；`app/core/`、`app/utils/` 空目录 |
| 待你 | **Embedding + RAG 真实验证** | 你已选定硅基流动方案。拿到 Key 后用 `scripts/check_embedding.py` 验证 5 项，再写入 `.env`、索引文档、跑一次真实检索。**这是"RAG 可以用了"的唯一证据来源** |
| 待你 | MySQL / LangSmith | 仍未真实验证，文档已如实标注 |
| 待你 | **Git 提交** | 本次修复已按你的要求提交到 Git（提交信息见提交记录）。`.env` 仍然只在本机、被 `.gitignore` 忽略，**从未进入版本库**。建议你后续每完成一块就提交一次 |

## 8.6 给面试的两个"修复故事"

1. **"我的测试全绿，但真实运行只有 1/10"**：定位到四类根因（Markdown 围栏导致 JSON 解析失败、工具错误无法自我纠正、外部 API 瞬时故障丢弃全部证据、模型参数与函数参数命名冲突直接崩溃），分别用输出清洗+修复重试、错误回灌+熔断、分层重试+部分结果降级、参数作为单一数据对象传入解决，成功率 1/10 → 6/6。
2. **"我给 Agent 加了结构化输出，成功率反而下降了"**：把 `claim_type` 写成严格枚举后模型返回的等价变体（`"conclusion"`、字段名叫 `claim`）会让整个决策校验失败。我把严格性重新分配——安全关键字段（action/tool_name/tool_input）保持严格，展示字段做归一化解析——并补上"非空输入却解析出 0 条结论就告警"的日志。**结论：严格性要用在正确的地方，宽容解析必须配可观测性，否则会变成静默数据丢失。**

---

## 8.7 中断点与恢复指引（下次从这里继续）

**当前状态（已存档）**

| 项 | 状态 |
|---|---|
| 全量测试 | **226 passed**（命令：`.\.venv\Scripts\python.exe -m pytest -q`） |
| 真实 Agent | **6/6 成功**，6/6 结构化结论（每个 5 条结论 + 3 条建议），0 条错误降级，图表链路返回 1 张图 |
| LLM | `.env` 已配置真实 Key（`deepseek-flash`，SDK 重试 2 次），**可用**；endpoint 有间歇性连接失败，失败时会返回已算出的证据 |
| Embedding | **仍未配置**（`EMBEDDING_PROVIDER=disabled`）。你已选定硅基流动方案，等 Key |
| Qdrant | 本地库为空（`{"collections": {}}`），知识库尚未真正索引 |
| MySQL / LangSmith | 未配置、未真实验证 |
| 本次改动范围 | `app/agent/*`、`app/tools/*`、`app/services/{llm,knowledge,persistence,dataset,profiler}`、`app/models/*`、`app/vectorstore/*`、`app/repositories/*`、`app/config/settings.py`、`app/ui/streamlit_app.py`、`i18n/*`、`tests/*`、`scripts/check_embedding.py`、本文档 |
| 未提交的内容 | `.env`（含真实 Key，被忽略）、`data/qdrant/`（被忽略） |

**下次继续的顺序**

1. **先确认环境没坏**：跑一次 `pytest -q`，应为 226 passed。若数字变了，先查清原因再往下做。
2. **补 Embedding（你的第 2 个问题）**：
   ```powershell
   .\.venv\Scripts\python.exe scripts\check_embedding.py `
     --base-url https://api.siliconflow.cn/v1 --api-key sk-你的Key `
     --model BAAI/bge-m3 --dimensions 1024
   ```
   5 项全过后把打印出的配置块写进 `.env`（`EMBEDDING_PROVIDER=openai_compatible`），重启 Streamlit，在"知识库"区点索引，再问一个文档里有的问题验证 Sources 是真实来源。
3. **继续 P1 剩余项**（顺序建议）：P1-4 i18n 剩余部分 → P1-5 上传错误信息 → P1-11/P1-13 评估可信度 → P1-6 文档一致性 → P2 工具链（CI/pyproject/ruff/mypy/coverage/LICENSE）。
4. **写真实验收报告** `docs/real_run_acceptance_v2.md`：真实成功率、真实 RAG 检索、真实 MySQL、真实 LangSmith Trace，成功就写成功、没验证就写没验证（这一步最能体现"不伪造 AI"）。

**下次开始时的第一句话建议这样说：**"继续 DataPilot AI 的 P1-4：把知识库区和 Evaluation 区的硬编码文案提取到 i18n，并加上 UI 字面量不得含中文的静态检查测试。"——这样我能直接接上，不用重新审查一遍。

