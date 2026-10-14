"""故障复现实验：亲眼看到"离线测试全绿但真实成功率 1/10"到底改了什么。

用途
----
项目在真实端到端验收时成功率只有 1/10（149 个离线测试全绿）。故障归因后定位到
四类根因，本脚本把它们**固化成可反复运行的复现实验**：

  实验 1  模型把 JSON 包进 Markdown 围栏 / 同一对象重复两次
  实验 2  工具报错后无法自我纠正（列名猜错）
  实验 3  网络中断导致已经算出的证据全部丢失
  实验 4  模型参数与函数形参同名，异常逃出整个 Agent

每个实验都打印「故障现象 → 为什么会发生 → 修复后的真实行为」。
**全部使用项目真实代码路径**，模型响应由 Mock 提供：不需要网络、不需要 API Key。

用法
----
python scripts/check_failure_modes.py

退出码：四个实验全部符合预期 0，否则 1。

为什么它不是 pytest 测试
------------------------
这四类行为在 tests/test_agent.py 里都已经有断言覆盖。本脚本的用途不同：
它是**给人看的诊断/教学工具**——把"故障长什么样"和"修复后长什么样"并排打印出来，
这样你能用"我跑过、我见过它怎么失败"的语气描述问题，而不是背结论。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 诊断脚本绝不能因为「打印」而崩掉。中文 Windows 的 GBK 代码页无法编码 ✓ / ✗，
# 一旦输出被管道或重定向，print 就会抛 UnicodeEncodeError，让人误以为是实验失败。
# 这里保留终端原有编码，只把无法编码的字符替换成 ?，让脚本在任何代码页下都能跑完。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import pandas as pd  # noqa: E402

from app.agent import AgentRequest, AnalystAgent  # noqa: E402
from app.agent.agent_models import AgentAction  # noqa: E402
from app.models.dataset_context import DatasetContext  # noqa: E402
from app.models.llm_models import LLMResponse  # noqa: E402
from app.services.llm_service import LLMServiceError  # noqa: E402
from app.services.profiler import profile_dataframe  # noqa: E402
from app.tools.registry import invoke_analysis_tool  # noqa: E402

# 重试日志是实验 3 的证据之一，但它会打乱打印顺序；实验里改用"消耗了几次模型调用"来证明重试。
logging.getLogger("app.agent.analyst_agent").setLevel(logging.CRITICAL)

SEP = "=" * 78

# 从真实运行里抓下来的模型原始输出：同一个 JSON 对象重复了两次，外面还包了代码块围栏。
# 这不是编造的样本——它就是当初让成功率掉到 1/10 的输入之一。
REAL_FENCED_OUTPUT = (
    '{"action": "tool_call", "tool_name": "get_dataset_overview", "tool_input": {}}\n'
    "```json\n"
    '{"action": "tool_call", "tool_name": "get_dataset_overview", "tool_input": {}}\n'
    "```"
)

FRAME = pd.DataFrame({"product": ["A", "B", "C"], "sales": [10, 30, 20]})
CONTEXT = DatasetContext("repro", "sales.csv", FRAME, profile_dataframe(FRAME, "sales.csv"))


class MockLLM:
    """按顺序返回预设响应；遇到 Exception 就抛出来，用来模拟网络/服务端故障。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return LLMResponse(content=value, model="mock", mode="llm")


def action_tool(name: str, arguments: dict) -> str:
    return json.dumps({"action": "tool_call", "tool_name": name, "tool_input": arguments})


def action_final(answer: str) -> str:
    return json.dumps({"action": "final", "final_answer": answer})


def run_agent(llm: MockLLM, question: str):
    return AnalystAgent(llm).run(AgentRequest(user_query=question, dataset_context=CONTEXT))


def experiment_1_fenced_json() -> bool:
    """根因一：模型把 JSON 包进 Markdown 围栏，直接解析会失败。"""
    print(SEP)
    print("实验 1：模型把 JSON 包进 Markdown 围栏（根因一）")
    print(SEP)
    print(f"模型返回的原始内容（{len(REAL_FENCED_OUTPUT)} 字符，来自真实运行）：")
    print("  " + repr(REAL_FENCED_OUTPUT[:92]) + " ...")
    print()

    print("修复前：直接 model_validate_json(原始内容)")
    naive_ok = True
    try:
        AgentAction.model_validate_json(REAL_FENCED_OUTPUT)
    except Exception as exc:  # noqa: BLE001 - 这里就是要展示"抛了什么"
        naive_ok = False
        print(f"  ✗ 失败：{type(exc).__name__}")
        print("  → 整轮决策作废，用户看到「模型输出格式不合法」")
    print()

    print("修复后：生成多个候选，按可能性依次尝试")
    parsed = False
    for index, candidate in enumerate(AnalystAgent._json_candidates(REAL_FENCED_OUTPUT), start=1):
        try:
            action = AgentAction.model_validate_json(candidate)
            parsed = True
            print(f"  候选{index}（{len(candidate):>4} 字符）✓ 成功 → action={action.action}, tool={action.tool_name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  候选{index}（{len(candidate):>4} 字符）✗ {type(exc).__name__}")
    print()
    print("结论：只做「去掉围栏」不够——围栏夹在两个对象中间，去不干净。")
    print("      真正救回来的是「括号配对扫出的单个 {...} 块」。")
    print()
    return (not naive_ok) and parsed


def experiment_2_recoverable_tool_error() -> bool:
    """根因二：工具报错后原代码直接终止，用户拿不到任何结果。"""
    print(SEP)
    print("实验 2：工具报错后自我纠正（根因二）")
    print(SEP)
    print("场景：模型猜错列名（数据里是 sales，它填了 revenue）")
    print()

    llm = MockLLM([
        action_tool("find_top_n", {"column": "revenue", "n": 1}),   # 第 1 轮：猜错列名
        action_tool("find_top_n", {"column": "sales", "n": 1}),     # 第 2 轮：看到错误后改成 sales
        action_final("B 的销售额最高，为 30。"),
    ])
    result = run_agent(llm, "销售额最高的是谁？")

    first_step = result.steps[0]
    print(f"最终结果：success={result.success}，共 {len(llm.requests)} 轮")
    print()
    print("第 1 轮（猜错列名）发生了什么：")
    print(f"  action_type = {first_step.action_type}")
    print(f"  参数        = {json.dumps(first_step.tool_input, ensure_ascii=False)}")
    print(f"  摘要        = {first_step.summary}")
    print()
    error = (first_step.tool_output or {}).get("error")
    print("系统回灌给模型的错误原文（这才是真实文案）：")
    print(f"  {json.dumps(error, ensure_ascii=False)}")
    print("  → 它只说列不存在，**没有列出可用列名**；模型是靠 prompt 里的数据集")
    print("    schema（字段列表）自己推断出应该用 sales 的。")
    print("  → 也注意：这次运行最终成功，所以 result.errors 是空的——")
    print("    可恢复错误不会进入最终错误列表，只作为步骤留在历史里。")
    print()
    prompt = llm.requests[1].user_prompt
    print("第 2 轮 prompt 里到底有什么：")
    for keyword, why in [("revenue", "被回灌的错误参数"),
                         ("does not exist", "错误原因"),
                         ("product", "数据集字段列表（来自 prompt 的 schema 段）")]:
        print(f"  {'✓ 包含' if keyword in prompt else '✗ 缺少'}「{keyword}」　← {why}")
    print()
    print(f"步骤序列：{[step.action_type for step in result.steps]}")
    print(f"实际用到的工具：{result.tools_used}")
    print()
    print("结论：错误被当成「观察值」喂回给模型，它自己改对了。")
    print("      修复前的代码会在第 1 轮直接终止，用户只拿到「参数不匹配」。")
    print()

    recovered = result.success and first_step.action_type == "error"
    recovered = recovered and "does not exist" in prompt and result.tools_used == ["find_top_n"]
    return bool(recovered)


def experiment_3_network_failure_keeps_evidence() -> bool:
    """根因三：网络中断时，已经算出的证据必须带出去，而不是丢掉。"""
    print(SEP)
    print("实验 3：网络中断时不再丢失已算出的证据（根因三）")
    print(SEP)
    print("场景：第 1 轮工具算出结果，第 2 轮调用模型时网络中断")
    print()

    llm = MockLLM([
        action_tool("find_top_n", {"column": "sales", "n": 1}),   # 第 1 轮：成功算出 B=30
        LLMServiceError("Connection reset by peer"),               # 之后每一次模型调用都失败
        LLMServiceError("Connection reset by peer"),
        LLMServiceError("Connection reset by peer"),
        LLMServiceError("Connection reset by peer"),
        LLMServiceError("Connection reset by peer"),
    ])
    remaining_before = len(llm.responses)
    result = run_agent(llm, "销售额最高的是谁？")
    attempts = remaining_before - len(llm.responses)

    print(f"success    = {result.success}　← 如实标记「未完成」")
    print(f"errors     = {[error.code for error in result.errors]}　← 结构化错误，不是抛异常")
    print(f"tools_used = {result.tools_used}　← 已算出的工具记录被带出来了")
    print(f"steps      = {[step.action_type for step in result.steps]}　← 步骤也带出来了")
    print(f"模型调用次数 = {attempts}（1 次工具调用 + 分层重试的若干次失败尝试）")
    print()
    print("界面会展示的「已计算证据」：")
    for line in [text for text in (result.final_answer or "").splitlines() if text.strip()][:6]:
        print(f"  {line[:104]}")
    print()
    print("结论：修复前 steps / tools_used 没有传给 _failure()，前一轮算出的 B=30 直接丢失，")
    print("      用户只看到一句「调用模型失败」；修复后 success=False，但证据被保住并明确标注未完成。")
    print()

    evidence_kept = '"sales": 30' in (result.final_answer or "")
    return (not result.success) and result.tools_used == ["find_top_n"] and evidence_kept


def experiment_4_argument_namespace() -> bool:
    """根因四：模型参数用 ** 展开时会与函数形参冲突，异常逃出整个 Agent。"""
    print(SEP)
    print("实验 4：参数命名空间冲突（根因四）—— 用项目真实的工具调用函数")
    print(SEP)
    print("场景：模型返回的 tool_input 里混进一个叫 name 的键，")
    print("      而 invoke_analysis_tool 自己的第二个形参就叫 name")
    print()

    print("① 修复前的写法：模型参数用 ** 展开")
    crashed = False
    try:
        invoke_analysis_tool(CONTEXT, "get_dataset_overview", **{"name": "oops"})
    except TypeError as exc:
        crashed = True
        print(f"  ✗ {type(exc).__name__}: {exc}")
        print("  → 异常发生在「参数绑定阶段」，比函数体第一行 try 还早")
        print("  → 所以函数内的 except、Agent 循环里的 try 都拦不到它")
        print("  → 异常逃出整个 Agent，页面直接显示 Python traceback")
    print()

    print("② 修复后的写法：模型参数装进一个数据对象（parameters=）")
    execution = invoke_analysis_tool(CONTEXT, "get_dataset_overview", parameters={"name": "oops"})
    code = execution.error.code if execution.error else None
    print(f"  没有崩溃：success={execution.success}")
    print(f"  返回结构化错误：{code}")
    print("  → 模型那个多余的 name 键只是「数据里的一个键」，碰不到系统参数")
    print()
    print("结论：这不只是修 bug，它是安全设计——可信参数（工具名、数据集、知识库 ID）")
    print("      由系统持有，模型只能往 parameters 这个口袋里塞数据。")
    print()
    return crashed and (not execution.success) and code == "invalid_tool_request"


def main() -> int:
    print()
    results = [
        ("实验 1  Markdown 围栏 JSON", experiment_1_fenced_json()),
        ("实验 2  工具错误自我纠正", experiment_2_recoverable_tool_error()),
        ("实验 3  网络中断保留证据", experiment_3_network_failure_keeps_evidence()),
        ("实验 4  参数命名空间隔离", experiment_4_argument_namespace()),
    ]
    print(SEP)
    print("汇总")
    print(SEP)
    for name, passed in results:
        print(f"  {'✓ 符合预期' if passed else '✗ 与预期不符'}　{name}")
    failed = [name for name, passed in results if not passed]
    print()
    if failed:
        print("以下实验与预期不符，说明相关行为被改动过：" + "、".join(failed))
        return 1
    print("四个故障复现实验全部符合预期：修复行为仍然生效。")
    print("面试用法：这就是「我跑过、我见过它怎么失败」的依据，而不是背结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
