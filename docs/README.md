# 项目文档

[返回项目首页](../README.md)

## 使用与设计

| 文档 | 内容 |
| --- | --- |
| [配置指南](configuration.md) | Chat LLM、Embedding、Qdrant、MySQL 与 LangSmith 的配置和自检 |
| [架构设计](architecture.md) | 数据流、工具边界、Agent 循环与隐私边界 |
| [事实证据校验](fact-evidence.md) | 调用编号、结果字段、值与单位核验，以及最新验证记录 |
| [评估说明](evaluation.md) | 评估数据集、指标、运行方法与已知局限 |
| [数据画像](dataset-profiler.md) | 字段推断、质量评分与文件读取限制 |
| [分析工具](analysis-tools.md) | 工具输入、输出与边界条件 |

## 验证与历史

以下文档保留当时的实现与验证情况；其中的测试数量和待办不代表当前状态。最新事实校验记录见 [fact-evidence.md](fact-evidence.md#本次验证记录2026-10-22)。

| 文档 | 用途 |
| --- | --- |
| [真实运行验收 v2](real_run_acceptance_v2.md) | 2026-10-10 的 Agent 与 RAG 真实服务验收 |
| [早期验收记录](real_run_acceptance.md) | 加固前的运行快照 |
| [审查与修复记录](code-review-and-next-steps.md) | 历史问题、定位过程与修复批次 |
| [面试演示稿](interview-guide.md) | 演示流程与技术问答；数字和结论须以当前验证记录为准 |
| [截图说明](images/README.md) | 演示截图内容与更新方法 |

## 代码入口

```text
app/
├── agent/        # 工具选择循环、结构化结论与事实证据核验
├── tools/        # 静态工具注册表、Pandas 分析与图表
├── services/     # 数据读取、画像、模型、知识库、Trace 和持久化
├── models/       # 数据集、分析与知识检索契约
├── knowledge/    # 文档读取与切块
├── vectorstore/  # Qdrant 适配
├── database/     # 数据表、连接与初始化
├── repositories/ # 元数据仓储
├── evaluation/   # 评估执行器与指标
└── ui/           # Streamlit 应用
data/             # 合成数据、知识文档与评估案例
i18n/             # 中文 / 英文界面文案
scripts/          # 服务自检与失败场景复现
tests/            # 离线回归测试
```
