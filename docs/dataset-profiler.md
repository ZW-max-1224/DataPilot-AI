# Dataset Profiler：实现说明

## 模块分工

- `dataset_service.py` 接收文件名和上传 bytes，负责格式/大小校验、解码、表头检查、CSV/Excel 读取和友好错误。它不接受文件路径，因此不能被用来遍历本机文件。
- `profiler.py` 接收 DataFrame，确定性生成 Pydantic `DatasetProfile`。输入 DataFrame 不会被修改。
- `dataset_profile.py` 是稳定的结构化结果契约，后续 API/Agent Tool 可以消费它，不必依赖 UI 字符串。
- `streamlit_app.py` 只做上传、调用 Service/Profiler 和展示，不承载分析算法。

## 文件限制与读取

支持 `.csv`、`.xlsx`、`.xls`；每文件最多 20 MiB、100,000 行、200 列。只读第一个 Excel 工作表。CSV 依次尝试 UTF-8 BOM/UTF-8 和 GB18030。缺文件、仅表头、无表头、重复或空列名、解析失败、尺寸超限都返回 `DatasetLoadError`。服务层会拒绝损坏或行格式不一致的 CSV。

限制是为了避免初学阶段的 UI 被超大表格拖垮；它不是生产级隔离/沙箱，也不能替代服务端资源配额。

## 语义类型规则

1. 已经是 Pandas datetime dtype 的列为 `datetime`。
2. object/string 列取至多 200 个非空样本，以 mixed-format datetime parser 尝试解析；至少 80% 可解析，且列名含 date/time 等线索或至少 80% 字符串符合日期形状，才认定为 `datetime`。这避免把普通分类文本都转换成日期。
3. bool 为 `categorical`；数值 dtype 为 `numeric`。
4. 文本描述类列名或长文本值归为 `text`。
5. region/category/product/customer/id 等线索归为 `categorical`。
6. 其他低基数值归为 `categorical`；证据不足的 object 列为 `unknown`。

语义推断是启发式，列名提示不是数据字典；可在未来增加用户校正和领域规则。

## 缺失、重复与摘要

- 列缺失率 = 该列空值数 / 总行数 × 100%。
- 全表缺失率 = 全部空单元格数 / (行数 × 列数) × 100%。
- 重复行 = `DataFrame.duplicated()` 判定的后续重复记录数；重复率 = 重复行 / 总行数 × 100%。
- 数值统计对每个数值列输出非空 count、mean、median、sample std、min、max；非有限数字输出为 null。
- 分类列 top values 最多 10 项，百分比以该列非空值为分母。
- 内存占用由 `memory_usage(index=True, deep=True)` 估算，包含索引及对象列估算值。

## Data Quality Score 规则

初始 100 分，三类扣分相加，最低 0 分：

| 指标 | 阈值 | 扣分 |
|---|---:|---:|
| 整体缺失率 | >20% / >5% / >0% | 30 / 15 / 5 |
| 重复行率 | >10% / >1% / >0% | 20 / 10 / 3 |
| unknown 列占比 | >30% / >10% / >0% | 15 / 7 / 2 |

每个指标只触发最高匹配的一档。Pydantic Profile 同时输出 `quality_issues` 和 penalty，因此 UI 能解释每次扣分。这个分数不是经过统计学习标定的通用数据质量标准，不检查业务规则、单位错误、异常值或因果有效性，只适合快速提示。

## Demo 数据

`data/sample_sales.csv` 是固定种子的 synthetic data，共 1,208 行。它为后续趋势、分组、Top-N、异常分析准备：8 月销售额整体较低，East 区域和 Electronics 额外走弱；Budget Tablet 利润率低；含少量空值、8 条精确重复行及一条极端销售记录。它不是现实业务数据，不能用于声称市场事实。

## 学习与面试表达

**Dataset Profiler 是什么？** 把上传表格转为确定、可检查的数据结构摘要。

**为什么先不用 LLM？** 缺失数、统计量和重复行应由 Pandas 对数据精确计算；LLM 后续负责理解问题和决定调用哪个工具。

**Pydantic 在这里做什么？** 它定义清晰的 Profile 输出契约并验证类型，使 UI、API 和后续 Agent 不依赖任意 dict 形状。

**语义类型是否绝对正确？** 不是。列名与值模式提供初步推断，80% 日期条件是保守门槛；用户确认、领域 schema 和测试集仍然重要。

**质量分代表什么？** 可解释的启发式检查结果，不是数据可信度的权威概率或统计结论。
