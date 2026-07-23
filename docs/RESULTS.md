# 已验证实验结果

本文件发布已经执行并可在仓库记录中追溯的实验结论。

- 项目路线与候选任务见 [`ROADMAP.md`](ROADMAP.md)。
- 完整终端输出、排错过程和阶段性观察见 `experiments/`。
- 原始测试集与逐样本评测数据见 `output/`。

这里不记录尚未实施的方案，也不将单次结果表述为普遍结论。

## 结果索引

| 实验 ID | 阶段 | 状态 | 主要结论 |
| --- | --- | --- |
| `baseline-smoke-v1` | Baseline 验证 | 已完成 | Markdown -> Chroma -> 检索 -> DeepSeek 的主链路可运行。 |
| `ragas-baseline-v1` | 评测框架 | 已完成 | 已生成测试集并获得四项有效 RAGAS 指标，可作为后续 A/B 对照。 |

## `baseline-smoke-v1`：基础链路验证

### 目标

确认本地 Embedding、DeepSeek API、Markdown 加载、文本切分、Chroma 索引、检索和回答生成可以串成完整闭环。

### 已验证事实

| 项目 | 结果 |
| --- | --- |
| Embedding 模型 | `Qwen3-Embedding-0.6B` 可正常加载并输出 1024 维向量。 |
| 语义区分验证 | 记录中的相近句相似度为 `0.8700`，无关句相似度为 `0.3364`。 |
| LLM 连通性 | DeepSeek OpenAI-compatible API 调用正常。 |
| 单文档冒烟测试 | 能检索到相关上下文，并生成 RAG 基本流程的正确步骤。 |
| 多文档 chunk 检查 | 7 篇 Markdown 得到 102 个 chunk，平均长度 `377.7`，最大 `494`，最小 `3`。 |

### 结论

基础 RAG 链路已经可运行，但最小 chunk 为 3，说明固定长度切分会制造极短片段；纯向量检索在语料扩展后也已出现语义接近但任务无关的召回现象。这两点是后续检索与切分实验的主要动机。

### 证据

- `experiments/基线测试结果记录.md`
- `scripts/test/test_embedding.py`
- `scripts/test/test_llm.py`
- `scripts/test/test_chunk_size.py`

## `ragas-baseline-v1`：RAGAS 评测基线

### 目标

建立一条真实调用 RAG 主链路的评测流程：测试集中的每个问题先执行 `rag_answer()`，再由 RAGAS 评估实际检索上下文和实际回答。

### 实验条件

| 项目 | 配置 / 范围 |
| --- | --- |
| 测试集 | `output/testset.csv`，当前版本 20 条样本。 |
| 评测结果 | `output/eval_result.csv`，当前版本 20 条逐样本结果。 |
| 被测系统 | 固定长度切分 + 本地 `Qwen3-Embedding-0.6B` + Chroma 相似度检索 + DeepSeek 生成。 |
| 检索参数 | `top_k=4`。 |
| 评审模型 | 配置中的 DeepSeek OpenAI-compatible 模型。 |
| RAGAS 指标 | Faithfulness、Answer Relevancy、LLM Context Precision with Reference、Context Recall。 |
| 运行配置 | `timeout=300`、`max_workers=4`。 |

### 汇总指标

| 指标 | 均值 | 解读 |
| --- | ---: | --- |
| Faithfulness | 0.8462 | 大部分回答可由检索上下文支撑，但仍存在缺少依据的生成风险。 |
| Answer Relevancy | 0.7579 | 当前最弱的核心指标，答案与问题的贴合度值得优先分析。 |
| LLM Context Precision with Reference | 0.8361 | 多数召回内容有用，但候选上下文中仍有噪声。 |
| Context Recall | 0.9333 | 参考答案所需的信息大多已被召回，是当前表现最好的维度。 |

### 结论

这组指标证明评测闭环已经可用，也为后续的检索优化提供了对照基线。它不证明某个优化一定有效，也不应被解读为跨语料、跨模型的通用分数。

下一次对比实验必须尽量保持以下条件不变：

- 同一份测试集及其版本。
- 同一套 RAGAS 指标与评审模型。
- 相同的主模型、Embedding 模型和生成温度。
- 除待验证策略外相同的切分、检索和运行参数。

### 复现边界

当前个人笔记、向量库和本地模型均不随 Git 同步。因此外部开发者可复现流程，但不能复现这组精确数值，除非使用相同语料、模型文件、API 配置和测试集。开源发布阶段应提供脱敏 demo 数据集与版本化评估集。

### 证据

- `output/testset.csv`
- `output/eval_result.csv`
- `experiments/基线测试结果记录.md`
- `experiments/初期结果评估记录.md`
- `scripts/TestsetGenerator.py`
- `scripts/evalute.py`

## 结果更新规则

新增实验时，不覆盖旧结果。请在本文件的“结果索引”中增加一行，并新建一个结果小节，至少包含：

```markdown
## `experiment-id`：实验名称

### 目标

### 实验条件

### 汇总指标

### 典型案例

### 结论

### 证据
```

当实验涉及检索策略、切分策略或生成 Prompt 时，还应补充与对照组相比的指标变化、延迟变化和至少三个代表性案例。
