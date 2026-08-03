# 已验证实验结果

本文件汇总已经执行、在仓库实验记录中可追溯的结论。它不记录尚未验证的设想，也不把单次结果表述为跨语料、跨模型的通用性能。

- 项目入口与运行说明见 [`../README.md`](../README.md)。
- 原始终端输出、排错过程和案例分析见 [`../experiments/`](../experiments/)。
- 下一步计划见 [`ROADMAP.md`](ROADMAP.md)。

## 评测边界

当前对照实验使用同一份 **20 条 RAGAS 合成样本**，评测脚本调用真实的 `rag_answer()` 链路，再使用 RAGAS 计算：

- Faithfulness：回答能否被实际检索上下文支持；
- Answer Relevancy：回答是否直接回应问题；
- LLM Context Precision with Reference：召回内容中有用信息的比例；
- Context Recall：参考答案所需信息是否被召回。

当前结果受以下因素共同影响：个人笔记语料、测试集、`Qwen3-Embedding-0.6B`、DeepSeek 生成与评测配置、Prompt、切分参数和 RAGAS 版本。`data/`、`chroma_db/` 与 `output/` 不随 Git 同步，因此外部读者可复现流程，但不能直接复现精确数值。

## 结果总表

| 指标 | Vector Baseline | Hybrid 0.7 / 0.3 | Parent-Child | Parent-Child + Structured | Metadata + Parent-Child + Structured |
| --- | ---: | ---: | ---: | ---: | ---: |
| Faithfulness | 0.8172 | 0.8375 | 0.8305 | 0.8922 | **0.9215** |
| Answer Relevancy | 0.7979 | 0.7374 | **0.8020** | 0.7439 | 0.7890 |
| Context Precision | 0.8444 | 0.8640 | 0.9737 | 0.9737 | **0.9792** |
| Context Recall | 0.9500 | 0.9333 | **1.0000** | 0.9900 | 0.9567 |

最后一列还记录了结构化输出的自定义统计：引用编号有效率 `1.0000`、JSON 解析失败率 `0.0000`、模型判定可回答 `20 / 20`。引用编号有效仅表示编号落在本次提供的上下文范围内，不等价于事实性完全正确。

## 实验索引

| 实验 ID | 状态 | 唯一变量 / 组合 | 主要结论 | 证据 |
| --- | --- | --- | --- | --- |
| `baseline-v1` | 已完成 | 固定长度 chunk + 向量检索 | 建立后续比较的稳定参照 | [基线记录](../experiments/基线测试结果记录.md) |
| `hybrid-bm25-v1` | 已完成 | `jieba` BM25 + RRF 融合 | 中文分词修复后可用，但当前语料未见稳定综合增益 | [混合检索记录](../experiments/检索优化BM25结果记录.md) |
| `parent-child-v1` | 已完成 | child 检索、parent 返回 | 上下文 Precision/Recall 显著改善 | [父子分块记录](../experiments/父子分块检索结果记录.md) |
| `structured-generation-v1` | 已完成 | JSON 回答、引用编号、可回答判断 | 提高可核查性与 Faithfulness，Relevancy 仍需优化 | [结构化生成记录](../experiments/结构化生成优化结果记录.md) |
| `self-query-metadata-v1` | 已完成 | `primary_tag` / `modified_date` 过滤 | 过滤链路可用，但直接返回 child 会造成上下文碎片化 | [元数据记录](../experiments/元数据metadata检索结果记录.md) |
| `self-query-parent-structured-v1` | 已完成 | metadata 过滤 + parent 映射 + JSON 生成 | 当前最高 Faithfulness 与 Context Precision | [元数据记录](../experiments/元数据metadata检索结果记录.md) |
| `multi-query-v1` | 已完成 | 生成多个改写查询 | 改写正常生成；当前小规模、主题集中的语料未观察到检索增益 | [多查询记录](../experiments/多查询检索结果记录.md) |

## 关键结论

### 1. 基线是必要的参照，不是最终策略

固定长度切分、向量检索和低温生成的链路已经可运行，且四项指标相对均衡。它为所有后续实验提供了共同对照：任何复杂模块都必须在相同测试集下优于或明确权衡基线，而不能仅凭单个问题的直观效果替换它。

### 2. 混合检索的收益依赖语料结构

默认 BM25 对中文按空格分词，会导致中文关键词检索失效；项目已通过 `jieba` 修复这一点。修复后，混合检索能增强词面召回，但当前约 22 篇、跨主题的笔记通常由单一文档即可回答，BM25 也会召回词面相近但任务无关的内容。因此 Hybrid 保留为候选策略，而非当前默认策略。

### 3. 父子分块解决了检索粒度与生成上下文的冲突

child chunk 使用较小粒度提升匹配精度，命中后映射为较大的 parent chunk 提供给 LLM。该方案把 Context Precision 提升到 `0.9737`，Context Recall 提升到 `1.0000`，并取得全策略最高的 Answer Relevancy `0.8020`。

### 4. 结构化生成将“回答”变成可检查的对象

结构化输出要求模型显式返回回答、引用编号和可回答判断。`Parent-Child + Structured` 的 Faithfulness 为 `0.8922`，高于基线；但 Answer Relevancy 从 `0.8020` 降至 `0.7439`，说明更严格的输出格式会改变回答风格，仍需要结合典型案例与人工抽查优化。

### 5. 元数据过滤必须与完整上下文协同

主题和日期 metadata 可由 SelfQueryRetriever 解析为 Chroma 过滤条件。直接将过滤后的 child chunk 交给 LLM 时，RAGAS 结果显著下降：Faithfulness `0.6809`、Context Recall `0.7429`。原因不是过滤机制失效，而是 child 的碎片化上下文不足以支持生成。

将被过滤的 child 映射回 parent，再配合结构化生成后，Faithfulness 达到 `0.9215`、Context Precision 达到 `0.9792`。这说明过滤负责缩小候选范围，parent 负责恢复上下文完整性，两者应组合使用。

### 6. 查询改写已验证执行链路，尚未验证业务收益

MultiQueryRetriever 能生成多个不同角度的查询；但在当前数据规模和主题分布下，改写前后仍会命中相同的 parent 文档。该结论不是“查询改写无效”，而是“当前实验条件不足以显示其收益”，应在语料扩展和问题表达更分散后重新评测。

## 结果更新规则

新增实验时，不覆盖已有记录。请在结果总表和实验索引中增加一项，并至少记录：

```markdown
## experiment-id

- 目标：
- 假设：
- 唯一变量：
- 固定条件：
- 语料与测试集版本：
- RAGAS 指标：
- 延迟、Token 与成本：
- 典型成功案例：
- 典型失败案例：
- 结论与下一步：
```

若只完成机制验证、未完成固定评测集上的量化对照，应明确标记为“链路验证”，不要升级为性能结论。
