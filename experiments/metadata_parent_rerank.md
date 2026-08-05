# 元数据+父子分块+rerank精排
```python
python scripts/evaluate.py
```
<details>
<summary>实验结果</summary>
| 模型 | R@1 | R@3 | R@5 | R@10==== ragas评测汇总 ====
{'faithfulness': 0.8981, 'answer_relevancy': 0.7606, 'llm_context_precision_with_reference': 0.8905, 'context_recall': 0.9500}

详细结果已保存至: RAG/output/eval_result_self_query_parent_rerank_structured.csv

==== 引用有效性统计（自定义，非ragas指标） ====
总问题数: 20
引用编号有效率: 1.0000（LLM给出的引用编号在合法范围内的比例）
JSON解析失败率: 0.0000（越低越好，说明结构化输出稳定性）
判定为'可回答'的问题数: 19 / 20
详细引用统计已保存至: RAG/output/citation_stats_self_query_parent_rerank_structured.csv
</details>
根据结果可知，当前的结果并没有追平或是超过不加入Rerank的最佳组合。
- 考虑到之前的的元数据过滤的方式，就已经是属于强相关的约束，过滤后的候选池本来就小，且都是高度相关的候选文档，Rerank的作用可能并不大。
- 其次就是，在当前的Rerank模型中，可能存在一些问题，比如：
  - Rerank模型本身的能力有限，无法有效区分高度相关的候选文档。
  - Rerank模型的训练数据可能与当前任务不完全匹配，导致其在特定领域或特定类型的问题上表现不佳。
  - Rerank模型的输入特征可能不足以捕捉到文档之间的微妙差异，从而影响排序效果。
因此，Rerank模型在这种高度相关的候选文档环境下，可能无法显著提升检索结果的质量。