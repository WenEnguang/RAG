## 父子分块实验记录

```python 
python build_parent_children_index.py
```
<details>
<summary>results output</summary>
    Step 1/3 加载文档 ...
  加载到 22 篇笔记
    Step 2/3 初始化embedding模型 ...
    Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 310/310 [00:00<00:00, 17854.96it/s]
    Step 3/3 构建父子分块索引（自动完成两级切分+embedding+持久化） ...
    Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:01<00:00,  8.69it/s]
    索引建立完成，collection=notes_parent_child
    父块原文已持久化到: RAG/output/parent_docstore

</details>
正如结果显示，现在的父子分块建立索引就正常进行完成了。
结果显示，一共切割了11批，batch_size是32/批，大概是11x32个child chunk。

```python
python RAG_pipiline.py
```
<details>
<summary>results output</summary>
    ===== 纯向量检索 =====
    [1] 长度366字符: # RAG基础概念笔记
    ## 什么是RAG
    RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索系统和大语...
    [2] 长度342字符: ## RAG的基本流程
    一个最简单的RAG系统包含以下几个步骤：
    1. 文档切分（Chunking）：把长文档切成较小的片段
    2. 向量化（Embeddin...
    [3] 长度290字符: 目前尚未出现完美的解决方案，新的想法仍在不断探索中。
    6. RAG 的本质与哲学思考
    RAG 本质上是一种在有限大模型上下文窗口下的**妥协策略**。它通过...
    [4] 长度494字符: 1. **向量存储与关联**：
    - 将每个生成的向量与其原始文本片段的对应关系保存起来。
    - **向量数据库**：专门设计用于存储和检索向量数据，能够高效地查...
    ===== 父子分块检索 =====
    [1] 长度675字符: # RAG基础概念笔记
    ## 什么是RAG
    RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索系统和大语...
    [2] 长度543字符: 1. **缺乏全局视角**：
    - RAG 侧重于检索与问题直接相关的局部片段。
    - 对于需要全局信息汇总的问题（例如“这篇文章中‘我’字共出现了多少次？”），...
</details>
根据结果显示，相较于长度而言，父子分块（675+543），但是纯向量（366+342+290+494），说明父子分块是复核预期的，“检索小块，返回大块”。
纯向量的返回结果是4条，但是父子分块返回的结果仅有2条。
ParentDocumentRetriever 内部的实际流程是：先按你设置的检索数量，去child向量库里检索出若干个child块（这一步默认用的是 similarity_search 的默认 k 值，你的代码目前没有显式设置这个数字），然后把这些child块映射回它们各自所属的parent块，再对parent做去重（如果两个不同的child恰好属于同一个parent，去重后只算一个）。修改之后，如下面结果显示：
<details>
<summary>results output</summary>
    ===== 纯向量检索 =====
    [1] 长度366字符: # RAG基础概念笔记
    ## 什么是RAG
    RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索系统和大语...
    [2] 长度342字符: ## RAG的基本流程
    一个最简单的RAG系统包含以下几个步骤：
    1. 文档切分（Chunking）：把长文档切成较小的片段
    2. 向量化（Embeddin...
    [3] 长度290字符: 目前尚未出现完美的解决方案，新的想法仍在不断探索中。
    6. RAG 的本质与哲学思考
    RAG 本质上是一种在有限大模型上下文窗口下的**妥协策略**。它通过...
    [4] 长度494字符: 1. **向量存储与关联**：
    - 将每个生成的向量与其原始文本片段的对应关系保存起来。
    - **向量数据库**：专门设计用于存储和检索向量数据，能够高效地查...
    ===== 父子分块检索 =====
    [1] 长度675字符: # RAG基础概念笔记
    ## 什么是RAG
    RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索系统和大语...
    [2] 长度543字符: 1. **缺乏全局视角**：
    - RAG 侧重于检索与问题直接相关的局部片段。
    - 对于需要全局信息汇总的问题（例如“这篇文章中‘我’字共出现了多少次？”），...
    [3] 长度1074字符: ### RAG Embedding：
    **RAG（retrieval augmented generation）检索、增强、生成。**RAG Embeddin...
    [4] 长度1180字符: 3.4. 向量空间与距离度量
    - **高维坐标系**：输出的固定长度数组可被视为一个高维坐标系中的一个点（例如，1536 维或 3072 维）。
    - **语义...
</details>
修改过后，结果显示已经正常。

### 评估结果
```python
python evaluate.py
```
<details>
<summary>results output</summary>
    ==== 评测汇总 ====
    {'faithfulness': 0.8305, 'answer_relevancy': 0.8020, 'llm_context_precision_with_reference': 0.9737, 'context_recall': 1.0000}

    详细结果已保存至: RAG/output/eval_result_parent_child.csv
</details>
目前结果很明显，四项指标全部都达到或接近目前实践过程中的最好结果。
| 指标 | baseline | hybrid(0.5/0.5)  | hybird(0.7/0.3) | parent_child | 
| :---: | :---: | :---: | :---: | :---: |
| Faithfulness| 0.8172 | 0.8137 | 0.8375 | 0.8305 |
| Answer Relevancy | 0.7979 | 0.7714 | 0.7374 | 0.8020 |
| Context Precision | 0.8444 | 0.8788 | 0.8640 | 0.9737 |
| Context Recall | 0.9500 | 0.9298 | 0.9333 | 1.0000 |
在Answer Relevancy 这一栏中，之前的数据一直很低，但是此次的结果说明，此次的父子分块处理有助于生成端去生成更好的答案，检索的质量对于生成的质量是大有裨益的。
