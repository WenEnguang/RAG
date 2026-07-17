# RAG — 个人笔记检索增强生成系统

一个面向个人笔记的 RAG（Retrieval-Augmented Generation，检索增强生成）实验项目。

项目目标不是直接做成一个完整应用，而是从最朴素的 Baseline 开始，把 RAG 的关键环节一步一步跑通、拆开、评估和优化：文档加载、切分、向量化、索引构建、检索、生成、量化评测。

当前阶段已经完成 **Phase 0 基线系统**：可以读取 Markdown 笔记，构建本地 Chroma 向量索引，根据用户问题检索相关片段，并调用 DeepSeek 生成基于资料的回答。同时，项目已经接入 RAGAS，用于对检索和生成效果进行量化评估。

## 当前进度

### 已完成

- Markdown 笔记加载：从 `data/notes/` 读取 `.md` 文件
- 固定长度切分：使用 `RecursiveCharacterTextSplitter`
- 本地 Embedding：使用 `Qwen3-Embedding-0.6B`
- 本地向量库：使用 Chroma 持久化到 `chroma_db/`
- 基础检索：使用 Chroma `similarity_search`
- LLM 生成：使用 DeepSeek OpenAI-compatible API
- 主链路封装：提供 `retrieve()`、`generate()`、`rag_answer()`
- 索引构建脚本：单独提供 `scripts/build_vectorstore.py`
- 验证脚本：Embedding、LLM、完整 RAG 流程均有验证脚本
- RAGAS 评测：自动生成测试集并进行量化评估
- 实验记录：在 `experiments/notebooks/` 中记录阶段性结果和问题

### 尚未完成

- 混合检索（向量 + BM25）
- Reranker 重排序
- 更稳健的 chunk 策略
- PDF 文档接入
- WebUI
- 更完整、稳定的评测基准

## 项目定位

这是一个学习型、实验型 RAG 项目，适合用来理解：

- RAG 的完整执行链路是什么
- chunk size、overlap、top-k 等参数如何影响结果
- 为什么纯向量检索在语料变多后会出现误召回
- 如何用 RAGAS 对 RAG 系统做量化评估
- 后续如何逐步优化检索、切分和生成效果

设计原则是：**先跑通，再拆开；先有 Baseline，再做实验；每次只优化一个关键环节，用结果说话。**

## 执行逻辑

### 索引构建阶段

```text
Markdown 笔记
  ↓
DirectoryLoader 加载 data/notes/*.md
  ↓
RecursiveCharacterTextSplitter 切分文档
  ↓
HuggingFaceEmbeddings 生成向量
  ↓
Chroma.from_documents 写入本地向量库
  ↓
索引持久化到 chroma_db/
```

对应脚本：

```bash
python scripts/build_vectorstore.py
```

### 用户查询阶段

```text
用户问题
  ↓
使用同一个 Embedding 模型向量化问题
  ↓
Chroma similarity_search 检索 top-k 个相关 chunk
  ↓
将检索片段与问题拼接进 Prompt
  ↓
调用 DeepSeek LLM
  ↓
返回基于上下文的回答
```

核心入口：

```python
from scripts.RAG_pipeline import rag_answer

result = rag_answer("什么是 RAG Embedding？")
print(result["answer"])
```

返回结构：

```python
{
    "question": "...",
    "retrieved_contexts": ["...", "..."],
    "answer": "...",
    "response": "..."
}
```

## 快速开始

### 环境要求

- Python >= 3.10
- 推荐有 CUDA 环境；小规模测试也可以使用 CPU
- 需要 DeepSeek API Key

### 安装依赖

当前项目还没有把完整依赖固化进 `pyproject.toml`，可以先手动安装：

```bash
pip install langchain langchain-community langchain-text-splitters \
    langchain-huggingface langchain-chroma langchain-openai \
    openai ragas pandas tqdm torch \
    pydantic-settings python-dotenv
```

### 配置 API Key

复制环境变量模板：

```bash
cp .env.example .env
```

填写 `.env`：

```env
DEEPSEEK_API_KEY=sk-你的真实key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

DeepSeek 使用 OpenAI 兼容接口。项目中：

- RAG 查询链路使用原生 `openai` client
- RAGAS 评测链路使用 `langchain_openai.ChatOpenAI`

### 准备 Embedding 模型

当前默认使用本地模型：

```text
../Pre_Models/Qwen3-Embedding-0.6B
```

配置位置：

```python
# config/settings.py
embedding_model = "../Pre_Models/Qwen3-Embedding-0.6B"
```

如果本地没有模型，首次运行可能会尝试从 HuggingFace 下载。国内网络较慢时，可以设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 准备笔记

将 Markdown 文件放到：

```text
data/notes/
```

当前项目的目标输入主要是个人 Markdown 笔记。`data/pdfs/` 已经预留，但 PDF 加载逻辑尚未接入。

## 常用命令

验证 Embedding 模型：

```bash
python scripts/test/test_embedding.py
```

验证 DeepSeek API：

```bash
python scripts/test/test_llm.py
```

构建或重建向量索引：

```bash
python scripts/build_vectorstore.py
```

运行完整 RAG 冒烟测试，并分析 chunk 质量：

```bash
python scripts/test/test_chunk_size.py
```

生成 RAGAS 测试集：

```bash
python scripts/TestsetGenerator.py
```

运行 RAGAS 量化评测：

```bash
python scripts/evalute.py
```

## 项目结构

```text
RAG/
├── config/
│   └── settings.py              # 统一配置：模型、路径、chunk、top-k 等
├── indexing/
│   └── vectorstore.py           # Chroma 工厂函数
├── scripts/
│   ├── RAG_pipeline.py          # 核心查询链路：retrieve/generate/rag_answer
│   ├── build_vectorstore.py     # 构建或重建向量索引
│   ├── TestsetGenerator.py      # 使用 RAGAS 自动生成测试集
│   ├── evalute.py               # RAGAS 量化评测脚本
│   └── test/
│       ├── test_embedding.py    # Embedding 模型验证
│       ├── test_llm.py          # DeepSeek API 验证
│       ├── test.py              # 早期 LCEL 全流程冒烟测试
│       └── test_chunk_size.py   # chunk 质量分析 + 全流程测试
├── output/
│   ├── testset.csv              # RAGAS 测试集输出
│   └── eval_result.csv          # RAGAS 评测结果输出
├── experiments/
│   └── notebooks/               # 实验记录与阶段性总结
├── data/
│   ├── notes/                   # Markdown 笔记，未提交到仓库
│   └── pdfs/                    # PDF 目录，暂未接入
└── chroma_db/                   # Chroma 本地向量库，未提交到仓库
```

## 核心模块说明

### `config/settings.py`

统一管理所有可调参数：

- DeepSeek API 配置
- Embedding 模型路径
- Chroma 持久化路径
- 数据目录
- 输出目录
- chunk 参数
- retrieval top-k

当前关键参数：

```python
chunk_size = 500
chunk_overlap = 50
retrieval_top_k = 4
chroma_collection_name = "notes"
llm_temperature = 0.0
```

### `indexing/vectorstore.py`

对 Chroma 做了一层薄封装：

- `build_vectorstore(chunks, embeddings)`：写入文档并建立索引
- `get_vectorstore(embeddings)`：连接已有本地向量库

这样后续如果要替换为 Qdrant、Milvus 或 PGVector，优先改这一层。

### `scripts/RAG_pipeline.py`

当前最重要的查询入口。

提供三个函数：

- `retrieve(question, top_k)`：从 Chroma 检索相关片段
- `generate(question, top_k, context)`：拼接 prompt 并调用 DeepSeek
- `rag_answer(question, top_k)`：完整执行检索 + 生成

Prompt 约束模型只能基于参考资料回答：

```text
如果参考资料中没有相关的信息，请直接回答“抱歉，我无法回答这个问题。”，不要编造答案。
```

### `scripts/build_vectorstore.py`

用于数据更新后的索引构建：

1. 加载 Markdown
2. 切分 chunk
3. 初始化 Embedding
4. 写入 Chroma

这个脚本会自动判断 CUDA/CPU：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### `scripts/evalute.py`

评测流程：

1. 读取 `output/testset.csv`
2. 对每个测试问题调用 `rag_answer()`
3. 收集实际检索片段和实际回答
4. 使用 RAGAS 指标打分
5. 输出到 `output/eval_result.csv`

当前使用的指标：

- Faithfulness
- ResponseRelevancy
- LLMContextPrecisionWithReference
- LLMContextRecall

## Phase 0 结果

### Embedding 验证

验证脚本：

```bash
python scripts/test/test_embedding.py
```

阶段性结果：

```text
向量维度: 1024
生成了 3 条向量
语义相近句子相似度: 0.8700
语义无关句子相似度: 0.3364
Embedding 模型验证通过
```

### LLM 验证

验证脚本：

```bash
python scripts/test/test_llm.py
```

阶段性结果：

```text
DeepSeek API 调用正常
能够用一句话解释 RAG
LLM 调用验证通过
```

### 全流程验证

验证脚本：

```bash
python scripts/test/test_chunk_size.py
```

当前在 7 篇 Markdown 笔记上的 chunk 分析：

```text
总 chunk 数量: 102
平均长度: 377.7
最大长度: 494
最小长度: 3

长度分布:
0-300: 22
300-600: 80
600+: 0
```

测试问题：

```text
RAG系统的基本流程有哪些步骤？
```

系统能够检索到相关上下文，并回答出：

```text
1. 文档切分
2. 向量化
3. 索引建立
4. 检索
5. 生成
```

## RAGAS 评测

当前项目已经可以通过 RAGAS 自动生成测试集并评估。

生成测试集：

```bash
python scripts/TestsetGenerator.py
```

运行评测：

```bash
python scripts/evalute.py
```

已有评测记录显示：

```text
answer_relevancy: 0.8625
context_recall: 1.0000
faithfulness: nan
llm_context_precision_with_reference: nan
```

这说明评测链路已经跑通，但指标稳定性还需要继续处理。尤其是 `faithfulness` 和 `context_precision` 出现 `nan`，后续需要检查 RAGAS 版本兼容、裁判模型输出格式、测试集字段和指标适配情况。

## 当前发现的问题

### 1. 固定长度切分比较粗糙

当前使用：

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
```

问题：

- 可能截断语义完整段落
- 可能切断标题和正文关系
- 可能产生极短 chunk
- 对代码块、多级标题、长段落适应性一般

### 2. 纯向量检索存在误召回

当笔记从 1 篇扩展到 7 篇后，已经观察到：

- 问 RAG 相关问题时，可能召回 Transformer 内容
- top-k 中会混入语义相似但任务不相关的片段
- 向量相似度本身不足以表达“这个片段是否真的应该被用于回答”

后续需要引入：

- BM25
- 混合检索
- Reranker
- 查询改写
- 元数据过滤

### 3. 查询链路中的设备选择还不够稳健

`build_vectorstore.py` 已经支持 CUDA/CPU 自动选择，但 `RAG_pipeline.py` 里仍然写死：

```python
model_kwargs={"device": "cuda"}
```

如果没有 GPU，查询链路可能无法正常运行。后续应统一设备选择逻辑。

### 4. 依赖没有完全固化

`pyproject.toml` 目前没有完整声明真实运行依赖。开源项目复现时，读者仍需要依赖 README 手动安装。

后续应补齐：

- `dependencies`
- 可选 GPU 说明
- RAGAS 兼容版本
- Python 版本约束

### 5. PDF 还只是预留

`settings.py` 中已有：

```python
pdfs_dir = "data/pdfs"
```

但当前加载逻辑只处理 Markdown：

```python
glob="**/*.md"
```

PDF 解析、清洗、元数据保留、页码引用都还没有实现。

## 下一阶段计划

### Phase 1：检索质量优化

优先级最高，因为当前最明显的问题是误召回。

计划方向：

- 引入 BM25 关键词检索
- 实现向量检索 + BM25 的混合召回
- 加入 Reranker 对候选 chunk 重排序
- 保留每个 chunk 的来源文件、标题、位置等元数据
- 对比优化前后的 RAGAS 指标变化

### Phase 2：文档切分优化

计划方向：

- 基于 Markdown 标题层级切分
- 保留标题上下文
- 避免代码块被截断
- 尝试父子 chunk 或多粒度 chunk
- 对比不同 chunk 策略对检索结果的影响

### Phase 3：PDF 接入

计划方向：

- 接入 PDF loader
- 保留页码信息
- 支持 Markdown + PDF 混合索引
- 在回答中输出来源定位

### Phase 4：交互入口

计划方向：

- Streamlit 或 Gradio WebUI
- 展示检索到的上下文
- 展示来源文件
- 支持调整 top-k、chunk 策略和检索策略

## 待办清单

- [x] 搭建 Phase 0 RAG baseline
- [x] 支持 Markdown 文档加载
- [x] 支持本地 Embedding 模型
- [x] 支持 Chroma 本地向量库
- [x] 支持 DeepSeek 生成回答
- [x] 拆分索引构建、查询、评测脚本
- [x] 接入 RAGAS 测试集生成
- [x] 接入 RAGAS 量化评测
- [ ] 修复查询链路 CUDA/CPU 设备选择问题
- [ ] 补齐 `pyproject.toml` 依赖
- [ ] 实现 BM25 检索
- [ ] 实现混合检索
- [ ] 接入 Reranker
- [ ] 优化 Markdown chunk 策略
- [ ] 接入 PDF 文档
- [ ] 增加 WebUI

## 项目当前结论

这个项目已经完成了一个可运行、可评估、可继续实验的 RAG 基线系统。

当前最大的价值不是“已经做成一个完美问答系统”，而是已经搭好了后续实验的骨架：每次调整切分、检索、排序或生成策略，都可以通过同一套构建脚本、查询入口和评测流程进行对比。

下一步最值得投入的是 **检索质量优化**。只有先解决“召回的上下文是否真的相关”，后面的 Prompt 优化、WebUI 和多格式接入才更有意义。
