# RAG — 个人笔记检索增强生成系统

一个基于 RAG（Retrieval-Augmented Generation，检索增强生成）的个人笔记问答系统。把你的 Markdown 笔记扔进去，用自然语言提问，让 LLM 基于笔记内容给出有据可查的回答——不瞎编。

## 项目定位

这不是一个"库"或"框架"，而是一个**实验性学习项目**。我会从最朴素的 Baseline 开始，分阶段逐步优化 RAG 管线的每一个环节（文档切分 → 向量化 → 检索 → 生成），记录每个阶段的做法、结果和踩坑经验。

适合人群：想从零理解 RAG 到底在做什么、每个参数和策略到底影响什么的人。

## 快速开始

### 环境要求

- Python >= 3.10
- 已安装的依赖（参考 pyproject.toml）：`langchain`、`langchain-community`、`langchain-text-splitters`、`langchain-huggingface`、`langchain-chroma`、`langchain-openai`、`pydantic-settings`、`python-dotenv`

```bash
pip install langchain langchain-community langchain-text-splitters \
    langchain-huggingface langchain-chroma langchain-openai \
    pydantic-settings python-dotenv
```

### 1. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

`.env` 内容：

```env
DEEPSEEK_API_KEY=sk-你的真实key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

> DeepSeek 使用 OpenAI 兼容接口，所以本项目通过 `langchain_openai.ChatOpenAI` 来调用，只是把 `base_url` 指向 `api.deepseek.com`。

### 2. 准备嵌入模型

本项目使用本地的 `Qwen3-Embedding-0.6B` 作为嵌入模型（1024 维向量），默认路径为项目上级目录的 `Pre_Models/Qwen3-Embedding-0.6B`。首次运行时会自动从 HuggingFace 下载（约 400MB）。

如需修改模型路径或切换其他模型，编辑 `config/settings.py` 中的 `embedding_model`。

> 国内下载慢？设置 HuggingFace 镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### 3. 准备笔记

在 `data/notes/` 目录下放入你的 Markdown 笔记文件。项目已有一篇测试笔记 `RAG基础概念笔记.md`。

### 4. 分步验证

项目提供了三个验证脚本，建议按顺序运行：

```bash
# Step 1: 验证嵌入模型是否正常加载
python scripts/test_embedding.py

# Step 2: 验证 DeepSeek API 是否连通
python scripts/test_llm.py

# Step 3: 全流程冒烟测试（加载→切分→向量化→检索→生成）
python scripts/test.py
```

## 项目架构

```
RAG/
├── config/               # 所有可调参数集中管理（Pydantic Settings）
│   └── settings.py       # LLM、Embedding、Chroma、切分、检索参数
├── indexing/             # 向量库抽象层
│   └── vectorstore.py    # Chroma 工厂函数（建索引 / 加载已有索引）
├── scripts/              # 验证脚本
│   ├── test_embedding.py # 验证本地嵌入模型
│   ├── test_llm.py       # 验证 DeepSeek API
│   └── test.py           # 全流程冒烟测试
├── experiments/          # 实验记录和 Notebook
├── data/                 # 笔记数据（gitignored）
│   ├── notes/            # Markdown 笔记
│   └── pdfs/             # PDF 文档（暂未接入）
├── chroma_db/            # Chroma 向量库持久化目录（gitignored）
└── Pre_Models/           # 本地嵌入模型文件（项目上级目录）
```

### 核心设计原则

**"改配置，不改代码"** —— `config/settings.py` 是所有可调参数的唯一来源。无论是换模型、调切分参数、改检索 top-k，都只改这里。这样后续做 A/B 实验时，不需要到处翻代码。

**工厂模式封装** —— `indexing/vectorstore.py` 对 Chroma 做了薄封装。业务代码只调用 `build_vectorstore()` 或 `get_vectorstore()`，不直接碰 Chroma 初始化细节。哪天想换 Qdrant / Milvus，改这一个文件就够了。

**LCEL 链式编排** —— 使用 LangChain Expression Language (`|` 运算符) 串联 RAG 各组件，每个环节（检索、格式化、拼 prompt、调 LLM、解析输出）是独立的可替换模块。

### RAG 数据流

```
用户问题
  │
  ▼
检索器 (Chroma Retriever, top-k=4)
  │  从 chroma_db/ 中检索最相关的 4 个 chunk
  ▼
格式化 (拼接 page_content)
  │  多个 chunk 用 "\n\n---\n\n" 分隔
  ▼
Prompt 模板
  │  参考资料 + 用户问题 → 完整的 prompt
  ▼
DeepSeek LLM (temperature=0.0)
  │  仅根据参考资料回答，找不到就说不知道
  ▼
字符串输出
```

### 技术栈

| 组件       | 选型                                    | 说明                   |
| ---------- | --------------------------------------- | ---------------------- |
| LLM        | DeepSeek (via langchain-openai)         | OpenAI 兼容接口        |
| Embedding  | Qwen3-Embedding-0.6B (HuggingFace 本地) | 1024 维，CPU 推理      |
| 向量数据库 | Chroma (本地 persist 模式)              | 无需服务器，开箱即用   |
| 框架       | LangChain + LCEL                        | 链式编排，组件可替换   |
| 配置管理   | Pydantic Settings + python-dotenv       | .env 管密钥，类管参数  |

---

## 进展记录

### Phase 0 — 基线验证 ✅

> **目标**：用最朴素的方式把 RAG 管线跑通，拿到一个可工作的 Baseline，后续每个 Phase 只优化一个环节，通过对比指标衡量提升。

#### 做了什么

1. **文档加载**：`DirectoryLoader` 加载 `data/notes/` 下的 `.md` 文件
2. **文档切分**：`RecursiveCharacterTextSplitter`，chunk_size=500，chunk_overlap=50（最朴素的固定长度切分）
3. **向量化**：`Qwen3-Embedding-0.6B` 本地模型，CPU 推理，归一化向量
4. **索引构建**：Chroma 本地 persist，collection 名 `notes`
5. **检索**：`as_retriever(k=4)`，最基本的向量相似度检索
6. **生成**：DeepSeek (`deepseek-chat`)，temperature=0.0，prompt 明确要求"仅根据参考资料回答，不知道就说不知道"

#### 验证结果

测试问题：**"RAG系统的基本流程有哪些步骤？"**

嵌入模型验证：

```
向量维度: 1024
句1 vs 句2 (语义相近，应高分): 0.8700
句1 vs 句3 (语义无关，应低分): 0.3364
✅ Embedding 模型验证通过
```

LLM 调用验证：

```
RAG（检索增强生成）是一种让大语言模型在生成回答前，先从外部知识库中
检索相关信息作为参考，从而提升答案准确性和时效性的技术方法。
✅ LLM 调用验证通过
```

全流程冒烟测试：

```
加载到 1 篇笔记 → 切分为 2 个 chunk → 索引建立完成

检索到的上下文:
  [1] ## RAG的基本流程 一个最简单的RAG系统包含以下几个步骤...
  [2] title: RAG基础概念笔记 tags: [RAG, LLM, 检索]...

LLM 生成回答:
根据现有笔记，RAG系统的基本流程包含以下步骤：
1. 文档切分（Chunking）：把长文档切成较小的片段
2. 向量化（Embedding）：把每个片段转换成向量表示
3. 索引建立（Indexing）：把向量存入向量数据库
4. 检索（Retrieval）：根据用户问题的向量，在库中找出最相似的片段
5. 生成（Generation）：把检索到的片段和用户问题一起交给LLM生成回答
```

#### 发现的问题 & 后续方向

1. **切分太粗暴**：`chunk_size=500` 的固定切分会把语义相关的句子拦腰截断。Phase 1 应该探索语义切分（如按标题/段落切分）或更合理的 chunk 大小。
2. **检索只看相似度**：纯向量检索可能漏掉精确的关键词匹配。后续考虑混合检索（Hybrid Search）或重排序（Reranker）。
3. **PDF 还没接入**：目录和配置已经预留了 PDF 路径，但加载逻辑还没写。
4. **没有量化评估**：目前只能人工判断"回答看起来对不对"。需要建立评估数据集和量化指标（如 RAGAS）。

#### 关键文件

- `scripts/test.py` — 全流程代码，Phase 0 的核心参考
- `config/settings.py` — 所有 Baseline 参数
- `experiments/notebooks/测试结果记录.md` — 原始运行输出

---

### Phase 1 — 待规划

方向候选：
- 文档切分优化（语义切分 / 父子 chunk / 多粒度索引）
- 检索优化（混合检索 / Reranker / 查询改写）
- PDF 文档接入
- 评估体系搭建（RAGAS）

---

## 待办

- [ ] 搭建量化评估体系（测试集 + 指标）
- [ ] 支持 PDF 文档
- [ ] 添加 WebUI（如 Streamlit / Gradio）
- [ ] Phase 1：文档切分优化
