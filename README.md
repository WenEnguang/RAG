# RAG — 个人笔记检索增强生成系统

一个基于 RAG（Retrieval-Augmented Generation，检索增强生成）的个人笔记问答系统。把你的 Markdown 笔记扔进去，用自然语言提问，让 LLM 基于笔记内容给出有据可查的回答——不瞎编。

## 项目定位

这不是一个"库"或"框架"，而是一个**实验性学习项目**。从最朴素的 Baseline 开始，分阶段逐步优化 RAG 管线的每一个环节（文档切分 → 向量化 → 检索 → 生成），记录每个阶段的做法、结果和踩坑经验。

适合人群：想从零理解 RAG 到底在做什么、每个参数和策略到底影响什么的人。

## 快速开始

### 环境要求

- Python >= 3.10
- GPU（可选）：嵌入模型和 LLM 评测在 CPU 上也能跑，但有 CUDA 会更快

### 安装依赖

```bash
pip install langchain langchain-community langchain-text-splitters \
    langchain-huggingface langchain-chroma langchain-openai \
    openai ragas pandas tqdm torch \
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

> DeepSeek 使用 OpenAI 兼容接口。项目中 RAG 查询链路用原生 `openai` 库直调，评测脚本用 `langchain_openai.ChatOpenAI`，两者都指向 `api.deepseek.com`。

### 2. 准备嵌入模型

本项目使用本地的 `Qwen3-Embedding-0.6B` 作为嵌入模型（1024 维向量），默认路径为项目上级目录的 `Pre_Models/Qwen3-Embedding-0.6B`。首次运行时会自动从 HuggingFace 下载（约 400MB）。

如需修改模型路径或切换其他模型，编辑 `config/settings.py` 中的 `embedding_model`。

> 国内下载慢？设置 HuggingFace 镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### 3. 准备笔记

在 `data/notes/` 目录下放入你的 Markdown 笔记文件。项目包含多篇测试笔记涵盖 RAG 基础概念、Embedding 原理、Transformer 架构等主题。

### 4. 分步验证

```bash
# Step 1: 验证嵌入模型是否正常加载
python scripts/test/test_embedding.py

# Step 2: 验证 DeepSeek API 是否连通
python scripts/test/test_llm.py

# Step 3: 构建向量索引（加载文档 → 切分 → 向量化 → 写入 Chroma）
python scripts/build_vectorstore.py

# Step 4: 全流程冒烟测试（含 chunk 质量分析）
python scripts/test/test_chunk_size.py
```

### 5. 运行 RAG 查询

```python
from scripts.RAG_pipeline import rag_answer

result = rag_answer("什么是 RAG Embedding？")
print(result["answer"])
# 返回 dict: {"question", "retrieved_contexts", "answer"}
```

### 6. 生成测试集 & 跑评测

```bash
# 自动从笔记生成测试集（RAGAS TestsetGenerator）
python scripts/TestsetGenerator.py

# 跑 RAGAS 量化评测
python scripts/evalute.py
```

---

## 项目架构

```
RAG/
├── config/
│   └── settings.py            # 所有可调参数（Pydantic Settings）
├── indexing/
│   └── vectorstore.py         # Chroma 工厂函数（建索引 / 加载索引）
├── scripts/
│   ├── RAG_pipeline.py        # ★ 核心：检索 + 生成主链路
│   ├── build_vectorstore.py   # 构建/重建向量索引
│   ├── TestsetGenerator.py    # 自动生成 RAGAS 测试集
│   ├── evalute.py             # RAGAS 量化评测脚本
│   └── test/
│       ├── test_embedding.py  # 验证本地嵌入模型
│       ├── test_llm.py        # 验证 DeepSeek API
│       ├── test.py            # Phase 0 LCEL 冒烟测试（保留参考）
│       └── test_chunk_size.py # chunk 切分质量分析
├── output/                    # 测试集 & 评测结果（gitignored）
├── experiments/               # 实验记录和 Notebook
├── data/                      # 笔记数据（gitignored）
│   ├── notes/                 # Markdown 笔记
│   └── pdfs/                  # PDF 文档（暂未接入）
└── chroma_db/                 # Chroma 向量库持久化目录（gitignored）
```

### 核心设计原则

**"改配置，不改代码"** —— `config/settings.py` 是所有可调参数的唯一来源。换模型、调切分参数、改检索 top-k，都只改这里。做 A/B 实验时不需要到处翻代码。

**工厂模式封装** —— `indexing/vectorstore.py` 对 Chroma 做了薄封装。业务代码只调用 `build_vectorstore()` 或 `get_vectorstore()`，不直接碰 Chroma 初始化细节。换 Qdrant / Milvus 改这一个文件就够了。

**模块级初始化** —— `RAG_pipeline.py` 在模块级别初始化 embedding 模型、Chroma 连接和 LLM client，作为全局单例。所有调用方 import 后直接使用，避免重复加载模型。

### RAG 数据流

```
用户问题
  │
  ▼
检索 (Chroma similarity_search, top-k=4)
  │  从 chroma_db/ 中检索最相关的 4 个 chunk
  ▼
Prompt 拼接
  │  "参考资料：{context}    用户提问：{question}"
  ▼
DeepSeek LLM (deepseek-v4-flash, temperature=0.0)
  │  仅根据参考资料回答，找不到就说"抱歉，我无法回答这个问题。"
  ▼
字符串输出
```

### 技术栈

| 组件       | 选型                                    | 说明                        |
| ---------- | --------------------------------------- | --------------------------- |
| LLM        | DeepSeek-V4-Flash (via openai)          | OpenAI 兼容接口，原生调用   |
| Embedding  | Qwen3-Embedding-0.6B (HuggingFace 本地) | 1024 维，支持 CPU/CUDA      |
| 向量数据库 | Chroma (本地 persist 模式)              | 无需服务器，开箱即用        |
| RAG 框架   | LangChain（文档加载/切分）+ 原生 OpenAI | 链式编排只在 ingestion 阶段 |
| 评测框架   | RAGAS                                    | Faithfulness, ResponseRelevancy, ContextPrecision/Recall |
| 配置管理   | Pydantic Settings + python-dotenv       | .env 管密钥，类管参数       |

---

## 进展记录

### Phase 0 — 基线验证 ✅

> **目标**：用最朴素的方式把 RAG 管线跑通，拿到可工作的 Baseline，后续每个 Phase 只优化一个环节，通过量化指标衡量提升。

#### 做了什么

1. **文档加载**：`DirectoryLoader` 加载 `data/notes/` 下的 `.md` 文件（从最初的 1 篇扩展到 7 篇）
2. **文档切分**：`RecursiveCharacterTextSplitter`，chunk_size=500，chunk_overlap=50
3. **向量化**：`Qwen3-Embedding-0.6B` 本地模型，支持 CUDA/CPU 自动切换，batch_size=32
4. **索引构建**：Chroma 本地 persist，collection 名 `notes`
5. **检索**：`similarity_search(k=4)`，最基本向量相似度检索
6. **生成**：DeepSeek-V4-Flash，temperature=0.0，原生 `openai` 库直调
7. **模块化重构**：将最初揉在一起的 LCEL 链拆分为独立模块（建索/检索/生成/评测各自独立）
8. **评估体系搭建**：集成 RAGAS，自动生成测试集 + 4 项指标量化评估

#### Chunk 分析结果（7 篇笔记，102 个 chunk）

```
总chunk数量: 102
平均长度: 377.7
最大长度: 494   最小长度: 3

长度分布:
  0-300:   22
  300-600: 80
  600+:     0
```

#### 验证结果（嵌入模型 + LLM + 全流程）

```
向量维度: 1024
句1 vs 句2 (语义相近，应高分): 0.8700
句1 vs 句3 (语义无关，应低分): 0.3364
✅ Embedding 模型验证通过

✅ LLM 调用验证通过

测试问题："RAG系统的基本流程有哪些步骤？"
检索到 4 个 chunk，LLM 准确回答 5 个步骤
✅ Phase 0 全流程冒烟测试通过
```

#### 发现的问题

1. **切分太粗暴**：`chunk_size=500` 的固定切分会把语义相关的句子拦腰截断，最小 chunk 只有 3 个字符。Phase 1 需要探索语义切分或更合理的 chunk 策略。
2. **大规模语料下检索精度下降**：笔记从 1 篇扩展到 7 篇后，提问 RAG 相关问题会检索到 Transformer 等不相关内容。纯向量检索不够，需要混合检索或 Reranker。
3. **PDF 还没接入**：目录和配置已预留，加载逻辑还没写。
4. **评测 LLM 依赖外部 API**：RAGAS 评分用 DeepSeek 当裁判，评分质量和成本受外部模型影响。

#### 关键文件

- `scripts/RAG_pipeline.py` — 当前 RAG 查询的核心入口
- `scripts/build_vectorstore.py` — 索引构建（数据更新后重跑）
- `scripts/evalute.py` — 量化评测（改配置后跑一遍看分数变化）
- `config/settings.py` — 所有 Baseline 参数
- `experiments/notebooks/基线测试结果记录.md` — 完整运行输出

---

### Phase 1 — 待规划

方向候选（按当前痛点优先级排序）：

1. **检索优化** — 混合检索（向量 + BM25）/ Reranker / 查询改写，解决大规模语料下的检索精度问题
2. **文档切分优化** — 语义切分 / 按标题层级切分 / 父子 chunk / 多粒度索引
3. **PDF 文档接入** — 利用已预留的 `pdfs_dir` 配置
4. **评测体系完善** — 增加更多 RAGAS 指标，建立固定的测试集基准

---

## 待办

- [x] 搭建量化评估体系（RAGAS 测试集 + 4 项指标）
- [x] 模块化重构（建索 / 检索 / 生成 / 评测各自独立）
- [ ] 检索优化：混合检索 / Reranker
- [ ] 支持 PDF 文档
- [ ] 添加 WebUI（如 Streamlit / Gradio）
- [ ] Phase 1：文档切分优化
