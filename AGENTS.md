# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) system for personal notes. Phase 0 baseline is working — documents load, split, embed, index into Chroma, retrieve, and generate via DeepSeek LLM. RAGAS-based quantitative evaluation is integrated. The project follows a phased experimentation approach; each phase optimizes a specific RAG component.

## Commands

```bash
# Verify embedding model loads and produces reasonable semantic similarity
python scripts/test/test_embedding.py

# Verify DeepSeek API connectivity
python scripts/test/test_llm.py

# Build/rebuild the vector index (load → split → embed → write to Chroma)
python scripts/build_vectorstore.py

# Full pipeline test with chunk quality analysis
python scripts/test/test_chunk_size.py

# Run RAG query (interactive / single-shot)
python scripts/RAG_pipeline.py

# Generate RAGAS test set from notes
python scripts/TestsetGenerator.py

# Run RAGAS quantitative evaluation (Faithfulness, ResponseRelevancy, ContextPrecision/Recall)
python scripts/evalute.py
```

No test framework, linter, or formatter is configured. Validation is done through the test scripts and RAGAS evaluation.

## Architecture

### Core Pipeline (`scripts/RAG_pipeline.py`)
The central RAG query module. Uses module-level singletons for embedding model, Chroma connection, and LLM client — import once, call anywhere. Three key functions:
- `retrieve(question, top_k)` — similarity search against Chroma
- `generate(question, top_k)` — calls DeepSeek with retrieved context + prompt
- `rag_answer(question, top_k)` — combines both, returns dict with question, retrieved_contexts, answer

Uses the raw `openai` client (not LangChain `ChatOpenAI`) for generation — direct `client.chat.completions.create()`.

### Configuration (`config/settings.py`)
Single source of truth for all tunable parameters via Pydantic `BaseSettings`. Design intent: change config, not code, when running A/B experiments.

- **LLM**: DeepSeek-V4-Flash via OpenAI-compatible API (raw `openai` client for query, `ChatOpenAI` for RAGAS evaluator)
- **Embedding**: Local HuggingFace model (`Qwen3-Embedding-0.6B`, 1024-dim vectors), expected at `../Pre_Models/Qwen3-Embedding-0.6B` relative to the project root
- **Vector store**: Chroma, persisted locally to `chroma_db/`
- **Chunking**: `RecursiveCharacterTextSplitter` with `chunk_size=500`, `chunk_overlap=50`
- **Retrieval**: top-k=4
- **Output**: test sets and eval results written to `output/`

Secrets come from `.env` (never committed). Copy `.env.example` → `.env` and fill in `DEEPSEEK_API_KEY`.

### Indexing (`indexing/vectorstore.py` + `scripts/build_vectorstore.py`)
- `indexing/vectorstore.py` — factory layer over Chroma: `build_vectorstore()` (ingestion) and `get_vectorstore()` (retrieval). Business code never touches Chroma directly.
- `scripts/build_vectorstore.py` — standalone script to build/rebuild the index. Auto-detects CUDA vs CPU. Use this after adding or modifying notes.

### Evaluation (`scripts/evalute.py` + `scripts/TestsetGenerator.py`)
- `TestsetGenerator.py` — uses RAGAS `TestsetGenerator` to auto-generate test questions from documents. Contains a monkey-patch for a ragas/langchain-community compatibility issue (deprecated `ChatVertexAI` import path).
- `evalute.py` — reads `output/testset.csv`, runs each question through `rag_answer()`, scores with 4 RAGAS metrics (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall) using DeepSeek as judge LLM. Results saved to `output/eval_result.csv`.

### Data Flow
1. `build_vectorstore.py`: `DirectoryLoader` loads `.md` → `RecursiveCharacterTextSplitter` chunks → `HuggingFaceEmbeddings` vectorizes (CUDA/CPU, batch_size=32, normalized) → `Chroma.from_documents` persists
2. `RAG_pipeline.py`: question → `vector_store.similarity_search(k=4)` → retrieved contexts + question → prompt template → DeepSeek LLM → answer string
3. `evalute.py`: testset questions → `rag_answer()` per question → RAGAS scoring (DeepSeek as evaluator LLM) → metrics CSV

The prompt enforces strict context-only answers: "如果参考资料中没有相关的信息，请直接回答'抱歉，我无法回答这个问题。'"

### Key Dependencies
- **langchain** ecosystem: `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-chroma`, `langchain-openai`
- **openai** — raw client for DeepSeek API calls in RAG pipeline
- **ragas** — test set generation + evaluation metrics
- **Chroma** — local vector database (no server needed)
- **pydantic-settings** + **python-dotenv** — config management

### Directory Layout
```
RAG/
├── config/              # Pydantic settings — all tunable parameters
├── indexing/            # Vector store factory (Chroma abstraction)
├── scripts/
│   ├── RAG_pipeline.py  # ★ Core: retrieve + generate entry point
│   ├── build_vectorstore.py  # Build/rebuild vector index
│   ├── TestsetGenerator.py   # RAGAS auto test set generation
│   ├── evalute.py            # RAGAS quantitative evaluation
│   └── test/                 # Verification + analysis scripts
├── output/              # Test sets & eval results (gitignored)
├── experiments/         # Experiment logs and notebooks
├── data/                # Notes (markdown) and PDFs (gitignored)
├── chroma_db/           # Persisted vector index (gitignored)
└── Pre_Models/          # Local embedding model files (outside project dir)
```

### Phase Roadmap
- **Phase 0** (current): Baseline — fixed-size chunking, naive vector retrieval, modularized pipeline, RAGAS evaluation framework in place. Chunk analysis done on 7-note corpus (102 chunks, avg 378 chars). Known issue: retrieval precision degrades with larger corpus.
- **Phase 1+** (planned): Retrieval improvements (hybrid search, reranking), chunking improvements (semantic/sentence-based), PDF ingestion.
