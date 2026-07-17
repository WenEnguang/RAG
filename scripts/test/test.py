"""
Phase 0 - 验证脚本 3/3（全流程冒烟测试）
目的：把 Loader -> Splitter -> Embedding -> Chroma -> Retriever -> LLM 串起来跑一遍，
验证 Phase 0 的目标——"对一篇md笔记问一个问题，能拿到检索结果+LLM回答"。

【需重点学习】本脚本涉及的LangChain核心概念：
1. Document 对象：page_content(正文) + metadata(元数据字典)，后面所有优化都围绕它展开
2. TextSplitter：这里先用最朴素的 RecursiveCharacterTextSplitter 作为baseline
3. Chroma.from_documents：一次性把切分好的Document写入向量库并建索引
4. as_retriever()：把向量库包装成统一的Retriever接口，后续换检索策略只需换这一行
5. LCEL (LangChain Expression Language)：用 | 符号把各个组件串成一条链，是LangChain的核心写法

运行：python scripts/step3_smoke_test.py
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from indexing.vectorstore import build_vectorstore


def build_baseline_chain():
    # ---------- 1. 加载文档 ----------
    print("Step 1/5 加载文档 ...")
    loader = DirectoryLoader(
        settings.notes_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"  加载到 {len(docs)} 篇笔记")

    # ---------- 2. 切分 ----------
    print("Step 2/5 切分文档 ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"  切分为 {len(chunks)} 个chunk（baseline朴素切分，Phase1后续会优化）")

    # ---------- 3. Embedding + 建索引 ----------
    print("Step 3/5 向量化并写入Chroma（本地persist模式） ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = build_vectorstore(chunks, embeddings)
    print(f"  索引建立完成，collection={settings.chroma_collection_name}")

    # ---------- 4. 检索器 ----------
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_top_k})

    # ---------- 5. Prompt + LLM，用LCEL组链 ----------
    print("Step 4/5 组装生成链 ...")
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.llm_temperature,
    )

    prompt = ChatPromptTemplate.from_template(
        """你是一个基于个人笔记回答问题的助手。请仅根据下面提供的"参考资料"回答问题，
不要编造资料中没有的内容。如果资料中没有相关信息，请直接说"根据现有笔记无法回答这个问题"。

参考资料：
{context}

问题：{question}

回答："""
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    # LCEL链：question进来 -> 并行取 context(检索+格式化) 和 question本身 -> 拼进prompt -> LLM -> 解析成字符串
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Step 5/5 链路组装完成\n")
    return chain, retriever


def main():
    chain, retriever = build_baseline_chain()

    question = "RAG系统的基本流程有哪些步骤？"
    print(f"【测试问题】{question}\n")

    print("【检索到的上下文】")
    retrieved = retriever.invoke(question)
    for i, doc in enumerate(retrieved):
        preview = doc.page_content.replace("\n", " ")[:60]
        print(f"  [{i+1}] {preview}...")

    print("\n【LLM生成回答】")
    answer = chain.invoke(question)
    print(answer)

    print("\n✅ Phase 0 全流程冒烟测试通过！")


if __name__ == "__main__":
    main()
