"""
Phase 0 - 验证脚本，测试目前系统的chunk_size参数是否合理
    相较于test.py文件，此时docs目录下的md笔记已经增加，并且文档的格式也更复杂了，包含了更多的标题以及代码块的内容
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
from time import time

def analyze_chunks(chunks):
    """
    分析chunk质量
    """
    print("\n========== Chunk分析 ==========")
    print(f"总chunk数量: {len(chunks)}")
    lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"平均长度: {sum(lengths)/len(lengths):.1f}")
    print(f"最大长度: {max(lengths)}")
    print(f"最小长度: {min(lengths)}")
    # 长度分布
    print("\n长度分布:")
    ranges = [
        (0,300),
        (300,600),
        (600,1000),
        (1000,1500),
        (1500,99999)
    ]
    for low, high in ranges:
        count = sum(1 for l in lengths if low <= l < high)
        print(f"{low}-{high}: {count}")

    print("\n========== 前10个chunk预览 ==========")
    for i, chunk in enumerate(chunks[:10]):
        print("\n----------------------")
        print(f"Chunk {i}")
        print("长度:",len(chunk.page_content))
        print("来源:",chunk.metadata.get("source"))
        print(chunk.page_content[:500])

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
    analyze_chunks(chunks)

    # ---------- 3. Embedding + 建索引 ----------
    print("Step 3/5 向量化并写入Chroma（本地persist模式） ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    start_time = time()
    vectorstore = build_vectorstore(chunks, embeddings)
    end_time = time()
    print(f"  索引建立完成，collection={settings.chroma_collection_name}，耗时：{end_time - start_time:.2f}秒")

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
