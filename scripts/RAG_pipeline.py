"""
RAG 查询主体：输入问题 -> 检索相关片段 -> 拼接prompt -> LLM生成答案
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI

from config.settings import settings
from indexing.vectorstore import build_vectorstore
from scripts.hybrid_search_optimize import build_hybrid_retriever
from scripts.build_parent_children_index import get_parent_child_retriever, PARENT_CHILD_COLLECTION_NAME
from scripts.structured_generate import generate_structured
from scripts.query_rewrite import build_multi_query_retriever
from scripts.self_query_retriever import build_self_query_retriever, get_parent_docs_from_children
from scripts.rerank import build_reranker, rerank_docs  
import pickle

try:
    with open(f"{settings.output_dir}/all_chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
except FileNotFoundError:
    print("未找到切分后的chunk文档列表，将使用默认值。")
    all_chunks = []

embedding_model = HuggingFaceEmbeddings(
    model_name = settings.embedding_model,
    model_kwargs = {"device": "cuda"},
    encode_kwargs = {"normalize_embeddings": True},
)

vector_store = Chroma(
    persist_directory = settings.chroma_persist_dir,
    collection_name = settings.chroma_collection_name,
    embedding_function = embedding_model,
)

parent_child_retriever = get_parent_child_retriever(embeddings=embedding_model, for_indexing=False)

metadata_vectorstore = Chroma(
    persist_directory=settings.chroma_persist_dir,
    collection_name=PARENT_CHILD_COLLECTION_NAME,
    embedding_function=embedding_model,
)

llm_client = OpenAI(
    api_key = settings.deepseek_api_key,
    base_url = settings.deepseek_base_url
)

langchain_llm = ChatOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    model=settings.llm_model,
    temperature=settings.llm_temperature,
)

# Rerank用的cross-encoder模型，懒加载：只有第一次真正调用use_rerank=True时
# 才会触发build_reranker()（内部会自动判断本地是否已有模型，没有则下载，
# 有则直接加载，这个判断逻辑已经封装在build_reranker内部，这里不需要重复判断）。
# 修正：初始值必须是None，不能是True——之前误把初始值设成True导致
# "is None"这个判断条件永远不成立，缓存形同虚设，每次调用都会重新加载模型
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print("首次使用Rerank，正在初始化cross-encoder模型 ...")
        _reranker = build_reranker(device="cuda")  # 没有GPU改成"cpu"
    return _reranker


prompt_template = """
    你是一个严谨的问答助手，严格按照用户的提问和检索到的相关片段来回答问题。
    如果参考资料中没有相关的信息，请直接回答“抱歉，我无法回答这个问题。”，不要编造答案。
    检索到的信息：{context}
    用户的提问: {question}
    回答：
"""


def hybird_retriever(question:str,
                    top_k:int,
                    use_hybrid:bool=False,
                    all_chunks:list=None,
                    vector_weight:float=0.7,
                    bm25_weight:float=0.3,
                    use_parent_child:bool=False,
                    use_multi_query:bool=False,
                    use_self_query:bool=False,
                    use_rerank:bool=False,
                    rerank_candidate_k:int=10,
):
    """
    统一检索入口。
    优先级：use_self_query > use_multi_query > use_parent_child > use_hybrid > 纯向量。

    use_rerank 说明：
        可叠加在任意上述检索方式之上的"后处理"开关，不是互斥分支。
        开启后，粗排阶段会多捞rerank_candidate_k个候选，再用cross-encoder精排，
        最终返回真正最相关的top_k个。
    """
    top_k = top_k or settings.retrieval_top_k
    fetch_k = rerank_candidate_k if use_rerank else top_k

    if use_self_query:
        sq_retriever = build_self_query_retriever(metadata_vectorstore, langchain_llm, top_k=fetch_k)
        child_docs = sq_retriever.invoke(question)
        docs = get_parent_docs_from_children(child_docs, parent_child_retriever)
    elif use_multi_query:
        mq_retriever = build_multi_query_retriever(base_retriever=parent_child_retriever, llm=langchain_llm)
        docs = mq_retriever.invoke(question)
    elif use_parent_child:
        docs = parent_child_retriever.invoke(question)
    elif use_hybrid:
        hybird_retriever = build_hybrid_retriever(vector_store, all_chunks, top_k=fetch_k,
                                                    vector_weight=vector_weight, bm25_weight=bm25_weight)
        docs = hybird_retriever.invoke(question)
    else:
        docs = vector_store.similarity_search(query=question, k=fetch_k)

    if use_rerank:
        reranker = get_reranker()
        docs = rerank_docs(reranker, question, docs, top_k=top_k)
    else:
        docs = docs[:top_k]

    return [doc.page_content for doc in docs]


def generate(question:str, top_k:int = None, context:list = None):
    prompt = prompt_template.format(context="\n".join(context), question=question)
    response = llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.llm_temperature,
    )
    return response.choices[0].message.content.strip()


def rag_answer(question:str, top_k:int = None, use_hybrid:bool=False, all_chunks:list=None,
                vector_weight:float=0.7, bm25_weight:float=0.3, use_parent_child:bool=False,
                use_structured:bool=False, use_multi_query:bool=False, use_self_query:bool=False,
                use_rerank:bool=False, rerank_candidate_k:int=10):
    # 修正：之前这版遗漏了use_rerank/rerank_candidate_k的接收和向下传递，
    # 导致__main__里调用rag_answer(..., use_rerank=True)会直接报TypeError
    contexts = hybird_retriever(
        question, top_k, use_hybrid=use_hybrid, all_chunks=all_chunks,
        vector_weight=vector_weight, bm25_weight=bm25_weight,
        use_parent_child=use_parent_child, use_multi_query=use_multi_query,
        use_self_query=use_self_query, use_rerank=use_rerank,
        rerank_candidate_k=rerank_candidate_k,
    )

    if use_structured:
        structured_result = generate_structured(
            llm_client, question, contexts,
            model=settings.llm_model, temperature=settings.llm_temperature,
        )
        return {
            "question": question, "retrieved_contexts": contexts,
            "answer": structured_result["answer"], "response": structured_result["answer"],
            "cited_indices": structured_result["cited_indices"],
            "is_answerable": structured_result["is_answerable"],
            "citation_valid": structured_result["citation_valid"],
            "raw_parse_failed": structured_result["raw_parse_failed"],
        }
    else:
        answer = generate(question, top_k, contexts)
        return {"question": question, "retrieved_contexts": contexts, "answer": answer, "response": answer}


if __name__ == "__main__":
    question = "帮我找一下关于Docker的笔记里，写了什么内容？"

    print("===== 不加Rerank（self_query + parent映射） =====")
    result = rag_answer(question, use_self_query=True)
    print(f"检索到 {len(result['retrieved_contexts'])} 条")
    for i, ctx in enumerate(result["retrieved_contexts"], 1):
        print(f"[{i}] {ctx[:80]}...")

    print("\n===== 加Rerank（self_query + parent映射 + rerank精排） =====")
    result_rr = rag_answer(question, use_self_query=True, use_rerank=True)
    print(f"检索到 {len(result_rr['retrieved_contexts'])} 条")
    for i, ctx in enumerate(result_rr["retrieved_contexts"], 1):
        print(f"[{i}] {ctx[:80]}...")