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
from scripts.self_query_retriever import build_self_query_retriever, get_parent_docs_from_children  # 新增：元数据过滤+parent映射
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

# 新增：指向"带元数据的child向量库"的独立Chroma实例，专供SelfQueryRetriever使用。
# 注意这里复用的是父子分块用的同一个collection（PARENT_CHILD_COLLECTION_NAME），
# 因为元数据是在build_parent_children_index.py里绑定到parent/child块上的，
# 不是baseline那个collection。Chroma允许多个实例指向同一个collection做读取，
# 这里不会产生冲突。
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
):
    """
    统一检索入口。
    优先级：use_self_query > use_multi_query > use_parent_child > use_hybrid > 纯向量。
    """
    top_k = top_k or settings.retrieval_top_k

    if use_self_query:
        # child检索候选数量设置得比top_k更大（10），因为多个child去重映射到
        # parent之后，数量会减少，需要留足冗余，这个逻辑和父子分块本身
        # search_kwargs的设计考虑一致
        sq_retriever = build_self_query_retriever(metadata_vectorstore, langchain_llm, top_k=10)
        child_docs = sq_retriever.invoke(question)
        # 关键新增：把过滤筛选出的child块，映射回它们所属的parent块，
        # 让"元数据过滤"和"父子分块大块返回"两个能力叠加，而不是只有过滤没有大块
        docs = get_parent_docs_from_children(child_docs, parent_child_retriever)
        docs = docs[:top_k]
    elif use_multi_query:
        mq_retriever = build_multi_query_retriever(base_retriever=parent_child_retriever, llm=langchain_llm)
        docs = mq_retriever.invoke(question)
        docs = docs[:top_k]
    elif use_parent_child:
        docs = parent_child_retriever.invoke(question)
        docs = docs[:top_k]
    elif use_hybrid:
        hybird_retriever = build_hybrid_retriever(vector_store, all_chunks, top_k=top_k,
                                                    vector_weight=vector_weight, bm25_weight=bm25_weight)
        docs = hybird_retriever.invoke(question)
        docs = docs[:top_k]
    else:
        docs = vector_store.similarity_search(query=question, k=top_k)

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
                use_structured:bool=False, use_multi_query:bool=False, use_self_query:bool=False):
    contexts = hybird_retriever(
        question, top_k, use_hybrid=use_hybrid, all_chunks=all_chunks,
        vector_weight=vector_weight, bm25_weight=bm25_weight,
        use_parent_child=use_parent_child, use_multi_query=use_multi_query,
        use_self_query=use_self_query,
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
    # 对比测试：同一批问题，分别看"父子分块（无过滤）"和"SelfQuery（带元数据过滤）"的差异

    print("=" * 70)
    print("测试1：带明确主题过滤意图的问题")
    print("=" * 70)
    question1 = "帮我找一下关于Docker的笔记里，写了什么内容？"

    print("\n----- 父子分块检索（无过滤，纯语义匹配） -----")
    result_pc = rag_answer(question1, use_parent_child=True)
    print(f"检索到 {len(result_pc['retrieved_contexts'])} 条")
    for i, ctx in enumerate(result_pc["retrieved_contexts"], 1):
        print(f"[{i}] {ctx[:80]}...")

    print("\n----- SelfQuery检索（元数据过滤，verbose会打印解析出的过滤条件） -----")
    result_sq = rag_answer(question1, use_self_query=True)
    print(f"检索到 {len(result_sq['retrieved_contexts'])} 条")
    for i, ctx in enumerate(result_sq["retrieved_contexts"], 1):
        print(f"[{i}] {ctx[:80]}...")

    print("\n\n" + "=" * 70)
    print("测试2：不带过滤意图的普通问题（验证SelfQuery在无过滤需求时是否退化为普通语义检索）")
    print("=" * 70)
    question2 = "光合作用的暗反应发生在哪里？"

    print("\n----- SelfQuery检索 -----")
    result_sq2 = rag_answer(question2, use_self_query=True)
    print(f"检索到 {len(result_sq2['retrieved_contexts'])} 条")
    for i, ctx in enumerate(result_sq2["retrieved_contexts"], 1):
        print(f"[{i}] {ctx[:80]}...")

    print("\n\n" + "=" * 70)
    print("测试3：带时间过滤意图的问题")
    print("=" * 70)
    question3 = "2026年7月18日修改过的笔记里，有没有关于咖啡的内容？"

    print("\n----- SelfQuery检索 -----")
    result_sq3 = rag_answer(question3, use_self_query=True)
    print(f"检索到 {len(result_sq3['retrieved_contexts'])} 条")
    for i, ctx in enumerate(result_sq3["retrieved_contexts"], 1):
        print(f"[{i}] {ctx[:80]}...")