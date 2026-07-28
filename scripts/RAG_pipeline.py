"""
RAG 查询主体：输入问题 -> 检索相关片段 -> 拼接prompt -> LLM生成答案
这是评测脚本要调用的核心函数，也是未来对外提供RAG服务的核心函数。
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from indexing.vectorstore import build_vectorstore
from scripts.hybrid_search_optimize import build_hybrid_retriever
from scripts.build_parent_children_index import get_parent_child_retriever  # 新增：父子分块
import pickle

# 加载切分后的chunk文档列表，用于BM25检索
try:
    with open(f"{settings.output_dir}/all_chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)
except FileNotFoundError:
    print("未找到切分后的chunk文档列表，将使用默认值。")
    all_chunks = []

# 初始化模型
embedding_model = HuggingFaceEmbeddings(
    model_name = settings.embedding_model,   # 本地嵌入模型
    model_kwargs = {"device": "cuda"},        # 没有GPU就改成 "cpu"
    encode_kwargs = {"normalize_embeddings": True},
)

# 初始化向量数据库
vector_store = Chroma(
    persist_directory = settings.chroma_persist_dir, # 本地持久化目录
    collection_name = settings.chroma_collection_name,  # 收集名称，作用是区分不同的向量集合
    embedding_function = embedding_model,       # 使用的嵌入模型
)

# 初始化父子分块检索器（独立的collection + 独立的docstore,只加载不重建索引）
# for_indexing=False：查询阶段不需要parent_splitter，索引已经在build_parent_child_index.py里建好了
parent_child_retriever = get_parent_child_retriever(embeddings=embedding_model, for_indexing=False)

# 初始化LLM，后期用于生成答案
llm_client = OpenAI(
    api_key = settings.deepseek_api_key,
    base_url = settings.deepseek_base_url
)

# prompt模板：按照需求进行凭借用户的提问和检索到的相关片段，形成一个完整的prompt,帮助后期的LLM的生成答案
prompt_template = """
    你是一个严谨的问答助手，严格按照用户的提问和检索到的相关片段来回答问题。
    如果参考资料中没有相关的信息，请直接回答“抱歉，我无法回答这个问题。”，不要编造答案。
    检索到的信息：{context}
    用户的提问: {question}
    回答：
"""

# def retrieve(question:str, top_k:int):
#     """
#     检索相关片段
#         :param question: 用户的提问
#         :param top_k: 检索的片段数量
#         :return: 检索到的相关片段列表
#     """
#     # 检索相关的片段
#     top_k = top_k or settings.retrieval_top_k  # 限制top_k的最大值
#     docs = vector_store.similarity_search(query=question, k=top_k)
#     return [doc.page_content for doc in docs]   # docs是Documents列表对象，返回的是每个Document的page_content属性，即文本内容，剩下的元数据属性暂时不需要

def hybird_retriever(question:str,
                    top_k:int,
                    use_hybrid:bool=False,
                    all_chunks:list=None,
                    vector_weight:float=0.5,
                    bm25_weight:float=0.5,
                    use_parent_child:bool=False,
):
    top_k = top_k or settings.retrieval_top_k

    # 使用父子分块检索器
    if use_parent_child:
        docs = parent_child_retriever.invoke(question)
        docs = docs[:top_k]
    # 使用混合检索器（向量+BM25）
    elif use_hybrid:
        hybird_retriever = build_hybrid_retriever(vector_store, all_chunks, top_k=top_k,
                                                    vector_weight=vector_weight, bm25_weight=bm25_weight)
        docs = hybird_retriever.invoke(question)
        docs = docs[:top_k]
    else:
        # 纯向量检索
        docs = vector_store.similarity_search(query=question, k=top_k)

    return [doc.page_content for doc in docs]   # docs是Documents列表对象，返回的是每个Document的page_content属性，即文本内容，剩下的元数据属性暂时不需要

def generate(question:str, top_k:int = None, context:list = None):
    """
    生成答案
        :param question: 用户的提问
        :param top_k: 检索的片段数量
        :param context: 提供的上下文信息
        :return: LLM生成的答案
    """
    # 生成阶段：将检索到的信息与用户提问拼接到一起，形成一个完整的prompt，传给LLM生成答案
    prompt = prompt_template.format(context="\n".join(context), question=question)
    response = llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=settings.llm_temperature,
    )
    # 这里可以查看response的结构，通常是一个字典，里面包含了生成的文本、token使用情况等信息
    return response.choices[0].message.content.strip()  # 返回LLM生成的答案

def rag_answer(question:str, top_k:int = None,use_hybrid:bool=False, all_chunks:list=None,
               vector_weight:float=0.5, bm25_weight:float=0.5,use_parent_child:bool=False):
    """完整RAG查询：检索 + 生成，一次调用拿到全部结果"""
    contexts = hybird_retriever(question, top_k, use_hybrid=use_hybrid, all_chunks=all_chunks,
                                vector_weight=vector_weight, bm25_weight=bm25_weight,
                                use_parent_child=use_parent_child)    # 检索内容
    answer = generate(question, top_k, contexts)  # 生成答案
    return {
        "question": question,
        "retrieved_contexts": contexts,
        "answer": answer,
        "response": answer,  # 兼容旧版本的字段名
    }

if __name__ == "__main__":
    # 测试RAG查询：分别跑一次纯向量和父子分块，直观对比检索到的内容长度差异
    question = "请问RAG的核心思路是什么？"
 
    print("===== 纯向量检索 =====")
    result = rag_answer(question)
    for i, ctx in enumerate(result["retrieved_contexts"], 1):
        print(f"[{i}] 长度{len(ctx)}字符: {ctx[:80]}...")
 
    print("\n===== 父子分块检索 =====")
    result_pc = rag_answer(question, use_parent_child=True)
    for i, ctx in enumerate(result_pc["retrieved_contexts"], 1):
        print(f"[{i}] 长度{len(ctx)}字符: {ctx[:80]}...")