"""
RAG 查询主体：输入问题 -> 检索相关片段 -> 拼接prompt -> LLM生成答案
这是评测脚本要调用的核心函数，也是未来对外提供RAG服务的核心函数。
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
 
from config.settings import settings
from indexing.vectorstore import build_vectorstore
from scripts.hybrid_search_optimize import build_hybrid_retriever
from scripts.build_parent_children_index import get_parent_child_retriever
from scripts.structured_generate import generate_structured  
from scripts.query_rewrite import build_multi_query_retriever
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

# 初始化父子分块检索器
parent_child_retriever = get_parent_child_retriever(
                                    embeddings=embedding_model, 
                                    for_indexing=False
                        )

# 初始化LLM，后期用于生成答案
llm_client = OpenAI(
    api_key = settings.deepseek_api_key,
    base_url = settings.deepseek_base_url
)

# LangChain封装的LLM，专供MultiQueryRetriever使用
langchain_llm = ChatOpenAI(
    api_key = settings.deepseek_api_key,
    base_url = settings.deepseek_base_url,
    model = settings.llm_model,
    temperature = settings.llm_temperature,
)

# prompt模板（非结构化模式使用，结构化模式用的是structured_generate.py里独立的模板）
prompt_template = """
    你是一个严谨的问答助手，严格按照用户的提问和检索到的相关片段来回答问题。
    如果参考资料中没有相关的信息，请直接回答“抱歉，我无法回答这个问题。”，不要编造答案。
    检索到的信息：{context}
    用户的提问: {question}
    回答：
"""

def hybird_retriever(
        question:str,
        top_k:int,
        use_hybrid:bool=False,
        all_chunks:list=None,
        vector_weight:float=0.7,
        bm25_weight:float=0.3,
        use_parent_child:bool=False,
        use_multi_query:bool=False,
):
    '''
    统一检索入口：根据参数选择不同的检索方式
        use_multi_query:
            True时：先使用简单返回时验证一下——以父子分块检索器作为改为后每个子查询
            的底层检索器（因为父子分块是目前验证过的最优检索方式），
            再用MultiQueryRetriever做查询改写，最后合并去重返回。
    '''
    
    top_k = top_k or settings.retrieval_top_k  # 限制top_k的最大值
    if use_multi_query:
        # 使用查询改写检索器
        mq_retriever = build_multi_query_retriever(
            base_retriever=parent_child_retriever,
            llm=langchain_llm
        )
        docs = mq_retriever.invoke(question)
        docs = docs[:top_k]  # 只取前top_k个结果
    elif use_parent_child:
        # 使用父子分块检索器
        docs = parent_child_retriever.invoke(question)
        docs = docs[:top_k]  # 只取前top_k个结果
    elif use_hybrid:
        # 使用混合检测器
        hybrid_retriever = build_hybrid_retriever(
            vector_store=vector_store,
            all_chunks=all_chunks,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight
        )
        docs = hybrid_retriever.invoke(question)
        docs = docs[:top_k]  # 只取前top_k个结果
    else:
        # 使用向量检索器
        docs = vector_store.similarity_search(query=question, k=top_k)
    return [doc.page_content for doc in docs]  # 返回每个Document的page_content属性，即文本内容，剩下的元数据属性暂时不需要 

def generate(question:str,top_k:int=None,context:list=None):
    """
    生成答案（非结构化，保持向后兼容）
    """
    prompt = prompt_template.format(context="\n".join(context), question=question)
    response = llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=settings.llm_temperature,
    )
    return response.choices[0].message.content.strip()

def rag_answer(question:str, top_k:int = None, use_hybrid:bool=False, 
               all_chunks:list=None,vector_weight:float=0.7, 
               bm25_weight:float=0.3, use_parent_child:bool=False,
                use_structured:bool=False,use_multi_query:bool=False):
    """
    完整RAG查询：检索 + 生成，一次调用拿到全部结果
 
    新增 use_structured 开关：
        True 时走 structured_generate.py 里的结构化生成逻辑，
        返回结果里会多出 cited_indices / is_answerable / citation_valid /
        raw_parse_failed 这几个字段，用于分析引用是否可信。
        False 时走原有的自由文本生成，行为和之前完全一致。
    """
    contexts = hybird_retriever(
        question, top_k, use_hybrid=use_hybrid, all_chunks=all_chunks,
        vector_weight=vector_weight, bm25_weight=bm25_weight,
        use_parent_child=use_parent_child, use_multi_query=use_multi_query,
    )
    if use_structured:
        structured_result = generate_structured(
            llm_client, question, contexts,
            model=settings.llm_model, temperature=settings.llm_temperature,
        )
        return {
            "question": question,
            "retrieved_contexts": contexts,
            "answer": structured_result["answer"],
            "response": structured_result["answer"],  # 兼容ragas评测，只取回答正文
            "cited_indices": structured_result["cited_indices"],
            "is_answerable": structured_result["is_answerable"],
            "citation_valid": structured_result["citation_valid"],
            "raw_parse_failed": structured_result["raw_parse_failed"],
        }
    else:
        answer = generate(question, top_k, contexts)
        return {
            "question": question,
            "retrieved_contexts": contexts,
            "answer": answer,
            "response": answer,
        }

if __name__ == "__main__":
    # 对比测试：父子分块 vs 查询改写(基于父子分块)，看检索到的内容数量/多样性差异
    question = "那个亚马逊雨林 它是不是在巴西那边啊 那它到底有多大"  # 用一个口语化问题测试改写效果
 
    print("===== 父子分块检索（不改写） =====")
    result = rag_answer(question, use_parent_child=True)
    print(f"检索到 {len(result['retrieved_contexts'])} 条")
    for i, ctx in enumerate(result["retrieved_contexts"], 1):
        print(f"[{i}] {ctx[:60]}...")
 
    print("\n===== 查询改写 + 父子分块检索 =====")
    result_mq = rag_answer(question, use_multi_query=True)
    print(f"检索到 {len(result_mq['retrieved_contexts'])} 条")
    for i, ctx in enumerate(result_mq["retrieved_contexts"], 1):
        print(f"[{i}] {ctx[:60]}...")

    