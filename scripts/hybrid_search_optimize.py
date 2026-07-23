"""
多路召回检索模块

实现思路：
    - 向量检索：语义相似度，捕获语义相关的内容
    - BM25检索：基于关键词匹配，捕获关键词相关的内容
    - EmsembleRetriever:将两路的结果按照按照权重进行融合（内部使用RRF、Reciprocal Rank Fusion算法）

用法：
    from hybird_retriever import build_hybrid_retriever
    retriever = build_hybrid_retriever(vectorstore, docs,top_k=4)
    retults = retriever.invoke(query)
"""

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def build_hybrid_retriever(vectorstore,all_chunks,top_k:int=4,vector_weight=0.5,bm25_weight=0.5):
    '''
    构建混合检索器
    
    Args:
        vectorstore:已经构建好的chroma向量库实例
        all_chunks:切分后的全部chunk文档列表（Document对象），BM25Retriever需要基于这批
            文档在内存里建立索引。不依赖向量库，所以需要单独把切分结果传进来。
            这批chunk应该和写入向量库时用的同一批（保持一致性，否则两路检索）
            面对的“候选池”不一致，对比没有任何意义。
        top_k:检索结果返回的top_k个chunk
        vector_weight:向量检索结果的权重
        bm25_weight:BM25检索结果的权重

    Returns:
        返回类型是 EnsembleRetriever，融合了向量检索和BM25检索的结果。
        融合后的检索器，用法和普通retriever一致：retriever.invoke(question)
    '''
    # 向量检索器：复用已有的chroma向量库
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )
    # BM25检索器
    # BM25Retriever.from_documents 会对中文做默认的英文分词处理，
    # 效果不一定理想，如果后续发现BM25这一路召回质量差，
    # 可以考虑传入自定义的中文分词函数（如jieba分词）作为preprocess_func参数
    bm25_retriever = BM25Retriever.from_documents(all_chunks, k=top_k)
    # 融合检索器
    emsemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )

    return emsemble_retriever