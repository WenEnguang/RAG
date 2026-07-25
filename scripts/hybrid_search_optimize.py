"""
多路召回检索模块（已修复中文分词问题）

实现思路：
    - 向量检索：语义相似度，捕获语义相关的内容
    - BM25检索：基于关键词匹配，捕获关键词相关的内容
    - EnsembleRetriever:将两路的结果按照权重进行融合（内部使用RRF算法）

用法：
    from hybrid_search_optimize import build_hybrid_retriever
    retriever = build_hybrid_retriever(vectorstore, docs, top_k=4)
    results = retriever.invoke(query)
"""

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import jieba


def _chinese_tokenizer(text: str) -> list:
    """
    中文分词函数，替代BM25默认的英文空格分词。
    默认分词会把整句中文当成一个不可分割的"单词"（因为中文没有空格），
    实测会导致BM25完全无法做关键词匹配，退化成"总是返回同样几个候选，
    和问题内容毫无关系"（表现为不同问题返回完全相同的chunk，来自同一篇文档）。
    用jieba分词后，BM25才能真正逐词比对，恢复"关键词匹配"这一路本该有的能力。
    """
    return list(jieba.cut(text))


def build_hybrid_retriever(vectorstore, all_chunks, top_k: int = 4, vector_weight=0.5, bm25_weight=0.5):
    '''
    构建混合检索器

    Args:
        vectorstore: 已经构建好的chroma向量库实例
        all_chunks: 切分后的全部chunk文档列表（Document对象），BM25Retriever需要基于这批
            文档在内存里建立索引。不依赖向量库，所以需要单独把切分结果传进来。
            这批chunk应该和写入向量库时用的同一批（保持一致性，否则两路检索
            面对的"候选池"不一致，对比没有任何意义）。
        top_k: 检索结果返回的top_k个chunk
        vector_weight: 向量检索结果的权重
        bm25_weight: BM25检索结果的权重

    Returns:
        返回类型是 EnsembleRetriever，融合了向量检索和BM25检索的结果。
        融合后的检索器，用法和普通retriever一致：retriever.invoke(question)
    '''
    # 向量检索器：复用已有的chroma向量库
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )
    # BM25检索器：显式传入中文分词函数，修复默认英文分词导致的失效问题
    bm25_retriever = BM25Retriever.from_documents(
        all_chunks,
        k=top_k,
        preprocess_func=_chinese_tokenizer,
    )
    # 融合检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )

    return ensemble_retriever