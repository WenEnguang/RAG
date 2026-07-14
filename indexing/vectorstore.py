"""
向量库工厂函数。

设计意图：业务代码（ingestion/retrieval）永远只调用这里的函数，
不直接碰 Chroma 的初始化细节，方便以后统一调整存储方式。
"""
from langchain_chroma import Chroma

from config.settings import settings


def build_vectorstore(chunks, embeddings) -> Chroma:
    """首次建索引用：把切分好的chunks写入本地向量库"""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
    )


def get_vectorstore(embeddings) -> Chroma:
    """连接到已存在的本地向量库（检索时用，不重新写入）"""
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )