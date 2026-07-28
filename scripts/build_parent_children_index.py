'''
父子分块（Parent-Children Chunking）：索引构建脚本
核心思路：
    - child:子块，切分的较小，专门用于向量检索，保证语义匹配精度
    - parent:父块，切分的较大，保留更加完整的上下文内容
    - 检索时实际匹配的是child的向量，但是命中后返回LLM的是该child所属的parent
    - LangChain的 ParentDocumentRetriever 封装了这套"子块检索、父块返回"的逻辑
持久化说明：
    - child的向量：存入独立的Chroma collection（"notes_parent_child"），
      不会覆盖你现有baseline用的"notes"collection，方便A/B对比
    - parent的原文：默认用InMemoryStore（进程结束就丢失），这里改用
      LocalFileStore落盘存储，保证评测脚本下次能直接加载复用，不用每次重建,不然评测时无法复现baseline的不准确
'''

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
 
from config.settings import settings

# 两级切分参数，作为模块级常量导出，方便RAG_pipeline.py里重建检索器时复用同一套参数
PARENT_CHUNK_SIZE = 1200
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 30

PARENT_CHILD_COLLECTION_NAME = "notes_parent_child"
PARENT_DOCSTORE_DIR = os.path.join(settings.output_dir, "parent_docstore")

def get_parent_child_retriever(
        embeddings:HuggingFaceEmbeddings,
        for_indexing:bool=False,
         child_search_k: int = 10,   # 新增：child检索候选数量，故意设置得比top_k大，
                                      # 因为去重后parent数量必然≤child数量，
                                      # 留足冗余才能保证最终拿到接近top_k个不同的parent
):
    '''
    构建/加载 ParentDocumentRetriever 检索器
    构建/加载 ParentDocumentRetriever。
 
    Args:
        embeddings: 复用你现有的embedding模型实例
        for_indexing: True表示本次是用来"建索引"（需要parent_splitter），
                      False表示本次只是"查询用"（不需要parent_splitter，
                      因为不会再调用add_documents）
    '''
    # 子块分割
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    # 子块向量存储（独立collection,对其他的collection不覆盖，方便A/B对比)
    child_vectorstore = Chroma(
        persist_directory=settings.chroma_persist_dir,
        collection_name=PARENT_CHILD_COLLECTION_NAME,
        embedding_function=embeddings,
    )

    os.makedirs(PARENT_DOCSTORE_DIR, exist_ok=True) 
    fs = LocalFileStore(PARENT_DOCSTORE_DIR)
    docstore = create_kv_docstore(fs)

    kwargs = dict(
        vectorstore=child_vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": child_search_k},   # 新增这一行
    )
    if for_indexing:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
        )
        kwargs["parent_splitter"] = parent_splitter
 
    return ParentDocumentRetriever(**kwargs)

def build_index():
    print("Step 1/3 加载文档 ...")
    loader = DirectoryLoader(
        settings.notes_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"  加载到 {len(docs)} 篇笔记")
 
    print("Step 2/3 初始化embedding模型 ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cuda"},  # 没有GPU改成 "cpu"
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        show_progress=True,
    )
 
    print("Step 3/3 构建父子分块索引（自动完成两级切分+embedding+持久化） ...")
    retriever = get_parent_child_retriever(embeddings, for_indexing=True)
    retriever.add_documents(docs)
    print(f"  索引建立完成，collection={PARENT_CHILD_COLLECTION_NAME}")
    print(f"  父块原文已持久化到: {PARENT_DOCSTORE_DIR}")
 
if __name__ == "__main__":
    build_index()
