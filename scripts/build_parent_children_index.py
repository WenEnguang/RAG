'''
父子分块（Parent-Children Chunking）：索引构建脚本（已支持元数据绑定）
 
核心思路：
    - child:子块，切分的较小，专门用于向量检索，保证语义匹配精度
    - parent:父块，切分的较大，保留更加完整的上下文内容
    - 检索时实际匹配的是child的向量，但是命中后返回LLM的是该child所属的parent
    - LangChain的 ParentDocumentRetriever 封装了这套"子块检索、父块返回"的逻辑
 
新增：
    - 在文档加载阶段（切分之前），把 notes_metadata.json 里预先生成的
      primary_tag（主标签）和 modified_date（修改日期）绑定到每篇笔记的
      Document.metadata 上。切分器会自动把这份metadata复制给切出来的
      每一个parent/child块，不需要对每个chunk单独处理。
    - 只写入metadata字段，不改动page_content本身，因此不会影响向量检索质量
      （原理见之前讨论：向量只由page_content算出，metadata是独立的过滤字段）
 
持久化说明：
    - child的向量：存入独立的Chroma collection（"notes_parent_child"）
    - parent的原文：LocalFileStore落盘存储
'''

import os
import json
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
NOTES_METADATA_PATH = os.path.join(settings.output_dir, "notes_metadata.json")

def get_parent_child_retriever(
        embeddings: HuggingFaceEmbeddings,
        for_indexing: bool = False,
):
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )
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
        search_kwargs={"k": 10},
    )
    if for_indexing:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
        )
        kwargs["parent_splitter"] = parent_splitter
 
    return ParentDocumentRetriever(**kwargs)

def attach_metadata(docs: list, metadata_path: str) -> list:
    """
    把 notes_metadata.json 里的 primary_tag / modified_date 绑定到对应文档的metadata上。
    docs里每个Document的metadata['source']是完整文件路径，需要提取出不含路径和
    扩展名的文件名，去匹配notes_metadata.json里的key。
    """
    if not os.path.exists(metadata_path):
        print(f"⚠️ 未找到 {metadata_path}，跳过元数据绑定，请先运行 auto_tag_notes.py")
        return docs
 
    with open(metadata_path, "r", encoding="utf-8") as f:
        notes_metadata = json.load(f)
 
    matched_count = 0
    for doc in docs:
        source_path = doc.metadata.get("source", "")
        filename_key = os.path.splitext(os.path.basename(source_path))[0]
 
        if filename_key in notes_metadata:
            doc.metadata["primary_tag"] = notes_metadata[filename_key]["primary_tag"]
            doc.metadata["modified_date"] = notes_metadata[filename_key]["modified_date"]
            matched_count += 1
        else:
            # 匹配不上的情况（比如新增了笔记但还没跑auto_tag_notes.py），
            # 给一个兜底值，避免SelfQueryRetriever因为字段缺失而报错
            doc.metadata["primary_tag"] = "未分类"
            doc.metadata["modified_date"] = "未知"
            print(f"⚠️ 未找到 {filename_key} 对应的元数据，使用兜底值")
 
    print(f"元数据绑定完成: {matched_count}/{len(docs)} 篇笔记成功匹配")
    return docs

def build_index():
    print("Step 1/4 加载文档 ...")
    loader = DirectoryLoader(
        settings.notes_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"  加载到 {len(docs)} 篇笔记")
 
    print("Step 2/4 绑定元数据（primary_tag / modified_date） ...")
    docs = attach_metadata(docs, NOTES_METADATA_PATH)
 
    print("Step 3/4 初始化embedding模型 ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        show_progress=True,
    )
 
    print("Step 4/4 构建父子分块索引（自动完成两级切分+embedding+持久化） ...")
    retriever = get_parent_child_retriever(embeddings, for_indexing=True)
    retriever.add_documents(docs)
    print(f"  索引建立完成，collection={PARENT_CHILD_COLLECTION_NAME}")
    print(f"  父块原文已持久化到: {PARENT_DOCSTORE_DIR}")
 
 
if __name__ == "__main__":
    build_index()