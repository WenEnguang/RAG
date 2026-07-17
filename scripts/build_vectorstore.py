import torch
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings
from indexing.vectorstore import build_vectorstore as save_to_chroma    

device = "cuda" if torch.cuda.is_available() else "cpu"

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

    print("Step 2/3 切分文档 ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"  切分为 {len(chunks)} 个chunk（baseline朴素切分，后续会优化）")
    print("Step 3/3 向量化并写入Chroma（本地persist模式） ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        show_progress=True,
    )
    vectorstore = save_to_chroma(chunks, embeddings)
    print(f"  索引建立完成，collection={settings.chroma_collection_name}")

if __name__ == "__main__":
    build_index()
    