from jieba import lcut
from langchain_community.retrievers.bm25 import default_preprocessing_func
from config.settings import settings
from sentence_transformers import SentenceTransformer
texts = [
    "RAG Embedding在检索中的作用是什么？",
    "请问腺苷是什么，它怎么让人感到疲劳？",
    "BM25和向量检索有什么区别？",
]

# 初始化嵌入模型
embedding_model = SentenceTransformer(settings.embedding_model)

for text in texts:
    print(f"\n原问题: {text}")
    print(f"默认分词: {default_preprocessing_func(text)}")
    print(f"Qwen Embedding分词: {embedding_model.encode(text)}")
    print(f"jieba分词: {lcut(text)}")