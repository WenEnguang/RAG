from jieba import lcut
from langchain_community.retrievers import BM25Retriever

from scripts.RAG_pipeline import all_chunks


def chinese_preprocess(text: str) -> list[str]:
    return [token for token in lcut(text) if token.strip()]


question = "请问腺苷是什么，它怎么让人感到疲劳？"

default_bm25 = BM25Retriever.from_documents(
    all_chunks,
    k=4,
)

jieba_bm25 = BM25Retriever.from_documents(
    all_chunks,
    k=4,
    preprocess_func=chinese_preprocess,
)

print("默认 BM25", default_bm25.invoke(question))
print("jieba BM25", jieba_bm25.invoke(question))