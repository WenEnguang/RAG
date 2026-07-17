"""
Phase 0 - 验证脚本 1/3
目的：验证本地 bge-small-zh embedding 模型能正常加载并向量化文本。

首次运行会自动从 HuggingFace 下载模型（约100MB），需要联网一次，
之后可离线使用。如果下载慢，可配置 HF 镜像（见README）。

运行：python scripts/step1_test_embedding.py
预期输出：打印向量维度 + 两句话的相似度分数
"""
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
import numpy as np


def main():
    print(f"正在加载模型: {settings.embedding_model} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cuda"},  # 个人笔记数据量小，CPU足够，没必要折腾GPU
        encode_kwargs={"normalize_embeddings": True},  # 归一化后可直接用点积算余弦相似度
    )

    texts = [
        "RAG是一种检索增强生成技术",
        "检索增强生成（RAG）结合了信息检索与大语言模型",
        "今天天气真不错，适合出去散步",
    ]

    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors)

    print(f"\n✅ 向量维度: {vectors.shape[1]}")
    print(f"✅ 生成了 {vectors.shape[0]} 条向量\n")

    # 验证语义相似度是否合理：前两句应该比第三句更相似
    sim_01 = np.dot(vectors[0], vectors[1])
    sim_02 = np.dot(vectors[0], vectors[2])
    print(f"句1 vs 句2 (语义相近，应该分数高): {sim_01:.4f}")
    print(f"句1 vs 句3 (语义无关，应该分数低): {sim_02:.4f}")

    assert sim_01 > sim_02, "❌ 相似度关系不符合预期，检查模型是否加载正确"
    print("\n✅ Embedding模型验证通过！")


if __name__ == "__main__":
    main()
