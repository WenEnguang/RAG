"""
评测脚本：拿testset.csv里的问题，逐条跑你的RAG主链路，收集实际的检索结果和生成答案，
再用ragas打分，量化你的RAG系统在retrieval和generation两端的表现。
每次改config.py里的参数（chunk_size, retrieval_top_k...）后重跑一遍这个脚本，
对比分数变化，就是最直接的A/B实验方式。
"""
import os
import ast

import pandas as pd
from tqdm import tqdm


import sys
import types

# --- ragas 0.4.3+ 的已知bug临时补丁 ---
# ragas/llms/base.py 内部无条件导入了 langchain_community.chat_models.vertexai.ChatVertexAI，
# 这个模块在新版langchain_community(1.x系)里已被彻底移除。
# 我们的项目完全不用VertexAI，这里手动伪造一个空模块骗过导入检查即可，不影响任何实际功能。
# 等ragas官方修复此问题后，可以删掉这段补丁。
_fake_module = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:  # 占位符，永远不会被真正调用
    pass
_fake_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _fake_module
# --- 补丁结束 ---
from config.settings import settings
from RAG_pipeline import rag_answer,embedding_model

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper

# ---- 1. 读取testset ----
testset_path = os.path.join(settings.output_dir, "testset.csv")
testset_df = pd.read_csv(testset_path)

# reference_contexts 列存的是字符串形式的list，读出来要转回真正的list
testset_df["reference_contexts"] = testset_df["reference_contexts"].apply(ast.literal_eval)

# ---- 2. 逐条跑RAG主链路，收集真实的检索结果和生成答案 ----
records = []
for _, row in tqdm(testset_df.iterrows(), total=len(testset_df), desc="跑RAG主链路"):
    result = rag_answer(row["user_input"])
    records.append({
        "user_input": row["user_input"],
        "retrieved_contexts": result["retrieved_contexts"],   # 你的系统实际检索到的
        "response": result["response"],                       # 你的系统实际生成的
        "reference": row["reference"],                         # testset里的标准答案
        "reference_contexts": row["reference_contexts"],       # testset里的标准检索片段
    })

evaluation_dataset = EvaluationDataset.from_list(records)
print(f"测评数据集：{evaluation_dataset}")

# ---- 3. 配置评测用的LLM（用DeepSeek当裁判）和嵌入模型 ----
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.llm_model,
        temperature=0,
    )
)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    embedding_model
)
# embedding_model = HuggingFaceEmbeddings(
#     model_name = settings.embedding_model,   # 本地嵌入模型
#     model_kwargs = {"device": "cuda"},        # 没有GPU就改成 "cpu"
#     encode_kwargs = {"normalize_embeddings": True},
# )

# ---- 4. 跑评测 ----
# Faithfulness / ResponseRelevancy 看生成端；ContextPrecision / ContextRecall 看检索端
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall()
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

# ---- 5. 保存结果，方便和下一次改配置后的结果做对比 ----
result_df = result.to_pandas()
output_path = os.path.join(settings.output_dir, "eval_result.csv")
result_df.to_csv(output_path, index=False)

print("\n==== 评测汇总 ====")
print(result)
print(f"\n详细结果已保存至: {output_path}")