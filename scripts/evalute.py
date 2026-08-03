"""
评测脚本：拿testset.csv里的问题，逐条跑你的RAG主链路，收集实际的检索结果和生成答案，
再用ragas打分，量化你的RAG系统在retrieval和generation两端的表现。

本版新增：结构化生成模式下的引用有效性统计（citation_valid相关），
这部分是ragas无法评估的自定义指标，单独统计、单独展示。
"""
import os
import ast

import pandas as pd
from tqdm import tqdm

import sys
import types

# --- ragas 0.4.3+ 的已知bug临时补丁 ---
_fake_module = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:
    pass
_fake_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _fake_module
# --- 补丁结束 ---
from config.settings import settings
from RAG_pipeline import rag_answer, embedding_model, all_chunks

from ragas.run_config import RunConfig
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper

# ---- 1. 读取testset ----
testset_path = os.path.join(settings.output_dir, "testset.csv")
testset_df = pd.read_csv(testset_path)
testset_df["reference_contexts"] = testset_df["reference_contexts"].apply(ast.literal_eval)

# ---- 2. 本次实验配置 ----
USE_HYBIRD = False
USE_PARENT_CHILD = False   # 父子分块检索方式
USE_STRUCTURED = True     # 结构化生成
USE_SELF_QUERY= True   # 元数据检索

records = []
citation_stats = []  # 单独收集引用相关的自定义统计，不进ragas

for _, row in tqdm(testset_df.iterrows(), total=len(testset_df), desc="跑RAG主链路"):
    result = rag_answer(
        question=row["user_input"],
        use_hybrid=USE_HYBIRD,
        all_chunks=all_chunks,
        use_parent_child=USE_PARENT_CHILD,
        use_structured=USE_STRUCTURED,
        use_self_query=USE_SELF_QUERY,
    )
    records.append({
        "user_input": row["user_input"],
        "retrieved_contexts": result["retrieved_contexts"],
        "response": result["response"],
        "reference": row["reference"],
        "reference_contexts": row["reference_contexts"],
    })

    if USE_STRUCTURED:
        citation_stats.append({
            "user_input": row["user_input"],
            "cited_indices": result.get("cited_indices"),
            "is_answerable": result.get("is_answerable"),
            "citation_valid": result.get("citation_valid"),
            "raw_parse_failed": result.get("raw_parse_failed"),
        })

evaluation_dataset = EvaluationDataset.from_list(records)
print(f"测评数据集：{evaluation_dataset}")

# ---- 3. 配置评测用的LLM和嵌入模型 ----
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.llm_model,
        temperature=0,
    ),
    bypass_n=True,
)
evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# ---- 4. 跑ragas评测 ----
my_config = RunConfig(timeout=300, max_workers=4)
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall()
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=my_config,
    raise_exceptions=False,
)

# ---- 5. 保存ragas结果 ----
result_df = result.to_pandas()
if USE_STRUCTURED:
    if USE_SELF_QUERY:
        suffix = "self_query_parent_structured"
    else:
        suffix = "parent_child_structured" if USE_PARENT_CHILD else "structured"
elif USE_PARENT_CHILD:
    suffix = "parent_child"
elif USE_HYBIRD:
    suffix = "hybird"
else:
    suffix = "baseline"
output_path = os.path.join(settings.output_dir, f"eval_result_{suffix}.csv")
result_df.to_csv(output_path, index=False)

print("\n==== ragas评测汇总 ====")
print(result)
print(f"\n详细结果已保存至: {output_path}")

# ---- 6. 单独汇总引用有效性统计（不属于ragas，自定义分析） ----
if USE_STRUCTURED:
    citation_df = pd.DataFrame(citation_stats)
    citation_output_path = os.path.join(settings.output_dir, f"citation_stats_{suffix}.csv")
    citation_df.to_csv(citation_output_path, index=False)

    total = len(citation_df)
    valid_citation_rate = citation_df["citation_valid"].sum() / total
    parse_fail_rate = citation_df["raw_parse_failed"].sum() / total
    answerable_count = citation_df["is_answerable"].sum()

    print("\n==== 引用有效性统计（自定义，非ragas指标） ====")
    print(f"总问题数: {total}")
    print(f"引用编号有效率: {valid_citation_rate:.4f}（LLM给出的引用编号在合法范围内的比例）")
    print(f"JSON解析失败率: {parse_fail_rate:.4f}（越低越好，说明结构化输出稳定性）")
    print(f"判定为'可回答'的问题数: {answerable_count} / {total}")
    print(f"详细引用统计已保存至: {citation_output_path}")