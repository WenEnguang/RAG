'''
查询改写模块（Mutil-Query Rewriting）

核心思路：
    - 用户原始问题不一定是最利于向量机检索的表达形式（口语化、模糊、一句多问）
    - 用LLM将原始问题改写成3个不同角度的版本，分别检索，再合并去重
    - 好处：缓解“单一问题覆盖不到相关内容”的语义盲区
    - langchain的MultiQueryRetriever封装了“生成多个改写问题+分别检索+合并去重”
    这套逻辑，只需要传入一个底层retriever（可以是纯向量、也可以是父子分块等任意已有
    的retriever）和一个用于生成改写问题的LLM

用法：
    from query_rewrite import build_multi_query_retriever
    retriever = build_multi_query_retriever(base_retirever, llm)
    results = retirever.invoke(query)  # 返回合并去重后的检索结果
'''

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever

def build_multi_query_retriever(base_retriever:BaseRetriever,llm) -> MultiQueryRetriever:
    '''
    构建多查询改写检索器
    Args:
        base_retriever: 任意已有的langchain retirever对象，作为改写后每个子查询实际执行
                        检索时使用的底层检索器。可以传入纯向量retriever(vector_store.as_retriever()),
                        也可以传入父子分块的parent_child_retriever,这样就可以“查询改写”和“目前验证过的最优检索方式”叠加使用。
        llm: 用于生成改写问题的LLM
    Returns:
        MultiQueryRetriever对象，调用invoke(query)即可返回合并去重后的检索结果
    '''
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )