"""
查询改写诊断脚本

目的：绕开日志系统（日志配置是否生效受很多外部因素干扰，不够可靠），
     直接调用 MultiQueryRetriever 内部负责"生成改写问题"的那一步，
     打印出LLM实际生成了哪几个改写版本，用最直接的方式确认这一步有没有真正执行。
"""
from RAG_pipeline import parent_child_retriever, langchain_llm
from scripts.query_rewrite import build_multi_query_retriever

question = "那个亚马逊雨林 它是不是在巴西那边啊 那它到底有多大"

mq_retriever = build_multi_query_retriever(
    base_retriever=parent_child_retriever, llm=langchain_llm
)

# 先看看这个对象上有哪些和"生成查询"相关的属性/方法，帮助确认实际接口
print("MultiQueryRetriever对象上的相关属性/方法：")
relevant_attrs = [a for a in dir(mq_retriever) if not a.startswith("_")]
print(relevant_attrs)

print("\n" + "=" * 60)

# 尝试直接调用生成改写问题的方法
if hasattr(mq_retriever, "generate_queries"):
    try:
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
        # 用no-op run manager代替None——LangChain内部需要一个真正的run_manager对象
        # 来记录执行过程（哪怕不需要真正记录），传None会导致内部调用.get_child()时报错
        noop_run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        queries = mq_retriever.generate_queries(question, noop_run_manager)
        print(f"generate_queries() 直接调用结果:")
        for i, q in enumerate(queries, 1):
            print(f"  改写问题{i}: {q}")
    except Exception as e:
        print(f"generate_queries() 调用失败: {e}")
elif hasattr(mq_retriever, "llm_chain"):
    try:
        result = mq_retriever.llm_chain.invoke({"question": question})
        print(f"llm_chain.invoke() 直接调用结果:\n{result}")
    except Exception as e:
        print(f"llm_chain.invoke() 调用失败: {e}")
else:
    print("未找到 generate_queries 或 llm_chain 属性，需要人工查看上面打印的属性列表，"
          "找到实际负责生成改写问题的方法名")