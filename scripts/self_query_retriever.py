"""
元数据自动过滤检索模块（SelfQueryRetriever）

核心思路：
    - 用户的自然语言问题里，可能隐含着过滤意图（比如"关于Docker的笔记"里的
      "关于Docker"，或者"最近更新的内容"里的时间限定）
    - SelfQueryRetriever会先让LLM分析问题，把这类意图解析成结构化的过滤条件
      （类似 primary_tag = 'Docker'），再结合向量检索，先筛选范围、再排相关性
    - 需要预先用AttributeInfo告诉LLM：有哪些元数据字段可用、每个字段是什么类型、
      代表什么含义——这样LLM才能正确地把自然语言"翻译"成过滤条件

本次新增：
    - get_parent_docs_from_children()：把SelfQueryRetriever检索到的child块，
      映射回它们所属的parent块，让"元数据过滤"和"父子分块大块返回"两个能力
      叠加使用，而不是二选一。

用法：
    from self_query_retriever import build_self_query_retriever, get_parent_docs_from_children
    retriever = build_self_query_retriever(vectorstore, llm)
    child_docs = retriever.invoke(question)
    parent_docs = get_parent_docs_from_children(child_docs, parent_child_retriever)
"""

# --- 已知问题：这几个组件在新版langchain里已经挪到了langchain_classic包 ---
# 和之前EnsembleRetriever/ParentDocumentRetriever/LocalFileStore遇到的情况一样，
# 优先尝试langchain_classic路径，失败再退回旧的langchain路径做兼容
try:
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
except ImportError:
    from langchain.retrievers.self_query.base import SelfQueryRetriever

try:
    from langchain_classic.chains.query_constructor.schema import AttributeInfo
except ImportError:
    from langchain.chains.query_constructor.schema import AttributeInfo

# --- Databricks导入bug的补丁（原因见下方说明） ---
# SelfQueryRetriever.from_llm() 内部的 _get_builtin_translator 函数，
# 为了"自动识别向量库类型"，会无条件把所有支持的向量库类型（包括Databricks、
# Pinecone等我们根本用不到的）全部import一遍再做isinstance判断。
# 新版langchain_community里DatabricksVectorSearch已经不再从这个路径导出，
# 导致这个自动识别函数在import阶段就直接崩溃——哪怕我们用的是Chroma，
# 跟Databricks毫无关系。
# 解法：跳过这个"自动识别"逻辑，直接明确指定我们用的翻译器是ChromaTranslator，
# 不给from_llm()机会去触发那段有问题的自动识别代码。
try:
    from langchain_classic.retrievers.self_query.chroma import ChromaTranslator
except ImportError:
    from langchain.retrievers.self_query.chroma import ChromaTranslator

try:
    from langchain_classic.chains.query_constructor.ir import Comparison
except ImportError:
    from langchain.chains.query_constructor.ir import Comparison
# --- 补丁结束 ---


class SafeChromaTranslator(ChromaTranslator):
    """
    ChromaTranslator的兼容包装：
    LangChain的查询构造器在解析日期类字段时（比如modified_date这种格式规整的
    YYYY-MM-DD字符串），有时会自动把值包装成 {"date": "2026-07-18", "type": "date"}
    这种结构化字典，这是给支持"日期范围比较"的高级翻译器准备的表示法。
    但Chroma的where过滤条件只接受基本类型（str/int/float/bool），不认识这种
    嵌套字典，会直接报ValueError。这里在传给父类实际处理之前，先把这种日期字典
    "拆包"还原成普通字符串，绕开这个不兼容问题。
    """
    def visit_comparison(self, comparison: Comparison) -> dict:
        if isinstance(comparison.value, dict) and "date" in comparison.value:
            comparison = Comparison(
                comparator=comparison.comparator,
                attribute=comparison.attribute,
                value=comparison.value["date"],
            )
        return super().visit_comparison(comparison)


def get_parent_docs_from_children(child_docs: list, parent_child_retriever) -> list:
    """
    把SelfQueryRetriever返回的child文档，映射回它们所属的parent文档。

    原理：ParentDocumentRetriever在建索引时，会给每个child块的metadata里
    记录一个doc_id字段（具体字段名由retriever.id_key决定，默认"doc_id"），
    指向它所属parent在docstore里的存储key。这里利用这份已经存在的映射关系，
    从docstore中取出完整的parent原文，替代掉child碎片化的内容，
    让"元数据过滤"和"大块返回"这两个能力叠加起来，而不是二选一。

    Args:
        child_docs: SelfQueryRetriever检索到的child级别Document列表
        parent_child_retriever: 已经构建好的ParentDocumentRetriever实例，
                                 用来获取docstore和id_key

    Returns:
        parent级别的Document列表（已去重，一个parent只返回一次，
        即使它对应多个命中的child块）
    """
    id_key = parent_child_retriever.id_key

    doc_ids = []
    seen = set()
    for doc in child_docs:
        doc_id = doc.metadata.get(id_key)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)

    if not doc_ids:
        # 找不到doc_id映射关系（理论上不应该发生，除非索引结构有问题），
        # 退化返回原始child结果，避免直接报错导致整条链路失败
        return child_docs

    parent_docs = parent_child_retriever.docstore.mget(doc_ids)
    parent_docs = [d for d in parent_docs if d is not None]  # mget可能返回None，过滤掉

    return parent_docs if parent_docs else child_docs


# 描述向量库里存的是什么内容，帮助LLM理解整体上下文
DOCUMENT_CONTENT_DESCRIPTION = "个人学习笔记的内容片段，涵盖AI技术、科学常识、历史文化等多个主题"

# 声明每个可用于过滤的元数据字段：字段名、类型、含义描述
METADATA_FIELD_INFO = [
    AttributeInfo(
        name="primary_tag",
        description="这篇笔记内容片段所属的主题标签，例如'RAG'、'Docker'、'黑洞'、"
                    "'咖啡因'等，代表笔记讨论的核心主题",
        type="string",
    ),
    AttributeInfo(
        name="modified_date",
        description="这篇笔记最后修改的日期，格式为YYYY-MM-DD，例如'2026-07-18'",
        type="string",
    ),
]


def build_self_query_retriever(vectorstore, llm, top_k: int = 4) -> SelfQueryRetriever:
    """
    构建元数据自动过滤检索器

    Args:
        vectorstore: 已经带有primary_tag/modified_date元数据的Chroma向量库实例
                     （必须是重建过、包含元数据的索引，旧的无元数据索引无法使用此功能）
        llm: 用于解析过滤意图的LangChain LLM对象
        top_k: 最终返回结果数量。注意：如果后续要接get_parent_docs_from_children()
               做parent映射去重，这里的top_k建议设置得比最终期望的parent数量更大
               一些（比如8~10），因为多个child去重后可能对应更少的parent数量，
               这个逻辑和父子分块本身的search_kwargs设计考虑是一致的。

    Returns:
        SelfQueryRetriever对象，用法和普通retriever一致：retriever.invoke(question)
    """
    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=METADATA_FIELD_INFO,
        search_kwargs={"k": top_k},
        structured_query_translator=SafeChromaTranslator(),  # 用修复过日期兼容问题的版本
        verbose=True,
    )