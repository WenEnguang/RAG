import os
import sys
import types
import inspect

# ---- 修补 ragas 对已废弃路径 langchain_community.chat_models.vertexai 的依赖 ----
# 原因：langchain-community 0.4.x 已移除该文件，ChatVertexAI 现在在 langchain-google-vertexai 包里，
# 但 ragas/llms/base.py 里的 import 语句还没跟上，导致 import ragas 直接报 ModuleNotFoundError。
# 这里在 ragas 被导入之前，手动往 sys.modules 注册一个转发模块，绕开这个死路径。
try:
    import langchain_community.chat_models.vertexai  # 如果哪天官方修好了，这里就不会报错
except ModuleNotFoundError:
    from langchain_google_vertexai import ChatVertexAI
    shim = types.ModuleType("langchain_community.chat_models.vertexai")
    shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = shim
# ---- 修补结束 ----

from config.settings import settings
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.testset.persona import Persona
from ragas.testset.graph import NodeType
from ragas.testset.transforms.engine import Parallel
from ragas.testset.transforms.extractors import EmbeddingExtractor, SummaryExtractor
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)

try:
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
except ImportError:
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

# 加载环境变量
_ = load_dotenv(find_dotenv())


def normalize_docs(raw_docs):
    """过滤远程数据中的空文档，并为无标题 Markdown 补一个稳定标题。"""
    docs = []
    for doc in raw_docs:
        content = (doc.page_content or "").strip()
        if len(content) < 50:
            continue

        has_markdown_heading = any(
            line.lstrip().startswith("#") for line in content.splitlines()
        )
        if not has_markdown_heading:
            source = os.path.basename(doc.metadata.get("source", "untitled.md"))
            content = f"# {source}\n\n{content}"

        docs.append(Document(page_content=content, metadata=doc.metadata))
    return docs


def build_safe_transforms(documents, llm, embedding_model):
    """
    对没有稳定 Markdown 标题层级的文档，直接在 DOCUMENT 节点抽取主题和实体。

    RAGAS 的长文档默认分支依赖 HeadlineSplitter 先创建 CHUNK 节点；当
    headlines 缺失时，Splitter 会失败。若只移除 Splitter，后续主题和实体
    提取仍只处理 CHUNK，最终不会产生可生成问题的场景。
    """
    def is_document(node):
        return node.type == NodeType.DOCUMENT

    return [
        SummaryExtractor(llm=llm, filter_nodes=is_document),
        CustomNodeFilter(llm=llm, filter_nodes=is_document),
        Parallel(
            EmbeddingExtractor(
                embedding_model=embedding_model,
                property_name="summary_embedding",
                embed_property_name="summary",
                filter_nodes=is_document,
            ),
            ThemesExtractor(llm=llm, filter_nodes=is_document),
            NERExtractor(llm=llm, filter_nodes=is_document),
        ),
        Parallel(
            CosineSimilarityBuilder(
                property_name="summary_embedding",
                new_property_name="summary_similarity",
                threshold=0.7,
                filter_nodes=is_document,
            ),
            OverlapScoreBuilder(threshold=0.01, filter_nodes=is_document),
        ),
    ]


def build_single_hop_synthesizer(llm):
    """
    主题是列表属性，可被 SingleHopSpecificQuerySynthesizer 用于构造场景。
    summary 是普通字符串，不能作为该 synthesizer 的问题种子。
    """
    kwargs = {"llm": llm}
    if "property_name" in inspect.signature(SingleHopSpecificQuerySynthesizer).parameters:
        kwargs["property_name"] = "themes"
    return SingleHopSpecificQuerySynthesizer(**kwargs)


# 读取数据
md_path = settings.notes_dir
loader = DirectoryLoader(
    md_path,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
docs = normalize_docs(loader.load())
if not docs:
    raise ValueError(f"没有从 {md_path} 加载到可用于生成测试集的 Markdown 文档")
# docs = docs[:2]  # 临时修改，只取前2个文档，避免测试集生成太慢

# 加载LLM和Embedding模型
generator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    ),
    # DeepSeek 的 OpenAI 兼容接口仅支持 n=1。
    bypass_n=True,
)
local_embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,   # 本地目录路径，正好对上
    model_kwargs={"device": "cuda"},        # 没有GPU就改成 "cpu"
    encode_kwargs={"normalize_embeddings": True},
)
generator_embedding = LangchainEmbeddingsWrapper(local_embeddings)
safe_transforms = build_safe_transforms(docs, generator_llm, generator_embedding)
query_distribution = [
    (build_single_hop_synthesizer(generator_llm), 1.0),
]
generation_run_config = RunConfig(
    timeout=90,
    max_workers=2,
)

my_personas = [
    Persona(
        name="RAG初学者",
        role_description="正在学习RAG技术的开发者，习惯用口语化的方式提问，"
                          "有时会用不太精确的术语描述自己想问的概念。",
    ),
    Persona(
        name="笔记整理者",
        role_description="习惯把学习过程记录成markdown笔记的人，"
                          "提问时经常会关联多篇笔记内容，喜欢问“A和B有什么区别”这类对比性问题。",
    ),
    Persona(
        name="严谨的复习者",
        role_description="复习知识点时会追问细节和原理的人，"
                          "提问方式偏正式、术语使用准确。",
    ),
]

# 生成测试集
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embedding,
    persona_list=my_personas,
)
dataset = generator.generate_with_langchain_docs(
    documents=docs,
    testset_size=20,
    transforms=safe_transforms,
    query_distribution=query_distribution,
    run_config=generation_run_config,
    raise_exceptions=True,
)
df = dataset.to_pandas()
if df.empty:
    raise RuntimeError(
        "RAGAS 没有生成任何测试样本。"
        f"已加载 {len(docs)} 篇文档，但没有得到可用 scenario。"
        "请检查远程 Markdown 是否包含可被模型提取的有效主题。"
    )

output_path = os.path.join(settings.output_dir, "testset.csv")  # 换成你想要的确定路径
os.makedirs(settings.output_dir, exist_ok=True)  # 确保输出目录存在
df.to_csv(output_path, index=False)
print(f"测试集已保存至: {output_path}")
