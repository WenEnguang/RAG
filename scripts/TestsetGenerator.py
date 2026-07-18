import os
import sys
import types

# ---- 修补 ragas 对已废弃路径 langchain_community.chat_models.vertexai 的依赖 ----
try:
    import langchain_community.chat_models.vertexai
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
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.testset.persona import Persona

_ = load_dotenv(find_dotenv())

# ---- 1. 读取文档 ----
md_path = settings.notes_dir
loader = DirectoryLoader(md_path, glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
print(f"共加载 {len(docs)} 篇笔记")

# ---- 2. 给 HeadlineSplitter 打容错补丁 ----
# 背景：ragas的HeadlinesExtractor基于LLM生成结果，对少数文档（原因不完全确定，
# 可能是内容结构、也可能是LLM返回的随机性）有一定概率提取失败，没能写入
# headlines属性。下游HeadlineSplitter遇到缺失属性时会直接raise ValueError，
# 且这一步不受generate_with_langchain_docs的raise_exceptions参数控制，
# 之前尝试"过滤短文档"证明和文档长度无必然关系，问题更可能出在个别节点的
# 随机性上。这里直接给split方法打补丁：遇到这个特定错误时，不再让它中断整个
# 生成流程，而是跳过对该节点的标题切分（该节点仍以整篇形式保留在知识图谱里，
# 依然可以参与后续的简单问答生成，只是不会被细分成多个子节点）。
from ragas.testset.transforms.splitters.headline import HeadlineSplitter

_original_split = HeadlineSplitter.split


async def _patched_split(self, node):
    try:
        return await _original_split(self, node)
    except ValueError as e:
        if "headlines" in str(e):
            source = node.properties.get("document_metadata", {}).get("source", "未知来源")
            print(f"  [容错跳过] 节点标题提取失败，跳过切分: {source}")
            return [], []
        raise


HeadlineSplitter.split = _patched_split
# ---- 补丁结束 ----


# ---- 3. 初始化LLM与Embedding ----
generator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
)

local_embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={"device": "cuda"},  # 没有GPU改成 "cpu"
    encode_kwargs={"normalize_embeddings": True},
)
generator_embedding = LangchainEmbeddingsWrapper(local_embeddings)

# ---- 4. 中文persona，避免多跳问题生成时退回英文默认人格 ----
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

# ---- 5. 生成测试集 ----
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embedding,
    persona_list=my_personas,
)

dataset = generator.generate_with_langchain_docs(
    documents=docs,
    testset_size=20,
    raise_exceptions=False,
)

df = dataset.to_pandas()

output_path = os.path.join(settings.output_dir, "testset.csv")
os.makedirs(settings.output_dir, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\n测试集已保存至: {output_path}")
print(f"共生成 {len(df)} 条测试问题")

# ---- 6. 快速质量自检 ----
print("\n===== 快速质量检查 =====")
non_ascii_check = df["user_input"].apply(lambda x: any(ord(c) > 127 for c in x))
english_only_rows = df[~non_ascii_check]
if len(english_only_rows) > 0:
    print(f"⚠️ 发现 {len(english_only_rows)} 条问题不含中文字符（可能是英文问题混入），建议人工检查：")
    for idx, row in english_only_rows.iterrows():
        print(f"  [{idx}] {row['user_input']}")
else:
    print("✅ 未发现明显的英文问题混入")