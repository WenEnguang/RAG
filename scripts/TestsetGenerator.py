import os
import sys
import types

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
from ragas.llms import llm_factory
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from openai import OpenAI
from ragas.testset import TestsetGenerator

# 加载环境变量
_ = load_dotenv(find_dotenv())

# 读取数据
md_path = settings.notes_dir
loader = DirectoryLoader(md_path, glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
# docs = docs[:2]  # 临时修改，只取前2个文档，避免测试集生成太慢

# 加载LLM和Embedding模型
client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url
)
generator_llm = llm_factory(
    client=client,
    model=settings.llm_model,
    provider="openai",
    max_tokens=4096
)
local_embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,   # 本地目录路径，正好对上
    model_kwargs={"device": "cuda"},        # 没有GPU就改成 "cpu"
    encode_kwargs={"normalize_embeddings": True},
)
generator_embedding = LangchainEmbeddingsWrapper(local_embeddings)

# 生成测试集
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embedding
)
dataset = generator.generate_with_langchain_docs(
    documents=docs,
    testset_size=10,
)
df = dataset.to_pandas()

output_path = os.path.join(settings.output_dir, "testset.csv")  # 换成你想要的确定路径
os.makedirs(settings.output_dir, exist_ok=True)  # 确保输出目录存在
df.to_csv(output_path, index=False)
print(f"测试集已保存至: {output_path}")