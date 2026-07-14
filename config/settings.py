"""
统一配置管理。
所有"可切换的策略参数"都应该集中在这里，方便后续做A/B实验时只改配置不改代码。
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv,find_dotenv

_ = load_dotenv(find_dotenv())  # 读取.env文件

# 项目目录路径
project_dir = Path(__file__).parent.parent

# 根目录路径
root_dir = Path(__file__).parent.parent.parent
# 嵌入模型目录路径
embedding_model_path = os.path.join(root_dir, 'Pre_Models')


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (DeepSeek，OpenAI兼容接口) ---
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.0  # RAG场景建议低温度，减少胡编

    # --- Embedding (本地模型) ---
    embedding_model: str = os.path.join(embedding_model_path, 'Qwen3-Embedding-0.6B')

    # --- 向量库（本地persist模式） ---
    chroma_persist_dir: str = os.path.join(project_dir, 'chroma_db')  # 本地存储目录
    chroma_collection_name: str = "notes"  

    # --- 数据目录 ---
    notes_dir: str = os.path.join(project_dir, 'data/notes')  # 笔记目录
    pdfs_dir: str = os.path.join(project_dir, 'data/pdfs')  # PDF目录

    # --- 切分参数（baseline，先用最朴素的固定长度切分） ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- 检索参数 ---
    retrieval_top_k: int = 4


settings = Settings()