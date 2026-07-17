"""
Phase 0 - 验证脚本 2/3
目的：验证 DeepSeek API 调用正常。

运行：python scripts/step2_test_llm.py
预期输出：模型对一个简单问题的回答
"""
from langchain_openai import ChatOpenAI
from config.settings import settings


def main():
    if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("sk-xxx"):
        raise ValueError("❌ 请先复制 .env.example 为 .env，并填入你的真实 DEEPSEEK_API_KEY")

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.llm_temperature,
    )

    print("正在调用 DeepSeek API ...")
    response = llm.invoke("用一句话解释什么是RAG（检索增强生成）")

    print(f"\n✅ 模型回答:\n{response.content}\n")
    print("✅ LLM调用验证通过！")


if __name__ == "__main__":
    main()
