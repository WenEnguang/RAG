'''
结构化生成模块(Structured Output)

核心思路：
    - 现有的generate()让LLM自由输出一段文本，没法验证“这句话是不是真的，是否有资料依据”，
    因为文本和资料之间是没有显式对应关系的。
    - 这次改为：给每一段检索到的资料编号，要求LLM按照固定的JSON格式输出，除了回答正文
    还必须明确指出“这个回答引用了第几段资料”。
    - 有了这个显式的引用编号，就可以写代码核查验证：LLM说应用来第几段，但是一共只检索到了
    其他资料——这种情况就可以说明LLM在编造引用，是幻觉的一个信号。
    - 使用Deep Seek/OpenAI兼容的JSON模式，(response_format={"type":"json_object"})
    来强制输出JSON，而不是LangChain的with_structured_output，因为项目里
    generate()用的是原生openai客户端直连，不经过LangChain的LLM封装层。
用法：
    from structured_generate import generate_structured
    result = generate_structured(llm_client, question, contexts, model, temperature)
    # result: {"answer": str, "cited_indices": list[int], "is_answerable": bool,
    #          "citation_valid": bool, "raw_parse_failed": bool}
'''

import json
from pydantic import BaseModel, Field, ValidationError
from typing import List

class RAGAnswer(BaseModel):
    answer:str=Field(description="基于检索到的资料生成的回答正文")
    cited_indices:List[int] = Field(
        default_factory=list,
        description="LLM在回答中引用的资料编号列表，编号从1开始"
                    "如果资料不足以回答，返回空列表"
    )
    is_answerable:bool = Field(
        description="检索到的资料足以回答用户的问题"
    )

STRUCTURED_PROMPT_TEMPLATE = """
你是一个严谨的回答助手，请严格按照下面编号的检索资料来回答用户的问题。

检索到的资料：
{numbered_contexts}
用户的提问：
{question}
请严格按照下面的JSON格式输出：
{{
    "answer":"你的回答正文，如果资料不足以回答，这里写'抱歉，我无法回答这个问题'"，
    "cited_indices":[引用到的资料编号，例如[1,3]，如果资料不足以回答，这里返回空列表]，
    "is_answerable":true/false, 表示资料是否足以回答问题。
}}
注意：
    - cited_indices里的编号必须是上面资料实际存在的编号，不要编造不存在的编号
    - 只有当你回答内容确实是参考了某段资料，才把该段编号写进cited_indices
    - 如果资料不足以回答问题，answer写'抱歉，我无法回答这个问题'，cited_indices返回空列表，is_answerable返回false
"""

def _format_numbered_context(contexts:List[str]) -> str:
    """给每一段检索到的资料加上编号，方便LLM在回答时引用"""
    return "\n\n".join(
        f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)
    )

def _validate_citations(cited_indices:List[int],num_contexts:int) -> bool:
    '''
    检查LLM给出的引用编号是否都在合法的范围内
    只要有一个编号超出了范围
    就判断这次引用不可信，是一个潜在的幻觉信号  
    '''
    if not cited_indices:
        return True # 没有引用（例如拒答的情况），不算无效引用
    return all(1 <= idx <= num_contexts for idx in cited_indices)

def generate_structured(
        llm_client,question:str,contexts:List[str],
        model:str,temperature:float=0.0,
        max_retries:int=3
) -> dict:
    """
    结构化生成：返回回答正文 + 引用编号 + 引用有效性校验效果

    Args:
        llm_client: openai.Client对象
        question: 用户的提问
        contexts: 检索到的资料列表，每个元素是一段文本
        model: LLM模型名称
        temperature: 生成温度，越低越确定性
        max_retries: 如果LLM输出的JSON无法解析，最多重试几次

    Returns:
        dict: {
            "answer": str, 回答正文
            "cited_indices": list[int], 引用编号列表
            "is_answerable": bool, 是否有足够资料回答
            "citation_valid": bool, 引用编号是否都在合法范围内
            "raw_parse_failed": bool, LLM输出的JSON是否无法解析
        }
    """
    numbered_contexts = _format_numbered_context(contexts)
    prompt = STRUCTURED_PROMPT_TEMPLATE.format(
        numbered_contexts=numbered_contexts,
        question=question
    )
    last_raw_response = ""
    for attempt in range(max_retries):
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{
                "role":"user",
                "content":prompt
            }],
            temperature=temperature,
            response_format={"type":"json_object"}  # 强制要求LLM输出JSON
        )
        raw_output = response.choices[0].message.content.strip()
        last_raw_response = raw_output
        try:
            parsed = RAGAnswer.model_validate_json(raw_output)
            citation_valid = _validate_citations(
                                                parsed.cited_indices, 
                                                len(contexts)
                            )
            return {
                "answer": parsed.answer,
                "cited_indices": parsed.cited_indices,
                "is_answerable": parsed.is_answerable,
                "citation_valid": citation_valid,
                "raw_parse_failed": False,
            }
        except (ValidationError, json.JSONDecodeError):
            # 解析失败，如果还有重试次数，下一轮继续；否则走兜底
            continue

    # 所有重试都解析失败，走兜底：把原始输出直接当作答案返回，
    # 保证上层调用不会因为解析失败而崩溃，但明确标记这次解析是失败的
    return {
        "answer": last_raw_response,
        "cited_indices": [],
        "is_answerable": None,
        "citation_valid": False,
        "raw_parse_failed": True,
    }
