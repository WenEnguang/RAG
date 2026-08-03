'''
自动打标签脚本

核心思路：
    - 用LLM每篇笔记全文，生成多个主题标签tags + 挑出一个最核心的主标签primary tag
    - tags：完整的标签，存成逗号分割字符串，用于人工查看/未来做关键词类特征，
        不用于chroma的精确元数据过滤（chroma不支持“列表包含某值”的模糊匹配）
    - primary tag：单值，专门给selfqueryRetriever做精确过滤使用。
    - 同时记录文件的修改时间，作为时间维度的元数据
输出：
    output/notes_metadata.json，格式：
    {
        "文件名（不含路径和扩展名）": {
            "tags": "标签1, 标签2, 标签3",
            "primary_tag": "标签1",
            "modified_date": "2026-07-15"
        },
        ...
    }
    后续build_parent_children_index.py/build_vectorstore.py会读取这份文件，
    把对应元数据附加到每个chunk的Document.metadata上。
'''
import os
import json
import datetime
from pydantic import BaseModel, Field, ValidationError
from typing import List
from openai import OpenAI

from config.settings import settings

class NoteTags(BaseModel):
    tags: List[str] = Field(
        description="该笔记涉及所有的主题标签，2-5个，每个标签1-4个词，"
                    "覆盖笔记里提到的重要主题，（如果笔记内容较多，混杂了多个不相关的主题，" \
                    "每个话题都应该有对应的标签）"
    )
    primary_tag: str = Field(
        description="该笔记最核心的主题标签，1个，1-4个词，必须是tags列表里的一个"
    )

TAGGING_PROMPT_TEMPLATE = """
请阅读下面这篇笔记的内容，为它生成主题标签。
 
笔记内容：
{content}
 
请以严格的JSON格式输出：
{{
    "tags": ["标签1", "标签2", ...],
    "primary_tag": "从tags中选出的最核心的一个标签"
}}
 
注意：
    - 如果笔记内容混杂多个不相关的主题（比如一篇笔记里同时讲了好几个不同的技术点），
      每个主题都应该生成对应标签，不要遗漏
    - primary_tag必须是tags列表里已经存在的值，不能是新造的词
    - 标签要具体、有区分度，避免"AI"、"技术"这种过于宽泛的词
"""

def get_note_tags(
        client:OpenAI,
        model:str,
        content:str,
        max_retries:int=3
) -> NoteTags:
    """
    调用LLM为单篇笔记生成标签，带重试和一致性校验
    """
    prompt = TAGGING_PROMPT_TEMPLATE.format(content=content)
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw_output = response.choices[0].message.content.strip()

        try:
            parsed = NoteTags.model_validate_json(raw_output)
            # 校验一致性：primary_tag必须真的在tags列表里，否则视为无效，重试
            if parsed.primary_tag in parsed.tags:
                return {
                    "tags": ", ".join(parsed.tags),
                    "primary_tag": parsed.primary_tag,
                }
            # primary_tag不在tags里，直接兜底取tags第一个，不算失败，只是降级处理
            if parsed.tags:
                return {
                    "tags": ", ".join(parsed.tags),
                    "primary_tag": parsed.tags[0],
                }
        except (ValidationError, json.JSONDecodeError):
            continue
 
    # 所有重试都失败，返回空标签，后续可以人工检查这些笔记
    return {"tags": "", "primary_tag": "未分类"}
 
 
def main():
    notes_dir = settings.notes_dir
    md_files = [f for f in os.listdir(notes_dir) if f.endswith(".md")]
    md_files.sort()
 
    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
 
    metadata = {}
    for filename in md_files:
        filepath = os.path.join(notes_dir, filename)
        key = os.path.splitext(filename)[0]  # 去掉.md后缀作为key
 
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
 
        print(f"正在处理: {filename} ...")
        tag_result = get_note_tags(client, settings.llm_model, content)
 
        mtime = os.path.getmtime(filepath)
        modified_date = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
 
        metadata[key] = {
            "tags": tag_result["tags"],
            "primary_tag": tag_result["primary_tag"],
            "modified_date": modified_date,
        }
        print(f"  tags: {tag_result['tags']}")
        print(f"  primary_tag: {tag_result['primary_tag']}")
        print(f"  modified_date: {modified_date}\n")
 
    output_path = os.path.join(settings.output_dir, "notes_metadata.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
 
    print(f"全部处理完成，共 {len(metadata)} 篇笔记")
    print(f"元数据已保存至: {output_path}")
 
    # 打印一下未分类的笔记，方便人工检查
    unclassified = [k for k, v in metadata.items() if v["primary_tag"] == "未分类"]
    if unclassified:
        print(f"\n⚠️ 以下 {len(unclassified)} 篇笔记打标签失败，建议人工检查: {unclassified}")
 
 
if __name__ == "__main__":
    main()
