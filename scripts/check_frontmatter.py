'''
文档结构诊断脚本
    目的：
        1. 有没有YAML frontmatter（文件开头的 --- ... --- 结构化信息块）
        2. 如果有，包含哪些字段（title/tags/date等）
        3. 如果没有，第一行/前几行长什么样（用于后续判断怎么补充元数据）
'''

import os
import re
from config.settings import settings

def check_frontmatter(file_path:str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # frontmatter正则匹配:文件开头是---，然后是任意内容，最后是---，中间的内容可以跨行
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)

    if match:
        frontmatter_raw = match.group(1)
        # 简单提取字段名（不做完整yaml解析，只看有哪些顶层字段）
        fields = re.findall(r"^(\w+):", frontmatter_raw, re.MULTILINE)
        return {
            "has_frontmatter": True,
            "fields": fields,
            "raw_preview": frontmatter_raw[:200],
        }
    else:
        # 没有frontmatter，看前100个字符长什么样
        preview = content[:100].replace("\n", " ")
        return {
            "has_frontmatter": False,
            "fields": [],
            "raw_preview": preview,
        }

def main():
    notes_dir = settings.notes_dir
    md_files = [f for f in os.listdir(notes_dir) if f.endswith(".md")]
    md_files.sort()
 
    print(f"共找到 {len(md_files)} 篇笔记\n")
 
    has_fm_count = 0
    no_fm_count = 0
 
    for filename in md_files:
        filepath = os.path.join(notes_dir, filename)
        info = check_frontmatter(filepath)
 
        if info["has_frontmatter"]:
            has_fm_count += 1
            print(f"✅ [有frontmatter] {filename}")
            print(f"   字段: {info['fields']}")
        else:
            no_fm_count += 1
            print(f"❌ [无frontmatter] {filename}")
            print(f"   开头预览: {info['raw_preview']}...")
        print()
 
    print("=" * 60)
    print(f"汇总: {has_fm_count} 篇有frontmatter, {no_fm_count} 篇没有")
    print(f"覆盖率: {has_fm_count / len(md_files) * 100:.1f}%")
 
 
if __name__ == "__main__":
    main()
 