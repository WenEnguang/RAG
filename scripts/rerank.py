'''
Rerank（重排序模块）

    核心思路：
        - 前面所有的检索方式（向量/混合/父子分块/元数据过滤）本质上都是“粗排”：
            用向量相似度或关键词快速从全量语料中筛选出候选，速度快但是精度有限
        - Rerank是精排：用一个专门来判断“查询文档”相关性的Cross-Encoder模型，对
            候选逐一精确打分，比相似度更准，但是计算成本也会更高（因为需要对每一对
            “问题+候选文档”单独跑一次模型推理，不能像向量检索那样提前算好索引再快速查找）。
        - 常用的做法是“先粗后精”：粗排多选出一个候选，精排再从中选出真正最相关的几条，兼顾
            速度和精度。
        - BAAI/bge-reranker-base,是一个开源的、中文支持较好的重排序模型，首次会自动下载，之后就会走本地缓存。

    本地存放说明：
    - 模型统一放在模型文件中，方便管理
    - build_reranker()先检查这个目录下有没有模型文件：
        - 没有，就从hf中下载
        - 有，走本地进行加载，不发起任何的网络请求

'''

import os
from config.settings import embedding_model_path
from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

def build_reranker(device:str="cuda"):
    local_dir = os.path.join(embedding_model_path,'bge-reranker-base')

    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        print(f"本地未找到reranker模型，正在下载到 {local_dir} ...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(repo_id=RERANKER_MODEL_NAME, local_dir=local_dir)
        print("下载完成")
    else:
        print(f"检测到本地已有reranker模型，直接从 {local_dir} 加载")

    return CrossEncoder(local_dir,device=device)


def rerank_docs(reranker:CrossEncoder,question:str,docs:list,top_k:int=4) -> list:
    '''
    对候选文档做精排，返回相关性最高的top-K个。
    '''
    if not docs:
        return docs

    # cross_encoder需要成对（问题、文档内容）的输入，对每一个候选逐一打分，
    # 这一步是精排耗时的主要来源，候选数量越多耗时越长。
    pairs = [
        (question,doc.page_content)
        for doc in docs
    ]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs,scores))
    scored_docs.sort(key=lambda x: x[1],reverse=True)

    return [doc for doc,score in scored_docs[:top_k]]