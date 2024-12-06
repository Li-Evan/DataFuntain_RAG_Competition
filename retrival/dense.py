'''
dense 检索, 单纯基于向量相似度
'''
import sys

sys.path.append('..')  # 添加父目录到Python路径
from tool.util import *
import chroma as chroma
from embedding.dense import *

def bge_en_m3(collection,query, limit_num=100):
    # 对问题进行嵌入
    embed_query = embedding_text(query, embedding_name="bge-en-m3")
    # 相似度查找
    similar_info = collection.query(
        query_embeddings=[embed_query],
        n_results=limit_num,
    )
    query_result = similar_info["documents"]
    return query_result


