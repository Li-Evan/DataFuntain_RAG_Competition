import dense
import bm25
import sys

sys.path.append('..')  # 添加父目录到Python路径
from tool.util import *
import chroma as chroma


def retrival(question, summary, type="dense", method="bge-m3"):
    if type == "dense":
        if method == "bge-m3":
            vector_db_name = "original_bge_m3"
            db_path = f"../vector_db/{vector_db_name}"
            collection_name = "content"
            collection = chroma.get_collection(db_path, collection_name)
            ref_doc = dense.bge_en_m3(collection, question)
    elif type == "sparse":
        if method == "bm25":
            ref_doc = bm25.bm25(question)
    else:
        ref_doc = []
    return ref_doc


def rrf_fusion(list1, list2, k=60):
    """
    Implement Reciprocal Rank Fusion for two ranked lists.

    Args:
        list1: First ranked list (highest scoring items first)
        list2: Second ranked list (highest scoring items first)
        k: Constant to prevent items with very low ranks from having too much impact

    Returns:
        A new list sorted by RRF scores
    """
    # 创建字典来存储每个元素的RRF分数
    rrf_scores = {}

    # 处理第一个列表
    for rank, item in enumerate(list1):
        rrf_scores[item] = 1 / (k + rank)

    # 处理第二个列表
    for rank, item in enumerate(list2):
        if item in rrf_scores:
            rrf_scores[item] += 1 / (k + rank)
        else:
            rrf_scores[item] = 1 / (k + rank)

    # 根据RRF分数排序并返回结果
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in sorted_items]

if __name__ == '__main__':
    # 示例用法
    list1 = ["问题A", "问题B", "问题C"]
    list2 = ["问题B", "问题D", "问题A"]

    result = rrf_fusion(list1, list2)
    print(result)