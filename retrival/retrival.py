import dense
import sys
sys.path.append('..')  # 添加父目录到Python路径
from tool.util import *
import chroma as chroma

def retrival(question,summary,type="dense",method="bge-m3"):
    if type=="dense":
        if method=="bge-m3":
            vector_db_name = "original_bge_m3"
            db_path = f"../vector_db/{vector_db_name}"
            collection_name = "content"
            collection = chroma.get_collection(db_path,collection_name)
            ref_doc = dense.bge_en_m3(collection,question)

    return ref_doc