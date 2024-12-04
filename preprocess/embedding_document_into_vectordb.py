import sys
sys.path.append('..')  # 添加父目录到Python路径
from tool.util import *
import chroma as chroma
from embedding.dense import *
import json

def _study_single_text(db_path, collection_name, text):
    print(f"Studying {text}")
    collection = chroma.get_collection(db_path, collection_name)
    # chroma.show_collection(collection)
    embed_text = embedding_text(text)
    # print(embed_text)
    chroma.add_single_collection(collection, text, embed_text)
    # chroma.show_collection(collection)

def study_all_text(db_path, collection_name, text_li: list):
    for text in text_li:
        # text = text.replace("\n"," ")
        _study_single_text(db_path, collection_name, text)
    print(f"study {text_li} done")

if __name__ == '__main__':
    # 1. 先读取所有的document并组装到一个list中
    import json

    documents = []
    input_filename = "../dataset/CORAL/deduplicate_passage_corpus.json"
    with open(input_filename, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                ref_string = data.get('ref_string', '')
                documents.append(ref_string)
            except:
                pass
    # documents = documents[:10]
    # print(len(documents))
    # 2. 使用Embedding模型进行同步Embedding
    embed_documents = embedding_text(documents,embedding_name="bge-m3")

    # 3. 把原始文档和Embedding后的文档一起放到vdb中
    vector_db_name = "original_bge_m3"
    db_path = f"../vector_db/{vector_db_name}"
    os.makedirs(db_path,exist_ok=True)
    collection_name = "content"
    collection = chroma.get_collection(db_path, collection_name)
    for document,embed_document in zip(documents,embed_documents):
        chroma.add_single_collection(collection, document, embed_document)
    chroma.count_collection(db_path)
    # chroma.show_collection(collection)