import chromadb
from openai import OpenAI
from chromadb.config import Settings
from chromadb import PersistentClient

# 根据db_path和collection_name获得一个collection对象
def get_collection(db_path, collection_name):
    settings = Settings()
    settings.persist_directory = db_path
    persistent_client = PersistentClient(path=db_path, settings=settings)  # 必须 path 和 settings 都写，少哪一个都不行
    collection = persistent_client.get_or_create_collection(name=collection_name)
    return collection

# 通过id删除一个collection中的元素
def del_collection_by_id(collection, id: str):
    collection.delete(
        ids=[id],
    )

# 查看一个collection中的所有东西
def show_collection(collection):
    print(collection.get())

# 通过collection中的现有的id获得下一个id的值
def _get_next_id(collection):
    # 获取所有现有的 IDs
    existing_ids = [id for id in collection.get()["ids"]]
    # print(existing_ids)
    # 如果没有现有文档，从 1 开始
    if not existing_ids:
        return "id1"
    # 将 ID 转换为数字，找到最大值
    max_id = max(int(id[2:]) for id in existing_ids)
    # 返回下一个 ID
    return f"id{max_id + 1}"

# 往collection中插入一个嵌入文本
def add_single_collection(collection, text, embed_text):
    id = _get_next_id(collection)
    collection.add(
        documents=[text],
        embeddings=[embed_text],
        ids=[id]  # ids 是用来唯一标识每个文档的。每个 id 在同一个集合（collection）中应该是唯一的，以便于区分和检索不同的文档。
    )

# 获取一个 vector_db 中每一个 collection 含有的数据条数以及所有 collection 的数据量之和
def count_collection(db_path):
    settings = Settings()
    settings.persist_directory = db_path
    client = PersistentClient(path=db_path, settings=settings)
    # 获取所有集合
    collections = client.list_collections()

    total_count = 0
    # 遍历每个集合并计算数据量
    for collection in collections:
        count = collection.count()
        print(f"Collection '{collection.name}' contains {count} items.")
        total_count += count
    # print(f"\nTotal number of items across all collections: {total_count}")

if __name__ == '__main__':
    db_path = "fd-new"
    collection_name = "content"  # 类似关系型数据库中的表
    show_collection(get_collection(db_path, collection_name))
