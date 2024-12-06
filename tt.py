import os
import chroma
if __name__ == '__main__':
    vector_db_name = "original_bge_m3"
    db_path = f"vector_db/{vector_db_name}"
    os.makedirs(db_path, exist_ok=True)
    collection_name = "content"
    collection = chroma.get_collection(db_path, collection_name)
    chroma.count_collection(db_path)
    # chroma.show_collection(collection)