import sys
sys.path.append('..')  # 添加父目录到Python路径
import os.path
from openai import OpenAI
from zhipuai import ZhipuAI
import random
from constant import *
from difflib import SequenceMatcher
from io import BytesIO
from pdf2image import convert_from_path
import os
import json
import tiktoken
from FlagEmbedding import FlagAutoModel
import torch

'''
https://huggingface.co/BAAI/bge-m3/blob/ecf34d3096437e53f2c163f370b4135980d2e024/README.md
'''

def _embedding_zhipu(text):
    api_key = ZHIPU_API
    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
        model=ZHIPU_EMBEDDING_MODEL,  # 填写需要调用的模型编码
        input=[
            text
        ],
    )
    return response.data[0].embedding


def _embedding_openai(text):
    api_key = OPENAI_API
    api_base = OPENAI_BASE_URL
    # 初始化 openai 的 embeddings 对象
    # embeddings = OpenAIEmbeddings()
    client = OpenAI(api_key=api_key, base_url=api_base)
    model = OPENIA_EMBEDDING_MODEL
    text = text.replace("\n", " ")
    embed_text = client.embeddings.create(input=[text], model=model).data[0].embedding
    return embed_text


def _embedding_bge_m3(text_list):
    model = FlagAutoModel.from_finetuned('/home/models/bge-m3',
                                         query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                         use_fp16=True,
                                         device="cuda:1")
    # model.to(device)

    embed_text = model.encode(text_list)['dense_vecs']
    # print(len(embed_text))
    return embed_text

def embedding_text(text, embedding_name=EMBEDDING_NAME):
    if embedding_name == "zhipu":
        return _embedding_zhipu(text)
    elif embedding_name == "openai":
        return _embedding_openai(text)
    elif embedding_name == "bge-m3":
        return _embedding_bge_m3(text)
    return _embedding_bge_m3(text)

if __name__ == '__main__':
    text = "I hope I can abtain HKPFS!"
    textli = []
    textli.append(text)
    textli.append(text+"xxx")
    textli.append(text+"yyy")
    textli.append(text+"zzz")
    etext = embedding_text(textli,embedding_name="bge-m3")
    print(len(etext))