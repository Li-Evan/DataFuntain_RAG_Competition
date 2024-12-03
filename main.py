from embedding.embedding import *
from summary.summary import *
from retrival.retrival import *
from rerank.rerank import *
from generate.generate import *
import os
import chroma
import json
# from rag import *


def rag_query(conv):
    # 0. 对输入进行相关的处理
    conv_id = conv["conv_id"]
    turns = conv["turns"]

    # 1. 拿到问题后先对之前的内容进行预处理
    pre_turns = turns[:-1]
    summary = summary_pre_turns(pre_turns)
    current_question = turns[-1]["question"]

    # 2. 基于当前问题和 summary 进行相关的检索相关文档
    reference_document_list = retrival(current_question,summary)

    # 3. 通过 rerank 得到最终的附加到 prompt 上的文档
    final_reference_document_list = rerank(reference_document_list)

    # 4. 根据相关的文档, 添加到 prompt 得到最终的回答
    with open("prompt/raw.txt", "r", encoding="utf-8") as f:
        final_prompt = f.read().format(
            question=current_question,
            reference_document=final_reference_document_list,
        )

     # 5. 根据生成模型进行生成
    # TODO: 这里可能还需要修改一下，要实验把之前的所有的轮次对话不做处理全部放进去的效果
    answer = generate(final_prompt)

if __name__ == '__main__':
    file_path = "xx"
    with open(file_path,"r",encoding="utf-8") as file:
        data = json.load(file)





