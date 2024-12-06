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
import sys

sys.path.append('..')  # 添加父目录到Python路径


def _split_sentences(text, split_char='。'):
    """
    将文本分割成句子
    """
    # 这里使用简单的标点符号分割，您可能需要根据实际情况调整
    return [s.strip() for s in text.split(split_char) if s.strip()]


def calculate_similarity(a, b):
    """
    计算两个字符串的相似度
    """
    return SequenceMatcher(None, a, b).ratio()


def find_single_best_match(sentences: list, target: str):
    """
    找到与知识点最匹配的句子
    """
    best_match = ""
    best_score = 0
    for sentence in sentences:
        score = calculate_similarity(target, sentence)
        if score > best_score:
            best_score = score
            best_match = sentence
    return best_match


def _talk_gpt(text_list: list):
    api_key = OPENAI_API
    api_base = OPENAI_BASE_URL

    client = OpenAI(api_key=api_key, base_url=api_base)

    message = []
    message.append({"role": "system", "content": "You are a helpful assistant."})
    for i, text in enumerate(text_list):
        if i % 2 == 0:
            role = "user"
        else:
            role = "assistant"
        message.append({"role": role, "content": text})
    # print(message)
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        # model="gpt-3.5-turbo",
        # stream: False,
        messages=message,
    )
    return completion.choices[0].message.content


def _talk_zhipu(text_list: list):
    api_key = ZHIPU_API
    client = ZhipuAI(api_key=api_key)  # 填写您自己的APIKey

    message = []
    message.append({"role": "system", "content": "You are a helpful assistant."})
    for i, text in enumerate(text_list):
        if i % 2 == 0:
            role = "user"
        else:
            role = "assistant"
        message.append({"role": role, "content": text})

    response = client.chat.completions.create(
        model=ZHIPU_MODEL,  # 填写需要调用的模型名称
        messages=message,
    )

    return response.choices[0].message.content


def talk_llm(text_list, llm_name=LLM_NAME):
    if llm_name == "openai":
        return _talk_gpt(text_list)
    elif llm_name == "zhipu":
        return _talk_zhipu(text_list)
    else:
        raise ValueError("Invalid LLM name.")
    # return _talk_gpt(text_list)


def pdf_to_images(pdf_path, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 将PDF转换为图片
    pages = convert_from_path(pdf_path)
    file_name = os.path.basename(pdf_path).split(".")[0]

    # 保存每一页为独立的图片文件
    for i, page in enumerate(pages):
        image_name = f"{file_name}_page_{i + 1}.jpg"
        image_path = os.path.join(output_folder, image_name)
        page.save(image_path, "JPEG")
        # print(f"Saved {image_name}")


def truncate_string(input_string, chunk_size=2000):
    # 检查输入字符串是否为空
    if not input_string:
        return []

    # 将字符串分割成指定长度的多个字符串
    chunks = [input_string[i:i + chunk_size] for i in range(0, len(input_string), chunk_size)]
    return chunks


def count_tokens(text, model="gpt-3.5-turbo"):
    # 加载与指定模型兼容的编码器
    encoding = tiktoken.encoding_for_model(model)
    # 计算字符串的 tokens 数量
    tokens = encoding.encode(text)
    return len(tokens)


def split_long_text(string_list, max_tokens=512, model="gpt-3.5-turbo", gap_sentence=1):
    split_string_list = []
    temp_list = []

    # 批量处理字符串，每次处理32个
    for i in range(0, len(string_list), gap_sentence):
        batch = string_list[i:i + gap_sentence]
        temp_string = " ".join(temp_list + batch)  # 拼接当前批次的字符串
        current_token_count = count_tokens(temp_string, model)
        if current_token_count > max_tokens:
            # 如果拼接后的字符串token数超过最大限制，保存之前的结果并重新开始拼接
            split_string_list.append(temp_string)
            temp_list = batch  # 开始新的拼接
        else:
            # 如果未超过最大token限制，继续累积当前批次的字符串
            temp_list += batch

    # 处理最后一组字符串
    if temp_list:
        split_string_list.append(" ".join(temp_list))

    return split_string_list


def reflect_doc_to_dict():
    file_path = r"dataset/CORAL/deduplicate_passage_corpus.json"
    reflect_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                ref_string = data.get('ref_string', '')
                ref_id = data.get('ref_id', '')
                reflect_dict[ref_string] = ref_id
            except:
                pass
    return reflect_dict


if __name__ == '__main__':

    file_path = r"C:\Users\Evan\Desktop\zju-rag-edu\algorithm\original_data\学术写作书籍\写作过程层面\2.做研究是有趣的：给学术新人的科研入门笔记 (刀熊) (Z-Library)_json\第三部分__深耕学术写作：从风格到结构.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, value in data.items():
        print(len(value))
        chunks = truncate_string(value)
        print(len(chunks))
