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


def split_long_text(string_list, max_tokens=512, model="gpt-3.5-turbo",gap_sentence=1):
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

if __name__ == '__main__':

    text = '''
    FSHF uefa.com . 22 january 2015 . retrieved 11 january 2018 . ^ `` foto e rrall\u00eb/ ja si e ka pritur ahmet zogu ekipin komb\u00ebtar t\u00eb futbollit '' [ rare picture / here 's how ahmet zogu hosted the national football team ] ( in albanian ) . citynews . 27 december 2015. archived from the original on 2018-01-12 . retrieved 11 january 2018 . ^ `` uefa statement on albania '' . uefa.com . 20 march 2008 . retrieved 11 january 2018 . ^ `` duka re-elected as albanian fa president '' . uefa.com . 10 march 2010 . retrieved 11 january 2018 . ^ bashkim tufa ( 25 august 2010 ) . `` 80 vjet futboll , blater d he platini n\u00eb tiran\u00eb '' [ 80 years of football , blatter and platini in tirana ) ] ( in albanian ) . sport ekspres . retrieved 24 august 2010 . ^ `` stadiumi i ri 60 milione euro '' [ new stadium 60 million euros ] ( in albanian ) . fshf.org . 3 february 2011. archived from the original on 2018-01-12 . retrieved 11 january 2018 . ^ inagurohet zyra e pare rajonale e fshf-se ne durres archived 2017-12-10 at the wayback machine fshf.org external links national football association - fshf fshf statutes , 2009 fshf documentary about albania football history albaniasoccer albania at fifa site albania at uefa site national football teams weltfussball albania sport albaniansport.net v t e football in albania albanian football association national teams men albania u-23 u-21 u-20 u-19 u-18 u-17 u-16 u-15 women albania u19 league competitions men level 1 kategoria superiore levels 2\u20134 kategoria e par\u00eb kategoria e dyt\u00eb kategoria e tret\u00eb women level 1 national championship youth level 1 kategoria superiore u-21 cup competitions men albanian cup albanian supercup independence cup supersport trophy women women 's cup awards albanian footballer of the year kategoria superiore talent of the season kategoria superiore fair play award kategoria superiore player of the month lists list of albania international footballers list of kategoria superiore all-time goalscorers list of kategoria superiore hat-tricks list of clubs list of venues ( by capacity ) foreign players transfers men 's clubs women 's clubs male players female players expatriate footballers managers referees venues records v t e futsal in albania futsal fshf league competitions premier league futsal cup national teams albania national futsal team list of clubs venues ( listed by capacity ) competitions trophys and awards records v t e albania national football team general albanian football association history managers captains venues home venues arena komb\u00ebtare elbasan arena loro bori\u00e7i stadium statistics all-time record records head to head european championship record results 1946\u201369 1970\u201399 2000\u201319 2020\u201329 matches albania v kosovo serbia v albania players all-time players other categories european championships 2016 other tournaments balkan cup 1946 1947 1948 malta international tournament 1998 2000 rivalries brotherly derby ( with kosovo ) other fshf teams men youth u23 u21 u20 u19 u18 u17 u16 u15 women senior u19 v t e national football associations of europe ( uefa ) current albania andorra armenia austria azerbaijan belarus belgium bosnia and herzegovina bulgaria croatia cyprus czech republic denmark england estonia faroe islands finland france georgia germany gibraltar greece hungary iceland israel italy kazakhstan kosovo latvia liechtenstein lithuania luxembourg malta moldova montenegro netherlands north macedonia northern ireland norway poland portugal republic of ireland romania russia san marino scotland serbia slovakia slovenia spain sweden switzerland turkey ukraine wales defunct east germany saarland serbia and montenegro soviet union yugoslavia v t e sports governing bodies in albania summer olympic sports aquatics diving swimming synchronized swimming water polo archery athletics badminton basketball boxing canoeing cycling equestrian fencing field hockey football golf gymnastics handball judo modern pentathlon rugby 7 's rowing sailing shooting practical shooting table tennis taekwondo tennis triathlon volleyball beach volleyball weightlifting wrestling winter olympic sports biathlon bobsleigh curling skating ( figure , speed & short track ) ice hockey luge skeleton skiing ( alpine , cross country , nordic combined , freestyle & jumping ) snowboarding other ioc recognised sports air sports auto racing bandy baseball billiard sports boules bowling bridge chess cricket dance sport floorball karate korfball lifesaving motorcycle racing mountaineering and climbing netball orienteering pelota vasca polo powerboating racquetball roller sports rugby softball sport climbing squash sumo surfing tug of war underwater sports water ski wushu paralympic sports others sports rugby league rugby union albanian national olympic committee categories : uefa member associations football in albania futsal in albania sports governing bodies in albania sports organizations established in 1930 1930 establishments in albania sport in tirana
    '''

    print(count_tokens(text))

