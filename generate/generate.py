import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _get_model_path(model_id=0):
    model_name = ""
    if(model_id == 0):
        model_name = "Llama-3.1-8B-Instruct"
    elif(model_id == 1):
        model_name = "Qwen2.5-7B-Instruct"
    elif(model_id == 2):
        model_name = "Mistral-7B-Instruct-v0.3"

    print("current_model: ", model_name)
    return f"/home/models/{model_name}"


# model_id 0-llama3.1; 1-qwen2.5; 2-mistral
def generate(prompt, temperature=1e-6, model_id=0):
    # 选择模型路径 0-llama3.1; 1-qwen2.5; 2-mistral
    model_path = _get_model_path(model_id)

    # 载入模型
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
    # 使用分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 设置填充token ID
    tokenizer.pad_token_id = tokenizer.eos_token_id

    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": prompt}
    ]

    # 自动填充好prompt模板，确保自然停止
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
 
    # 生成输出
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id = tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)
    del model
    return response
