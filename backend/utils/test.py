# 使用LLM降噪
import time
import openai
import pandas as pd
from openai import OpenAI
import json


# 配置信息
model_name = "deepseek-chat"
base_url="https://api.deepseek.com"
api_key="sk-56b66cad3490428dab565396bb33a0bd"

def create_prompt(text):
    return {
        "role": "system",
        "content": "你是一个反欺诈专家，请判断消息是否为欺诈信息并修正标签。要求仅输出'欺诈'或'非欺诈'"
    }, {
        "role": "user",
        "content": text
    }
def clean(text):
    prompt=create_prompt(text)
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        stream=False
    )
    print("模型输入:"+json.dumps(prompt,ensure_ascii=False,indent=4))
    print("模型输出:"+response.choices[0].message.content)

if __name__ == "__main__":
    client = OpenAI(api_key=api_key, base_url=base_url)
    clean("您好，这里是上海市松江区公安局，我们发现您的招商卡涉及犯罪，需要进行资金清查。我们需要您配合我们的调查，并下载一个“Zom”APP，以便我们远程操控您的手机进行相关操作")