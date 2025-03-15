# 使用LLM降噪
import time
import openai
import pandas as pd
from openai import OpenAI

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

def clean(number):
    data = pd.read_csv(f"../raw_data/labels.csv")
    contents = data["content"].tolist()

    for text in contents:
        prompt=create_prompt(text)

        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            stream=False
        )
        clean_labels.append(response.choices[0].message.content)
        time.sleep(0.5)

    data['label']=cleaned_labels
    data.to_csv(f"label0{number}-cleaned",index=false)

if __name__ == "__main__":
    client = OpenAI(api_key=api_key, base_url=base_url)
