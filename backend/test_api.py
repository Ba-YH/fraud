from openai import OpenAI

def get_result(text):
    response = client.chat.completions.create(
        model="model",
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
        max_tokens=50
    )
    print("文本："+text+"\n预测结果为："+response.choices[0].message.content)

if __name__=="__main__":
    client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:5000/v1")
    get_result("这里是华南农业大学泰山区快递站")