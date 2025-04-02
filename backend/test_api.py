from openai import OpenAI

client = OpenAI(
    api_key="sk-5aB3cD9eF2gH5jK8mNpQ4rS7tU2vW9xY3zA6kL",
    base_url="http://42.194.147.11:5000/v1"
)
def get_result(text):
    try:
        response = client.chat.completions.create(
            model="model",
            messages=[{"role": "user", "content": text}],
            max_tokens=50
        )
        content = response.choices[0].message.content
        print(content)
    except Exception as e:
        print(f"请求失败：{str(e)}")

if __name__=="__main__":
    text = input("请输入待检测文本：")
    get_result(text)