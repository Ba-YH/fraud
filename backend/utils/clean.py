# 使用LLM降噪
import time
import openai
import pandas as pd
from openai import OpenAI

# LLM 配置信息
model_name = "deepseek-reasoner"
base_url="https://api.deepseek.com"
api_key="sk-56b66cad3490428dab565396bb33a0bd"

# 数据文件信息
source_data_path="../dataset/label01234-raw.csv"
target_data_path="../dataset/label01234-clean.csv"


def create_prompt(text):
    system_prompt = """
**欺诈信息分类指令**

请对以下文本进行分类，判断其属于以下五类中的哪一类：

1. "非欺诈信息"
   - 特征：不涉及任何欺诈行为，属于正常交流或合法商业活动

2. "冒充公检法及政府机关类"
   - 特征：假装司法机关/政府部门人员，要求配合调查、缴纳保证金或提供敏感信息
   - 示例："您涉嫌洗钱犯罪，请立即转账至安全账户配合调查"

3. "贷款、代办信用卡类"
   - 特征：以低门槛贷款/信用卡办理为诱饵，要求预付费用或窃取个人信息
   - 示例："无抵押贷款，当天放款，仅需提供验证码"

4. "冒充客服服务"
   - 特征：伪装电商平台/金融机构客服，以退款、账户异常等理由实施诈骗
   - 示例："您购买的商品有质量问题，点击链接办理三倍赔偿"

5. "冒充领导、熟人类"
   - 特征：仿冒上级/亲友身份，要求紧急转账或提供敏感信息
   - 示例："我是张总，正在开会，急需转账8万元到这个账户"

**输出要求**：
1. 仅输出对应标签编号或名称（如"非欺诈信息"或"冒充客服服务"）
2. 不需要任何解释或附加内容
"""
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": text}
    ]

def clean():
    # cnt=0
    clean_labels=[]
    data = pd.read_csv(source_data_path)
    contents = data["content"].tolist()

    for text in contents:
        prompt=create_prompt(text)

        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            stream=False
        )
        response=response.choices[0].message.content;
        clean_labels.append(response)
        time.sleep(0.5)

        # print("待检测文本："+text)
        # print("LLM 预测标签："+response)
        # line=""
        # for i in range(1, 100):
        #     line+="- "
        # print(line+"\n")

    data['label']=cleaned_labels
    data.to_csv(target_data_path,index=false)

if __name__ == "__main__":
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    clean()