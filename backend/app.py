from flask import Flask, render_template, request, jsonify
import os,sys
from path import Path
import dill
from backend.utils.process import *

app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)

best_model = None
MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILENAME = "best_model.dill"

LABEL_MAPPING = {
    0: "非欺诈信息",
    1: "冒充公检法及政府机关类",
    2: "贷款、代办信用卡类",
    3: "冒充客服服务",
    4: "冒充领导、熟人类"
}

# 模型加载
def load_model() -> None:
    global best_model
    model_path = MODEL_DIR / MODEL_FILENAME
    try:
        with open(model_path, "rb") as f:
            best_model = dill.load(f)
        app.logger.info("模型加载成功")
    except Exception as e:
        app.logger.error(f"模型加载失败: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')


# 处理接口
@app.route("/process", methods=["POST"])
def get_result():
    try:
        input_text = request.json.get("text", "").strip()
        if not input_text:
            return jsonify({
                "status": "error",
                "message": "输入文本不能为空"
            })

        # 在线处理 -> 模型输入 -> 获取预测结果
        processed_text = process_text(input_text)

        print(processed_text)
        prediction = best_model.predict(processed_text.reshape(1, -1))
        label = prediction[0]

        return jsonify({
            "status": "success",
            "result": LABEL_MAPPING.get(label[0], "未知分类")
        })

    except Exception as e:
        app.logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "内部服务器错误"
        })

if __name__ == '__main__':
    try:
        load_model()
        app.run(debug=False)
    except Exception as e:
        app.logger.error(f"应用启动失败: {str(e)}")
        sys.exit(1)