import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime, timedelta
import json
import dill
import numpy as np
from pathlib import Path
from backend.utils.process import *

app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)
DEEPSEEK_API_URL = "https://api.deepseek.com"
# 模型相关
MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILENAME = "best_model.dill"
best_model = None
stop_words=[]

# 数据存储
HISTORY_FILE = Path(__file__).parent / "data" / "history.json"
HISTORY_FILE.parent.mkdir(exist_ok=True)

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
    print(model_path)
    try:
        with open(model_path, "rb") as f:
            best_model = dill.load(f)
        app.logger.info("模型加载成功")
    except Exception as e:
        app.logger.error(f"模型加载失败: {str(e)}")
        raise


def save_detection_history(text, result, confidence):
    """保存检测历史"""
    try:
        history = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)

        history.append({
            'id': len(history) + 1,
            'text': text,
            'result': result,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # 只保留最近1000条记录
        history = history[-1000:]

        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        app.logger.error(f"保存历史记录失败: {str(e)}")

def delete_detection_history(record_id):
    """删除检测历史记录"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)

            # 过滤掉要删除的记录
            updated_history = [record for record in history if record['id'] != record_id]

            # 将更新后的历史记录写回文件
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(updated_history, f, ensure_ascii=False, indent=2)

            return True  # 删除成功
        else:
            app.logger.error("历史记录文件不存在")
            return False  # 文件不存在
    except Exception as e:
        app.logger.error(f"删除历史记录失败: {str(e)}")
        return False  # 删除失败
@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/batch')
def batch_detect():
    """批量检测页面"""
    return render_template('batch.html')


@app.route('/history')
def history():
    """历史记录页面"""
    return render_template('history.html')


@app.route('/statistics')
def statistics_page():
    return render_template('statistics.html')


@app.route('/api/history')
def get_history():
    """获取历史记录API"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return jsonify({'status': 'success', 'data': history})
        return jsonify({'status': 'success', 'data': []})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/delete_record/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    """删除历史记录API"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)

            # 根据 ID 删除记录
            history = [record for record in history if record['id'] != record_id]

            # 将更新后的历史记录写回文件
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False)

            return '', 204  # 返回204 No Content
        return jsonify({'status': 'error', 'message': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500



@app.route('/api/statistics')
def get_statistics():
    try:
        # 读取历史记录数据
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)

        # 如果历史记录为空，返回默认数据
        if not history:
            return jsonify({
                'total_detections': 0,
                'fraud_ratio': 0,
                'high_risk_count': 0,
                'type_distribution': {},
                'daily_trend': {'dates': [], 'total_counts': [], 'fraud_counts': []},
                'risk_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'confidence_distribution': {}
            })

        # 计算总检测次数和欺诈比例
        total_detections = len(history)
        fraud_count = sum(1 for record in history if record['result'] != '非欺诈信息')
        fraud_ratio = fraud_count / total_detections if total_detections > 0 else 0

        # 计算高风险信息数量
        high_risk_count = sum(1 for record in history if record['confidence'] >= 0.8)

        # 统计欺诈类型分布
        type_distribution = {}
        for record in history:
            result = record['result']
            type_distribution[result] = type_distribution.get(result, 0) + 1

        # 统计每日检测趋势（最近7天）
        daily_trend = {'dates': [], 'total_counts': [], 'fraud_counts': []}
        today = datetime.now().date()
        for i in range(6, -1, -1):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            daily_records = [r for r in history if r['timestamp'].split()[0] == date_str]
            daily_trend['dates'].append(date_str)
            daily_trend['total_counts'].append(len(daily_records))
            daily_trend['fraud_counts'].append(
                sum(1 for r in daily_records if r['result'] != '非欺诈信息')
            )

        # 统计风险等级分布
        risk_distribution = {
            'high': sum(1 for r in history if r['confidence'] >= 0.8),
            'medium': sum(1 for r in history if 0.5 <= r['confidence'] < 0.8),
            'low': sum(1 for r in history if r['confidence'] < 0.5)
        }

        # 统计置信度分布（按0.1区间划分）
        confidence_distribution = {}
        for i in range(0, 10):
            lower = i * 0.1
            upper = (i + 1) * 0.1
            label = f'{lower:.1f}-{upper:.1f}'
            count = sum(1 for r in history if lower <= r['confidence'] < upper)
            confidence_distribution[label] = count
        # 处理1.0的情况
        confidence_distribution['1.0'] = sum(1 for r in history if r['confidence'] == 1.0)

        return jsonify({
            'total_detections': total_detections,
            'fraud_ratio': fraud_ratio,
            'high_risk_count': high_risk_count,
            'type_distribution': type_distribution,
            'daily_trend': daily_trend,
            'risk_distribution': risk_distribution,
            'confidence_distribution': confidence_distribution
        })

    except Exception as e:
        app.logger.error(f'获取统计数据时出错: {str(e)}')
        return jsonify({'error': '获取统计数据失败'}), 500


@app.route('/process', methods=['POST'])
def process():
    """处理单条文本检测"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({
                'status': 'error',
                'message': '输入文本不能为空'
            })

        # 文本预处理
        processed_text = process_text(preprocess_text(text))

        # 模型预测
        prediction = best_model.predict_proba(processed_text.reshape(1, -1))
        label_idx = prediction.argmax()
        confidence = float(prediction.max())

        result = LABEL_MAPPING[label_idx]

        # 保存历史记录
        save_detection_history(text, result, confidence)

        return jsonify({
            'status': 'success',
            'result': result,
            'confidence': confidence
        })
    except Exception as e:
        app.logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': '内部服务器错误'
        })


@app.route('/batch_process', methods=['POST'])
def batch_process():
    """处理批量文本检测"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '没有上传文件'
            })

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': '没有选择文件'
            })

        if not file.filename.endswith('.txt'):
            return jsonify({
                'status': 'error',
                'message': '只支持txt文件'
            })

        # 读取文件内容
        content = file.read().decode('utf-8')
        texts = [text.strip() for text in content.split('\n') if text.strip()]

        results = []
        for text in texts:
            processed_text = process_text(preprocess_text(text))
            label,pro= best_model.predict(processed_text.reshape(1, -1))
            label_idx = label
            confidence = float(pro)
            result = LABEL_MAPPING[label_idx]

            results.append({
                'text': text,
                'result': result,
                'confidence': confidence
            })

            # 保存历史记录
            save_detection_history(text, result, confidence)

        return jsonify({
            'status': 'success',
            'data': results
        })
    except Exception as e:
        app.logger.error(f"批量处理请求时发生错误: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': '内部服务器错误'
        })


def preprocess_text(text):
    global stop_words
    with open("utils/stop_words.txt", "r", encoding="UTF-8") as f:
        stop_words = set(word.strip() for word in f.readlines())

    words = jieba.lcut(text)
    return [word for word in words if word not in stop_words]

# @app.route('/api/deepseek/generate', methods=['POST'])
# def deepseek_generate():


if __name__ == '__main__':
    try:
        # 确保必要的目录存在
        MODEL_DIR.mkdir(exist_ok=True)
        HISTORY_FILE.parent.mkdir(exist_ok=True)

        # 加载模型
        load_model()

        # 启动应用
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        app.logger.error(f"应用启动失败: {str(e)}")
        raise