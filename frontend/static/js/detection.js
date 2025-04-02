document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('input-text');
    const charCount = document.querySelector('.char-count');
    const detectBtn = document.getElementById('detect-btn');
    const resultSection = document.getElementById('result-section');

    // 字数统计
    textarea.addEventListener('input', function() {
        const length = this.value.length;
        charCount.textContent = `${length}/500`;

        if (length > 500) {
            this.value = this.value.substring(0, 500);
            charCount.textContent = '500/500';
        }
    });

    // 检测按钮点击事件
    detectBtn.addEventListener('click', async function() {
        const text = textarea.value.trim();

        if (!text) {
            showToast('请输入需要检测的文本内容');
            return;
        }

        // 显示加载状态
        detectBtn.disabled = true;
        detectBtn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> 检测中...';

        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (data.status === 'success') {
                showResult(data);
            } else {
                showToast(data.message || '检测失败，请稍后重试');
            }
        } catch (error) {
            console.error('Error:', error);
            showToast('网络错误，请稍后重试');
        } finally {
            // 恢复按钮状态
            detectBtn.disabled = false;
            detectBtn.innerHTML = '<i class="bx bx-search-alt"></i> 开始检测';
        }
    });

//document.getElementById('deepseek-generate-btn').addEventListener('click', function() {
//    const inputText = document.getElementById('input-text').value;
//
//    if (inputText.trim() === "") {
//        alert("请输入需要生成欺诈文本的基础内容！");
//        return;
//    }
//
//    // 发送请求到 DeepSeek 生成欺诈文本的接口
//    fetch('/api/deepseek/generate', {
//        method: 'POST',
//        headers: {
//            'Content-Type': 'application/json'
//        },
//        body: JSON.stringify({ text: inputText })
//    })
//    .then(response => {
//        if (!response.ok) {
//            throw new Error('网络响应不是 OK');
//        }
//        return response.json();
//    })
//    .then(data => {
//        // 更新文本框以显示生成的欺诈文本
//        document.getElementById('input-text').value = data.generated_text || "未生成文本";
//    })
//    .catch(error => {
//        console.error('发生错误:', error);
//        alert('生成欺诈文本失败，请稍后再试。');
//    });
//});


    // 显示检测结果
    function showResult(data) {
        const fraudType = document.getElementById('fraud-type');
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceValue = document.getElementById('confidence-value');
        const riskLevel = document.getElementById('risk-level').querySelector('.badge');
        const suggestionText = document.getElementById('suggestion-text');

        // 设置欺诈类型
        fraudType.textContent = data.result;

        // 设置置信度（假设API返回的置信度在0-1之间）
        const confidence = Math.round(data.confidence * 100) || 85; // 如果API没有返回置信度，使用默认值
        confidenceBar.style.width = `${confidence}%`;
        confidenceValue.textContent = `${confidence}%`;

        // 设置风险等级
        let riskClass = '';
        let riskText = '';
        if (data.result != '非欺诈信息') {
            if (confidence >= 70) {
                riskClass = 'bg-danger';
                riskText = '高风险';
            } else {
                riskClass = 'bg-warning text-dark';
                riskText = '中风险';
            }
        } else {
            riskClass = 'bg-success';
            riskText = '低风险';
        }

        riskLevel.className = `badge ${riskClass}`;
        riskLevel.textContent = riskText;

        // 设置建议文本
        const suggestions = {
            '非欺诈信息': '该信息未发现明显的欺诈特征，可以正常处理。',
            '冒充公检法及政府机关类': '请注意：这可能是诈骗分子冒充公检法或政府机关的欺诈信息。建议：\n1. 不要相信对方的身份声明\n2. 不要点击任何链接或下载文件\n3. 及时向当地警方报案',
            '贷款、代办信用卡类': '这可能是非法贷款或信用卡诈骗。建议：\n1. 不要相信无抵押、低息贷款的承诺\n2. 不要提供个人银行信息\n3. 通过正规银行办理相关业务',
            '冒充客服服务': '这可能是假冒客服的诈骗信息。建议：\n1. 不要相信主动联系的"客服"\n2. 通过官方渠道验证对方身份\n3. 不要向对方转账或提供验证码',
            '冒充领导、熟人类': '这可能是冒充熟人的诈骗信息。建议：\n1. 通过其他渠道核实对方身份\n2. 不要轻信转账要求\n3. 提高警惕，保护个人信息'
        };

        suggestionText.textContent = suggestions[data.result] || '建议提高警惕，谨防上当受骗。';

        // 显示结果区域
        resultSection.style.display = 'block';

        // 平滑滚动到结果区域
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    // 显示提示消息
    function showToast(message) {
        // 如果已经存在toast，先移除
        const existingToast = document.querySelector('.toast-container');
        if (existingToast) {
            existingToast.remove();
        }

        // 创建新的toast
        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';

        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;

        toastContainer.appendChild(toast);
        document.body.appendChild(toastContainer);

        // 3秒后自动移除
        setTimeout(() => {
            toastContainer.remove();
        }, 3000);
    }
});