document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const selectedFile = document.querySelector('.selected-file');
    const filename = selectedFile.querySelector('.filename');
    const removeFile = selectedFile.querySelector('.remove-file');
    const uploadBtn = document.getElementById('upload-btn');
    const resultsSection = document.getElementById('results-section');
    const resultsTable = document.getElementById('results-table');
    const exportBtn = document.getElementById('export-btn');

    let currentFile = null;

    // 文件拖放处理
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-color)';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border-color)';

        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    // 点击上传
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
    });

    // 移除文件
    removeFile.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        selectedFile.style.display = 'none';
        uploadBtn.disabled = true;
    });

    // 处理文件
    function handleFile(file) {
        if (!file) return;

        if (!file.name.endsWith('.txt')) {
            showToast('只支持 .txt 文件');
            return;
        }

        currentFile = file;
        filename.textContent = file.name;
        selectedFile.style.display = 'flex';
        uploadBtn.disabled = false;
    }

    // 上传并检测
    uploadBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        const formData = new FormData();
        formData.append('file', currentFile);

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> 检测中...';

        try {
            const response = await fetch('/batch_process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                showResults(data.data);
            } else {
                showToast(data.message || '检测失败');
            }
        } catch (error) {
            console.error('Error:', error);
            showToast('网络错误，请稍后重试');
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="bx bx-upload"></i> 开始检测';
        }
    });

    // 显示结果
    function showResults(results) {
        // 清空表格
        resultsTable.innerHTML = '';

        // 计算统计数据
        const total = results.length;
        const fraudCount = results.filter(r => r.result !== '非欺诈信息').length;
        const fraudRatio = ((fraudCount / total) * 100).toFixed(1);

        // 更新统计显示
        document.getElementById('total-count').textContent = total;
        document.getElementById('fraud-count').textContent = fraudCount;
        document.getElementById('fraud-ratio').textContent = `${fraudRatio}%`;

        // 填充表格
        results.forEach((result, index) => {
            const row = document.createElement('tr');

            // 设置风险等级
            let riskLevel = '';
            let riskClass = '';
            if (result.confidence >= 0.9) {
                riskLevel = '高风险';
                riskClass = 'bg-danger';
            } else if (result.confidence >= 0.7) {
                riskLevel = '中风险';
                riskClass = 'bg-warning text-dark';
            } else {
                riskLevel = '低风险';
                riskClass = 'bg-success';
            }

            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${result.text}</td>
                <td>${result.result}</td>
                <td>${Math.round(result.confidence * 100)}%</td>
                <td><span class="badge ${riskClass}">${riskLevel}</span></td>
            `;

            resultsTable.appendChild(row);
        });

        // 显示结果区域
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // 导出结果
    exportBtn.addEventListener('click', () => {
        const rows = Array.from(resultsTable.querySelectorAll('tr'));
        let csv = '序号,文本内容,检测结果,置信度,风险等级\n';

        rows.forEach(row => {
            const cells = Array.from(row.querySelectorAll('td'));
            const rowData = cells.map(cell => {
                let text = cell.textContent;
                // 如果单元格内容包含逗号，用引号包裹
                if (text.includes(',')) {
                    text = `"${text}"`;
                }
                return text;
            });
            csv += rowData.join(',') + '\n';
        });

        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);

        link.setAttribute('href', url);
        link.setAttribute('download', `检测结果_${new Date().toISOString().split('T')[0]}.csv`);
        link.style.display = 'none';

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});