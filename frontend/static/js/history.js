document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const typeFilter = document.getElementById('type-filter');
    const riskFilter = document.getElementById('risk-filter');
    const historyTable = document.getElementById('history-table');
    const prevPageBtn = document.getElementById('prev-page');
    const nextPageBtn = document.getElementById('next-page');
    const currentPageSpan = document.getElementById('current-page');
    const startIndexSpan = document.getElementById('start-index');
    const endIndexSpan = document.getElementById('end-index');
    const totalCountSpan = document.getElementById('total-count');

    let allHistory = [];
    let filteredHistory = [];
    let currentPage = 1;
    const itemsPerPage = 10;

    // 获取历史数据
    async function fetchHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();

            if (data.status === 'success') {
                allHistory = data.data;
                filteredHistory = [...allHistory];
                updateTable();
            } else {
                showError('获取历史记录失败');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('网络错误，请稍后重试');
        }
    }

    // 更新表格显示
    function updateTable() {
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = Math.min(startIndex + itemsPerPage, filteredHistory.length);
        const pageData = filteredHistory.slice(startIndex, endIndex);

        // 更新分页信息
        startIndexSpan.textContent = filteredHistory.length ? startIndex + 1 : 0;
        endIndexSpan.textContent = endIndex;
        totalCountSpan.textContent = filteredHistory.length;
        currentPageSpan.textContent = currentPage;

        // 更新分页按钮状态
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = endIndex >= filteredHistory.length;

        // 清空表格
        historyTable.innerHTML = '';

        if (pageData.length === 0) {
            historyTable.innerHTML = `
                <tr class="text-center">
                    <td colspan="5">暂无数据</td>
                </tr>
            `;
            return;
        }

        // 填充数据
        pageData.forEach(item => {
            const row = document.createElement('tr');

            // 设置风险等级
            let riskLevel = '';
            let riskClass = '';
            if (item.confidence >= 0.9) {
                riskLevel = '高风险';
                riskClass = 'bg-danger';
            } else if (item.confidence >= 0.7) {
                riskLevel = '中风险';
                riskClass = 'bg-warning text-dark';
            } else {
                riskLevel = '低风险';
                riskClass = 'bg-success';
            }

            row.innerHTML = `
                <td>${item.timestamp}</td>
                <td>${item.text}</td>
                <td>${item.result}</td>
                <td>${Math.round(item.confidence * 100)}%</td>
                <td><span class="badge ${riskClass}">${riskLevel}</span></td>
            `;

            historyTable.appendChild(row);
        });
    }

    // 应用过滤器
    function applyFilters() {
        const searchTerm = searchInput.value.toLowerCase();
        const selectedType = typeFilter.value;
        const selectedRisk = riskFilter.value;

        filteredHistory = allHistory.filter(item => {
            // 文本搜索
            const matchesSearch = item.text.toLowerCase().includes(searchTerm);

            // 类型过滤
            const matchesType = !selectedType || item.result === selectedType;

            // 风险等级过滤
            let matchesRisk = true;
            if (selectedRisk) {
                const confidence = item.confidence;
                switch (selectedRisk) {
                    case 'high':
                        matchesRisk = confidence >= 0.9;
                        break;
                    case 'medium':
                        matchesRisk = confidence >= 0.7 && confidence < 0.9;
                        break;
                    case 'low':
                        matchesRisk = confidence < 0.7;
                        break;
                }
            }

            return matchesSearch && matchesType && matchesRisk;
        });

        currentPage = 1;
        updateTable();
    }

    // 显示错误信息
    function showError(message) {
        historyTable.innerHTML = `
            <tr class="text-center">
                <td colspan="5">
                    <div class="text-danger">
                        <i class='bx bx-error-circle'></i>
                        ${message}
                    </div>
                </td>
            </tr>
        `;
    }

    // 事件监听
    searchInput.addEventListener('input', applyFilters);
    typeFilter.addEventListener('change', applyFilters);
    riskFilter.addEventListener('change', applyFilters);

    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            updateTable();
        }
    });

    nextPageBtn.addEventListener('click', () => {
        const maxPage = Math.ceil(filteredHistory.length / itemsPerPage);
        if (currentPage < maxPage) {
            currentPage++;
            updateTable();
        }
    });

    // 初始化加载
    fetchHistory();
});