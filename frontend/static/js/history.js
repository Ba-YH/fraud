document.addEventListener('DOMContentLoaded', function() {
    const historyTable = document.getElementById('history-table');
    const searchInput = document.getElementById('search-input');
    const typeFilter = document.getElementById('type-filter');
    const riskFilter = document.getElementById('risk-filter');
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

    // 初始化加载历史记录
    fetchHistory();

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
            console.error('错误:', error);
            showError('网络错误，请稍后重试');
        }
    }

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
                    <td colspan="7">暂无数据</td>
                </tr>
            `;
            return;
        }

        // 填充数据
        pageData.forEach(item => {
            const row = document.createElement('tr');
            const riskLevel = getRiskLevel(item.confidence,item.result);
            row.innerHTML = `
                <td>${item.id}</td>
                <td>${item.timestamp}</td>
                <td>${item.text}</td>
                <td>${item.result}</td>
                <td>${Math.round(item.confidence * 100)}%</td>
                <td><span class="badge ${riskLevel.class}">${riskLevel.label}</span></td>
                <td>
                    <button class="btn btn-danger btn-sm delete-record" data-id="${item.id}">删除</button>
                </td>
            `;
            historyTable.appendChild(row);
        });

        bindDeleteButtons();
    }

    function getRiskLevel(confidence,result) {

        if (result != '非欺诈信息') {
            if (confidence >= 0.7) {
                return { label: '高风险', class: 'bg-danger' };
            } else {
                return { label: '中风险', class: 'bg-warning text-dark' };
            }
        } else {
            return { label: '低风险', class: 'bg-success' };
        }
    }

    function bindDeleteButtons() {
        const deleteButtons = document.querySelectorAll('.delete-record');
        deleteButtons.forEach(button => {
            button.addEventListener('click', function() {
                const recordId = this.getAttribute('data-id');
                deleteRecord(recordId);
            });
        });
    }

    async function deleteRecord(recordId) {
        // 使用 SweetAlert2 弹出确认框
        const { isConfirmed } = await Swal.fire({
            title: '确认删除',
            text: '您确定要删除这条记录吗？',
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: '删除',
            cancelButtonText: '取消'
        });

        if (isConfirmed) {
            try {
                const response = await fetch(`/delete_record/${recordId}`, { method: 'DELETE' });
                if (response.ok) {
                    Swal.fire('已删除!', '记录已删除', 'success');
                    fetchHistory(); // 重新加载数据
                } else {
                    Swal.fire('删除失败', '请重试', 'error');
                }
            } catch (error) {
                console.error('错误:', error);
                Swal.fire('删除失败', '请重试', 'error');
            }
        } else {
            Swal.fire('已取消', '删除操作已取消', 'info');
        }
    }



    function applyFilters() {
        const searchTerm = searchInput.value.toLowerCase();
        const selectedType = typeFilter.value;
        const selectedRisk = riskFilter.value;

        filteredHistory = allHistory.filter(item => {
            const matchesSearch = item.text.toLowerCase().includes(searchTerm);
            const matchesType = !selectedType || item.result === selectedType;
            const matchesRisk = applyRiskFilter(item.confidence, selectedRisk, item.result);
            return matchesSearch && matchesType && matchesRisk;
        });

        currentPage = 1;
        updateTable();
    }

    function applyRiskFilter(confidence, selectedRisk, result) {
        if (!selectedRisk) return true;

        // 先判断是否为非欺诈信息
        if (result === '非欺诈信息') {
            return selectedRisk === 'low';
        }

        // 欺诈信息的风险等级判断
        switch (selectedRisk) {
            case 'high':
                return confidence >= 0.7;
            case 'medium':
                return confidence < 0.7;
            case 'low':
                return false; // 欺诈信息不会有低风险
            default:
                return true;
        }
    }

    function showError(message) {
        historyTable.innerHTML = `
            <tr class="text-center">
                <td colspan="7">
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
});
