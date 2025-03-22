document.addEventListener('DOMContentLoaded', function() {
    // 获取所有图表的 canvas 元素
    const typeDistributionChart = document.getElementById('type-distribution-chart');
    const dailyTrendChart = document.getElementById('daily-trend-chart');
    const riskDistributionChart = document.getElementById('risk-distribution-chart');
    const confidenceDistributionChart = document.getElementById('confidence-distribution-chart');

    // 初始化图表
    let charts = {};

    // 获取统计数据并更新图表
    async function fetchStatistics() {
        try {
            const response = await fetch('/api/statistics');
            if (!response.ok) {
                throw new Error('获取统计数据失败');
            }
            const data = await response.json();
            updateStatistics(data);
        } catch (error) {
            console.error('获取统计数据时出错:', error);
            showError();
        }
    }

    // 更新统计数据和图表
    function updateStatistics(data) {
        // 更新总览卡片
        document.getElementById('total-detections').textContent = data.total_detections;
        document.getElementById('fraud-ratio').textContent = (data.fraud_ratio * 100).toFixed(1);
        document.getElementById('high-risk-count').textContent = data.high_risk_count;

        // 更新欺诈类型分布图表
        updateTypeDistributionChart(data.type_distribution);

        // 更新每日检测趋势图表
        updateDailyTrendChart(data.daily_trend);

        // 更新风险等级分布图表
        updateRiskDistributionChart(data.risk_distribution);

        // 更新置信度分布图表
        updateConfidenceDistributionChart(data.confidence_distribution);
    }

    // 更新欺诈类型分布图表
    function updateTypeDistributionChart(data) {
        if (charts.typeDistribution) {
            charts.typeDistribution.destroy();
        }

        charts.typeDistribution = new Chart(typeDistributionChart, {
            type: 'pie',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    data: Object.values(data),
                    backgroundColor: [
                        '#FF6384',
                        '#36A2EB',
                        '#FFCE56',
                        '#4BC0C0',
                        '#9966FF'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    title: {
                        display: false
                    }
                }
            }
        });
    }

    // 更新每日检测趋势图表
    function updateDailyTrendChart(data) {
        if (charts.dailyTrend) {
            charts.dailyTrend.destroy();
        }

        charts.dailyTrend = new Chart(dailyTrendChart, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: '检测总数',
                    data: data.total_counts,
                    borderColor: '#36A2EB',
                    fill: false
                }, {
                    label: '欺诈数量',
                    data: data.fraud_counts,
                    borderColor: '#FF6384',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // 更新风险等级分布图表
    function updateRiskDistributionChart(data) {
        if (charts.riskDistribution) {
            charts.riskDistribution.destroy();
        }

        charts.riskDistribution = new Chart(riskDistributionChart, {
            type: 'doughnut',
            data: {
                labels: ['高风险', '中风险', '低风险'],
                datasets: [{
                    data: [data.high, data.medium, data.low],
                    backgroundColor: [
                        '#FF6384',
                        '#FFCE56',
                        '#4BC0C0'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
    }

    // 更新置信度分布图表
    function updateConfidenceDistributionChart(data) {
        if (charts.confidenceDistribution) {
            charts.confidenceDistribution.destroy();
        }

        charts.confidenceDistribution = new Chart(confidenceDistributionChart, {
            type: 'bar',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    label: '检测数量',
                    data: Object.values(data),
                    backgroundColor: '#36A2EB'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // 显示错误信息
    function showError() {
        const cards = document.querySelectorAll('.card');
        cards.forEach(card => {
            card.innerHTML = `
                <div class="card-body text-center text-danger">
                    <i class='bx bx-error-circle'></i>
                    <p>获取数据失败，请稍后重试</p>
                </div>
            `;
        });
    }

    // 初始化加载数据
    fetchStatistics();

    // 每5分钟自动刷新一次数据
    setInterval(fetchStatistics, 5 * 60 * 1000);
});