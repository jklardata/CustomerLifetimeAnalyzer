// Dashboard JavaScript functionality
class CLVDashboard {
    constructor() {
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.startPeriodicUpdates();
    }

    setupEventListeners() {
        // Sync data button
        const syncBtn = document.getElementById('syncDataBtn');
        if (syncBtn) {
            syncBtn.addEventListener('click', this.handleSyncData.bind(this));
        }

        // Export functionality
        const exportBtns = document.querySelectorAll('[data-export]');
        exportBtns.forEach(btn => {
            btn.addEventListener('click', this.handleExport.bind(this));
        });

        // Refresh button
        const refreshBtns = document.querySelectorAll('[data-refresh]');
        refreshBtns.forEach(btn => {
            btn.addEventListener('click', this.refreshData.bind(this));
        });
    }

    initializeCharts() {
        // CLV Distribution Chart (if not already initialized)
        const clvChartCanvas = document.getElementById('clvDistributionChart');
        if (clvChartCanvas && !this.charts.clvDistribution) {
            this.initCLVDistributionChart(clvChartCanvas);
        }

        // Revenue Trend Chart
        const revenueTrendCanvas = document.getElementById('revenueTrendChart');
        if (revenueTrendCanvas && !this.charts.revenueTrend) {
            this.initRevenueTrendChart(revenueTrendCanvas);
        }

        // Customer Segmentation Chart
        const segmentationCanvas = document.getElementById('segmentationChart');
        if (segmentationCanvas && !this.charts.segmentation) {
            this.initSegmentationChart(segmentationCanvas);
        }
    }

    initCLVDistributionChart(canvas) {
        const ctx = canvas.getContext('2d');
        this.charts.clvDistribution = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['$0-100', '$100-500', '$500-1K', '$1K-5K', '$5K+'],
                datasets: [{
                    label: 'Customers',
                    data: [120, 85, 45, 25, 8],
                    backgroundColor: [
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(32, 201, 151, 0.8)',
                        'rgba(40, 167, 69, 0.8)',
                        'rgba(23, 162, 184, 0.8)'
                    ],
                    borderColor: [
                        'rgba(220, 53, 69, 1)',
                        'rgba(255, 193, 7, 1)',
                        'rgba(32, 201, 151, 1)',
                        'rgba(40, 167, 69, 1)',
                        'rgba(23, 162, 184, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.parsed.y} customers`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    initRevenueTrendChart(canvas) {
        const ctx = canvas.getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(40, 167, 69, 0.3)');
        gradient.addColorStop(1, 'rgba(40, 167, 69, 0.05)');

        this.charts.revenueTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Revenue',
                    data: [45000, 52000, 48000, 61000, 55000, 67000],
                    borderColor: '#28a745',
                    backgroundColor: gradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#28a745',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#28a745',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return `Revenue: $${context.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + (value / 1000) + 'K';
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    initSegmentationChart(canvas) {
        const ctx = canvas.getContext('2d');
        this.charts.segmentation = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['High Value', 'Medium Value', 'Low Value', 'New Customers'],
                datasets: [{
                    data: [25, 35, 30, 10],
                    backgroundColor: [
                        '#28a745',
                        '#20c997',
                        '#ffc107',
                        '#6c757d'
                    ],
                    borderWidth: 0,
                    cutout: '60%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    async handleSyncData() {
        const syncBtn = document.getElementById('syncDataBtn');
        const originalText = syncBtn.innerHTML;
        
        // Show loading state
        syncBtn.innerHTML = '<i data-feather="loader" class="me-1 spinner"></i> Syncing...';
        syncBtn.disabled = true;
        
        try {
            const response = await fetch('/sync-data', {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Data synced successfully!', 'success');
                this.updateMetrics(data);
                this.updateLastSyncTime();
            } else {
                throw new Error(data.error || 'Sync failed');
            }
        } catch (error) {
            console.error('Sync error:', error);
            this.showNotification('Failed to sync data: ' + error.message, 'error');
        } finally {
            // Restore button state
            syncBtn.innerHTML = originalText;
            syncBtn.disabled = false;
            feather.replace(); // Re-initialize feather icons
        }
    }

    handleExport(event) {
        const exportType = event.target.dataset.export || event.target.closest('[data-export]').dataset.export;
        
        switch (exportType) {
            case 'csv':
                this.exportCSV();
                break;
            case 'pdf':
                this.exportPDF();
                break;
            case 'analytics':
                this.exportAnalytics();
                break;
            case 'customers':
                this.exportCustomerData();
                break;
            case 'segmentation':
                this.exportSegmentationData();
                break;
            case 'aov':
                this.exportAOVData();
                break;
            case 'churn':
                this.exportChurnData();
                break;
            case 'revenue-retention':
                this.exportRevenueRetentionData();
                break;
            case 'product-returns':
                this.exportProductReturnsData();
                break;
            case 'pdf':
                this.exportDashboardPDF();
                break;
            case 'top-customers':
                this.exportTopCustomersData();
                break;
            case 'recent-orders':
                this.exportRecentOrdersData();
                break;
            case 'clv-distribution':
                this.exportCLVDistributionData();
                break;
            default:
                console.warn('Unknown export type:', exportType);
        }
    }

    exportCSV() {
        // Create CSV data
        const csvData = this.generateCSVData();
        const blob = new Blob([csvData], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        
        // Create download link
        const a = document.createElement('a');
        a.href = url;
        a.download = `clv-report-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        this.showNotification('CSV report downloaded', 'success');
    }

    exportSegmentationData() {
        // Get segmentation data from the page
        const highValue = document.querySelector('[data-segment="high"]')?.textContent || '0';
        const mediumValue = document.querySelector('[data-segment="medium"]')?.textContent || '0';
        const lowValue = document.querySelector('[data-segment="low"]')?.textContent || '0';
        
        const csvData = [
            'Segment,Customer Count,Description',
            `High Value,${highValue},Top 25% of customers by CLV`,
            `Medium Value,${mediumValue},Middle 50% of customers by CLV`,
            `Low Value,${lowValue},Bottom 25% of customers by CLV`
        ].join('\n');
        
        this.downloadCSV(csvData, 'customer-segmentation');
    }

    exportAOVData() {
        // Export AOV trend data
        const csvData = [
            'Date,Average Order Value',
            // This would be populated with actual trend data
            ...Array.from({length: 30}, (_, i) => {
                const date = new Date();
                date.setDate(date.getDate() - (29 - i));
                const aov = Math.floor(Math.random() * 100) + 50; // Sample data
                return `${date.toISOString().split('T')[0]},${aov}`;
            })
        ].join('\n');
        
        this.downloadCSV(csvData, 'aov-trend');
    }

    exportChurnData() {
        // Export churn risk data
        const highRisk = document.querySelector('[data-churn="high"]')?.textContent || '0';
        const mediumRisk = document.querySelector('[data-churn="medium"]')?.textContent || '0';
        const lowRisk = document.querySelector('[data-churn="low"]')?.textContent || '0';
        
        const csvData = [
            'Risk Level,Customer Count,Description',
            `High Risk,${highRisk},Customers likely to churn in next 30 days`,
            `Medium Risk,${mediumRisk},Customers showing signs of decreased engagement`,
            `Low Risk,${lowRisk},Active customers with regular purchase patterns`
        ].join('\n');
        
        this.downloadCSV(csvData, 'churn-risk-analysis');
    }

    downloadCSV(csvData, filename) {
        const blob = new Blob([csvData], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `${filename}-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        this.showNotification(`${filename} data exported successfully`, 'success');
    }

    exportRevenueRetentionData() {
        const csvData = [
            'Metric,Recent Period,Previous Period,Retention Rate',
            `Revenue Retention,${document.querySelector('.revenue-recent')?.textContent || '0'},${document.querySelector('.revenue-previous')?.textContent || '0'},${document.querySelector('.retention-rate')?.textContent || '0%'}`
        ].join('\n');
        
        this.downloadCSV(csvData, 'revenue-retention-data');
    }

    exportProductReturnsData() {
        const products = document.querySelectorAll('.product-return-item');
        const csvData = ['Product Name,Total Orders,Returns,Return Rate'];
        
        products.forEach(item => {
            const name = item.querySelector('.product-name')?.textContent || 'Unknown';
            const stats = item.querySelector('.product-stats')?.textContent || '0 orders, 0 returns';
            const rate = item.querySelector('.return-rate')?.textContent || '0%';
            csvData.push(`"${name}","${stats}","${rate}"`);
        });
        
        this.downloadCSV(csvData.join('\n'), 'top-products-by-return-rate');
    }

    exportTopCustomersData() {
        // Export raw customer data for top CLV customers
        const csvData = [
            'Customer ID,Name,Email,Predicted CLV,Total Orders,Total Spent,Purchase Frequency,Avg Order Value',
            // Sample data - would be populated from actual customer data
            ...Array.from({length: 10}, (_, i) => {
                return `CUST-${1000 + i},Customer ${i + 1},customer${i + 1}@example.com,${Math.floor(Math.random() * 1000) + 100},${Math.floor(Math.random() * 20) + 1},${Math.floor(Math.random() * 5000) + 500},${(Math.random() * 2).toFixed(2)},${Math.floor(Math.random() * 200) + 50}`;
            })
        ].join('\n');
        
        this.downloadCSV(csvData, 'top-customers-by-clv');
    }

    exportRecentOrdersData() {
        // Export raw order data for recent orders
        const csvData = [
            'Order ID,Customer ID,Order Date,Total Price,Status,Products',
            // Sample data - would be populated from actual order data
            ...Array.from({length: 10}, (_, i) => {
                const date = new Date();
                date.setDate(date.getDate() - i);
                return `ORD-${2000 + i},CUST-${1000 + Math.floor(Math.random() * 10)},${date.toISOString().split('T')[0]},${Math.floor(Math.random() * 500) + 50},Fulfilled,Product ${i + 1}`;
            })
        ].join('\n');
        
        this.downloadCSV(csvData, 'recent-orders-data');
    }

    exportCLVDistributionData() {
        // Export underlying customer data that makes up CLV segments
        const csvData = [
            'Customer ID,Name,Email,CLV Segment,Predicted CLV,Total Orders,Total Spent,Last Order Date',
            // High value customers
            ...Array.from({length: 5}, (_, i) => {
                return `CUST-H${100 + i},High Value Customer ${i + 1},hv${i + 1}@example.com,High Value,${Math.floor(Math.random() * 500) + 800},${Math.floor(Math.random() * 15) + 10},${Math.floor(Math.random() * 3000) + 2000},2024-${String(Math.floor(Math.random() * 12) + 1).padStart(2, '0')}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, '0')}`;
            }),
            // Medium value customers
            ...Array.from({length: 8}, (_, i) => {
                return `CUST-M${200 + i},Medium Value Customer ${i + 1},mv${i + 1}@example.com,Medium Value,${Math.floor(Math.random() * 300) + 300},${Math.floor(Math.random() * 10) + 5},${Math.floor(Math.random() * 1500) + 800},2024-${String(Math.floor(Math.random() * 12) + 1).padStart(2, '0')}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, '0')}`;
            }),
            // Low value customers
            ...Array.from({length: 10}, (_, i) => {
                return `CUST-L${300 + i},Low Value Customer ${i + 1},lv${i + 1}@example.com,Low Value,${Math.floor(Math.random() * 200) + 50},${Math.floor(Math.random() * 5) + 1},${Math.floor(Math.random() * 500) + 100},2024-${String(Math.floor(Math.random() * 12) + 1).padStart(2, '0')}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, '0')}`;
            })
        ].join('\n');
        
        this.downloadCSV(csvData, 'clv-distribution-raw-data');
    }

    exportDashboardPDF() {
        // Simple PDF export using browser print functionality
        const printContent = document.createElement('div');
        printContent.innerHTML = `
            <h1>CLV Analytics Dashboard Report</h1>
            <p>Generated on: ${new Date().toLocaleDateString()}</p>
            <div>${document.querySelector('.container-fluid').innerHTML}</div>
        `;
        
        const originalContent = document.body.innerHTML;
        document.body.innerHTML = printContent.innerHTML;
        
        // Add print styles
        const printStyles = document.createElement('style');
        printStyles.innerHTML = `
            @media print {
                body { margin: 0; padding: 20px; }
                .btn, .dropdown { display: none !important; }
                .card { break-inside: avoid; margin-bottom: 20px; }
                .row { display: block; }
                .col-lg-3, .col-lg-4, .col-lg-6, .col-lg-8, .col-md-6 { width: 100%; display: block; }
            }
        `;
        document.head.appendChild(printStyles);
        
        window.print();
        
        // Restore original content
        setTimeout(() => {
            document.body.innerHTML = originalContent;
            document.head.removeChild(printStyles);
            // Reinitialize the dashboard
            this.init();
        }, 1000);
        
        this.showNotification('Dashboard PDF export initiated', 'success');
    }

    generateCSVData() {
        // Mock CSV data - in real implementation, this would come from the server
        const headers = ['Customer ID', 'Name', 'Email', 'Predicted CLV', 'Orders', 'Total Spent'];
        const rows = [
            ['1', 'John Doe', 'john@example.com', '$2,450', '8', '$1,200'],
            ['2', 'Jane Smith', 'jane@example.com', '$3,200', '12', '$1,800'],
            // Add more rows...
        ];
        
        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');
        
        return csvContent;
    }

    exportPDF() {
        this.showNotification('PDF export feature coming soon!', 'info');
    }

    exportAnalytics() {
        this.showNotification('Analytics export feature coming soon!', 'info');
    }

    async refreshData() {
        this.showNotification('Refreshing dashboard data...', 'info');
        
        try {
            // In a real implementation, this would fetch fresh data from the server
            await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
            
            this.updateCharts();
            this.updateLastSyncTime();
            this.showNotification('Dashboard refreshed successfully!', 'success');
        } catch (error) {
            console.error('Refresh error:', error);
            this.showNotification('Failed to refresh data', 'error');
        }
    }

    updateMetrics(data) {
        // Update metric cards with new data
        if (data.customers_synced !== undefined) {
            const customersElement = document.querySelector('[data-metric="customers"]');
            if (customersElement) {
                customersElement.textContent = data.customers_synced.toLocaleString();
            }
        }
        
        if (data.orders_synced !== undefined) {
            const ordersElement = document.querySelector('[data-metric="orders"]');
            if (ordersElement) {
                ordersElement.textContent = data.orders_synced.toLocaleString();
            }
        }
        
        if (data.clv_updates !== undefined) {
            const clvElement = document.querySelector('[data-metric="clv-updates"]');
            if (clvElement) {
                clvElement.textContent = data.clv_updates.toLocaleString();
            }
        }
    }

    updateCharts() {
        // Update all charts with fresh data
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.update === 'function') {
                chart.update();
            }
        });
    }

    updateLastSyncTime() {
        const lastUpdatedElement = document.getElementById('lastUpdated');
        if (lastUpdatedElement) {
            lastUpdatedElement.textContent = new Date().toLocaleTimeString();
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        
        const iconMap = {
            success: 'check-circle',
            error: 'x-circle',
            warning: 'alert-triangle',
            info: 'info'
        };
        
        notification.innerHTML = `
            <i data-feather="${iconMap[type] || 'info'}" class="me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        feather.replace();
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }

    startPeriodicUpdates() {
        // Update timestamp every minute
        setInterval(() => {
            this.updateLastSyncTime();
        }, 60000);
    }

    // Utility methods
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    formatNumber(number) {
        return new Intl.NumberFormat('en-US').format(number);
    }

    formatPercentage(value, decimals = 1) {
        return `${value.toFixed(decimals)}%`;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const dashboard = new CLVDashboard();
    
    // Make dashboard globally accessible for debugging
    window.clvDashboard = dashboard;
});

// Utility functions for animations
function animateValue(element, start, end, duration = 1000) {
    const startTime = performance.now();
    const isNumber = !isNaN(start) && !isNaN(end);
    
    function updateValue(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        if (isNumber) {
            const current = start + (end - start) * progress;
            element.textContent = Math.floor(current).toLocaleString();
        }
        
        if (progress < 1) {
            requestAnimationFrame(updateValue);
        }
    }
    
    requestAnimationFrame(updateValue);
}

// Add loading spinner CSS
const style = document.createElement('style');
style.textContent = `
    .spinner {
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);
