async function getMetrics(file = null, url = null) {
    const algorithm = document.getElementById('algorithm').value;
    if (!algorithm) {
        showError('Please select an algorithm');
        return;
    }

    const formData = new FormData();
    formData.append('algorithm', algorithm);
    
    if (file) {
        formData.append('file', file);
    } else if (url) {
        formData.append('url', url);
    }

    try {
        showLoading();
        const response = await fetch('/metrics', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            showError(data.error || 'Unknown error occurred');
            return;
        }

        displayMetrics(data.metrics, data.dataset_info);
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while processing your request');
    } finally {
        hideLoading();
    }
}

function displayMetrics(metrics, datasetInfo) {
    const metricsSection = document.getElementById('metrics');
    let html = `
        <h2><i class="fas fa-chart-line"></i> Results</h2>
        
        ${datasetInfo.is_default ? `
        <div class="dataset-info">
            <p class="info-badge">USING THE GIVEN DATASETs</p>
        </div>
        ` : ''}
        
        <div class="dataset-stats">
            <div class="stat-item">
                <i class="fas fa-database"></i>
                <span>Samples: ${datasetInfo.n_samples}</span>
            </div>
            <div class="stat-item">
                <i class="fas fa-columns"></i>
                <span>Features: ${datasetInfo.n_features}</span>
            </div>
            <div class="stat-item">
                <i class="fas fa-tags"></i>
                <span>Classes: ${datasetInfo.n_classes}</span>
            </div>
        </div>
        
        <div class="metrics-grid">
    `;

    // main metrics
    const mainMetrics = ['accuracy', 'precision', 'recall', 'f1_score'];
    mainMetrics.forEach(metric => {
        html += `
            <div class="metric-card">
                <h3>${metric.replace('_', ' ').toUpperCase()}</h3>
                <div class="metric-value">${(metrics[metric] * 100).toFixed(2)}%</div>
            </div>
        `;
    });

    html += `</div>`;

    // confusion matrix
    if (metrics.confusion_matrix) {
        html += `
            <div class="confusion-matrix">
                <h3>Confusion Matrix</h3>
                <div class="matrix-container">
                    ${createConfusionMatrixHTML(metrics.confusion_matrix)}
                </div>
            </div>
        `;
    }

    metricsSection.innerHTML = html;
}

function createConfusionMatrixHTML(matrix) {
    let html = '<table class="matrix-table">';
    matrix.forEach(row => {
        html += '<tr>';
        row.forEach(cell => {
            html += `<td>${cell}</td>`;
        });
        html += '</tr>';
    });
    html += '</table>';
    return html;
}

function showError(message) {
    const metricsSection = document.getElementById('metrics');
    metricsSection.innerHTML = `
        <div class="error-message">
            <i class="fas fa-exclamation-circle"></i>
            ${message}
        </div>
    `;
}

function showLoading() {
    const metricsSection = document.getElementById('metrics');
    metricsSection.innerHTML = `
        <div class="loading">
            <i class="fas fa-spinner fa-spin"></i>
            Processing...
        </div>
    `;
}

function hideLoading() {
    // error handling
}

// upload handling
document.getElementById('file-upload').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const fileName = file?.name || 'No file chosen';
    document.getElementById('file-name').textContent = fileName;
    
    // file processing
    if (file) {
        getMetrics(file);
    }
});

// import handling
async function importFromURL() {
    const url = document.getElementById('dataset-url').value;
    if (!url) {
        showError('Please enter a valid URL');
        return;
    }
    await getMetrics(null, url);
}

// grag and drop for upload
const dropZone = document.querySelector('.custom-file-upload');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    dropZone.classList.add('highlight');
}

function unhighlight(e) {
    dropZone.classList.remove('highlight');
}

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    
    document.getElementById('file-upload').files = dt.files;
    document.getElementById('file-name').textContent = file.name;
    
    getMetrics(file);
}
