// API 请求工具类 - OpenCV Platform
const API_BASE = '/api/v1';
const CACHE_INTERVALS = {
    models: 10000,
    datasets: 10000
};

class API {
    static _modelCache = null;
    static _modelCacheAt = 0;
    static _datasetCache = null;
    static _datasetCacheAt = 0;

    // 通用请求助手
    static async _fetchJson(path, options = {}) {
        const response = await fetch(`${API_BASE}${path}`, options);
        if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || response.statusText);
        }
        return response.json();
    }

    static _cloneArray(items) {
        return Array.isArray(items) ? items.map(item => ({ ...item })) : [];
    }

    // ==================== 系统信息 ====================
    static async getSystemInfo() {
        return this._fetchJson('/system/info');
    }

    static async healthCheck() {
        return this._fetchJson('/system/health');
    }

    // ==================== 数据准备模块 ====================
    // 数据集管理
    static async listDatasets({ force = false } = {}) {
        const now = Date.now();
        if (!force && this._datasetCache && (now - this._datasetCacheAt) < CACHE_INTERVALS.datasets) {
            return {
                total: this._datasetCache.total,
                datasets: this._cloneArray(this._datasetCache.datasets)
            };
        }

        const response = await this._fetchJson('/datasets/list');
        const datasets = Array.isArray(response?.datasets) ? response.datasets : [];
        const payload = {
            total: response?.total ?? datasets.length,
            datasets
        };
        this._datasetCache = payload;
        this._datasetCacheAt = now;
        return {
            total: payload.total,
            datasets: this._cloneArray(payload.datasets)
        };
    }

    static invalidateDatasetCache() {
        this._datasetCache = null;
        this._datasetCacheAt = 0;
    }

    static async uploadDataset(file, name = null, taskType = 'detect') {
        const formData = new FormData();
        formData.append('file', file);
        if (name) formData.append('name', name);
        formData.append('task_type', taskType);

        const result = await this._fetchJson('/datasets/upload', {
            method: 'POST',
            body: formData
        });
        this.invalidateDatasetCache();
        return result;
    }

    static async deleteDataset(name) {
        return this._fetchJson(`/datasets/${name}`, { method: 'DELETE' });
    }

    // 标注项目管理
    static async createAnnotationProject(name, taskType, classes) {
        return this._fetchJson('/annotation/projects', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ name, task_type: taskType, classes: JSON.stringify(classes) })
        });
    }

    static async addAnnotationImages(projectName, files) {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));

        return this._fetchJson(`/annotation/projects/${projectName}/images`, {
            method: 'POST',
            body: formData
        });
    }

    static async getAnnotationProjectImages(projectName) {
        return this._fetchJson(`/annotation/projects/${projectName}/images`);
    }

    static async saveAnnotation(projectName, imageName, annotations) {
        return this._fetchJson(`/annotation/projects/${projectName}/image/${imageName}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ annotations: JSON.stringify(annotations) })
        });
    }

    static async getAnnotationTasks() {
        return this._fetchJson('/annotation/tasks');
    }

    // SAM 智能标注
    static async loadSAMModel(modelType = 'vit_b') {
        return this._fetchJson('/sam/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ model_type: modelType })
        });
    }

    static async getSAMStatus() {
        return this._fetchJson('/sam/status');
    }

    static async setSAMImage(imagePath) {
        return this._fetchJson('/sam/set-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ image_path: imagePath })
        });
    }

    static async samPredict(points, labels) {
        return this._fetchJson('/sam/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
                points: JSON.stringify(points),
                labels: JSON.stringify(labels)
            })
        });
    }

    // ==================== 模型训练模块 ====================
    static async startTraining(config) {
        return this._fetchJson('/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
    }

    static async getTrainingStatus(taskId) {
        return this._fetchJson(`/training/status/${taskId}`);
    }

    static async listTrainingTasks() {
        return this._fetchJson('/training/tasks');
    }

    static async cancelTraining(taskId) {
        return this._fetchJson(`/training/cancel/${taskId}`, { method: 'POST' });
    }

    // 模型导出
    static async exportModel(modelPath, format, options = {}) {
        return this._fetchJson('/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_path: modelPath,
                format: format,
                ...options
            })
        });
    }

    static async getExportFormats() {
        return this._fetchJson('/export/formats');
    }

    // ==================== 推理测试模块 ====================
    static async inferImage(file, options = {}) {
        const formData = new FormData();
        formData.append('file', file);

        if (options.model_name) formData.append('model_name', options.model_name);
        if (options.confidence !== undefined) formData.append('confidence', options.confidence);
        if (options.draw_results !== undefined) formData.append('draw_results', options.draw_results);

        const response = await fetch(`${API_BASE}/inference/image`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || response.statusText);
        }
        return response.json();
    }

    static async inferBatch(files, options = {}) {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));

        if (options.model_name) formData.append('model_name', options.model_name);
        if (options.confidence !== undefined) formData.append('confidence', options.confidence);

        const response = await fetch(`${API_BASE}/inference/batch`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || response.statusText);
        }
        return response.json();
    }

    // 监控
    static async addCamera(name, url, modelName, confidence) {
        return this._fetchJson('/monitor/camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
                name, url,
                model_name: modelName || '',
                confidence: confidence || 0.25
            })
        });
    }

    static async listCameras() {
        return this._fetchJson('/monitor/cameras');
    }

    // ==================== 智能解决方案模块 ====================
    static async listSolutions() {
        return this._fetchJson('/solutions/list');
    }

    static async runSolution(solutionType, file, options = {}) {
        const formData = new FormData();
        formData.append('file', file);

        Object.keys(options).forEach(key => {
            formData.append(key, typeof options[key] === 'object' ? JSON.stringify(options[key]) : options[key]);
        });

        const response = await fetch(`${API_BASE}/solutions/${solutionType}`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || response.statusText);
        }
        return response.json();
    }

    // ==================== 模型管理 ====================
    static async listModels({ force = false } = {}) {
        const now = Date.now();
        if (!force && this._modelCache && (now - this._modelCacheAt) < CACHE_INTERVALS.models) {
            return this._cloneArray(this._modelCache);
        }

        const response = await this._fetchJson('/models/list');
        const models = Array.isArray(response) ? response : (response?.models ?? []);
        this._modelCache = models || [];
        this._modelCacheAt = now;
        return this._cloneArray(this._modelCache);
    }

    static invalidateModelCache() {
        this._modelCache = null;
        this._modelCacheAt = 0;
    }

    static async uploadModel(file) {
        const formData = new FormData();
        formData.append('file', file);

        const result = await this._fetchJson('/models/upload', {
            method: 'POST',
            body: formData
        });
        this.invalidateModelCache();
        return result;
    }
}

// 工具函数
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    const container = document.querySelector('.container');
    if (!container) {
        console.warn('Missing container element for alert');
        return;
    }
    container.insertBefore(alertDiv, container.firstChild);

    setTimeout(() => alertDiv.remove(), 5000);
}

function formatBytes(bytes) {
    if (!bytes || typeof bytes !== 'number' || Number.isNaN(bytes)) {
        return '0 Bytes';
    }
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`;
}

function formatDate(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return Number.isNaN(date.getTime()) ? '-' : date.toLocaleString();
}

if (typeof window !== 'undefined') {
    window.API = API;
    window.showAlert = window.showAlert || showAlert;
    window.formatBytes = window.formatBytes || formatBytes;
    window.formatDate = window.formatDate || formatDate;
}
