// API 请求工具类
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

    // ---------- 系统信息 ----------
    static async getSystemInfo() {
        return this._fetchJson('/system/info');
    }

    static async healthCheck() {
        return this._fetchJson('/system/health');
    }

    // ---------- 推理相关 ----------
    static async inferImage(file, options = {}) {
        const formData = new FormData();
        formData.append('file', file);

        if (options.model_name) formData.append('model_name', options.model_name);
        if (options.model_path) formData.append('model_path', options.model_path);
        if (options.confidence !== undefined) formData.append('confidence', options.confidence);
        if (options.iou_threshold !== undefined) formData.append('iou_threshold', options.iou_threshold);
        if (options.img_size !== undefined) formData.append('img_size', options.img_size);

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
        if (options.model_path) formData.append('model_path', options.model_path);
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

    // ---------- 训练相关 ----------
    static async startTraining(config) {
        return this._fetchJson('/training/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
    }

    static async getTrainingStatus(taskId) {
        return this._fetchJson(`/training/status/${taskId}`);
    }

    static async listTrainingTasks() {
        return this._fetchJson('/training/tasks');
    }

    // ---------- 模型相关 ----------
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

    static getLocalModelPath(modelName) {
        if (!this._modelCache) return null;
        const record = this._modelCache.find(model => model.name === modelName);
        return record ? record.path : null;
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

    static async exportModel(config) {
        return this._fetchJson('/models/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
    }

    // ---------- 数据集相关 ----------
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

    static async refreshDatasets() {
        const response = await this._fetchJson('/datasets/refresh', { method: 'POST' });
        const datasets = Array.isArray(response?.datasets) ? response.datasets : [];
        const payload = {
            total: response?.total ?? datasets.length,
            datasets
        };
        this._datasetCache = payload;
        this._datasetCacheAt = Date.now();
        return {
            total: payload.total,
            datasets: this._cloneArray(payload.datasets)
        };
    }

    static async uploadDataset(file) {
        const formData = new FormData();
        formData.append('file', file);

        const result = await this._fetchJson('/datasets/upload', {
            method: 'POST',
            body: formData
        });
        this.invalidateDatasetCache();
        return result;
    }

    // ---------- Label Studio 相关 ----------
    static async checkLabelStudio() {
        return this._fetchJson('/labelstudio/check');
    }

    static async listLabelStudioProjects() {
        return this._fetchJson('/labelstudio/projects');
    }

    static async createLabelStudioProject(title, description = '') {
        const params = new URLSearchParams({ title, description });
        return this._fetchJson(`/labelstudio/projects/create?${params}`, {
            method: 'POST'
        });
    }

    static async exportLabelStudioAnnotations(projectId, datasetName, format = 'YOLO') {
        const params = new URLSearchParams({ dataset_name: datasetName, format });
        return this._fetchJson(`/labelstudio/export/${projectId}?${params}`, {
            method: 'POST'
        });
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
