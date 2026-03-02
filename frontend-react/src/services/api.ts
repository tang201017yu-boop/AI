import axios, { AxiosInstance, AxiosError } from 'axios';
import type { ApiResponse, SystemInfo, Dataset, DatasetUploadResponse, Model, TrainingConfig, TrainingStatus, InferenceRequest, InferenceResult, AnnotationProject } from '../types';

const api: AxiosInstance = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证 token
    return config;
  },
  (error) => Promise.reject(error)
);

// 响应拦截器
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiResponse<unknown>>) => {
    const message = error.response?.data?.message || error.message || '请求失败';
    console.error('API Error:', message);
    return Promise.reject(new Error(message));
  }
);

export default api;

// ============ 系统 API ============
export const systemApi = {
  getInfo: () => api.get<ApiResponse<SystemInfo>>('/system/info'),
  getHealth: () => api.get('/system/health'),
};

// ============ 数据集 API ============
export const datasetApi = {
  list: () => api.get<ApiResponse<{datasets: Dataset[]}>>('/datasets/list'),
  get: (id: string) => api.get<ApiResponse<Dataset>>(`/datasets/${id}`),
  upload: (formData: FormData) =>
    api.post<ApiResponse<DatasetUploadResponse>>('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  delete: (id: string) => api.delete(`/datasets/${id}`),
  // 数据集统计
  getStatistics: (name: string) => api.get(`/datasets/${name}/statistics`),
  getClassDistribution: (name: string) => api.get(`/datasets/${name}/class-distribution`),
  getSpatialDistribution: (name: string) => api.get(`/datasets/${name}/spatial-distribution`),
  getDimensionAnalysis: (name: string) => api.get(`/datasets/${name}/dimension-analysis`),
  getSplitDetails: (name: string) => api.get(`/datasets/${name}/split-details`),
  // 图片浏览
  getImages: (name: string, params?: { split?: string; page?: number; pageSize?: number; sort?: string; labeled?: string }) =>
    api.get(`/datasets/${name}/images`, { params }),
  getImage: (name: string, filename: string) =>
    api.get(`/datasets/${name}/images/${filename}`),
};

// ============ 模型 API ============
export const modelApi = {
  list: () => api.get<ApiResponse<Model[]>>('/models/list'),
  get: (id: string) => api.get<ApiResponse<Model>>(`/models/${id}`),
  upload: (formData: FormData) =>
    api.post<ApiResponse<Model>>('/models/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  delete: (id: string) => api.delete(`/models/${id}`),
};

// ============ 训练 API ============
export const trainingApi = {
  start: (config: TrainingConfig) =>
    api.post<ApiResponse<{ task_id: string }>>('/training/start', config),
  status: (taskId: string) =>
    api.get<ApiResponse<TrainingStatus>>(`/training/status/${taskId}`),
  stop: (taskId: string) => api.post(`/training/stop/${taskId}`),
  getHistory: () => api.get('/training/history'),
};

// ============ 推理 API ============
export const inferenceApi = {
  image: (file: File, model_name?: string, confidence: number = 0.25, iou_threshold: number = 0.45) => {
    const formData = new FormData();
    formData.append('file', file);
    if (model_name) formData.append('model_name', model_name);
    formData.append('confidence', String(confidence));
    formData.append('iou_threshold', String(iou_threshold));
    return api.post<ApiResponse<InferenceResult>>('/inference/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  video: (request: InferenceRequest) =>
    api.post<ApiResponse<InferenceResult>>('/inference/video', request),
  batch: (requests: InferenceRequest[]) =>
    api.post<ApiResponse<InferenceResult[]>>('/inference/batch', { requests }),
};

// ============ 标注 API ============
export const annotationApi = {
  listProjects: () => api.get<ApiResponse<AnnotationProject[]>>('/annotation/projects'),
  getProject: (id: string) => api.get<ApiResponse<AnnotationProject>>(`/annotation/projects/${id}`),
  createProject: (data: Partial<AnnotationProject>) =>
    api.post<ApiResponse<AnnotationProject>>('/annotation/projects', data),
  getImages: (projectId: string) =>
    api.get(`/annotation/projects/${projectId}/images`),
  saveAnnotations: (projectId: string, imageId: string, annotations: unknown) =>
    api.post(`/annotation/projects/${projectId}/images/${imageId}/annotations`, { annotations }),
};

// ============ 解决方案 API ============
export const solutionsApi = {
  list: () => api.get('/solutions/list'),
  objectCounting: (formData: FormData) =>
    api.post('/solutions/object-counting', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  heatmap: (formData: FormData) =>
    api.post('/solutions/heatmap', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  speedEstimation: (formData: FormData) =>
    api.post('/solutions/speed-estimation', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  distanceCalculation: (formData: FormData) =>
    api.post('/solutions/distance-calculation', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  objectBlur: (formData: FormData) =>
    api.post('/solutions/object-blur', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  objectCrop: (formData: FormData) =>
    api.post('/solutions/object-crop', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  queueManagement: (formData: FormData) =>
    api.post('/solutions/queue-management', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
};
