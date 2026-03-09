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
  // 获取用户上传的模型
  getUserModels: () => api.get<ApiResponse<{ models: any[]; total: number }>>('/training/models'),
  // 上传模型
  upload: (formData: FormData) =>
    api.post<ApiResponse<any>>('/training/models/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  get: (id: string) => api.get<ApiResponse<Model>>(`/models/${id}`),
  delete: (id: string) => api.delete(`/training/models/${id}`),
  // 获取已加载到内存的模型列表
  getLoadedModels: () => api.get<ApiResponse<{ models: any[]; total: number }>>('/models/loaded'),
  // 预加载模型到内存
  loadModel: (modelName: string, device: string = 'cpu') =>
    api.post<ApiResponse<{ success: boolean; message: string }>>('/models/load', {
      model_name: modelName,
      device: device,
    }),
  // 获取预训练基础模型列表
  getPretrainedModels: () => api.get<ApiResponse<{ models: any[]; installed: string[] }>>('/models/pretrained'),
};

// ============ 训练 API ============
export const trainingApi = {
  start: (config: TrainingConfig) =>
    api.post<ApiResponse<{ task_id: string }>>('/training/start', config),
  status: (taskId: string) =>
    api.get<ApiResponse<TrainingStatus>>(`/training/status/${taskId}`),
  stop: (taskId: string) => api.post(`/training/stop/${taskId}`),
  // 获取训练历史列表 - 使用 experiments API
  getHistory: () => api.get('/experiments'),
  // 获取单个实验详情
  getExperiment: (taskId: string) =>
    api.get<ApiResponse<any>>(`/experiments/${taskId}`),
  // 获取实验验证结果图片
  getExperimentResults: (taskId: string) =>
    api.get<ApiResponse<any>>(`/experiments/${taskId}/results`),
  // 导出实验模型
  exportExperimentModel: (taskId: string, format: string) =>
    api.post<ApiResponse<any>>(`/experiments/${taskId}/export?format=${format}`),
  // 使用实验模型推理
  inferWithExperiment: (taskId: string, data: { image_url: string; conf_threshold?: number; iou_threshold?: number }) =>
    api.post<ApiResponse<any>>(`/experiments/${taskId}/infer`, data),
  // 继续训练
  resumeTraining: (taskId: string, data: { epochs?: number; batch_size?: number; resume_from_best?: boolean }) =>
    api.post<ApiResponse<any>>(`/experiments/${taskId}/resume`, data),
  // 删除实验
  deleteExperiment: (taskId: string) =>
    api.delete<ApiResponse<any>>(`/experiments/${taskId}`),
  getChartData: (taskId: string) =>
    api.get<ApiResponse<any>>(`/training/chart-data/${taskId}`),
  getSystemInfo: () =>
    api.get<ApiResponse<any>>('/training/system-info'),
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
  deleteProject: (id: string) =>
    api.delete<ApiResponse<void>>(`/annotation/projects/${id}`),
  getImages: (projectId: string) =>
    api.get(`/annotation/projects/${projectId}/images`),
  addImages: (projectId: string, files: File[]) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    return api.post(`/annotation/projects/${projectId}/images`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  saveAnnotations: (projectId: string, imageId: string, annotations: unknown) =>
    api.post(`/annotation/projects/${projectId}/images/${imageId}/annotations`, { annotations }),
  getAnnotations: (projectId: string, imageId: string) =>
    api.get(`/annotation/projects/${projectId}/image/${imageId}`),
};

// ============ SAM 智能标注 API ============
export const samApi = {
  // 获取SAM模型状态
  getStatus: () => api.get('/sam/status'),
  // 加载SAM模型
  loadModel: (modelType: string = 'vit_b') =>
    api.post(`/sam/load?model_type=${modelType}`),
  // 设置图片
  setImage: (imagePath: string) =>
    api.post('/sam/set-image', { image_path: imagePath }),
  // 点点击分割预测
  predict: (points: number[][], labels: number[]) =>
    api.post('/sam/predict', { points, labels }),
  // 边界框分割预测
  predictBox: (bbox: number[]) =>
    api.post('/sam/predict-box', { bbox }),
  // 自动标注 (YOLO检测 + SAM分割)
  autoLabel: (imagePath: string, classNames: string[], modelName: string = 'yolo11n.pt', confidence: number = 0.25) => {
    const formData = new FormData();
    formData.append('image_path', imagePath);
    formData.append('class_names', JSON.stringify(classNames));
    formData.append('model_name', modelName);
    formData.append('confidence', String(confidence));
    return api.post('/sam/auto-label', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  // 批量自动标注
  batchAutoLabel: (imagePaths: string[], classNames: string[], modelName: string = 'yolo11n.pt', confidence: number = 0.25) => {
    const formData = new FormData();
    formData.append('image_paths', JSON.stringify(imagePaths));
    formData.append('class_names', JSON.stringify(classNames));
    formData.append('model_name', modelName);
    formData.append('confidence', String(confidence));
    return api.post('/sam/batch-auto-label', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  // 批量 SAM 标注 - 针对特定类别
  batchSamLabel: (file: File, className: string, modelName?: string, confidence?: number) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('class_name', className);
    if (modelName) formData.append('model_name', modelName);
    formData.append('confidence', String(confidence || 0.25));
    return api.post('/sam/batch-sam-label', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  // 导出YOLO格式
  exportYolo: (annotations: any[], outputDir: string, imagePath: string) => {
    const formData = new FormData();
    formData.append('annotations', JSON.stringify(annotations));
    formData.append('output_dir', outputDir);
    formData.append('image_path', imagePath);
    return api.post('/sam/export-yolo', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  // 获取支持的模型列表
  getModels: () => api.get('/sam/models'),
};

// ============ 项目 API ============
export interface Project {
  id: string;
  name: string;
  description: string;
  cover_image?: string;
  status: string;
  created_at: string;
  updated_at: string;
  settings?: Record<string, any>;
  models_count?: number;
}

export const projectApi = {
  // 创建项目
  create: (data: { name: string; description?: string; cover_image?: string; task_type?: string; settings?: Record<string, any> }) =>
    api.post<ApiResponse<Project>>('/training/projects', data),
  // 获取项目列表
  list: () => api.get<ApiResponse<{ projects: Project[]; total: number }>>('/training/projects'),
  // 获取项目详情
  get: (projectId: string) =>
    api.get<ApiResponse<Project>>(`/training/projects/${projectId}`),
  // 更新项目
  update: (projectId: string, data: { name?: string; description?: string; cover_image?: string; settings?: Record<string, any> }) =>
    api.put<ApiResponse<Project>>(`/training/projects/${projectId}`, data),
  // 删除项目
  delete: (projectId: string) =>
    api.delete<ApiResponse<void>>(`/training/projects/${projectId}`),
  // 恢复项目
  restore: (projectId: string) =>
    api.post<ApiResponse<Project>>(`/training/projects/${projectId}/restore`),
  // 获取回收站
  getRecycleBin: () =>
    api.get<ApiResponse<{ items: Project[]; expired: Project[]; total: number }>>('/training/projects/recycle-bin'),
  // 清空回收站
  emptyRecycleBin: () =>
    api.post<ApiResponse<{ message: string }>>('/training/projects/recycle-bin/empty'),
  // 获取项目模型列表
  getModels: (projectId: string) =>
    api.get<ApiResponse<{ models: any[]; total: number }>>(`/training/projects/${projectId}/models`),
  // 获取项目活动日志
  getActivity: (projectId: string) =>
    api.get<ApiResponse<{ activities: any[]; total: number }>>(`/training/projects/${projectId}/activity`),
};

// ============ 解决方案 API ============
export const solutionsApi = {
  list: () => api.get('/solutions/list'),
  objectCounting: (formData: FormData) =>
    api.post('/solutions/object-counting', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  // 热力图（异步任务，需要轮询）
  heatmap: async (formData: FormData) => {
    // 提交任务
    const response = await api.post('/solutions/heatmap', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    const taskId = response.data?.task_id;
    if (!taskId) {
      return response;
    }

    // 轮询等待完成
    const maxAttempts = 300; // 最多5分钟
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      try {
        const statusResponse = await api.get(`/solutions/heatmap/status/${taskId}`);
        const status = statusResponse.data;
        if (status.status === 'completed') {
          return { data: { success: true, ...status } };
        } else if (status.status === 'failed') {
          return { data: { success: false, message: status.message || '处理失败' } };
        }
      } catch (e) {
        // 继续轮询
      }
    }
    return { data: { success: false, message: '处理超时' } };
  },
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
