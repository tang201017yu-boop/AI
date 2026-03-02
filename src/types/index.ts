// 系统相关
export interface SystemInfo {
  total_models: number;
  total_datasets: number;
  gpu_available: boolean;
  gpu_info?: string;
  ultralytics_version: string;
}

// 数据集
export interface Dataset {
  id?: string;
  name: string;
  path: string;
  task_type?: string;
  num_images?: number;
  image_count?: number;
  num_classes?: number;
  classes?: string[];
  split?: { total?: number; train?: number; val?: number; test?: number };
  created_at: string;
  updated_at?: string;
}

export interface DatasetUploadResponse {
  success: boolean;
  dataset_id: string;
  message: string;
}

// 模型
export interface Model {
  id?: string;
  name: string;
  path: string;
  size?: number;
  type?: string;
  model_type?: string;
  task?: string;
  input_shape?: number[];
  classes?: string[];
  created_at: string;
  metrics?: ModelMetrics;
}

export interface ModelMetrics {
  precision?: number;
  recall?: number;
  mAP50?: number;
  mAP50_95?: number;
}

// 训练
export interface TrainingConfig {
  model_type: string;
  dataset_id: string;
  epochs: number;
  batch_size: number;
  image_size: number;
  optimizer: string;
  learning_rate: number;
}

export interface TrainingStatus {
  status: 'idle' | 'running' | 'completed' | 'failed';
  current_epoch: number;
  total_epochs: number;
  loss?: number;
  metrics?: Record<string, number>;
}

// 推理
export interface InferenceRequest {
  model_id: string;
  image_path?: string;
  image_data?: string;
  conf_threshold?: number;
  iou_threshold?: number;
}

export interface InferenceResult {
  success: boolean;
  image_path: string;
  detections: Detection[];
  inference_time: number;
}

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // x1, y1, x2, y2
}

// 标注
export interface AnnotationProject {
  id: string;
  name: string;
  description?: string;
  dataset_id?: string;
  classes?: string[];
  status?: 'pending' | 'in_progress' | 'completed';
  progress?: number;
  path?: string;
  created_at: string;
  updated_at?: string;
}

// 通用
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}
