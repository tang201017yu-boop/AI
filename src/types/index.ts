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
  project_name: string;      // 项目名称
  dataset_path: string;     // 数据集路径
  model_type?: string;       // 模型类型
  epochs?: number;           // 训练轮数
  batch_size?: number;      // 批大小
  img_size?: number;        // 输入图片尺寸
  device?: string;          // 设备
  optimizer?: string;       // 优化器
  lr0?: number;             // 初始学习率
  lrf?: number;             // 最终学习率因子
  warmup_epochs?: number;   // 预热轮数
  patience?: number;       // 早停耐心值
  amp?: boolean;           // 混合精度
  workers?: number;         // 数据加载线程数
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

// SAM 智能标注
export interface SAMStatus {
  available: boolean;
  loaded: boolean;
  model_type?: string;
  supported_models: Record<string, SAMModelInfo>;
}

export interface SAMModelInfo {
  name: string;
  speed: string;
  size: string;
  description: string;
}

export interface SAMPrediction {
  success: boolean;
  masks: number[][][];
  mask_image?: string;
  score: number;
  all_scores?: number[];
}

export interface SAMAutoLabelResult {
  success: boolean;
  message: string;
  annotations: SAMAnnotation[];
  preview_image?: string;
  total_detections: number;
  total_annotations: number;
}

export interface SAMAnnotation {
  class: string;
  class_id: number;
  bbox: number[];
  segmentation: string;
  confidence: number;
  segment_score?: number;
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

// ============ 标注交互类型 ============
export interface AnnotationPoint {
  x: number;
  y: number;
  label: 1 | 0; // 1=前景, 0=背景
}

export interface AnnotationBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface AnnotationMask {
  polygons: number[];
  color: string;
}

export type AnnotationTool = 'point' | 'box' | 'select' | 'auto' | 'polygon';

// 标注状态
export interface AnnotationState {
  tool: AnnotationTool;
  points: AnnotationPoint[];
  boxes: AnnotationBox[];
  masks: AnnotationMask[];
  selectedId: string | null;
  currentClass: string;
}

// 快捷键配置
export interface AnnotationShortcut {
  key: string;
  description: string;
  action: string;
}

// 类别建议
export interface ClassSuggestion {
  name: string;
  count: number;
  color: string;
}
