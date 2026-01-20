"""
数据模型定义
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DetectionResult(BaseModel):
    """检测结果"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")


class InferenceRequest(BaseModel):
    """推理请求"""
    model_name: Optional[str] = "yolo11n.pt"
    model_path: Optional[str] = None
    confidence: Optional[float] = 0.25
    iou_threshold: Optional[float] = 0.45
    img_size: Optional[int] = 640


class InferenceResponse(BaseModel):
    """推理响应"""
    success: bool
    message: str
    detections: List[DetectionResult] = []
    inference_time: float
    image_shape: List[int]


class TrainingConfig(BaseModel):
    """训练配置"""
    project_name: str
    dataset_path: str
    model_type: str = "yolo11n"
    model_path: Optional[str] = None
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    device: str = "cpu"  # cpu 或 cuda
    patience: int = 50
    save_period: int = 10
    pretrained: bool = True
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    # GPU 优化配置
    amp: bool = True  # 混合精度训练 (AMP)
    workers: int = 4  # 数据加载线程数
    pin_memory: bool = True  # 固定内存 (GPU训练加速)
    cache: str = "disk"  # 缓存类型: none, ram, disk
    rect: bool = False  # 矩形训练 (减少padding)
    cos_lr: bool = True  # 余弦学习率
    close_mosaic: int = 10  # 最后N个epoch关闭mosaic增强


class TrainingStatus(BaseModel):
    """训练状态"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float
    current_epoch: int
    total_epochs: int
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    gpu_memory: Optional[str] = None  # GPU 内存使用情况


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    path: str
    size: int
    created_at: datetime
    model_type: str
    task: str  # detect, segment, classify, pose
    input_shape: Optional[List[int]] = None
    classes: Optional[List[str]] = None


class DatasetInfo(BaseModel):
    """数据集信息"""
    name: str
    path: str
    num_images: int
    num_classes: int
    classes: List[str]
    split: Dict[str, int] = Field(default_factory=dict)  # train/val/test 数量
    created_at: datetime


class ExportConfig(BaseModel):
    """模型导出配置"""
    model_path: str
    format: str = "onnx"  # onnx, torchscript, coreml, saved_model, pb, tflite, edgetpu, tfjs
    img_size: List[int] = [640, 640]
    batch_size: int = 1
    optimize: bool = False
    half: bool = False  # FP16 量化
    simplify: bool = True  # ONNX 简化
    dynamic: bool = False  # 动态输入
    opset: int = 12  # ONNX opset 版本



class SystemInfo(BaseModel):
    """系统信息"""
    app_name: str
    version: str
    python_version: str
    ultralytics_version: str
    total_models: int
    total_datasets: int
    gpu_available: bool
    gpu_info: Optional[str] = None


# ==================== Solutions 相关模型 ====================

class ObjectCountingRequest(BaseModel):
    """对象计数请求"""
    model_name: Optional[str] = "yolo11n.pt"
    region_points: Optional[List[List[int]]] = None  # [(x1,y1), (x2,y2), ...]
    show_in: bool = True
    show_out: bool = True
    classes: Optional[List[int]] = None
    conf: float = 0.25


class HeatmapRequest(BaseModel):
    """热图生成请求"""
    model_name: Optional[str] = "yolo11n.pt"
    colormap: int = 2  # cv2.COLORMAP_JET
    classes: Optional[List[int]] = None
    conf: float = 0.25


class SpeedEstimationRequest(BaseModel):
    """速度估算请求"""
    model_name: Optional[str] = "yolo11n.pt"
    region_points: Optional[List[List[int]]] = None
    classes: Optional[List[int]] = None
    conf: float = 0.25


class DistanceCalculationRequest(BaseModel):
    """距离计算请求"""
    model_name: Optional[str] = "yolo11n.pt"
    classes: Optional[List[int]] = None
    conf: float = 0.25


class ObjectBlurRequest(BaseModel):
    """对象模糊请求"""
    model_name: Optional[str] = "yolo11n.pt"
    classes: Optional[List[int]] = None
    conf: float = 0.25
    blur_ratio: float = 50


class ObjectCropRequest(BaseModel):
    """对象裁剪请求"""
    model_name: Optional[str] = "yolo11n.pt"
    classes: Optional[List[int]] = None
    conf: float = 0.25


class QueueManagementRequest(BaseModel):
    """队列管理请求"""
    model_name: Optional[str] = "yolo11n.pt"
    region_points: Optional[List[List[int]]] = None
    classes: Optional[List[int]] = None
    conf: float = 0.25


class ParkingManagementRequest(BaseModel):
    """停车管理请求"""
    model_name: Optional[str] = "yolo11n.pt"
    parking_slots: List[List[int]]  # 每个停车位的四个点坐标
    conf: float = 0.25


class VisionEyeRequest(BaseModel):
    """VisionEye 请求"""
    model_name: Optional[str] = "yolo11n.pt"
    vision_point: List[int] = [20, 20]  # 视觉焦点坐标 (x, y)
    classes: Optional[List[int]] = None
    conf: float = 0.25


class WorkoutMonitoringRequest(BaseModel):
    """健身监控请求"""
    model_name: Optional[str] = "yolo11n-pose.pt"
    kpts_indices: List[int] = [6, 8, 10]  # 关键点索引：肩-肘-腕
    up_angle: float = 145.0
    down_angle: float = 90.0
    conf: float = 0.25


class AnalyticsRequest(BaseModel):
    """数据分析请求"""
    model_name: Optional[str] = "yolo11n.pt"
    analytics_type: str = "line"  # line, bar, area, pie
    classes: Optional[List[int]] = None
    conf: float = 0.25


class SolutionResponse(BaseModel):
    """Solutions 通用响应"""
    success: bool
    message: str
    results: Optional[Dict[str, Any]] = None
    output_path: Optional[str] = None
    total_frames: Optional[int] = None
