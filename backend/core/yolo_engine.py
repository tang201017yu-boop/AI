"""
YOLO 引擎 - YOLO Model Engine
处理模型加载、推理和训练的核心功能
"""
import os
import time
import copy
import threading
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
from PIL import Image

try:
    import torch
    from ultralytics import YOLO
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    SummaryWriter = None
    print(f"Warning: torch/ultralytics not installed - {e}")
except Exception as e:
    TORCH_AVAILABLE = False
    SummaryWriter = None
    print(f"Warning: torch/ultralytics initialization failed - {e}")

from .config import settings

logger = logging.getLogger(__name__)


class YOLOEngine:
    """YOLO 模型引擎 - 核心模型处理类"""

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")

        self.models: Dict[str, YOLO] = {}
        self.training_tasks: Dict[str, 'TrainingStatus'] = {}
        self.training_futures: Dict[str, Future] = {}
        self.training_lock = threading.Lock()
        self.model_cache_lock = threading.Lock()
        self.model_aliases: Dict[str, str] = {}
        self.model_metadata_cache: Dict[str, Tuple[float, 'ModelInfo']] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=settings.MAX_TRAINING_WORKERS,
            thread_name_prefix="yolo-train"
        )

        # GPU 优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass

        self._index_existing_models()

    def _index_existing_models(self) -> None:
        """索引已存在的模型"""
        for path in self._iter_model_paths():
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                continue
            self.model_aliases.setdefault(resolved.name, str(resolved))

    def _iter_model_paths(self):
        """迭代所有模型文件路径"""
        seen = set()
        for root in settings.model_search_paths:
            if not root.exists():
                continue
            for path in root.glob("**/*.pt"):
                try:
                    resolved = path.resolve()
                except FileNotFoundError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved

    def _resolve_model_path(self, model_identifier: Optional[str]) -> str:
        """解析模型标识符"""
        if not model_identifier:
            return settings.DEFAULT_MODEL

        identifier = model_identifier.strip()
        alias = Path(identifier).name

        with self.model_cache_lock:
            mapped = self.model_aliases.get(identifier) or self.model_aliases.get(alias)
            if mapped and Path(mapped).exists():
                return str(Path(mapped).resolve())

        candidate = Path(identifier)
        if candidate.exists():
            return str(candidate.resolve())

        # 搜索已配置的路径
        for root in settings.model_search_paths:
            direct = root / alias
            if direct.exists():
                return str(direct.resolve())
            nested = next(root.glob(f"**/{alias}"), None)
            if nested and nested.exists():
                return str(nested.resolve())

        return identifier

    def load_model(self, model_identifier: Optional[str] = None) -> YOLO:
        """加载模型"""
        resolved_path = self._resolve_model_path(model_identifier)
        cache_key = resolved_path

        with self.model_cache_lock:
            if cache_key in self.models:
                return self.models[cache_key]

        try:
            model = YOLO(resolved_path)
        except Exception as e:
            raise FileNotFoundError(f"无法加载模型 {model_identifier}: {e}")

        with self.model_cache_lock:
            self.models[cache_key] = model

        return model

    def infer(
        self,
        image_path: str,
        model_identifier: str = None,
        confidence: float = None,
        iou_threshold: float = None,
        img_size: int = None
    ) -> Dict[str, Any]:
        """执行推理"""
        start_time = time.time()

        model_identifier = model_identifier or settings.DEFAULT_MODEL
        confidence = confidence or settings.CONFIDENCE_THRESHOLD
        iou_threshold = iou_threshold or settings.IOU_THRESHOLD
        img_size = img_size or settings.DEFAULT_IMG_SIZE

        model = self.load_model(model_identifier)
        results = model.predict(
            source=image_path,
            conf=confidence,
            iou=iou_threshold,
            imgsz=img_size,
            verbose=False
        )

        detections = []
        image_shape = None
        if len(results) > 0:
            boxes = results[0].boxes
            # 获取图片尺寸
            try:
                orig_img = results[0].orig_img
                image_shape = list(orig_img.shape)
            except:
                image_shape = [0, 0, 3]

            for i in range(len(boxes)):
                box = boxes[i]
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        inference_time = time.time() - start_time

        return {
            "success": True,
            "message": "推理完成",
            "detections": detections,
            "inference_time": inference_time,
            "image_shape": image_shape
        }

    def train(self, config: 'TrainingConfig') -> str:
        """开始训练（异步）"""
        task_id = f"train_{int(time.time())}"

        status = TrainingStatus(
            task_id=task_id,
            status="pending",
            progress=0.0,
            current_epoch=0,
            total_epochs=config.epochs,
            created_at=__import__('datetime').datetime.now(),
            updated_at=__import__('datetime').datetime.now()
        )

        with self.training_lock:
            self.training_tasks[task_id] = status

        config_copy = copy.deepcopy(config)
        future = self.executor.submit(self._train_worker, task_id, config_copy)
        self.training_futures[task_id] = future
        print(f"[{task_id}] 训练任务已提交到线程池，等待执行...")

        return task_id

    def resume_training(self, checkpoint_path: str, dataset_path: str = None) -> str:
        """
        从检查点恢复训练

        Args:
            checkpoint_path: 检查点文件路径 (.pt)
            dataset_path: 数据集路径（可选，如果检查点已包含则不需要）

        Returns:
            task_id: 训练任务 ID
        """
        from pathlib import Path

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        # 从检查点获取训练信息
        try:
            checkpoint_model = YOLO(str(checkpoint))
            # Ultralytics 检查点包含训练历史
            if hasattr(checkpoint_model, 'resume'):
                # 使用模型自带的 resume 功能
                pass
        except Exception as e:
            raise ValueError(f"无法加载检查点: {e}")

        # 生成新的任务 ID
        task_id = f"resume_{int(time.time())}"

        # 尝试从检查点获取已训练的轮数
        start_epoch = 0
        if hasattr(checkpoint_model, 'trainer') and checkpoint_model.trainer is not None:
            trainer = checkpoint_model.trainer
            if hasattr(trainer, 'epoch'):
                start_epoch = trainer.epoch
            if hasattr(trainer, 'args') and hasattr(trainer.args, 'epochs'):
                total_epochs = trainer.args.epochs
            else:
                total_epochs = 100
        else:
            total_epochs = 100

        # 创建状态
        status = TrainingStatus(
            task_id=task_id,
            status="pending",
            progress=min(100.0, start_epoch / max(total_epochs, 1) * 100.0),
            current_epoch=start_epoch,
            total_epochs=total_epochs,
            created_at=__import__('datetime').datetime.now(),
            updated_at=__import__('datetime').datetime.now()
        )

        with self.training_lock:
            self.training_tasks[task_id] = status

        # 创建恢复训练配置
        config = TrainingConfig(
            project_name=checkpoint.parent.parent.name,  # 从路径提取项目名
            model_type=checkpoint.parent.parent.name,
            epochs=total_epochs,
            resume=str(checkpoint)  # 设置恢复路径
        )
        if dataset_path:
            config.dataset_path = dataset_path

        config_copy = copy.deepcopy(config)
        future = self.executor.submit(self._train_worker, task_id, config_copy)
        self.training_futures[task_id] = future

        return task_id

    def _train_worker(self, task_id: str, config: 'TrainingConfig'):
        """训练工作函数"""
        print(f"[{task_id}] _train_worker 开始执行")
        try:
            model_type = config.model_type or "yolo11n"
            total_epochs = config.epochs

            print(f"[{task_id}] 开始训练 - 模型: {model_type}")

            # 检查是否是恢复训练
            if config.resume and Path(config.resume).exists():
                print(f"[{task_id}] 从检查点恢复: {config.resume}")
                model = YOLO(config.resume)
            else:
                base_model_path = self._resolve_model_path(
                    config.model_path or f"{model_type}.pt"
                )
                model = YOLO(base_model_path)

            with self.training_lock:
                status = self.training_tasks.get(task_id)
                if status:
                    status.status = "running"

            # 自动调整 batch size (大显存 GPU)
            batch_size = config.batch_size
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_memory >= 14 and config.batch_size <= 16:
                    batch_size = 32
                    print(f"[{task_id}] 自动增大 batch size 到 32 ({total_memory:.1f}GB)")

            # 获取项目路径
            project_dir = settings.MODELS_DIR / config.project_name
            project_dir.mkdir(parents=True, exist_ok=True)

            # 创建 TensorBoard 日志目录
            log_dir = project_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # 初始化 TensorBoard writer
            writer = None
            if SummaryWriter is not None:
                try:
                    writer = SummaryWriter(str(log_dir))
                    print(f"[{task_id}] TensorBoard 日志: {log_dir}")
                except Exception as e:
                    print(f"[{task_id}] 初始化 TensorBoard 失败: {e}")

            # 训练回调 - 收集指标
            def epoch_callback(trainer):
                nonlocal writer
                epoch_index = getattr(trainer, "epoch", 0) + 1

                # 获取损失
                losses = {}
                if hasattr(trainer, 'loss_items'):
                    loss_items = trainer.loss_items
                    if loss_items is not None:
                        losses = {
                            "box_loss": float(loss_items[0]) if len(loss_items) > 0 else 0,
                            "cls_loss": float(loss_items[1]) if len(loss_items) > 1 else 0,
                            "dfl_loss": float(loss_items[2]) if len(loss_items) > 2 else 0
                        }

                # 获取指标
                metrics = getattr(trainer, "metrics", {})

                # TensorBoard 日志记录
                if writer is not None:
                    try:
                        # 记录损失
                        if losses:
                            writer.add_scalars('Loss/box', {'train': losses.get("box_loss", 0)}, epoch_index)
                            writer.add_scalars('Loss/cls', {'train': losses.get("cls_loss", 0)}, epoch_index)
                            writer.add_scalars('Loss/dfl', {'train': losses.get("dfl_loss", 0)}, epoch_index)

                        # 记录性能指标
                        if metrics:
                            if 'metrics/mAP50(B)' in metrics:
                                writer.add_scalars('mAP/mAP50', {'val': metrics['metrics/mAP50(B)']}, epoch_index)
                            if 'metrics/mAP50-95(B)' in metrics:
                                writer.add_scalars('mAP/mAP50-95', {'val': metrics['metrics/mAP50-95(B)']}, epoch_index)
                            if 'metrics/precision(B)' in metrics:
                                writer.add_scalars('Precision', {'val': metrics['metrics/precision(B)']}, epoch_index)
                            if 'metrics/recall(B)' in metrics:
                                writer.add_scalars('Recall', {'val': metrics['metrics/recall(B)']}, epoch_index)

                        # 记录学习率
                        if hasattr(trainer, 'optimizer'):
                            lr = trainer.optimizer.param_groups[0].get('lr', 0)
                            writer.add_scalars('Learning_Rate', {'lr': lr}, epoch_index)

                        writer.flush()
                    except Exception as e:
                        print(f"[{task_id}] TensorBoard 写入失败: {e}")

                # 获取系统统计
                gpu_mem = None
                gpu_util = None
                sys_mem = None
                if torch.cuda.is_available():
                    try:
                        allocated = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved = torch.cuda.memory_reserved(0) / (1024**3)
                        gpu_mem = f"{allocated:.1f}GB / {reserved:.1f}GB"

                        # GPU 利用率 (近似)
                        if hasattr(torch.cuda, 'utilization'):
                            gpu_util = torch.cuda.utilization(0)
                    except:
                        pass

                # 获取系统内存
                try:
                    import psutil
                    process = psutil.Process()
                    sys_mem = process.memory_info().rss / (1024**3)
                except:
                    pass

                with self.training_lock:
                    status = self.training_tasks.get(task_id)
                    if status:
                        status.current_epoch = epoch_index
                        status.progress = min(100.0, epoch_index / max(total_epochs, 1) * 100.0)
                        status.gpu_memory = gpu_mem
                        status.gpu_utilization = gpu_util
                        status.system_memory = sys_mem

                        # 添加指标到历史
                        status.add_metrics(epoch_index, metrics, losses)

                        # 更新最新指标
                        status.metrics = {
                            "latest": metrics,
                            "losses": losses
                        }

            model.add_callback("on_train_epoch_end", epoch_callback)

            # 完成回调
            def finish_callback(trainer):
                nonlocal writer
                # 关闭 TensorBoard writer
                if writer is not None:
                    try:
                        writer.close()
                        print(f"[{task_id}] TensorBoard 日志已保存")
                    except Exception as e:
                        print(f"[{task_id}] 关闭 TensorBoard 失败: {e}")

                with self.training_lock:
                    status = self.training_tasks.get(task_id)
                    if status:
                        status.status = "completed"
                        status.progress = 100.0

                        # 设置最佳检查点路径
                        best_pt = Path(settings.MODELS_DIR) / config.project_name / "train" / "weights" / "best.pt"
                        if best_pt.exists():
                            status.checkpoint_path = str(best_pt)

                        # 最终最佳指标已保存在 best_metrics 中

            model.add_callback("on_train_end", finish_callback)

            # 执行训练
            device = config.device
            if device == "auto":
                device = "0" if torch.cuda.is_available() else "cpu"

            # 自动匹配优化器
            optimizer = config.optimizer
            if optimizer == "auto":
                # 根据模型类型自动选择最佳优化器
                model_type_lower = config.model_type.lower() if config.model_type else ""
                if any(x in model_type_lower for x in ['yolo26', 'yolo11', 'yolov8', 'yolov10']):
                    optimizer = "AdamW"  # YOLO 系列推荐使用 AdamW
                elif any(x in model_type_lower for x in ['yolox', 'yolov5']):
                    optimizer = "SGD"  # YOLOX/YOLOv5 传统上使用 SGD
                else:
                    optimizer = "AdamW"  # 默认使用 AdamW
                print(f"[{task_id}] 自动选择优化器: {optimizer}")

            # 构建训练参数
            train_kwargs = {
                'data': str(config.dataset_path),
                'epochs': config.epochs,
                'batch': batch_size,
                'imgsz': config.img_size,
                'device': device,
                'patience': config.patience,
                'save_period': config.save_period,
                'project': str(settings.MODELS_DIR / config.project_name),
                'name': 'train',
                'exist_ok': True,
                'pretrained': config.pretrained,
                'optimizer': optimizer,
                'amp': config.amp if hasattr(config, 'amp') else True,
                'workers': config.workers if hasattr(config, 'workers') else 8,
                'cache': settings.DEFAULT_CACHE,
                # 学习率参数
                'lr0': getattr(config, 'lr0', 0.01),
                'lrf': getattr(config, 'lrf', 0.01),
                'warmup_epochs': getattr(config, 'warmup_epochs', 3.0),
                'warmup_bias_lr': getattr(config, 'warmup_bias_lr', 0.1),
                # 马赛克增强参数
                'mosaic': getattr(config, 'mosaic', 1.0),
                'multi_scale': getattr(config, 'multi_scale', 0.0),  # 使用 multi_scale 而不是 mosaic_scale
                'close_mosaic': getattr(config, 'close_mosaic', 10),  # 使用 close_mosaic 而不是 close_mosaic_epochs
            }

            results = model.train(**train_kwargs)

            print(f"[{task_id}] 训练完成")

        except Exception as e:
            print(f"[{task_id}] 训练失败: {e}")
            import traceback
            traceback.print_exc()
            with self.training_lock:
                status = self.training_tasks.get(task_id)
                if status:
                    status.status = "failed"
                    status.error_message = str(e)

    def get_training_status(self, task_id: str) -> Optional['TrainingStatus']:
        """获取训练状态"""
        with self.training_lock:
            status = self.training_tasks.get(task_id)
            return copy.deepcopy(status) if status else None

    def list_training_statuses(self) -> List['TrainingStatus']:
        """列出所有训练任务"""
        with self.training_lock:
            return [status.copy(deep=True) for status in self.training_tasks.values()]

    def export_model(self, model_path: str, format: str = "onnx", **kwargs) -> Dict[str, Any]:
        """导出模型"""
        try:
            model = YOLO(model_path)
            export_path = model.export(
                format=format,
                imgsz=kwargs.get("imgsz", 640),
                half=kwargs.get("half", False),
                simplify=kwargs.get("simplify", True),
            )
            return {
                "success": True,
                "message": "导出成功",
                "export_path": str(export_path)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"导出失败: {str(e)}",
                "export_path": None
            }

    def get_device_info(self) -> Tuple[bool, Optional[str]]:
        """获取 GPU 信息"""
        try:
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return True, f"{name} ({memory:.1f}GB)"
            return False, None
        except:
            return False, None


# 数据类型定义 (避免循环导入)
@dataclass
class TrainingStatus:
    """训练状态"""
    task_id: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    created_at: Any
    updated_at: Any
    metrics: Optional[Dict[str, Any]] = None
    metrics_history: List[Dict[str, Any]] = None  # 完整的指标历史
    losses_history: Dict[str, List[float]] = None  # 损失曲线历史
    error_message: Optional[str] = None
    gpu_memory: Optional[str] = None
    gpu_utilization: Optional[float] = None
    system_memory: Optional[float] = None
    best_metrics: Dict[str, float] = None  # 最佳指标
    checkpoint_path: Optional[str] = None

    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []
        if self.losses_history is None:
            self.losses_history = {"box_loss": [], "cls_loss": [], "dfl_loss": []}
        if self.best_metrics is None:
            self.best_metrics = {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0}

    def copy(self, deep: bool = False):
        """创建副本"""
        import copy
        if deep:
            return copy.deepcopy(self)
        else:
            new_status = copy.copy(self)
            new_status.metrics_history = list(self.metrics_history)
            new_status.losses_history = {k: list(v) for k, v in self.losses_history.items()}
            new_status.best_metrics = dict(self.best_metrics)
            return new_status

    def add_metrics(self, epoch: int, metrics: Dict[str, Any], losses: Dict[str, float]):
        """添加指标记录"""
        # 记录损失
        for loss_name, loss_value in losses.items():
            if loss_name in self.losses_history:
                self.losses_history[loss_name].append(loss_value)

        # 记录完整指标
        record = {
            "epoch": epoch,
            "metrics": dict(metrics),
            "losses": dict(losses),
            "timestamp": time.time()
        }
        self.metrics_history.append(record)

        # 更新最佳指标
        if "metrics/mAP50(B)" in metrics:
            if metrics["metrics/mAP50(B)"] > self.best_metrics["mAP50"]:
                self.best_metrics["mAP50"] = metrics["metrics/mAP50(B)"]
        if "metrics/mAP50-95(B)" in metrics:
            if metrics["metrics/mAP50-95(B)"] > self.best_metrics["mAP50-95"]:
                self.best_metrics["mAP50-95"] = metrics["metrics/mAP50-95(B)"]
        if "metrics/precision(B)" in metrics:
            if metrics["metrics/precision(B)"] > self.best_metrics["precision"]:
                self.best_metrics["precision"] = metrics["metrics/precision(B)"]
        if "metrics/recall(B)" in metrics:
            if metrics["metrics/recall(B)"] > self.best_metrics["recall"]:
                self.best_metrics["recall"] = metrics["metrics/recall(B)"]

    def get_chart_data(self) -> Dict[str, Any]:
        """获取图表数据"""
        return {
            "epochs": list(range(1, len(self.metrics_history) + 1)),
            "losses": {
                name: list(values) for name, values in self.losses_history.items()
            },
            "metrics_history": self.metrics_history,
            "best_metrics": self.best_metrics
        }


@dataclass
class TrainingConfig:
    """训练配置"""
    project_name: str
    dataset_path: str
    model_type: str = "yolo11n"
    model_path: Optional[str] = None
    epochs: int = 100
    batch_size: int = 32
    img_size: int = 640
    device: str = "auto"
    patience: int = 100
    save_period: int = -1
    pretrained: bool = True
    optimizer: str = "auto"
    amp: bool = True
    workers: int = 8
    resume: Optional[str] = None  # 恢复训练的检查点路径
    multi_scale: bool = False
    seed: int = 0
    deterministic: bool = False
    plots: bool = True
    val: bool = True

    # ==================== 优化参数 ====================
    # 学习率配置
    lr0: float = 0.01  # 初始学习率
    lrf: float = 0.01  # 最终学习率（相对于 lr0）
    warmup_epochs: float = 3.0  # 预热轮数，设置为 0 可立即使用较高学习率
    warmup_bias_lr: float = 0.1  # 预热期间 bias 的学习率
    momentum: float = 0.937
    weight_decay: float = 0.0005
    cos_lr: bool = False

    # ==================== 数据增强参数 ====================
    # 颜色变换
    hsv_h: float = 0.015  # 色相
    hsv_s: float = 0.7  # 饱和度
    hsv_v: float = 0.4  # 亮度

    # 几何变换
    degrees: float = 0.0  # 旋转角度
    translate: float = 0.1  # 平移比例
    scale: float = 0.5  # 缩放比例
    shear: float = 0.0  # 剪切角度
    perspective: float = 0.0  # 透视变换
    flipud: float = 0.0  # 垂直翻转
    fliplr: float = 0.5  # 水平翻转

    # 高级增强
    mosaic: float = 1.0  # 马赛克增强 (0-1)
    multi_scale: float = 0.0  # 多尺度训练 (0-1)
    close_mosaic: int = 10  # 关闭马赛克增强的轮数
    mixup: float = 0.0  # 混合增强
    copy_paste: float = 0.0  # 复制粘贴（分割任务）
    auto_augment: str = "randaugment"
    dropout: float = 0.0  # 分类任务 dropout

    # ==================== 损失函数权重 ====================
    box: float = 7.5  # 边界框损失权重
    cls: float = 0.5  # 分类损失权重
    dfl: float = 1.5  # 分布焦点损失权重

    # 验证参数
    val_conf: float = 0.001  # 验证时的置信度阈值
    val_iou: float = 0.6  # 验证时的 IoU 阈值
    val_rect: bool = True  # 验证时使用矩形图像

    def copy(self, deep: bool = False):
        """创建副本"""
        import copy
        return copy.deepcopy(self) if deep else copy.copy(self)


class ModelInfo:
    """模型信息"""
    name: str
    path: str
    size: int
    created_at: Any
    model_type: str = "yolo"
    task: str = "detect"
    input_shape: List[int] = None
    classes: List[str] = None


# 全局实例
yolo_engine = YOLOEngine() if TORCH_AVAILABLE else None
