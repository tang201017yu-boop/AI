"""
YOLO 模型服务
"""
import os
import time
import json
import shutil
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

from backend.core.config import settings
from backend.models.schemas import (
    DetectionResult, InferenceResponse, TrainingConfig,
    TrainingStatus, ModelInfo, ExportConfig
)


logger = logging.getLogger(__name__)


class YOLOService:
    """YOLO 模型服务"""
    
    def __init__(self):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics YOLO is not installed")
        
        self.models: Dict[str, YOLO] = {}
        self.training_tasks: Dict[str, TrainingStatus] = {}
        self.training_futures: Dict[str, Future] = {}
        self.training_lock = threading.Lock()
        self.model_cache_lock = threading.Lock()
        self.model_aliases: Dict[str, str] = {}
        self.model_metadata_cache: Dict[str, Tuple[float, ModelInfo]] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=settings.MAX_TRAINING_WORKERS,
            thread_name_prefix="yolo-train"
        )
        self._index_existing_models()
    
    def _index_existing_models(self) -> None:
        for path in self._iter_model_paths():
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                continue
            self.model_aliases.setdefault(resolved.name, str(resolved))

    def _resolve_model_path(self, model_identifier: Optional[str]) -> str:
        """解析模型标识符，返回可用于 YOLO 加载的路径"""
        if not model_identifier:
            default_alias = Path(settings.DEFAULT_MODEL).name
            with self.model_cache_lock:
                mapped = self.model_aliases.get(default_alias)
            return mapped or settings.DEFAULT_MODEL

        identifier = model_identifier.strip()
        alias = Path(identifier).name

        with self.model_cache_lock:
            mapped = self.model_aliases.get(identifier) or self.model_aliases.get(alias)
            if mapped and Path(mapped).exists():
                return str(Path(mapped).resolve())

        candidate = Path(identifier)
        if candidate.exists():
            resolved = str(candidate.resolve())
            with self.model_cache_lock:
                self.model_aliases[alias] = resolved
            return resolved

        if candidate.is_absolute():
            return identifier

        located = self._search_model_in_paths(alias)
        if located:
            resolved = str(located)
            with self.model_cache_lock:
                self.model_aliases[alias] = resolved
            return resolved

        return identifier

    def _search_model_in_paths(self, name: str) -> Optional[Path]:
        search_name = Path(name).name
        for root in settings.model_search_paths:
            if not root.exists():
                continue
            direct = root / search_name
            if direct.exists():
                return direct.resolve()
            nested = next(root.glob(f"**/{search_name}"), None)
            if nested and nested.exists():
                return nested.resolve()
        return None

    def _iter_model_paths(self):
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

    @staticmethod
    def _to_existing_path(candidate: Any) -> Optional[Path]:
        if not candidate:
            return None
        if isinstance(candidate, Path) and candidate.exists():
            return candidate
        if isinstance(candidate, str):
            path = Path(candidate)
            if path.exists():
                return path
        return None

    def _extract_model_file(self, model: 'YOLO') -> Optional[Path]:
        candidates = [
            getattr(model, "ckpt_path", None),
            getattr(model, "weights", None),
        ]
        overrides = getattr(model, "overrides", None)
        if isinstance(overrides, dict):
            candidates.extend([
                overrides.get("weights"),
                overrides.get("model"),
            ])
        for candidate in candidates:
            path = self._to_existing_path(candidate)
            if path is not None:
                return path
        return None

    def _register_model_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            return
        with self.model_cache_lock:
            self.model_aliases[resolved.name] = str(resolved)

    def _cache_model_file(self, model_identifier: Optional[str], source_path: Path) -> None:
        try:
            target_dir = settings.MODELS_DIR
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_path.name
            if not target_path.exists() or source_path.stat().st_mtime > target_path.stat().st_mtime:
                shutil.copy2(source_path, target_path)
            alias = Path(model_identifier).name if model_identifier else source_path.name
            self._register_model_file(target_path)
            with self.model_cache_lock:
                self.model_aliases[alias] = str(target_path.resolve())
        except Exception as exc:  # pragma: no cover - best effort caching
            logger.debug("Failed to cache model %s: %s", source_path, exc)

    def _extract_model_classes(self, model: 'YOLO') -> List[str]:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            try:
                return [names[key] for key in sorted(names, key=lambda k: int(k))]
            except Exception:
                return list(names.values())
        if isinstance(names, list):
            return [str(item) for item in names if item is not None]
        return []

    def _update_model_metadata_cache(self, path: Path, model: Optional['YOLO']) -> Optional[ModelInfo]:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None

        task = "detect"
        classes: Optional[List[str]] = None
        if model is not None:
            task = getattr(model, "task", "detect") or "detect"
            classes = self._extract_model_classes(model)

        info = ModelInfo(
            name=path.name,
            path=str(path),
            size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            model_type="yolo",
            task=task,
            input_shape=None,
            classes=classes
        )
        with self.model_cache_lock:
            self.model_metadata_cache[str(path)] = (stat.st_mtime, info)
            self.model_aliases.setdefault(path.name, str(path))
        return info

    def _get_cached_model_info(self, path: Path) -> Optional[ModelInfo]:
        key = str(path)
        with self.model_cache_lock:
            cached = self.model_metadata_cache.get(key)
        if not cached:
            return None
        cached_mtime, info = cached
        try:
            current_mtime = path.stat().st_mtime
        except FileNotFoundError:
            with self.model_cache_lock:
                self.model_metadata_cache.pop(key, None)
            return None
        if cached_mtime == current_mtime:
            return info.copy(deep=True)
        with self.model_cache_lock:
            self.model_metadata_cache.pop(key, None)
        return None

    def _load_model_metadata(self, path: Path) -> Optional[ModelInfo]:
        try:
            model = YOLO(str(path))
            info = self._update_model_metadata_cache(path, model)
            return info or self._get_cached_model_info(path)
        except Exception as exc:  # pragma: no cover - metadata extraction best effort
            logger.debug("Failed to load metadata for %s: %s", path, exc)
            try:
                stat = path.stat()
            except FileNotFoundError:
                return None
            info = ModelInfo(
                name=path.name,
                path=str(path),
                size=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                model_type="yolo",
                task="unknown",
                input_shape=None,
                classes=None
            )
            with self.model_cache_lock:
                self.model_metadata_cache[str(path)] = (stat.st_mtime, info)
            return info

    def _register_training_artifacts(self, save_dir: Path) -> None:
        weights_dir = save_dir / "weights"
        if not weights_dir.exists():
            return
        for filename in ("best.pt", "last.pt"):
            weight_path = weights_dir / filename
            if not weight_path.exists():
                continue
            self._register_model_file(weight_path)
            self._load_model_metadata(weight_path)
    
    def load_model(self, model_identifier: Optional[str]) -> 'YOLO':
        """加载或获取缓存模型"""
        resolved_path = self._resolve_model_path(model_identifier)
        cache_key = resolved_path

        with self.model_cache_lock:
            cached_model = self.models.get(cache_key)
            if cached_model is not None:
                return cached_model

        try:
            model = YOLO(resolved_path)
        except Exception as e:
            identifier = model_identifier or settings.DEFAULT_MODEL
            raise FileNotFoundError(f"无法加载模型 {identifier}: {e}")

        weight_path = self._extract_model_file(model)
        if weight_path and weight_path.exists():
            cache_key = str(weight_path.resolve())
            self._cache_model_file(model_identifier, weight_path)
            self._update_model_metadata_cache(weight_path, model)
        else:
            candidate_path = Path(resolved_path)
            if candidate_path.exists():
                self._register_model_file(candidate_path)
                self._update_model_metadata_cache(candidate_path, model)

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
    ) -> InferenceResponse:
        """执行推理"""
        try:
            start_time = time.time()
            
            # 使用默认值
            model_identifier = model_identifier or settings.DEFAULT_MODEL
            confidence = confidence or settings.CONFIDENCE_THRESHOLD
            iou_threshold = iou_threshold or settings.IOU_THRESHOLD
            img_size = img_size or settings.DEFAULT_IMG_SIZE
            
            # 加载模型
            model = self.load_model(model_identifier)
            
            # 执行推理
            results = model.predict(
                source=image_path,
                conf=confidence,
                iou=iou_threshold,
                imgsz=img_size,
                verbose=False
            )
            
            # 解析结果
            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    
                    detection = DetectionResult(
                        class_id=cls_id,
                        class_name=model.names[cls_id],
                        confidence=conf,
                        bbox=xyxy
                    )
                    detections.append(detection)
            
            inference_time = time.time() - start_time
            
            # 获取图像尺寸
            img = Image.open(image_path)
            image_shape = [img.height, img.width, 3]
            
            return InferenceResponse(
                success=True,
                message="Inference completed successfully",
                detections=detections,
                inference_time=inference_time,
                image_shape=image_shape
            )
            
        except Exception as e:
            return InferenceResponse(
                success=False,
                message=f"Inference failed: {str(e)}",
                detections=[],
                inference_time=0.0,
                image_shape=[0, 0, 0]
            )
    
    def _train_worker(self, task_id: str, config: TrainingConfig):
        """执行实际训练任务的工作函数"""
        model = None
        epoch_callback = None

        # GPU 优化：RTX 5080 (Ada Lovelace 架构) 特定优化
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            is_rtx_5080 = "5080" in gpu_name

            # 启用 cuDNN benchmark (卷积算法自动优化)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

            # TF32 on Ada Lovelace 加速 (FP32 计算加速 8x)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # 启用 Flash Attention (如果可用)
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass

            # 根据显存自动调整 batch size
            if config.device in ["0", "cuda", "auto"]:
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if is_rtx_5080 or total_memory >= 14:  # 16GB 显存
                    if config.batch_size <= 16:
                        print(f"[{task_id}] RTX 5080 ({total_memory:.1f}GB) - 自动增大 batch size 为 32")
                        # 注意: 这里不能修改 config，需要在传递参数时处理

        def sanitize_metrics(metrics_dict: Any) -> Dict[str, float]:
            sanitized: Dict[str, float] = {}
            if isinstance(metrics_dict, dict):
                for key, value in metrics_dict.items():
                    try:
                        if isinstance(value, (int, float)):
                            sanitized[key] = float(value)
                        elif hasattr(value, "item"):
                            sanitized[key] = float(value.item())
                        elif hasattr(value, "detach"):
                            sanitized[key] = float(value.detach().cpu().item())
                        elif isinstance(value, (list, tuple)) and value:
                            sanitized[key] = float(value[-1])
                        elif isinstance(value, np.ndarray):
                            sanitized[key] = float(value.mean())
                    except Exception:
                        continue
            return sanitized

        try:
            model_type = config.model_type or "yolo11n"
            total_epochs = config.epochs or settings.DEFAULT_EPOCHS

            print(f"[{task_id}] 开始训练")
            print(f"[{task_id}] 模型类型: {model_type}")

            # 解析数据集路径 - 支持数据集名称或完整路径
            dataset_path_resolved = Path(config.dataset_path)
            if not dataset_path_resolved.exists():
                # 尝试作为数据集名称解析
                for search_path in settings.dataset_search_paths:
                    candidate = search_path / config.dataset_path
                    if candidate.exists():
                        dataset_path_resolved = candidate
                        break
                    # 也检查 data.yaml
                    yaml_candidate = search_path / config.dataset_path / "data.yaml"
                    if yaml_candidate.exists():
                        dataset_path_resolved = yaml_candidate
                        break

            if not dataset_path_resolved.exists():
                raise FileNotFoundError(f"数据集不存在: {config.dataset_path}")

            # 如果是 data.yaml 文件，直接使用；否则查找 data.yaml
            if dataset_path_resolved.is_file() and dataset_path_resolved.name in ['data.yaml', 'data.yml']:
                config.dataset_path = str(dataset_path_resolved)
            else:
                config.dataset_path = str(dataset_path_resolved / "data.yaml")

            print(f"[{task_id}] 数据集路径: {config.dataset_path}")

            base_model_identifier = config.model_path or (
                f"{model_type}.pt" if config.pretrained else f"{model_type}.yaml"
            )
            base_model_path = self._resolve_model_path(base_model_identifier)
            print(f"[{task_id}] 基础模型: {base_model_path}")

            model = YOLO(base_model_path)

            with self.training_lock:
                status = self.training_tasks.get(task_id)
                if status:
                    status.status = "running"
                    status.progress = 0.0
                    status.total_epochs = total_epochs
                    status.updated_at = datetime.now()

            def _epoch_callback(trainer):
                epoch_index = getattr(trainer, "epoch", 0) + 1
                metrics = sanitize_metrics(getattr(trainer, "metrics", {}))

                # 获取 GPU 内存使用情况
                gpu_memory_str = None
                if torch.cuda.is_available():
                    try:
                        allocated = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved = torch.cuda.memory_reserved(0) / (1024**3)
                        gpu_memory_str = f"{allocated:.1f}GB / {reserved:.1f}GB"
                    except:
                        pass

                with self.training_lock:
                    status_inner = self.training_tasks.get(task_id)
                    if not status_inner:
                        return
                    status_inner.current_epoch = epoch_index
                    status_inner.progress = min(100.0, epoch_index / max(total_epochs, 1) * 100.0)
                    if metrics:
                        status_inner.metrics = {"latest": metrics}
                    if gpu_memory_str:
                        status_inner.gpu_memory = gpu_memory_str
                    status_inner.updated_at = datetime.now()

            epoch_callback = _epoch_callback
            model.add_callback("on_train_epoch_end", epoch_callback)

            print(f"[{task_id}] 模型加载成功，开始训练...")

            # GPU 优化配置
            device = config.device
            if device == "auto":
                device = "0" if torch.cuda.is_available() else "cpu"

            # 多 GPU 支持
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    print(f"[{task_id}] 检测到 {num_gpus} 个 GPU，使用多 GPU 训练")
                    device = list(range(num_gpus))  # 使用所有 GPU

            # RTX 5080 自动优化 batch size
            batch_size = config.batch_size
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_memory >= 14:  # 16GB+ 显存
                    if config.batch_size <= 16:
                        batch_size = 32
                        print(f"[{task_id}] RTX 5080 ({total_memory:.1f}GB) - batch size 自动调整为 32")
                    elif config.batch_size < 64:
                        batch_size = 64
                        print(f"[{task_id}] 大显存 ({total_memory:.1f}GB) - batch size 自动调整为 64")

            # 训练参数 - 包含所有 GPU 优化
            train_kwargs = {
                "data": str(config.dataset_path),
                "epochs": config.epochs,
                "batch": batch_size,
                "imgsz": config.img_size,
                "device": device,
                "patience": config.patience,
                "save_period": config.save_period,
                "project": str(settings.MODELS_DIR / config.project_name),
                "name": "train",
                "exist_ok": True,
                "pretrained": config.pretrained,
                "optimizer": config.optimizer,
                "lr0": config.lr0,
                "lrf": config.lrf,
                "verbose": True,
                # GPU 优化参数
                "amp": config.amp,  # 混合精度训练
                "workers": config.workers,  # 数据加载线程
                "cache": config.cache,  # 缓存
                "rect": config.rect,  # 矩形训练
                "cos_lr": config.cos_lr,  # 余弦学习率
                "close_mosaic": config.close_mosaic,  # 最后N个epoch关闭mosaic
            }

            # 如果是单 GPU 且使用 pin_memory，不需要单独设置
            # YOLO 会自动处理

            results = model.train(**train_kwargs)

            save_dir = getattr(results, "save_dir", None)
            if save_dir:
                try:
                    save_dir_path = Path(save_dir)
                    if save_dir_path.exists():
                        self._register_training_artifacts(save_dir_path)
                except Exception as artifact_exc:  # pragma: no cover
                    logger.debug("Failed to register training artifacts: %s", artifact_exc)

            print(f"[{task_id}] 训练完成")

            final_metrics = sanitize_metrics(getattr(results, "results_dict", {}))

            with self.training_lock:
                status = self.training_tasks.get(task_id)
                if status:
                    status.status = "completed"
                    status.progress = 100.0
                    status.current_epoch = total_epochs
                    status.metrics = {"final_metrics": final_metrics}
                    status.updated_at = datetime.now()

        except Exception as e:
            print(f"[{task_id}] 训练失败: {e}")
            import traceback
            traceback.print_exc()

            with self.training_lock:
                status = self.training_tasks.get(task_id)
                if status:
                    status.status = "failed"
                    status.error_message = str(e)
                    status.updated_at = datetime.now()
            raise

        finally:
            if model is not None and epoch_callback is not None:
                try:
                    model.remove_callback("on_train_epoch_end", epoch_callback)
                except Exception:
                    try:
                        callbacks = model.callbacks.get("on_train_epoch_end", [])
                        if epoch_callback in callbacks:
                            callbacks.remove(epoch_callback)
                    except Exception:
                        pass
    
    def train(self, config: TrainingConfig) -> str:
        """开始训练（异步）"""
        task_id = f"train_{int(time.time())}"

        status = TrainingStatus(
            task_id=task_id,
            status="pending",
            progress=0.0,
            current_epoch=0,
            total_epochs=config.epochs,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        with self.training_lock:
            self.training_tasks[task_id] = status

        config_copy = config.copy(deep=True)
        future = self.executor.submit(self._train_worker, task_id, config_copy)

        with self.training_lock:
            self.training_futures[task_id] = future

        future.add_done_callback(lambda f, tid=task_id: self._handle_training_completion(tid, f))

        print(f"训练任务已创建: {task_id}")

        return task_id

    def _handle_training_completion(self, task_id: str, future: Future):
        """训练任务完成后的清理操作"""
        exc = future.exception()
        with self.training_lock:
            self.training_futures.pop(task_id, None)
            status = self.training_tasks.get(task_id)

        if exc:
            print(f"[{task_id}] 训练任务异常: {exc}")
            if status and status.status != "failed":
                with self.training_lock:
                    status.status = "failed"
                    status.error_message = str(exc)
                    status.updated_at = datetime.now()

    def get_training_status(self, task_id: str) -> Optional[TrainingStatus]:
        """获取训练状态"""
        with self.training_lock:
            status = self.training_tasks.get(task_id)
            return status.copy(deep=True) if status else None

    def list_training_statuses(self) -> List[TrainingStatus]:
        """获取所有训练任务状态快照"""
        with self.training_lock:
            return [status.copy(deep=True) for status in self.training_tasks.values()]
    
    def export_model(self, config: ExportConfig) -> Dict[str, Any]:
        """导出模型"""
        try:
            model = YOLO(config.model_path)
            
            export_path = model.export(
                format=config.format,
                imgsz=config.img_size,
                batch=config.batch_size,
                optimize=config.optimize,
                half=config.half,
                simplify=config.simplify,
                dynamic=config.dynamic,
                opset=config.opset
            )
            
            return {
                "success": True,
                "message": "Model exported successfully",
                "export_path": str(export_path)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Export failed: {str(e)}",
                "export_path": None
            }
    
    def list_models(self) -> List[ModelInfo]:
        """列出所有模型"""
        model_infos: List[ModelInfo] = []
        for path in self._iter_model_paths():
            info = self._get_cached_model_info(path)
            if info is None:
                info = self._load_model_metadata(path)
            if info:
                model_infos.append(info)
        model_infos.sort(key=lambda item: item.name.lower())
        return model_infos
    
    def get_device_info(self) -> Tuple[bool, Optional[str]]:
        """获取设备信息"""
        try:
            gpu_available = torch.cuda.is_available()
            gpu_info = None
            if gpu_available:
                device_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_info = f"{device_name} ({total_memory:.1f}GB)"
            return gpu_available, gpu_info
        except:
            return False, None


# 全局服务实例
yolo_service = YOLOService() if ULTRALYTICS_AVAILABLE else None
