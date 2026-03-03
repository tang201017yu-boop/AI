# -*- coding: utf-8 -*-
"""
YOLO 引擎模块 - YOLO Model Engine
处理模型加载、推理和训练的核心功能模块
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

# 尝试导入 PyTorch 和 YOLO
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True

    # TensorBoard 可能不存在，单独处理
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None

    # 检测GPU显存，如果不足则禁用CUDA
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 2:
            print(f"[YOLO引擎] GPU显存不足 ({gpu_memory:.1f}GB < 2GB)，禁用CUDA")
            torch.cuda.is_available = lambda: False
except ImportError as e:
    TORCH_AVAILABLE = False
    SummaryWriter = None
    print(f"Warning: torch/ultralytics not installed - {e}")
except Exception as e:
    TORCH_AVAILABLE = False
    SummaryWriter = None
    print(f"Warning: torch/ultralytics initialization failed - {e}")

from .config import settings

# 创建日志记录器
logger = logging.getLogger(__name__)


class YOLOEngine:
    """
    YOLO 模型引擎类
    提供模型加载、推理、训练等核心功能
    支持模型缓存、异步训练、多设备等功能
    """

    def __init__(self):
        """
        初始化 YOLO 引擎
        - 检查 PyTorch 可用性
        - 配置 GPU/CPU 设备
        - 初始化线程池
        - 索引已有模型
        """
        logger.info("[YOLO引擎] 初始化 YOLO 引擎")

        # 检查 PyTorch 是否可用
        if not TORCH_AVAILABLE:
            error_msg = "PyTorch is not available"
            logger.error(f"[YOLO引擎] {error_msg}")
            raise ImportError(error_msg)

        # 模型缓存字典（key: 模型路径+设备, value: YOLO模型对象）
        self.models: Dict[str, YOLO] = {}

        # 训练任务字典（key: 任务ID, value: 训练状态）
        self.training_tasks: Dict[str, 'TrainingStatus'] = {}

        # 训练任务 Future 对象字典
        self.training_futures: Dict[str, Future] = {}

        # 训练任务线程锁（保证线程安全）
        self.training_lock = threading.Lock()

        # 模型缓存锁（保证模型加载线程安全）
        self.model_cache_lock = threading.Lock()

        # 模型别名字典（用于快速查找模型）
        self.model_aliases: Dict[str, str] = {}

        # 模型元数据缓存
        self.model_metadata_cache: Dict[str, Tuple[float, 'ModelInfo']] = {}

        # 训练线程池（最多同时训练 MAX_TRAINING_WORKERS 个任务）
        self.executor = ThreadPoolExecutor(
            max_workers=settings.MAX_TRAINING_WORKERS,
            thread_name_prefix="yolo-train"
        )

        # 默认使用 GPU（如果可用）
        self.default_device = "0" if torch.cuda.is_available() else "cpu"
        logger.info(f"[YOLO引擎] 默认设备: {self.default_device}")

        # 清空 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # GPU 优化配置（仅在有足够显存时启用）
        if torch.cuda.is_available():
            try:
                # 检查GPU显存是否足够（至少4GB）
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory >= 4:
                    logger.info(f"[YOLO引擎] GPU显存充足 ({gpu_memory:.1f}GB)，启用优化")
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    try:
                        torch.backends.cuda.enable_flash_sdp(True)
                    except:
                        pass
                else:
                    logger.warning(f"[YOLO引擎] GPU显存不足 ({gpu_memory:.1f}GB)，使用CPU")
            except:
                pass

        # 索引已存在的模型文件
        self._index_existing_models()
        logger.info("[YOLO引擎] 初始化完成")

    def _index_existing_models(self) -> None:
        """
        索引已存在的模型文件
        扫描模型搜索路径，建立模型别名映射
        """
        logger.debug("[YOLO引擎] 索引已有模型")

        for path in self._iter_model_paths():
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                continue
            # 建立模型名到路径的映射
            self.model_aliases.setdefault(resolved.name, str(resolved))

        logger.info(f"[YOLO引擎] 已索引 {len(self.model_aliases)} 个模型")

    def _iter_model_paths(self):
        """
        迭代所有模型文件路径
        遍历配置中的模型搜索路径，查找 .pt 文件

        Yields:
            Path: 模型文件路径
        """
        seen = set()  # 已处理的文件路径集合

        for root in settings.model_search_paths:
            # 跳过不存在的路径
            if not root.exists():
                continue

            # 递归查找所有 .pt 文件
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
        """
        解析模型标识符为实际文件路径
        支持：模型名、别名、完整路径

        Args:
            model_identifier: 模型标识符（名称/路径）

        Returns:
            str: 模型的完整文件路径
        """
        # 如果没有指定，使用默认模型
        if not model_identifier:
            return settings.DEFAULT_MODEL

        # 去除首尾空格
        identifier = model_identifier.strip()

        # 提取文件名（不含路径和扩展名）
        alias = Path(identifier).name

        # 从缓存中查找
        with self.model_cache_lock:
            mapped = self.model_aliases.get(identifier) or self.model_aliases.get(alias)
            if mapped and Path(mapped).exists():
                return str(Path(mapped).resolve())

        # 尝试直接作为路径
        candidate = Path(identifier)
        if candidate.exists():
            return str(candidate.resolve())

        # 在配置的搜索路径中查找
        for root in settings.model_search_paths:
            # 直接查找
            direct = root / alias
            if direct.exists():
                return str(direct.resolve())
            # 嵌套查找
            nested = next(root.glob(f"**/{alias}"), None)
            if nested and nested.exists():
                return str(nested.resolve())

        # 如果都找不到，返回原始标识符
        logger.warning(f"[YOLO引擎] 模型未找到: {model_identifier}")
        return identifier

    def load_model(self, model_identifier: Optional[str] = None, device: str = None) -> 'YOLO':
        """
        加载 YOLO 模型（带缓存）

        Args:
            model_identifier: 模型名称或路径
            device: 设备 "cpu" 或 "cuda"/"0"，默认使用 CPU

        Returns:
            YOLO: 加载的模型对象
        """
        # 默认使用 CPU
        if device is None:
            device = "cpu"

        # 解析模型路径
        resolved_path = self._resolve_model_path(model_identifier)

        # 生成缓存键（路径+设备组合）
        cache_key = f"{resolved_path}_{device}"

        # 检查缓存
        with self.model_cache_lock:
            if cache_key in self.models:
                logger.debug(f"[YOLO引擎] 从缓存加载模型: {resolved_path}")
                return self.models[cache_key]

        # 加载模型
        try:
            logger.info(f"[YOLO引擎] 加载模型: {resolved_path}, 设备: {device}")
            model = YOLO(resolved_path)
            # 移动到指定设备（如果可用）
            if device and device != "cpu":
                try:
                    model.to(device)
                except Exception as e:
                    logger.warning(f"[YOLO引擎] 无法移动到设备 {device}，使用 CPU: {e}")
                    model.to("cpu")
            logger.info(f"[YOLO引擎] 模型加载成功")
        except Exception as e:
            error_msg = f"无法加载模型 {model_identifier}: {e}"
            logger.error(f"[YOLO引擎] {error_msg}")
            raise FileNotFoundError(error_msg)

        # 存入缓存
        with self.model_cache_lock:
            self.models[cache_key] = model

        return model

    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """
        获取所有已加载的模型列表

        Returns:
            List: 已加载模型的信息列表
        """
        models_info = []
        with self.model_cache_lock:
            for cache_key, model in self.models.items():
                try:
                    # 提取模型名称
                    model_name = cache_key.split('_')[0] if '_' in cache_key else cache_key

                    # 获取模型信息
                    info = {
                        "cache_key": cache_key,
                        "model_name": model_name,
                        "task": getattr(model, 'task', 'detect'),
                        "device": cache_key.split('_')[-1] if '_' in cache_key else 'cpu',
                        "loaded": True,
                    }

                    # 尝试获取类别信息
                    try:
                        if hasattr(model, 'model') and hasattr(model.model, 'names'):
                            info["classes"] = list(model.model.names.values()) if model.model.names else []
                            info["num_classes"] = len(model.model.names)
                    except:
                        pass

                    models_info.append(info)
                except Exception as e:
                    logger.warning(f"[YOLO引擎] 获取模型信息失败 {cache_key}: {str(e)}")

        return models_info

    def get_model_status(self, model_identifier: str = None, device: str = None) -> Dict[str, Any]:
        """
        获取模型加载状态

        Args:
            model_identifier: 模型标识符
            device: 设备

        Returns:
            Dict: 模型状态信息
        """
        if device is None:
            device = "cpu"

        resolved_path = self._resolve_model_path(model_identifier)
        cache_key = f"{resolved_path}_{device}"

        with self.model_cache_lock:
            is_loaded = cache_key in self.models

        return {
            "model_identifier": model_identifier,
            "device": device,
            "loaded": is_loaded,
            "cache_key": cache_key,
        }

    def infer(
        self,
        image_path: str,
        model_identifier: str = None,
        confidence: float = None,
        iou_threshold: float = None,
        img_size: int = None,
        device: str = None
    ) -> Dict[str, Any]:
        """
        执行图片推理

        Args:
            image_path: 图片文件路径
            model_identifier: 模型名称
            confidence: 置信度阈值
            iou_threshold: IOU阈值
            img_size: 输入尺寸
            device: 设备 "cpu" 或 "cuda"，默认使用CPU

        Returns:
            Dict: 推理结果，包含检测框、置信度、类别等信息
        """
        logger.info(f"[YOLO引擎] 开始推理: {image_path}")
        start_time = time.time()

        # 默认使用 CPU（避免 CUDA 内存问题）
        device = device or self.default_device

        # 使用默认配置
        model_identifier = model_identifier or settings.DEFAULT_MODEL
        confidence = confidence or settings.CONFIDENCE_THRESHOLD
        iou_threshold = iou_threshold or settings.IOU_THRESHOLD
        img_size = img_size or settings.DEFAULT_IMG_SIZE

        # 加载模型
        model = self.load_model(model_identifier, device=device)

        # 执行推理
        logger.debug(f"[YOLO引擎] 推理参数: conf={confidence}, iou={iou_threshold}, imgsz={img_size}, device={device}")

        results = model.predict(
            source=image_path,
            conf=confidence,
            iou=iou_threshold,
            imgsz=img_size,
            device=device,
            verbose=False
        )

        # 处理推理结果
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

            # 遍历检测框
            for i in range(len(boxes)):
                box = boxes[i]
                detections.append({
                    "class_id": int(box.cls[0]),  # 类别 ID
                    "class_name": model.names[int(box.cls[0])],  # 类别名称
                    "confidence": float(box.conf[0]),  # 置信度
                    "bbox": box.xyxy[0].tolist()  # 边界框坐标 [x1, y1, x2, y2]
                })

        # 计算推理耗时
        inference_time = time.time() - start_time
        logger.info(f"[YOLO引擎] 推理完成: {len(detections)} 个检测, 耗时 {inference_time:.3f}s")

        return {
            "success": True,
            "message": "推理完成",
            "detections": detections,
            "inference_time": inference_time,
            "image_shape": image_shape
        }

    def train(self, config: 'TrainingConfig') -> str:
        """
        开始训练（异步）

        Args:
            config: 训练配置对象

        Returns:
            str: 训练任务 ID
        """
        # 生成任务 ID
        task_id = f"train_{int(time.time())}"
        logger.info(f"[YOLO引擎] 开始训练: task_id={task_id}")

        # 创建训练状态对象
        status = TrainingStatus(
            task_id=task_id,
            status="pending",  # 等待中
            progress=0.0,
            current_epoch=0,
            total_epochs=config.epochs,
            project_name=config.project_name,
            created_at=__import__('datetime').datetime.now(),
            updated_at=__import__('datetime').datetime.now()
        )

        # 保存训练任务状态
        with self.training_lock:
            self.training_tasks[task_id] = status

        # 复制配置（避免后续修改影响）
        config_copy = copy.deepcopy(config)

        # 提交训练任务到线程池
        future = self.executor.submit(self._train_worker, task_id, config_copy)
        self.training_futures[task_id] = future

        logger.info(f"[YOLO引擎] 训练任务已提交到线程池: task_id={task_id}")

        return task_id

    def resume_training(self, checkpoint_path: str, dataset_path: str = None) -> str:
        """
        从检查点恢复训练

        Args:
            checkpoint_path: 检查点文件路径 (.pt)
            dataset_path: 数据集路径（可选，如果检查点已包含则不需要）

        Returns:
            str: 训练任务 ID
        """
        from pathlib import Path

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            error_msg = f"检查点文件不存在: {checkpoint_path}"
            logger.error(f"[YOLO引擎] {error_msg}")
            raise FileNotFoundError(error_msg)

        # 加载检查点模型
        try:
            checkpoint_model = YOLO(str(checkpoint))
        except Exception as e:
            error_msg = f"无法加载检查点: {e}"
            logger.error(f"[YOLO引擎] {error_msg}")
            raise ValueError(error_msg)

        # 生成新的任务 ID
        task_id = f"resume_{int(time.time())}"
        logger.info(f"[YOLO引擎] 恢复训练: task_id={task_id}, checkpoint={checkpoint_path}")

        # 尝试从检查点获取已训练的轮数
        start_epoch = 0
        total_epochs = 100  # 默认值

        if hasattr(checkpoint_model, 'trainer') and checkpoint_model.trainer is not None:
            trainer = checkpoint_model.trainer
            if hasattr(trainer, 'epoch'):
                start_epoch = trainer.epoch
            if hasattr(trainer, 'args') and hasattr(trainer.args, 'epochs'):
                total_epochs = trainer.args.epochs

        # 创建训练状态
        status = TrainingStatus(
            task_id=task_id,
            status="pending",
            progress=min(100.0, start_epoch / max(total_epochs, 1) * 100.0),
            current_epoch=start_epoch,
            total_epochs=total_epochs,
            project_name=checkpoint.parent.parent.name,
            created_at=__import__('datetime').datetime.now(),
            updated_at=__import__('datetime').datetime.now()
        )

        with self.training_lock:
            self.training_tasks[task_id] = status

        # 创建恢复训练配置
        config = TrainingConfig(
            project_name=checkpoint.parent.parent.name,
            model_type=checkpoint.parent.parent.name,
            epochs=total_epochs,
            resume=str(checkpoint)
        )
        if dataset_path:
            config.dataset_path = dataset_path

        # 提交训练任务
        config_copy = copy.deepcopy(config)
        future = self.executor.submit(self._train_worker, task_id, config_copy)
        self.training_futures[task_id] = future

        logger.info(f"[YOLO引擎] 恢复训练任务已提交: task_id={task_id}")

        return task_id

    def cancel_training(self, task_id: str) -> Dict[str, Any]:
        """
        取消训练任务

        Args:
            task_id: 训练任务 ID

        Returns:
            Dict: 操作结果
        """
        logger.info(f"[YOLO引擎] 尝试取消训练: task_id={task_id}")

        with self.training_lock:
            # 检查任务是否存在
            if task_id not in self.training_tasks:
                return {"success": False, "message": f"任务 {task_id} 不存在"}

            # 更新任务状态
            status = self.training_tasks.get(task_id)
            if status:
                status.status = "cancelled"
                status.error_message = "用户取消训练"

            # 尝试取消 future
            future = self.training_futures.get(task_id)
            if future:
                cancelled = future.cancel()
                logger.info(f"[YOLO引擎] 任务取消结果: {cancelled}")
                if not cancelled:
                    # 如果无法取消，标记为已请求取消
                    return {"success": True, "message": "已发送取消请求", "cancelled": False}
                return {"success": True, "message": "训练已取消", "cancelled": True}

            return {"success": True, "message": "任务已标记为取消"}

    def _train_worker(self, task_id: str, config: 'TrainingConfig'):
        """
        训练工作函数（在线程池中执行）

        Args:
            task_id: 训练任务 ID
            config: 训练配置
        """
        logger.info(f"[YOLO引擎] _train_worker 开始执行: task_id={task_id}")

        try:
            # 获取模型类型和训练轮数
            model_type = config.model_type or "yolo11n"
            total_epochs = config.epochs

            logger.info(f"[YOLO引擎] 训练配置: 模型={model_type}, epochs={total_epochs}")

            # 检查是否是恢复训练
            if config.resume and Path(config.resume).exists():
                logger.info(f"[YOLO引擎] 从检查点恢复: {config.resume}")
                model = YOLO(config.resume)
            else:
                # 加载基础模型
                base_model_path = self._resolve_model_path(
                    config.model_path or f"{model_type}.pt"
                )
                model = YOLO(base_model_path)

            # 更新训练状态
            with self.training_lock:
                status = self.training_tasks.get(task_id)
                if status:
                    status.status = "running"

            # 自动调整 batch size (根据 GPU 显存)
            batch_size = config.batch_size
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_memory >= 14 and config.batch_size <= 16:
                    batch_size = 32
                    logger.info(f"[YOLO引擎] 自动增大 batch size 到 32 ({total_memory:.1f}GB)")

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
                    logger.info(f"[YOLO引擎] TensorBoard 日志: {log_dir}")
                except Exception as e:
                    logger.warning(f"[YOLO引擎] 初始化 TensorBoard 失败: {e}")

            # 训练回调 - 收集指标
            epoch_counter = [0]  # 使用列表以便在回调中修改
            trainer_ref = [None]  # 保存 trainer 引用

            def train_start_callback(trainer):
                """训练开始时的回调，用于获取 trainer 引用"""
                trainer_ref[0] = trainer
                logger.info(f"[YOLO引擎] 训练已开始，获取 trainer 引用")

            def epoch_callback(trainer):
                try:
                    nonlocal writer
                    epoch_counter[0] += 1
                    epoch_index = epoch_counter[0]

                    # 获取训练状态
                    with self.training_lock:
                        status = self.training_tasks.get(task_id)
                        if not status:
                            logger.warning(f"[YOLO引擎] 任务状态不存在: {task_id}")
                            return

                        # 更新进度
                        status.current_epoch = epoch_index
                        status.progress = min(100.0, epoch_index / max(total_epochs, 1) * 100.0)
                        status.status = "running"

                        logger.info(f"[YOLO引擎] 更新训练进度: epoch={epoch_index}/{total_epochs}, progress={status.progress:.1f}%")

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
                            logger.warning(f"[YOLO引擎] TensorBoard 写入失败: {e}")

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

                    # 更新训练状态
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

                except Exception as e:
                    logger.error(f"[YOLO引擎] Epoch 回调执行失败: {e}")

            # 注册训练回调 - 尝试多个回调事件
            model.add_callback("on_train_start", train_start_callback)
            model.add_callback("on_train_epoch_end", epoch_callback)
            model.add_callback("on_fit_epoch_end", epoch_callback)

            logger.info(f"[YOLO引擎] 已注册训练回调")

            # 完成回调
            def finish_callback(trainer):
                nonlocal writer
                # 关闭 TensorBoard writer
                if writer is not None:
                    try:
                        writer.close()
                        logger.info(f"[YOLO引擎] TensorBoard 日志已保存")
                    except Exception as e:
                        logger.warning(f"[YOLO引擎] 关闭 TensorBoard 失败: {e}")

                with self.training_lock:
                    status = self.training_tasks.get(task_id)
                    if status:
                        status.status = "completed"
                        status.progress = 100.0

                        # 设置最佳检查点路径
                        best_pt = Path(settings.MODELS_DIR) / config.project_name / "train" / "weights" / "best.pt"
                        if best_pt.exists():
                            status.checkpoint_path = str(best_pt)

            # 注册完成回调
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
                logger.info(f"[YOLO引擎] 自动选择优化器: {optimizer}")

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
                'multi_scale': getattr(config, 'multi_scale', 0.0),
                'close_mosaic': getattr(config, 'close_mosaic', 10),
            }

            logger.info(f"[YOLO引擎] 开始训练: epochs={config.epochs}, batch={batch_size}, device={device}")

            # 执行训练
            results = model.train(**train_kwargs)

            logger.info(f"[YOLO引擎] 训练完成: task_id={task_id}")

        except Exception as e:
            logger.exception(f"[YOLO引擎] 训练失败: {e}")
            import traceback
            traceback.print_exc()

            with self.training_lock:
                status = self.training_tasks.get(task_id)
                if status:
                    status.status = "failed"
                    status.error_message = str(e)

    def get_training_status(self, task_id: str) -> Optional['TrainingStatus']:
        """
        获取训练状态

        Args:
            task_id: 训练任务 ID

        Returns:
            TrainingStatus: 训练状态对象
        """
        with self.training_lock:
            status = self.training_tasks.get(task_id)
            if not status:
                return None

            # 如果训练正在进行中，尝试从训练目录轮询进度
            if status.status == "running":
                try:
                    # 使用 project_name 获取训练目录
                    project_name = status.project_name or task_id.replace("train_", "")
                    train_dir = settings.MODELS_DIR / project_name / "train"

                    if train_dir.exists():
                        # 检查 results.csv 文件
                        results_file = train_dir / "results.csv"
                        if results_file.exists():
                            import csv
                            with open(results_file, 'r') as f:
                                reader = csv.reader(f)
                                rows = list(reader)

                                if len(rows) > 1:  # 有数据行
                                    last_row = rows[-1]
                                    # 解析 epoch (通常是第1列或第0列)
                                    try:
                                        # Ultralytics results.csv 格式: epoch, train/box_loss, train/cls_loss, ...
                                        current_epoch = int(float(last_row[0])) + 1
                                        status.current_epoch = current_epoch
                                        status.progress = min(100.0, current_epoch / max(status.total_epochs, 1) * 100.0)

                                        # 尝试解析指标
                                        if len(last_row) > 3:
                                            metrics = {}
                                            try:
                                                metrics['metrics/mAP50(B)'] = float(last_row[5]) if last_row[5] else 0
                                                metrics['metrics/mAP50-95(B)'] = float(last_row[6]) if last_row[6] else 0
                                                metrics['metrics/precision(B)'] = float(last_row[7]) if last_row[7] else 0
                                                metrics['metrics/recall(B)'] = float(last_row[8]) if last_row[8] else 0
                                                status.metrics = {"latest": metrics}
                                            except (IndexError, ValueError):
                                                pass

                                        logger.debug(f"[YOLO引擎] 轮询更新进度: epoch={current_epoch}/{status.total_epochs}")
                                    except (IndexError, ValueError) as e:
                                        logger.warning(f"[YOLO引擎] 解析 results.csv 失败: {e}")

                        # 检查是否完成
                        weights_dir = train_dir / "weights"
                        if weights_dir.exists():
                            best_pt = weights_dir / "best.pt"
                            if best_pt.exists():
                                status.status = "completed"
                                status.progress = 100.0
                                status.checkpoint_path = str(best_pt)
                except Exception as e:
                    logger.warning(f"[YOLO引擎] 轮询训练进度失败: {e}")

            return copy.deepcopy(status)

    def list_training_statuses(self) -> List['TrainingStatus']:
        """
        列出所有训练任务

        Returns:
            List[TrainingStatus]: 训练状态列表
        """
        with self.training_lock:
            return [status.copy(deep=True) for status in self.training_tasks.values()]

    def export_model(self, model_path: str, format: str = "onnx", **kwargs) -> Dict[str, Any]:
        """
        导出模型

        Args:
            model_path: 模型文件路径
            format: 导出格式
            **kwargs: 其他导出参数

        Returns:
            Dict: 导出结果
        """
        logger.info(f"[YOLO引擎] 导出模型: {model_path}, 格式={format}")

        try:
            model = YOLO(model_path)
            export_path = model.export(
                format=format,
                imgsz=kwargs.get("imgsz", 640),
                half=kwargs.get("half", False),
                simplify=kwargs.get("simplify", True),
            )
            logger.info(f"[YOLO引擎] 导出成功: {export_path}")
            return {
                "success": True,
                "message": "导出成功",
                "export_path": str(export_path)
            }
        except Exception as e:
            error_msg = f"导出失败: {str(e)}"
            logger.error(f"[YOLO引擎] {error_msg}")
            return {
                "success": False,
                "message": error_msg,
                "export_path": None
            }

    def get_device_info(self) -> Tuple[bool, Optional[str]]:
        """
        获取 GPU 信息

        Returns:
            Tuple[bool, Optional[str]]: (是否可用, GPU信息)
        """
        try:
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info = f"{name} ({memory:.1f}GB)"
                logger.debug(f"[YOLO引擎] GPU信息: {info}")
                return True, info
            return False, None
        except:
            return False, None


# 数据类型定义 (避免循环导入)

@dataclass
class TrainingStatus:
    """
    训练状态数据类
    存储训练任务的实时状态信息
    """
    task_id: str  # 任务 ID
    status: str  # 状态 (pending/running/completed/failed)
    progress: float  # 进度百分比
    current_epoch: int  # 当前轮数
    total_epochs: int  # 总轮数
    created_at: Any  # 创建时间
    updated_at: Any  # 更新时间
    project_name: Optional[str] = None  # 项目名称，用于轮询进度
    metrics: Optional[Dict[str, Any]] = None  # 最新指标
    metrics_history: List[Dict[str, Any]] = None  # 完整的指标历史
    losses_history: Dict[str, List[float]] = None  # 损失曲线历史
    error_message: Optional[str] = None  # 错误信息
    gpu_memory: Optional[str] = None  # GPU 显存使用
    gpu_utilization: Optional[float] = None  # GPU 利用率
    system_memory: Optional[float] = None  # 系统内存使用
    best_metrics: Dict[str, float] = None  # 最佳指标
    checkpoint_path: Optional[str] = None  # 检查点路径

    def __post_init__(self):
        """初始化默认值"""
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
        """
        添加指标记录

        Args:
            epoch: 轮数
            metrics: 性能指标
            losses: 损失值
        """
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
        """
        获取图表数据

        Returns:
            Dict: 用于绘制图表的数据
        """
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
    """
    训练配置数据类
    包含所有训练相关参数
    """
    project_name: str  # 项目名称
    dataset_path: str  # 数据集路径
    model_type: str = "yolo11n"  # 模型类型
    model_path: Optional[str] = None  # 模型文件路径
    epochs: int = 100  # 训练轮数
    batch_size: int = 32  # 批大小
    img_size: int = 640  # 输入图片尺寸
    device: str = "auto"  # 设备
    patience: int = 100  # 早停耐心值
    save_period: int = -1  # 保存周期
    pretrained: bool = True  # 是否使用预训练权重
    optimizer: str = "auto"  # 优化器
    amp: bool = True  # 混合精度训练
    workers: int = 8  # 数据加载线程数
    resume: Optional[str] = None  # 恢复训练路径
    multi_scale: bool = False  # 多尺度训练
    seed: int = 0  # 随机种子
    deterministic: bool = False  # 确定性训练
    plots: bool = True  # 是否生成图表
    val: bool = True  # 是否验证

    # ==================== 优化参数 ====================
    lr0: float = 0.01  # 初始学习率
    lrf: float = 0.01  # 最终学习率（相对于 lr0）
    warmup_epochs: float = 3.0  # 预热轮数
    warmup_bias_lr: float = 0.1  # 预热期间 bias 的学习率
    momentum: float = 0.937  # 动量
    weight_decay: float = 0.0005  # 权重衰减
    cos_lr: bool = False  # 余弦学习率调度

    # ==================== 数据增强参数 ====================
    hsv_h: float = 0.015  # 色相
    hsv_s: float = 0.7  # 饱和度
    hsv_v: float = 0.4  # 亮度
    degrees: float = 0.0  # 旋转角度
    translate: float = 0.1  # 平移比例
    scale: float = 0.5  # 缩放比例
    shear: float = 0.0  # 剪切角度
    perspective: float = 0.0  # 透视变换
    flipud: float = 0.0  # 垂直翻转
    fliplr: float = 0.5  # 水平翻转
    mosaic: float = 1.0  # 马赛克增强
    multi_scale: float = 0.0  # 多尺度训练
    close_mosaic: int = 10  # 关闭马赛克增强的轮数
    mixup: float = 0.0  # 混合增强
    copy_paste: float = 0.0  # 复制粘贴
    auto_augment: str = "randaugment"  # 自动增强
    dropout: float = 0.0  # Dropout

    # ==================== 损失函数权重 ====================
    box: float = 7.5  # 边界框损失权重
    cls: float = 0.5  # 分类损失权重
    dfl: float = 1.5  # 分布焦点损失权重

    # ==================== 验证参数 ====================
    val_conf: float = 0.001  # 验证置信度阈值
    val_iou: float = 0.6  # 验证 IOU 阈值
    val_rect: bool = True  # 验证时使用矩形图像

    def copy(self, deep: bool = False):
        """创建配置副本"""
        import copy
        return copy.deepcopy(self) if deep else copy.copy(self)


class ModelInfo:
    """
    模型信息数据类
    存储模型元数据
    """
    name: str  # 模型名称
    path: str  # 模型路径
    size: int  # 文件大小
    created_at: Any  # 创建时间
    model_type: str = "yolo"  # 模型类型
    task: str = "detect"  # 任务类型
    input_shape: List[int] = None  # 输入形状
    classes: List[str] = None  # 类别列表


# 创建全局 YOLO 引擎实例
logger.info("[YOLO引擎] 创建全局引擎实例")
yolo_engine = YOLOEngine() if TORCH_AVAILABLE else None

logger.info("[YOLO引擎] 模块加载完成")
