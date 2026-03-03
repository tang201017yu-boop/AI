# -*- coding: utf-8 -*-
"""
训练服务模块 - Training Service
提供模型训练、实验管理、模型导出等功能
"""
import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

# 尝试导入 YOLO 模型
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.core.config import settings

# 创建日志记录器
logger = logging.getLogger(__name__)


class TrainingService:
    """
    训练服务类
    管理模型训练任务、实验记录、训练状态查询等功能
    """

    # 支持的导出格式
    EXPORT_FORMATS = {
        "onnx": {"name": "ONNX", "ext": ".onnx", "description": "跨平台推理框架"},
        "torchscript": {"name": "TorchScript", "ext": ".torchscript", "description": "PyTorch 静态图"},
        "coreml": {"name": "CoreML", "ext": ".mlpackage", "description": "Apple 设备"},
        "tflite": {"name": "TensorFlow Lite", "ext": ".tflite", "description": "移动设备"},
        "edgetpu": {"name": "Edge TPU", "ext": ".tflite", "description": "Google Edge TPU"},
        "tfjs": {"name": "TensorFlow.js", "ext": ".json", "description": "浏览器"},
        "tensorrt": {"name": "TensorRT", "ext": ".engine", "description": "NVIDIA GPU 优化"},
        "openvino": {"name": "OpenVINO", "ext": ".xml", "description": "Intel CPU/GPU"},
        "rknn": {"name": "RKNN", "ext": ".rknn", "description": "瑞芯微 NPU"},
        "engine": {"name": "Engine", "ext": ".engine", "description": "TensorRT 引擎"},
    }

    def __init__(self):
        """初始化训练服务"""
        logger.info("[训练] 初始化训练服务")
        self.experiments: Dict[str, Dict] = {}  # 实验记录字典
        self._load_experiments()  # 加载已有实验

    def _load_experiments(self):
        """
        加载已有的实验列表
        从 models/experiments 目录读取历史实验记录
        """
        experiments_dir = Path(settings.MODELS_DIR) / "experiments"

        # 检查实验目录是否存在
        if not experiments_dir.exists():
            logger.info("[训练] 实验目录不存在，将创建")
            experiments_dir.mkdir(parents=True, exist_ok=True)
            return

        # 遍历实验目录
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                # 保存实验基本信息
                self.experiments[exp_dir.name] = {
                    "name": exp_dir.name,
                    "path": str(exp_dir),
                    "created_at": datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat()
                }

        logger.info(f"[训练] 已加载 {len(self.experiments)} 个历史实验")

    def start_training(
        self,
        project_name: str,
        dataset_path: str,
        model_type: str = "yolo11n",
        epochs: int = 100,
        batch_size: int = 32,
        img_size: int = 640,
        device: str = "auto",
        optimizer: str = "auto",
        amp: bool = True,
        workers: int = 8,
        # 微调参数
        lr0: float = 0.01,
        lrf: float = 0.01,
        warmup_epochs: float = 3.0,
        warmup_bias_lr: float = 0.1,
        mosaic: float = 1.0,
        mosaic_scale: tuple = (0.1, 1.5),
        close_mosaic_epochs: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        开始模型训练

        Args:
            project_name: 项目名称
            dataset_path: 数据集路径（支持数据集名称或完整路径）
            model_type: 模型类型 (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
            epochs: 训练轮数
            batch_size: 批大小
            img_size: 输入图片尺寸
            device: 设备 (auto/cpu/cuda/0)
            optimizer: 优化器 (auto/AdamW/SGD/Adam/NAdam/RAdam/RMSProp)
                - auto: 根据模型类型自动选择（推荐）
                - AdamW: YOLO 系列推荐，稳定性好
                - SGD: 传统优化器，收敛稳定
                - Adam: 自适应学习率，适合大多数场景
                - NAdam/RAdam/RMSProp: 其他优化器选项
            amp: 是否使用混合精度训练
            workers: 数据加载线程数

        微调参数:
            lr0: 初始学习率 (微调时建议 0.001-0.01)
            lrf: 最终学习率因子 (相对于 lr0)
            warmup_epochs: 预热轮数 (设置为 0 可立即使用高学习率)
            warmup_bias_lr: 预热期间 bias 的学习率
            mosaic: 马赛克增强概率 (0-1)
            mosaic_scale: 马赛克缩放范围
            close_mosaic_epochs: 关闭马赛克的轮数

        Returns:
            Dict: 训练启动结果
        """
        logger.info(f"[训练] 开始训练: 项目={project_name}, 模型={model_type}, 数据集={dataset_path}")
        logger.info(f"[训练] 训练参数: epochs={epochs}, batch_size={batch_size}, img_size={img_size}, device={device}")

        # 检查 Ultralytics 是否安装
        if not ULTRALYTICS_AVAILABLE:
            error_msg = "Ultralytics 未安装，无法进行训练"
            logger.error(f"[训练] {error_msg}")
            return {"success": False, "message": error_msg}

        try:
            # 导入训练引擎和配置
            from backend.core.yolo_engine import yolo_engine, TrainingConfig
            from backend.core.config import settings
            from pathlib import Path

            # 检查训练引擎是否可用
            if yolo_engine is None:
                error_msg = "训练引擎初始化失败，请检查 PyTorch 和 Ultralytics 安装"
                logger.error(f"[训练] {error_msg}")
                return {"success": False, "message": error_msg}

            # 解析数据集路径 - 支持数据集名称或完整路径
            dataset_path_resolved = dataset_path
            dataset_path_obj = Path(dataset_path)

            # 如果路径不存在，尝试作为数据集名称解析
            if not dataset_path_obj.exists():
                logger.debug(f"[训练] 路径不存在，尝试解析为数据集名称: {dataset_path}")

                dataset_dir = settings.DATASETS_DIR / dataset_path
                if dataset_dir.exists():
                    # 优先在根目录查找 data.yaml 或 data.yml
                    yaml_found = False
                    for name in ["data.yaml", "data.yml"]:
                        yaml_file = dataset_dir / name
                        if yaml_file.exists():
                            dataset_path_resolved = str(yaml_file)
                            yaml_found = True
                            logger.info(f"[训练] 找到数据集配置文件: {yaml_file}")
                            break

                    # 如果根目录没找到，递归查找子目录
                    if not yaml_found:
                        yaml_files = sorted(dataset_dir.rglob("data.yaml")) + sorted(dataset_dir.rglob("data.yml"))
                        if yaml_files:
                            dataset_path_resolved = str(yaml_files[0])
                            logger.info(f"[训练] 在子目录找到配置文件: {yaml_files[0]}")
                        else:
                            error_msg = f"数据集 {dataset_path} 中未找到 data.yaml 文件"
                            logger.error(f"[训练] {error_msg}")
                            return {"success": False, "message": error_msg}

                    # 检查并修复 data.yaml 中的路径
                    yaml_path = Path(dataset_path_resolved)
                    if yaml_path.exists():
                        import yaml as pyyaml
                        try:
                            with open(yaml_path, 'r', encoding='utf-8') as f:
                                yaml_content = pyyaml.safe_load(f)

                            if yaml_content:
                                yaml_dir = yaml_path.parent
                                needs_fix = False
                                fixed_paths = {}

                                # 修复 train/val 路径中的 ../
                                for key in ['train', 'val']:
                                    if key in yaml_content:
                                        path_val = yaml_content[key]
                                        if path_val and '..' in path_val:
                                            # 移除 .. 并重新计算路径
                                            fixed_path = path_val.replace('../', '')
                                            full_path = yaml_dir / fixed_path
                                            if full_path.exists():
                                                yaml_content[key] = fixed_path
                                                needs_fix = True
                                                fixed_paths[key] = fixed_path

                                # 修复 path 字段
                                if 'path' in yaml_content:
                                    path_val = yaml_content['path']
                                    if path_val and '..' in path_val:
                                        fixed_path = path_val.replace('../', '')
                                        yaml_content['path'] = fixed_path
                                        needs_fix = True

                                # 写回修复后的文件
                                if needs_fix:
                                    with open(yaml_path, 'w', encoding='utf-8') as f:
                                        pyyaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
                                    logger.info(f"[训练] 已修复 data.yaml 路径: {fixed_paths}")
                        except Exception as e:
                            logger.warning(f"[训练] 无法修复 data.yaml path: {e}")

            # 合并所有参数
            all_params = {
                # 基础配置
                "project_name": project_name,
                "dataset_path": dataset_path_resolved,
                "model_type": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
                "device": device,
                "pretrained": True,
                "optimizer": optimizer,
                # 优化参数
                "lr0": lr0,
                "lrf": lrf,
                "warmup_epochs": warmup_epochs,
                "warmup_bias_lr": warmup_bias_lr,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "cos_lr": False,
                # 设备配置
                "amp": amp,
                "workers": workers,
                # 马赛克参数
                "mosaic": mosaic,
                "multi_scale": 0.0,
                "close_mosaic": close_mosaic_epochs
            }

            # 合并 kwargs 中的所有额外参数
            all_params.update(kwargs)

            # 创建训练配置对象
            config = TrainingConfig(**all_params)

            # 调用引擎开始训练
            task_id = yolo_engine.train(config)
            logger.info(f"[训练] 训练任务已提交: task_id={task_id}")

            # 保存实验信息
            experiment = {
                "task_id": task_id,
                "project_name": project_name,
                "model_type": model_type,
                "dataset_path": dataset_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": optimizer,
                "status": "running",
                "created_at": datetime.now().isoformat(),
                "fine_tune_params": {
                    "lr0": lr0,
                    "lrf": lrf,
                    "warmup_epochs": warmup_epochs,
                    "mosaic": mosaic,
                    "close_mosaic": close_mosaic_epochs
                }
            }
            self.experiments[task_id] = experiment

            logger.info(f"[训练] 训练已启动: task_id={task_id}")

            return {
                "success": True,
                "message": "训练已开始",
                "task_id": task_id,
                "experiment": experiment
            }

        except Exception as e:
            error_msg = f"启动训练失败: {str(e)}"
            logger.error(f"[训练] {error_msg}")
            import traceback
            logger.error(f"[训练] 详细错误: {traceback.format_exc()}")
            return {
                "success": False,
                "message": error_msg
            }

    def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取训练任务状态

        Args:
            task_id: 训练任务 ID

        Returns:
            Dict: 训练状态信息
        """
        logger.debug(f"[训练] 查询训练状态: task_id={task_id}")

        from backend.core.yolo_engine import yolo_engine

        if yolo_engine is None:
            error_msg = "训练引擎未初始化"
            logger.error(f"[训练] {error_msg}")
            return {"success": False, "message": error_msg}

        try:
            # 获取训练状态
            status = yolo_engine.get_training_status(task_id)
            if status:
                logger.debug(f"[训练] 任务状态: {status.status}, 进度: {status.progress:.1f}%")
                return {
                    "success": True,
                    "status": {
                        "task_id": status.task_id,
                        "status": status.status,
                        "progress": status.progress,
                        "current_epoch": status.current_epoch,
                        "total_epochs": status.total_epochs,
                        "metrics": status.metrics,
                        "gpu_memory": status.gpu_memory,
                        "error_message": status.error_message,
                        "updatedAt": status.updated_at.isoformat() if hasattr(status, 'updated_at') and status.updated_at else None
                    }
                }
        except Exception as e:
            error_msg = f"获取训练状态失败: {str(e)}"
            logger.error(f"[训练] {error_msg}")

        error_msg = "任务不存在"
        logger.warning(f"[训练] {error_msg}: task_id={task_id}")
        return {"success": False, "message": error_msg}

    def list_training_tasks(self) -> List[Dict[str, Any]]:
        """
        列出所有训练任务

        Returns:
            List[Dict]: 训练任务列表
        """
        logger.debug("[训练] 查询所有训练任务")

        from backend.core.yolo_engine import yolo_engine

        if yolo_engine is None:
            # 返回空列表而不是崩溃
            logger.warning("[训练] 训练引擎未初始化")
            return []

        try:
            tasks = yolo_engine.list_training_statuses()
            logger.debug(f"[训练] 找到 {len(tasks)} 个训练任务")
            return tasks
        except Exception as e:
            error_msg = f"获取训练任务列表失败: {str(e)}"
            logger.error(f"[训练] {error_msg}")
            return []

    def cancel_training(self, task_id: str) -> Dict[str, Any]:
        """
        取消训练任务

        Args:
            task_id: 训练任务 ID

        Returns:
            Dict: 操作结果
        """
        logger.info(f"[训练] 尝试取消训练: task_id={task_id}")

        # 调用 YOLO 引擎取消训练
        if yolo_engine:
            result = yolo_engine.cancel_training(task_id)
            if result.get("success"):
                logger.info(f"[训练] 训练已取消: task_id={task_id}")
                return result

        return {"success": False, "message": "取消训练失败"}

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        比较多个实验的结果

        Args:
            experiment_ids: 实验 ID 列表

        Returns:
            Dict: 实验比较结果
        """
        logger.info(f"[训练] 比较实验: {experiment_ids}")

        experiments = []
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                experiments.append(self.experiments[exp_id])

        # 获取每个实验的结果
        results = []
        for exp in experiments:
            exp_dir = Path(exp["path"]) / "weights"
            if exp_dir.exists():
                best_pt = exp_dir / "best.pt"
                if best_pt.exists():
                    results.append({
                        "name": exp["name"],
                        "model_path": str(best_pt),
                        "mAP50": None,  # 需要加载模型获取
                        "mAP50_95": None
                    })

        logger.info(f"[训练] 找到 {len(results)} 个可比较的实验")

        return {
            "success": True,
            "experiments": results,
            "message": "实验比较功能开发中"
        }

    def get_model_comparison(self) -> Dict[str, Any]:
        """
        获取模型对比信息

        Returns:
            Dict: 模型列表信息
        """
        logger.debug("[训练] 获取模型列表")

        from backend.core.yolo_engine import yolo_engine

        # 获取所有模型文件
        models = yolo_engine._iter_model_paths()
        model_list = []

        for path in models:
            try:
                stat = path.stat()
                model_list.append({
                    "name": path.name,
                    "path": str(path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
            except Exception as e:
                logger.warning(f"[训练] 读取模型信息失败: {path}, 错误: {e}")

        logger.info(f"[训练] 找到 {len(model_list)} 个模型")

        return {
            "success": True,
            "models": model_list
        }


class ExportService:
    """
    模型导出服务类
    提供模型导出为各种格式的功能
    """

    # 支持的导出格式
    EXPORT_FORMATS = {
        "onnx": {"name": "ONNX", "ext": ".onnx", "platforms": ["Windows", "Linux", "macOS"]},
        "torchscript": {"name": "TorchScript", "ext": ".torchscript", "platforms": ["PyTorch"]},
        "coreml": {"name": "CoreML", "ext": ".mlmodel", "platforms": ["macOS", "iOS"]},
        "tflite": {"name": "TensorFlow Lite", "ext": ".tflite", "platforms": ["Android", "iOS"]},
        "tensorrt": {"name": "TensorRT", "ext": ".engine", "platforms": ["NVIDIA GPU"]},
        "openvino": {"name": "OpenVINO", "ext": ".xml", "platforms": ["Intel CPU/GPU"]},
        "ncnn": {"name": "NCNN", "ext": ".param", "platforms": ["Mobile"]},
        "paddle": {"name": "Paddle", "ext": "_model", "platforms": ["Baidu"]},
    }

    def __init__(self):
        """初始化导出服务"""
        logger.info("[导出] 初始化导出服务")

    def export_model(
        self,
        model_path: str,
        format: str = "onnx",
        img_size: int = 640,
        half: bool = False,
        simplify: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        导出模型到指定格式

        Args:
            model_path: 模型文件路径
            format: 导出格式 (onnx/torchscript/coreml/tflite/tensorrt/openvino/ncnn/paddle)
            img_size: 输入图片尺寸
            half: 是否使用 FP16 量化
            simplify: 是否简化模型

        Returns:
            Dict: 导出结果
        """
        logger.info(f"[导出] 开始导出模型: {model_path}, 格式={format}")

        # 检查 Ultralytics 是否安装
        if not ULTRALYTICS_AVAILABLE:
            error_msg = "Ultralytics 未安装，无法导出模型"
            logger.error(f"[导出] {error_msg}")
            return {"success": False, "message": error_msg}

        try:
            # 导入训练引擎
            from backend.core.yolo_engine import yolo_engine

            # 调用引擎导出模型
            result = yolo_engine.export_model(
                model_path=model_path,
                format=format,
                imgsz=img_size,
                half=half,
                simplify=simplify,
                **kwargs
            )

            if result.get("success"):
                logger.info(f"[导出] 模型导出成功: {result.get('export_path')}")
            else:
                logger.error(f"[导出] 模型导出失败: {result.get('message')}")

            return result

        except Exception as e:
            error_msg = f"导出失败: {str(e)}"
            logger.error(f"[导出] {error_msg}")
            return {"success": False, "message": error_msg}

    def get_export_formats(self) -> Dict[str, Any]:
        """
        获取支持的导出格式列表

        Returns:
            Dict: 导出格式信息
        """
        logger.debug("[导出] 获取导出格式列表")
        return {
            "success": True,
            "formats": self.EXPORT_FORMATS
        }

    def get_recommended_format(self, target: str) -> str:
        """
        根据目标平台获取推荐的导出格式

        Args:
            target: 目标平台 (windows/linux/macos/android/ios/nvidia/intel/mobile)

        Returns:
            str: 推荐的导出格式
        """
        logger.debug(f"[导出] 获取推荐格式: target={target}")

        recommendations = {
            "windows": "onnx",
            "linux": "onnx",
            "macos": "coreml",
            "android": "tflite",
            "ios": "coreml",
            "nvidia": "tensorrt",
            "intel": "openvino",
            "mobile": "ncnn"
        }

        target_lower = target.lower()
        for key, fmt in recommendations.items():
            if key in target_lower:
                logger.debug(f"[导出] 推荐格式: {fmt}")
                return fmt

        # 默认返回 ONNX
        logger.debug("[导出] 使用默认格式: onnx")
        return "onnx"


# 创建全局服务实例
training_service = TrainingService()
export_service = ExportService()

logger.info("[训练] 训练服务模块加载完成")
