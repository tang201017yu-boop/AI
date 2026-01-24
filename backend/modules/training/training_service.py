"""
训练服务 - Training Service
提供云端训练、远程训练、实验管理等功能
"""
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.core.config import settings


class TrainingService:
    """训练服务"""

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
        self.experiments: Dict[str, Dict] = {}
        self._load_experiments()

    def _load_experiments(self):
        """加载实验列表"""
        experiments_dir = Path(settings.MODELS_DIR) / "experiments"
        if experiments_dir.exists():
            for exp_dir in experiments_dir.iterdir():
                if exp_dir.is_dir():
                    self.experiments[exp_dir.name] = {
                        "name": exp_dir.name,
                        "path": str(exp_dir),
                        "created_at": datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat()
                    }

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
        开始训练

        Args:
            project_name: 项目名称
            dataset_path: 数据集路径
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
        """
        if not ULTRALYTICS_AVAILABLE:
            return {"success": False, "message": "Ultralytics 未安装"}

        try:
            from backend.core.yolo_engine import yolo_engine, TrainingConfig
            from backend.core.config import settings
            from pathlib import Path

            # 检查训练引擎是否可用
            if yolo_engine is None:
                return {"success": False, "message": "训练引擎初始化失败，请检查 PyTorch 和 Ultralytics 安装"}

            # 解析数据集路径 - 支持数据集名称或完整路径
            dataset_path_resolved = dataset_path
            dataset_path_obj = Path(dataset_path)
            if not dataset_path_obj.exists():
                # 尝试作为数据集名称解析
                dataset_dir = settings.DATASETS_DIR / dataset_path
                if dataset_dir.exists():
                    # 优先在根目录查找 data.yaml 或 data.yml
                    yaml_found = False
                    for name in ["data.yaml", "data.yml"]:
                        yaml_file = dataset_dir / name
                        if yaml_file.exists():
                            dataset_path_resolved = str(yaml_file)
                            yaml_found = True
                            break

                    # 如果根目录没找到，递归查找子目录
                    if not yaml_found:
                        yaml_files = sorted(dataset_dir.rglob("data.yaml")) + sorted(dataset_dir.rglob("data.yml"))
                        if yaml_files:
                            dataset_path_resolved = str(yaml_files[0])
                        else:
                            return {"success": False, "message": f"数据集 {dataset_path} 中未找到 data.yaml 文件"}

                    print(f"[训练] 使用数据集配置文件: {dataset_path_resolved}")

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

                                # 如果 path 是相对的，也需要修复
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
                                    print(f"[训练] 已修复 data.yaml 路径: {fixed_paths}")
                        except Exception as e:
                            print(f"[训练] 警告: 无法修复 data.yaml path: {e}")

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

            # 创建训练配置
            config = TrainingConfig(**all_params)

            # 开始训练
            task_id = yolo_engine.train(config)

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

            return {
                "success": True,
                "message": "训练已开始",
                "task_id": task_id,
                "experiment": experiment
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"启动训练失败: {str(e)}"
            }

    def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """获取训练状态"""
        from backend.core.yolo_engine import yolo_engine

        if yolo_engine is None:
            return {"success": False, "message": "训练引擎未初始化"}

        try:
            status = yolo_engine.get_training_status(task_id)
            if status:
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
            print(f"获取训练状态失败: {e}")

        return {"success": False, "message": "任务不存在"}

    def list_training_tasks(self) -> List[Dict[str, Any]]:
        """列出所有训练任务"""
        from backend.core.yolo_engine import yolo_engine

        if yolo_engine is None:
            # 返回空列表而不是崩溃
            return []

        try:
            return yolo_engine.list_training_statuses()
        except Exception as e:
            print(f"获取训练任务列表失败: {e}")
            return []

    def cancel_training(self, task_id: str) -> Dict[str, Any]:
        """取消训练"""
        # TODO: 实现取消训练功能
        return {"success": False, "message": "取消训练功能开发中"}

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """比较实验"""
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

        return {
            "success": True,
            "experiments": results,
            "message": "实验比较功能开发中"
        }

    def get_model_comparison(self) -> Dict[str, Any]:
        """获取模型对比信息"""
        from backend.core.yolo_engine import yolo_engine

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
            except:
                pass

        return {
            "success": True,
            "models": model_list
        }


class ExportService:
    """模型导出服务"""

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
        pass

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
        导出模型

        Args:
            model_path: 模型路径
            format: 导出格式
            img_size: 输入尺寸
            half: FP16 量化
            simplify: 简化模型
        """
        if not ULTRALYTICS_AVAILABLE:
            return {"success": False, "message": "Ultralytics 未安装"}

        try:
            from backend.core.yolo_engine import yolo_engine
            return yolo_engine.export_model(
                model_path=model_path,
                format=format,
                imgsz=img_size,
                half=half,
                simplify=simplify,
                **kwargs
            )
        except Exception as e:
            return {"success": False, "message": f"导出失败: {str(e)}"}

    def get_export_formats(self) -> Dict[str, Any]:
        """获取支持的导出格式"""
        return {
            "success": True,
            "formats": self.EXPORT_FORMATS
        }

    def get_recommended_format(self, target: str) -> str:
        """获取推荐的导出格式"""
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
                return fmt

        return "onnx"  # 默认


# 全局实例
training_service = TrainingService()
export_service = ExportService()
