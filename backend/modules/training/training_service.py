# -*- coding: utf-8 -*-
"""
训练服务模块 - Training Service
提供模型训练、实验管理、模型导出等功能
"""
import os
import time
import json
import csv
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
        self.experiments_file = Path(settings.MODELS_DIR) / "experiments.json"  # 持久化文件
        self._load_experiments()  # 加载已有实验
        self._reconcile_stale_running_experiments()

    def _experiment_start_timestamp(self, exp: Dict[str, Any]) -> Optional[float]:
        raw = exp.get("created_at")
        if not raw or not isinstance(raw, str):
            return None
        try:
            s = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            return None

    def _max_epoch_from_results_csv(self, project_name: str) -> Optional[int]:
        """
        results.csv 中 epoch 列为 Ultralytics 的 epoch 序号；显示用 epoch_display = epoch + 1（与图表一致）。
        返回已完成的「显示用」最大 epoch，无文件或为空则返回 None。
        """
        results_file = Path(settings.MODELS_DIR) / project_name / "train" / "results.csv"
        if not results_file.exists():
            return None

        def safe_float(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        max_ep = 0
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row = {k.strip(): v.strip() for k, v in row.items()}
                    epoch = safe_float(row.get('epoch', row.get('                   epoch', '')))
                    if epoch is None:
                        continue
                    ep = int(epoch) + 1
                    max_ep = max(max_ep, ep)
            return max_ep if max_ep > 0 else None
        except Exception as e:
            logger.debug(f"[训练] 读取 results.csv 推断 epoch 失败: {e}")
            return None

    def _training_finished_on_disk(self, exp: Dict[str, Any]) -> bool:
        """根据磁盘上的 results.csv / 权重判断本次实验是否已完成（用于修正未落盘的 running 状态）"""
        project_name = exp.get("project_name")
        expected_epochs = int(exp.get("epochs") or 0)
        if not project_name or expected_epochs <= 0:
            return False

        weights_dir = Path(settings.MODELS_DIR) / project_name / "train" / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        if not best_pt.exists() and not last_pt.exists():
            return False

        csv_max = self._max_epoch_from_results_csv(project_name)
        if csv_max is not None and csv_max >= expected_epochs:
            return True

        start_ts = self._experiment_start_timestamp(exp)
        if start_ts is None:
            return False
        chosen = best_pt if best_pt.exists() else last_pt
        try:
            mtime = chosen.stat().st_mtime
        except OSError:
            return False
        # 磁盘上无权重的旧项目被复用同名时，避免因旧文件误判：权重须不早于实验创建时间（允许时钟偏差）
        if mtime + 120 < start_ts:
            return False
        # 无可读 csv、仅有本实验开始后写出的权重时，兜底认为已结束（finish 回调漏写但仍写了权重）
        return csv_max is None

    def _is_task_active_in_yolo_engine(self, task_id: str) -> bool:
        try:
            from backend.core.yolo_engine import yolo_engine
            if yolo_engine is None:
                return False
            st = yolo_engine.get_training_status(task_id)
            if not st:
                return False
            status = getattr(st, "status", None)
            return status in ("pending", "running")
        except Exception:
            return False

    def _maybe_complete_stale_experiment(self, task_id: str) -> bool:
        """
        若该 task 持久化为 running，但引擎中无活跃训练且磁盘已表明本轮结束，则改为 completed。
        返回是否在内存中做了修改（是否仍需 _save_experiments 由调用方决定）。
        """
        exp = self.experiments.get(task_id)
        if not isinstance(exp, dict) or exp.get("status") != "running":
            return False
        if self._is_task_active_in_yolo_engine(task_id):
            return False
        if not self._training_finished_on_disk(exp):
            return False

        exp["status"] = "completed"
        project_name = exp.get("project_name")
        if project_name:
            wd = Path(settings.MODELS_DIR) / project_name / "train" / "weights"
            best_pt = wd / "best.pt"
            last_pt = wd / "last.pt"
            if best_pt.exists():
                exp["checkpoint_path"] = str(best_pt)
            elif last_pt.exists():
                exp["checkpoint_path"] = str(last_pt)
        logger.info("[训练] 已自动修正异常的「训练中」状态为已完成: task_id=%s project=%s", task_id, project_name)
        return True

    def _reconcile_stale_running_experiments(self) -> None:
        """
        持久化中为 running，但进程内已无训练任务且磁盘显示已结束时，更正为 completed 并保存。
        解决 Ultralytics on_train_end 未同步 experiments.json、epoch 已满仍显示训练中等问题。
        """
        changed = False
        for tid in list(self.experiments.keys()):
            if self._maybe_complete_stale_experiment(tid):
                changed = True
        if changed:
            self._save_experiments()

    def _load_experiments(self):
        """
        加载已有的实验列表
        从持久化文件加载实验记录
        """
        # 优先从持久化文件加载
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, 'r', encoding='utf-8') as f:
                    self.experiments = json.load(f)
                logger.info(f"[训练] 已从持久化文件加载 {len(self.experiments)} 个实验")
                return
            except Exception as e:
                logger.warning(f"[训练] 加载实验文件失败: {e}")

        # 回退到从目录加载
        experiments_dir = Path(settings.MODELS_DIR) / "experiments"
        if not experiments_dir.exists():
            experiments_dir.mkdir(parents=True, exist_ok=True)
            return

        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                self.experiments[exp_dir.name] = {
                    "name": exp_dir.name,
                    "path": str(exp_dir),
                    "created_at": datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat()
                }

        logger.info(f"[训练] 已加载 {len(self.experiments)} 个历史实验")

    def _save_experiments(self):
        """
        保存实验列表到持久化文件
        """
        try:
            with open(self.experiments_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiments, f, ensure_ascii=False, indent=2)
            logger.debug(f"[训练] 实验列表已保存到 {self.experiments_file}")
        except Exception as e:
            logger.error(f"[训练] 保存实验列表失败: {e}")

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

        # 验证必需参数
        if not project_name:
            error_msg = "项目名称不能为空"
            logger.error(f"[训练] {error_msg}")
            return {"success": False, "message": error_msg}

        if not dataset_path:
            error_msg = "数据集路径不能为空"
            logger.error(f"[训练] {error_msg}")
            return {"success": False, "message": error_msg}

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

            # 判断是否为绝对路径
            is_absolute = dataset_path_obj.is_absolute()

            # 如果路径不存在，尝试作为数据集名称解析
            if not dataset_path_obj.exists():
                logger.debug(f"[训练] 路径不存在，尝试解析为数据集名称: {dataset_path}")

                # 如果是绝对路径但不存在，直接报错
                if is_absolute:
                    error_msg = f"数据集路径不存在: {dataset_path}"
                    logger.error(f"[训练] {error_msg}")
                    return {"success": False, "message": error_msg}

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
                            yaml_found = True
                        else:
                            # 尝试在嵌套子目录中查找
                            nested_dir = dataset_dir / dataset_path
                            if nested_dir.exists():
                                for name in ["data.yaml", "data.yml"]:
                                    yaml_file = nested_dir / name
                                    if yaml_file.exists():
                                        dataset_path_resolved = str(yaml_file)
                                        logger.info(f"[训练] 在嵌套目录找到配置文件: {yaml_file}")
                                        yaml_found = True
                                        break

                    if not yaml_found:
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
            project_id = kwargs.get('project_id')  # 获取项目ID
            experiment = {
                "task_id": task_id,
                "project_name": project_name,
                "project_id": project_id,  # 关联的项目ID
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
            self._save_experiments()  # 持久化保存

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

        # 1. 首先尝试从 experiments 中获取历史任务状态
        experiment = self.experiments.get(task_id)
        if experiment:
            if self._maybe_complete_stale_experiment(task_id):
                self._save_experiments()
            # 2. 检查 yolo_engine 中是否有正在运行的任务
            try:
                if yolo_engine:
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
                logger.warning(f"[训练] 从 yolo_engine 获取状态失败: {e}")

            # yolo_engine 中没有，从 experiments 返回历史状态
            return {
                "success": True,
                "status": {
                    "task_id": task_id,
                    "status": experiment.get("status", "completed"),
                    "progress": 100.0 if experiment.get("status") == "completed" else 0.0,
                    "current_epoch": experiment.get("epochs", 0),
                    "total_epochs": experiment.get("epochs", 0),
                    "metrics": experiment.get("metrics", {}),
                    "best_metrics": experiment.get("best_metrics", {}),
                    "project_name": experiment.get("project_name", ""),
                    "created_at": experiment.get("created_at", ""),
                    "updatedAt": experiment.get("created_at", "")
                }
            }

        # 3. 如果 experiments 也没有，检查 yolo_engine（可能是新的训练任务）
        try:
            if yolo_engine:
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
            logger.warning(f"[训练] 从 yolo_engine 获取状态失败: {e}")

        error_msg = "任务不存在"
        logger.warning(f"[训练] {error_msg}: task_id={task_id}")
        return {"success": False, "message": error_msg}

    def get_chart_data(self, task_id: str) -> Dict[str, Any]:
        """
        获取训练图表数据

        Args:
            task_id: 训练任务 ID

        Returns:
            图表数据
        """
        logger.debug(f"[训练] 获取图表数据: task_id={task_id}")

        from backend.core.yolo_engine import yolo_engine

        # 1. 优先从 yolo_engine 内存中获取（训练正在进行时）
        if yolo_engine:
            try:
                status = yolo_engine.get_training_status(task_id)
                if status:
                    chart_data = status.get_chart_data()
                    return {"success": True, "data": chart_data}
            except Exception as e:
                logger.warning(f"[训练] 从 yolo_engine 获取图表数据失败: {e}")

        # 2. 回退：从磁盘 results.csv 读取历史训练数据
        experiment = self.experiments.get(task_id)
        project_name = experiment.get("project_name") if experiment else None
        if not project_name:
            # 尝试从 task_id 推断（兼容旧格式）
            project_name = task_id.replace("train_", "")

        from backend.core.config import settings
        import csv, math

        train_dir = settings.MODELS_DIR / project_name / "train"
        results_file = train_dir / "results.csv"

        if not results_file.exists():
            if not experiment:
                return {"success": False, "message": "任务不存在"}
            # 实验存在但尚无 results.csv（可能还未开始第一个 epoch）
            return {"success": True, "data": {"epochs": [], "losses": {}, "metrics_history": [], "best_metrics": {}}}

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader]

            if not rows:
                return {"success": True, "data": {"epochs": [], "losses": {}, "metrics_history": [], "best_metrics": {}}}

            def safe_float(v):
                try:
                    val = float(v)
                    return None if (math.isnan(val) or math.isinf(val)) else val
                except (TypeError, ValueError):
                    return None

            epochs = []
            box_loss, cls_loss, dfl_loss = [], [], []
            metrics_history = []
            best_map50 = 0.0
            best_metrics = {}

            for row in rows:
                # strip whitespace from keys
                row = {k.strip(): v.strip() for k, v in row.items()}
                epoch = safe_float(row.get('epoch', row.get('                   epoch', '')))
                if epoch is None:
                    continue
                ep = int(epoch) + 1
                epochs.append(ep)

                bl = safe_float(row.get('train/box_loss'))
                cl = safe_float(row.get('train/cls_loss'))
                dl = safe_float(row.get('train/dfl_loss'))
                box_loss.append(bl)
                cls_loss.append(cl)
                dfl_loss.append(dl)

                map50   = safe_float(row.get('metrics/mAP50(B)'))
                map5095 = safe_float(row.get('metrics/mAP50-95(B)'))
                prec    = safe_float(row.get('metrics/precision(B)'))
                rec     = safe_float(row.get('metrics/recall(B)'))

                record = {
                    "epoch": ep,
                    "metrics/mAP50(B)": map50,
                    "metrics/mAP50-95(B)": map5095,
                    "metrics/precision(B)": prec,
                    "metrics/recall(B)": rec,
                    "train/box_loss": bl,
                    "train/cls_loss": cl,
                    "train/dfl_loss": dl,
                }
                metrics_history.append(record)

                if map50 is not None and map50 > best_map50:
                    best_map50 = map50
                    best_metrics = {
                        "epoch": ep,
                        "metrics/mAP50(B)": map50,
                        "metrics/mAP50-95(B)": map5095,
                        "metrics/precision(B)": prec,
                        "metrics/recall(B)": rec,
                    }

            chart_data = {
                "epochs": epochs,
                "losses": {
                    "box_loss": box_loss,
                    "cls_loss": cls_loss,
                    "dfl_loss": dfl_loss,
                },
                "metrics_history": metrics_history,
                "best_metrics": best_metrics,
            }
            return {"success": True, "data": chart_data}

        except Exception as e:
            error_msg = f"读取训练历史数据失败: {str(e)}"
            logger.error(f"[训练] {error_msg}")
            return {"success": False, "message": error_msg}

    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息

        Returns:
            系统信息（GPU、内存等）
        """
        logger.debug("[训练] 获取系统信息")

        import torch
        import psutil

        info = {
            "gpu_available": torch.cuda.is_available(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_total": psutil.virtual_memory().total,
            "memory_used": psutil.virtual_memory().used,
            "memory_percent": psutil.virtual_memory().percent,
        }

        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
                info["gpu_memory_used"] = torch.cuda.memory_allocated(0)
                info["gpu_memory_reserved"] = torch.cuda.memory_reserved(0)
                info["gpu_utilization"] = 0  # 需要 nvidia-ml-py3 获取
            except Exception as e:
                logger.warning(f"[训练] 获取GPU信息失败: {e}")

        return {"success": True, "data": info}

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
