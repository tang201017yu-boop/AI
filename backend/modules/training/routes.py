# -*- coding: utf-8 -*-
"""
训练模块路由 - Training Routes
提供模型训练、实验管理、模型导出、验证等接口
"""
import logging
import time
import os
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, Request, Query, Body
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

from backend.core.config import settings
from backend.modules.training.training_service import training_service, export_service
from backend.modules.training.project_service import project_service
from backend.modules.training.model_service import model_service

# 创建日志记录器
logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()


# ==================== 训练管理 ====================

class TrainingRequest(BaseModel):
    """训练请求数据模型"""
    project_name: str  # 项目名称
    project_id: Optional[str] = None  # 项目ID（关联训练项目）
    dataset_path: str  # 数据集路径
    model_type: str = "yolo11n"  # 模型类型
    epochs: int = 100  # 训练轮数
    batch_size: int = 32  # 批大小
    img_size: int = 640  # 输入图片尺寸
    device: str = "auto"  # 设备
    optimizer: str = "auto"  # 优化器
    amp: bool = True  # 混合精度训练
    workers: int = 8  # 数据加载线程数
    # 微调参数
    lr0: float = 0.01  # 初始学习率
    lrf: float = 0.01  # 最终学习率因子
    warmup_epochs: float = 3.0  # 预热轮数
    warmup_bias_lr: float = 0.1  # 预热期间 bias 学习率
    mosaic: float = 1.0  # 马赛克增强概率
    close_mosaic_epochs: int = 10  # 关闭马赛克轮数
    # 增强参数
    hsv_h: float = 0.015  # 色相增强
    hsv_s: float = 0.7  # 饱和度增强
    hsv_v: float = 0.4  # 亮度增强
    degrees: float = 0.0  # 旋转角度
    translate: float = 0.1  # 平移比例
    scale: float = 0.5  # 缩放比例
    shear: float = 0.0  # 剪切角度
    perspective: float = 0.0  # 透视变换
    flipud: float = 0.0  # 垂直翻转
    fliplr: float = 0.5  # 水平翻转
    mixup: float = 0.0  # 混合增强
    copy_paste: float = 0.0  # 复制粘贴增强
    # 损失函数权重
    box: float = 7.5  # 边界框损失权重
    cls: float = 0.5  # 分类损失权重
    dfl: float = 1.5  # 分布焦点损失权重
    # 训练控制
    patience: int = 100  # 早停耐心值
    save_period: int = -1  # 保存周期
    resume: bool = False  # 是否恢复训练
    # 模型路径（可选）
    model_path: Optional[str] = None  # 自定义模型路径


@router.post("/training/start")
async def start_training(request: TrainingRequest):
    """
    开始训练接口

    优化器说明:
    - optimizer: 优化器类型，支持 SGD, Adam, AdamW, NAdam, RAdam, RMSProp，设置为 "auto" 自动匹配
    - AdamW 是 YOLO 系列模型的推荐优化器

    微调参数说明:
    - lr0: 初始学习率，微调时建议 0.001-0.01
    - lrf: 最终学习率因子 (相对于 lr0)
    - warmup_epochs: 预热轮数，设置为 0 可立即使用较高学习率（适合微调）
    - warmup_bias_lr: 预热期间 bias 的学习率
    - mosaic: 马赛克增强概率 (0-1)，处理小目标时建议保持开启
    - close_mosaic_epochs: 训练后期关闭马赛克增强的轮数

    Returns:
        训练任务 ID 和实验信息
    """
    logger.info(f"[训练] 收到训练请求: 项目={request.project_name}, 数据集={request.dataset_path}, 模型={request.model_type}")

    try:
        # 转换为字典，排除 None 值
        config = {k: v for k, v in request.model_dump().items() if v is not None}

        # 调用训练服务启动训练
        result = training_service.start_training(**config)

        if result["success"]:
            logger.info(f"[训练] 训练任务已启动: task_id={result.get('task_id')}")
            return result
        else:
            error_msg = result.get("message", "启动训练失败")
            logger.error(f"[训练] 启动训练失败: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[训练] 训练请求异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.post("/training/resume")
async def resume_training(
    checkpoint_path: str,
    dataset_path: str = None
):
    """
    从检查点恢复训练接口

    Args:
        checkpoint_path: 检查点文件路径 (.pt)
        dataset_path: 数据集路径（可选）

    Returns:
        恢复后的训练任务 ID
    """
    logger.info(f"[训练] 恢复训练: checkpoint={checkpoint_path}")

    from backend.core.yolo_engine import yolo_engine

    try:
        # 调用引擎恢复训练
        task_id = yolo_engine.resume_training(checkpoint_path, dataset_path)
        logger.info(f"[训练] 恢复训练成功: task_id={task_id}")
        return {
            "success": True,
            "message": "恢复训练已开始",
            "task_id": task_id
        }
    except FileNotFoundError as e:
        logger.error(f"[训练] 检查点文件不存在: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"[训练] 恢复训练参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"[训练] 恢复训练失败: {e}")
        raise HTTPException(status_code=500, detail=f"恢复训练失败: {str(e)}")


@router.get("/training/status/{task_id}")
async def get_training_status(task_id: str):
    """
    获取训练状态接口

    Args:
        task_id: 训练任务 ID

    Returns:
        训练状态信息
    """
    logger.info(f"[训练] 查询训练状态: task_id={task_id}")

    result = training_service.get_training_status(task_id)
    logger.info(f"[训练] 查询结果: {result}")
    if result["success"]:
        return result

    error_msg = result.get("message", "任务不存在")
    logger.warning(f"[训练] 任务不存在: task_id={task_id}")
    raise HTTPException(status_code=404, detail=error_msg)


@router.get("/training/tasks")
async def list_training_tasks():
    """
    列出所有训练任务接口

    Returns:
        训练任务列表
    """
    logger.debug("[训练] 查询所有训练任务")

    tasks = training_service.list_training_tasks()
    logger.debug(f"[训练] 找到 {len(tasks)} 个训练任务")

    return {"success": True, "tasks": tasks}


@router.post("/training/cancel/{task_id}")
async def cancel_training(task_id: str):
    """
    取消训练接口

    Args:
        task_id: 训练任务 ID

    Returns:
        取消结果
    """
    logger.info(f"[训练] 尝试取消训练: task_id={task_id}")

    result = training_service.cancel_training(task_id)
    if result["success"]:
        logger.info(f"[训练] 训练已取消: task_id={task_id}")
        return result

    logger.warning(f"[训练] 取消失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result["message"])


@router.get("/training/chart-data/{task_id}")
async def get_training_chart_data(task_id: str):
    """
    获取训练图表数据接口

    Args:
        task_id: 训练任务 ID

    Returns:
        图表数据（损失曲线、性能指标等）
    """
    logger.debug(f"[训练] 获取图表数据: task_id={task_id}")

    result = training_service.get_chart_data(task_id)
    if result["success"]:
        return result

    error_msg = result.get("message", "任务不存在")
    logger.warning(f"[训练] 获取图表数据失败: task_id={task_id}")
    raise HTTPException(status_code=404, detail=error_msg)


@router.get("/training/system-info")
async def get_system_info():
    """
    获取系统信息接口

    Returns:
        系统信息（GPU、内存等）
    """
    logger.debug("[训练] 获取系统信息")

    result = training_service.get_system_info()
    return result


@router.post("/training/stop/{task_id}")
async def stop_training(task_id: str):
    """
    停止训练接口

    Args:
        task_id: 训练任务 ID

    Returns:
        停止结果
    """
    logger.info(f"[训练] 尝试停止训练: task_id={task_id}")

    result = training_service.cancel_training(task_id)
    if result["success"]:
        logger.info(f"[训练] 训练已停止: task_id={task_id}")
        return result

    logger.warning(f"[训练] 停止失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result["message"])


# ==================== 模型评估 ====================

@router.post("/training/validate")
async def validate_model(
    model_path: str,
    data: str = "coco.yaml",  # 数据集配置文件
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.6,
    rect: bool = True,
    split: str = "val"
):
    """
    验证模型性能接口

    返回详细的评估指标：
    - mAP@0.5, mAP@0.5:0.95
    - Precision, Recall
    - F1 score
    - 各类别 AP
    - 推理速度统计

    Args:
        model_path: 模型文件路径
        data: 数据集配置文件
        imgsz: 输入图片尺寸
        conf: 置信度阈值
        iou: IOU 阈值
        rect: 是否使用矩形推理
        split: 验证集划分

    Returns:
        验证指标结果
    """
    logger.info(f"[训练] 验证模型: {model_path}")

    from backend.core.yolo_engine import yolo_engine

    try:
        # 加载模型
        model = yolo_engine.load_model(model_path)
        logger.debug(f"[训练] 模型加载成功")

        # 运行验证
        results = model.val(
            data=data,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            rect=rect,
            split=split
        )

        # 提取评估指标
        metrics = {
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            "mAP50_95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            "mAP75": float(results.box.map75) if hasattr(results.box, 'map75') else 0.0,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
            "f1": float(results.box.f1) if hasattr(results.box, 'f1') else 0.0,
            "per_class_ap": dict(zip(results.box.ap_class_index, results.box.all_ap)) if hasattr(results.box, 'ap_class_index') else {}
        }

        # 速度指标
        if hasattr(results, 'speed'):
            metrics["speed"] = {
                "preprocess_ms": results.speed.get('preprocess', 0),
                "inference_ms": results.speed.get('inference', 0),
                "loss_ms": results.speed.get('loss', 0),
                "postprocess_ms": results.speed.get('postprocess', 0)
            }

        logger.info(f"[训练] 验证完成: mAP50={metrics['mAP50']:.4f}, mAP50-95={metrics['mAP50_95']:.4f}")

        return {
            "success": True,
            "model_path": model_path,
            "metrics": metrics,
            "message": "验证完成"
        }

    except Exception as e:
        logger.exception(f"[训练] 验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")


@router.get("/training/metrics/{task_id}")
async def get_training_metrics(task_id: str):
    """
    获取训练任务的详细指标接口

    返回：
    - 损失曲线 (box_loss, cls_loss, dfl_loss)
    - 精度指标 (mAP50, mAP50-95, precision, recall)
    - F1 曲线

    Args:
        task_id: 训练任务 ID

    Returns:
        训练指标数据
    """
    logger.debug(f"[训练] 获取训练指标: task_id={task_id}")

    from backend.core.yolo_engine import yolo_engine

    status = yolo_engine.get_training_status(task_id)
    if not status:
        logger.warning(f"[训练] 任务不存在: task_id={task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    chart_data = status.get_chart_data()

    logger.debug(f"[训练] 指标获取成功: epoch={status.current_epoch}/{status.total_epochs}")

    return {
        "success": True,
        "task_id": task_id,
        "status": status.status,
        "current_epoch": status.current_epoch,
        "total_epochs": status.total_epochs,
        "best_metrics": status.best_metrics,
        "chart_data": {
            "epochs": chart_data["epochs"],
            "losses": chart_data["losses"],
            "metrics_history": chart_data["metrics_history"]
        }
    }


# ==================== 实验管理 ====================

@router.get("/experiments")
async def get_experiments():
    """
    获取所有实验接口

    Returns:
        实验列表
    """
    logger.debug("[训练] 查询所有实验")

    experiments = list(training_service.experiments.values())
    logger.info(f"[训练] 找到 {len(experiments)} 个实验")

    return {
        "success": True,
        "experiments": experiments
    }


@router.get("/experiments/{task_id}")
async def get_experiment(task_id: str):
    """
    获取单个实验详情接口

    Args:
        task_id: 实验 ID

    Returns:
        实验详情
    """
    logger.debug(f"[训练] 查询实验详情: task_id={task_id}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    # 获取图表数据
    from backend.core.yolo_engine import yolo_engine
    chart_data = None
    status = None

    if yolo_engine:
        status = yolo_engine.get_training_status(task_id)
        if status:
            chart_data = status.get_chart_data()

    # 构建返回数据
    result = {
        "success": True,
        "data": {
            **experiment,
            "chart_data": chart_data,
            "best_metrics": status.best_metrics if status else experiment.get("best_metrics"),
            "checkpoint_path": status.checkpoint_path if status else experiment.get("checkpoint_path")
        }
    }
    return result


@router.delete("/experiments/{task_id}")
async def delete_experiment(task_id: str):
    """
    删除实验接口

    Args:
        task_id: 实验 ID

    Returns:
        删除结果
    """
    from pathlib import Path
    import shutil

    logger.info(f"[训练] 删除实验: task_id={task_id}")

    # 检查实验是否存在
    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    project_name = experiment.get("project_name")

    # 从 experiments 中删除
    if task_id in training_service.experiments:
        del training_service.experiments[task_id]
        training_service._save_experiments()

    # 删除实验目录（可选，保留模型文件）
    if project_name:
        project_dir = Path(settings.MODELS_DIR) / project_name
        # 检查是否还有其他实验使用这个项目
        other_experiments = [e for e in training_service.experiments.values() if e.get("project_name") == project_name]
        if not other_experiments and project_dir.exists():
            try:
                # 只删除 train 目录，保留其他文件
                train_dir = project_dir / "train"
                if train_dir.exists():
                    shutil.rmtree(train_dir)
                logger.info(f"[训练] 已删除实验目录: {project_dir}")
            except Exception as e:
                logger.warning(f"[训练] 删除实验目录失败: {e}")

    return {"success": True, "message": "实验已删除"}


@router.get("/experiments/{task_id}/results")
async def get_experiment_results(task_id: str):
    """
    获取实验验证结果图片接口

    Args:
        task_id: 实验 ID

    Returns:
        验证结果图片列表
    """
    from pathlib import Path

    logger.debug(f"[训练] 查询验证结果: task_id={task_id}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    project_name = experiment.get("project_name")
    if not project_name:
        raise HTTPException(status_code=400, detail="实验缺少项目名称")

    train_dir = Path(settings.MODELS_DIR) / project_name / "train"
    if not train_dir.exists():
        return {"success": True, "data": {"images": [], "path": str(train_dir)}}

    # 查找验证结果图片
    result_images = []
    image_patterns = [
        "confusion_matrix*.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "labels.jpg",
        "results.png"
    ]

    for pattern in image_patterns:
        for img_path in train_dir.glob(pattern):
            result_images.append({
                "name": img_path.name,
                "url": f"/models/{project_name}/train/{img_path.name}"
            })

    return {"success": True, "data": {"images": result_images, "path": str(train_dir)}}


@router.post("/experiments/{task_id}/export")
async def export_experiment_model(
    task_id: str,
    format: str = Query("onnx", description="导出格式: onnx/torchscript/tflite/pytorch")
):
    """
    导出实验模型接口

    Args:
        task_id: 实验 ID
        format: 导出格式

    Returns:
        导出结果
    """
    from pathlib import Path as PathLib

    logger.info(f"[训练] 导出模型: task_id={task_id}, format={format}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    project_name = experiment.get("project_name")
    checkpoint_path = experiment.get("checkpoint_path")

    if not checkpoint_path:
        # 查找默认路径
        best_pt = PathLib(settings.MODELS_DIR) / project_name / "train" / "weights" / "best.pt"
        if best_pt.exists():
            checkpoint_path = str(best_pt)

    if not checkpoint_path or not PathLib(checkpoint_path).exists():
        raise HTTPException(status_code=400, detail="模型文件不存在")

    # 调用导出
    from backend.core.yolo_engine import yolo_engine
    result = yolo_engine.export_model(checkpoint_path, format=format)

    return {"success": True, "data": result}


@router.post("/experiments/{task_id}/infer")
async def infer_with_experiment_model(
    task_id: str,
    image_url: str = Body(..., description="图片URL"),
    conf_threshold: float = Body(0.25, description="置信度阈值"),
    iou_threshold: float = Body(0.45, description="IOU阈值")
):
    """
    使用实验模型进行推理接口

    Args:
        task_id: 实验 ID
        image_url: 图片URL
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值

    Returns:
        推理结果
    """
    from pathlib import Path as PathLib

    logger.info(f"[训练] 推理测试: task_id={task_id}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    project_name = experiment.get("project_name")
    checkpoint_path = experiment.get("checkpoint_path")

    if not checkpoint_path:
        best_pt = PathLib(settings.MODELS_DIR) / project_name / "train" / "weights" / "best.pt"
        if best_pt.exists():
            checkpoint_path = str(best_pt)

    if not checkpoint_path or not PathLib(checkpoint_path).exists():
        raise HTTPException(status_code=400, detail="模型文件不存在")

    # 调用推理
    from backend.core.yolo_engine import yolo_engine
    result = yolo_engine.predict(
        model_path=checkpoint_path,
        source=image_url,
        conf=conf_threshold,
        iou=iou_threshold
    )

    return {"success": True, "data": result}


@router.post("/experiments/{task_id}/resume")
async def resume_training(
    task_id: str,
    epochs: int = Body(100, description="继续训练的轮数"),
    batch_size: int = Body(None, description="批量大小"),
    resume_from_best: bool = Body(True, description="是否从最佳权重继续")
):
    """
    继续训练接口

    Args:
        task_id: 实验 ID
        epochs: 继续训练的轮数
        batch_size: 批量大小
        resume_from_best: 是否从最佳权重继续

    Returns:
        新的训练任务信息
    """
    from pathlib import Path as PathLib

    logger.info(f"[训练] 继续训练: task_id={task_id}, epochs={epochs}")

    experiment = training_service.experiments.get(task_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")

    project_name = experiment.get("project_name")
    model_type = experiment.get("model_type")

    # 查找模型路径
    if resume_from_best:
        model_path = PathLib(settings.MODELS_DIR) / project_name / "train" / "weights" / "best.pt"
    else:
        model_path = PathLib(settings.MODELS_DIR) / project_name / "train" / "weights" / "last.pt"

    if not model_path.exists():
        raise HTTPException(status_code=400, detail="模型权重文件不存在")

    # 调用训练
    from backend.core.yolo_engine import yolo_engine
    result = yolo_engine.train(
        model=str(model_path),
        data=experiment.get("dataset_path"),
        epochs=epochs,
        batch_size=batch_size or experiment.get("batch_size"),
        project=project_name + "_continue",
        name=datetime.now().strftime("train_%H%M%S"),
        exist_ok=True,
        resume=True
    )

    return {"success": True, "data": result}


@router.post("/experiments/compare")
async def compare_experiments(experiment_ids: str):
    """
    比较实验接口

    Args:
        experiment_ids: 实验 ID 列表（JSON 字符串）

    Returns:
        实验比较结果
    """
    logger.info(f"[训练] 比较实验: {experiment_ids}")

    ids = json.loads(experiment_ids)
    result = training_service.compare_experiments(ids)

    return result


@router.get("/models/compare")
async def get_model_comparison():
    """
    获取模型对比接口

    Returns:
        模型列表信息
    """
    logger.debug("[训练] 获取模型对比")

    return training_service.get_model_comparison()


# ==================== 模型导出 ====================

@router.post("/export")
async def export_model(
    model_path: str,
    format: str = "onnx",
    img_size: int = 640,
    half: bool = False,
    simplify: bool = True
):
    """
    导出模型接口

    Args:
        model_path: 模型文件路径
        format: 导出格式
        img_size: 输入图片尺寸
        half: 是否使用 FP16 量化
        simplify: 是否简化模型

    Returns:
        导出结果
    """
    logger.info(f"[训练] 导出模型: {model_path}, 格式={format}")

    result = export_service.export_model(
        model_path=model_path,
        format=format,
        img_size=img_size,
        half=half,
        simplify=simplify
    )

    if result["success"]:
        logger.info(f"[训练] 导出成功: {result.get('export_path')}")
        return result

    logger.error(f"[训练] 导出失败: {result.get('message')}")
    raise HTTPException(status_code=500, detail=result["message"])


@router.get("/export/formats")
async def get_export_formats():
    """
    获取支持的导出格式接口

    Returns:
        导出格式列表
    """
    logger.debug("[训练] 获取导出格式")

    return export_service.get_export_formats()


@router.get("/export/recommended")
async def get_recommended_format(target: str):
    """
    获取推荐导出格式接口

    Args:
        target: 目标平台

    Returns:
        推荐格式信息
    """
    logger.debug(f"[训练] 获取推荐格式: target={target}")

    fmt = export_service.get_recommended_format(target)

    return {"success": True, "format": fmt, "format_info": export_service.EXPORT_FORMATS.get(fmt)}


# ==================== 训练监控 ====================

@router.get("/training/{task_id}/metrics")
async def get_training_metrics(task_id: str):
    """
    获取训练指标历史接口（用于图表展示）

    返回:
    - losses: 损失曲线历史 (box_loss, cls_loss, dfl_loss)
    - metrics: 性能指标历史 (mAP50, mAP50-95, precision, recall)
    - best_metrics: 最佳指标

    Args:
        task_id: 训练任务 ID

    Returns:
        训练指标和图表数据
    """
    logger.debug(f"[训练] 获取训练指标: task_id={task_id}")

    from backend.core.yolo_engine import yolo_engine

    status = yolo_engine.get_training_status(task_id)
    if not status:
        logger.warning(f"[训练] 任务不存在: task_id={task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    # 提取图表数据
    chart_data = status.get_chart_data() if hasattr(status, 'get_chart_data') else {
        "epochs": [],
        "losses": {"box_loss": [], "cls_loss": [], "dfl_loss": []},
        "metrics_history": [],
        "best_metrics": status.best_metrics if hasattr(status, 'best_metrics') else {}
    }

    # 构建性能指标序列
    metrics_series = {
        "mAP50": [],
        "mAP50-95": [],
        "precision": [],
        "recall": []
    }

    for record in chart_data.get("metrics_history", []):
        m = record.get("metrics", {})
        metrics_series["mAP50"].append(m.get("metrics/mAP50(B)", 0))
        metrics_series["mAP50-95"].append(m.get("metrics/mAP50-95(B)", 0))
        metrics_series["precision"].append(m.get("metrics/precision(B)", 0))
        metrics_series["recall"].append(m.get("metrics/recall(B)", 0))

    logger.debug(f"[训练] 指标数据获取成功: {len(chart_data['epochs'])} 个 epoch")

    return {
        "success": True,
        "task_id": task_id,
        "status": {
            "task_id": status.task_id,
            "status": status.status,
            "progress": status.progress,
            "current_epoch": status.current_epoch,
            "total_epochs": status.total_epochs,
            "best_metrics": chart_data["best_metrics"],
            "checkpoint_path": status.checkpoint_path,
            "error_message": status.error_message
        },
        "chart_data": {
            "epochs": chart_data["epochs"],
            "losses": chart_data["losses"],
            "metrics": metrics_series
        },
        "system_stats": {
            "gpu_memory": status.gpu_memory,
            "gpu_utilization": status.gpu_utilization,
            "system_memory": status.system_memory
        }
    }


@router.get("/training/{task_id}/checkpoints")
async def get_training_checkpoints(task_id: str):
    """
    获取训练检查点列表接口

    Args:
        task_id: 训练任务 ID

    Returns:
        检查点列表
    """
    logger.debug(f"[训练] 获取检查点: task_id={task_id}")

    from backend.core.yolo_engine import yolo_engine
    from pathlib import Path

    status = yolo_engine.get_training_status(task_id)
    if not status:
        logger.warning(f"[训练] 任务不存在: task_id={task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    checkpoints = []

    # 从项目目录查找检查点
    if status.task_id.startswith("train_"):
        timestamp = status.task_id.replace("train_", "")
        project_name = f"train_{timestamp}"

        weights_dir = Path(settings.MODELS_DIR) / project_name / "train" / "weights"
        if weights_dir.exists():
            for weight_file in weights_dir.glob("*.pt"):
                stat = weight_file.stat()
                checkpoints.append({
                    "name": weight_file.name,
                    "path": str(weight_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": stat.st_ctime
                })

    logger.info(f"[训练] 找到 {len(checkpoints)} 个检查点")

    return {
        "success": True,
        "checkpoints": sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)
    }


@router.get("/training/{task_id}/system-stats")
async def get_system_stats(task_id: str):
    """
    获取实时系统统计接口

    Args:
        task_id: 训练任务 ID

    Returns:
        系统资源使用情况
    """
    logger.debug(f"[训练] 获取系统统计: task_id={task_id}")

    from backend.core.yolo_engine import yolo_engine
    import psutil

    status = yolo_engine.get_training_status(task_id)
    if not status:
        logger.warning(f"[训练] 任务不存在: task_id={task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    stats = {
        "gpu_memory": status.gpu_memory,
        "gpu_utilization": status.gpu_utilization,
        "system_memory": status.system_memory
    }

    # 如果没有实时数据，返回当前系统状态
    if stats["system_memory"] is None:
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            stats["system_memory"] = round(mem_info.rss / (1024**3), 2)
        except:
            pass

    # GPU 实时信息
    if stats["gpu_memory"] is None:
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                stats["gpu_memory"] = f"{allocated:.1f}GB / {reserved:.1f}GB"
        except:
            pass

    return {
        "success": True,
        "task_id": task_id,
        "stats": stats
    }


# ==================== TensorBoard 和日志 ====================

@router.get("/training/{task_id}/tensorboard")
async def get_tensorboard_data(task_id: str):
    """
    获取 TensorBoard 日志数据接口

    返回:
    - log_dir: 日志目录
    - logs: 日志条目列表

    Args:
        task_id: 训练任务 ID

    Returns:
        日志数据
    """
    logger.debug(f"[训练] 获取 TensorBoard 数据: task_id={task_id}")

    from backend.core.yolo_engine import yolo_engine
    from pathlib import Path
    import json

    status = yolo_engine.get_training_status(task_id)
    if not status:
        logger.warning(f"[训练] 任务不存在: task_id={task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    logs = []
    log_dir = None

    # 从项目目录查找日志
    if status.task_id.startswith("train_"):
        timestamp = status.task_id.replace("train_", "")
        project_name = f"train_{timestamp}"

        # 查找各种日志目录
        possible_dirs = [
            settings.MODELS_DIR / project_name / "train" / "logs",
            settings.MODELS_DIR / project_name / "logs",
            settings.MODELS_DIR / project_name / "train",
        ]

        for dir_path in possible_dirs:
            if dir_path.exists():
                log_dir = str(dir_path)
                # 读取日志文件
                for log_file in dir_path.glob("*.log"):
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        log_entry = json.loads(line)
                                        logs.append(log_entry)
                                    except json.JSONDecodeError:
                                        # 非 JSON 格式，尝试解析
                                        if 'Epoch' in line or 'loss' in line.lower():
                                            logs.append({
                                                "time": str(datetime.now().isoformat()),
                                                "message": line
                                            })
                                    except:
                                        pass
                    except Exception as e:
                        logger.error(f"[训练] 读取日志文件失败: {log_file}, 错误: {e}")
                break

    # 如果没有找到日志，返回模拟数据
    if not logs and status.status == "running":
        logs = [
            {"time": status.started_at.isoformat() if status.started_at else None, "epoch": 0, "message": "训练开始"},
            {"time": str(datetime.now().isoformat()), "epoch": status.current_epoch, "message": f"正在训练 Epoch {status.current_epoch}/{status.total_epochs}"}
        ]

    logger.debug(f"[训练] 获取 {len(logs)} 条日志")

    return {
        "success": True,
        "task_id": task_id,
        "log_dir": log_dir,
        "logs": logs[-100:]  # 只返回最近的100条日志
    }


@router.get("/training/{task_id}/export/logs")
async def export_training_logs(task_id: str):
    """
    导出训练日志接口

    Args:
        task_id: 训练任务 ID

    Returns:
        训练日志 JSON 文件
    """
    logger.info(f"[训练] 导出日志: task_id={task_id}")

    from backend.core.yolo_engine import yolo_engine
    from fastapi.responses import StreamingResponse
    from datetime import datetime

    status = yolo_engine.get_training_status(task_id)
    if not status:
        logger.warning(f"[训练] 任务不存在: task_id={task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    # 构建日志数据
    log_data = {
        "task_id": task_id,
        "exported_at": datetime.now().isoformat(),
        "status": status.status,
        "progress": status.progress,
        "epochs": {
            "current": status.current_epoch,
            "total": status.total_epochs
        },
        "metrics": status.metrics,
        "chart_data": status.get_chart_data() if hasattr(status, 'get_chart_data') else {}
    }

    def generate():
        yield json.dumps(log_data, ensure_ascii=False, indent=2)

    logger.info(f"[训练] 日志导出成功: task_id={task_id}")

    return StreamingResponse(
        generate(),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=training_logs_{task_id}.json"}
    )


@router.get("/training/{task_id}/config")
async def get_training_config(task_id: str):
    """
    获取训练配置接口

    Args:
        task_id: 训练任务 ID

    Returns:
        训练配置信息
    """
    logger.debug(f"[训练] 获取训练配置: task_id={task_id}")

    from backend.core.yolo_engine import yolo_engine

    status = yolo_engine.get_training_status(task_id)
    if not status:
        logger.warning(f"[训练] 任务不存在: task_id={task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")

    # 返回训练配置信息
    config = {
        "task_id": task_id,
        "status": status.status,
        "project_name": getattr(status, 'project_name', task_id),
        "epochs": status.total_epochs,
        "current_epoch": status.current_epoch,
        "metrics": status.metrics,
        "started_at": status.started_at.isoformat() if status.started_at else None,
        "best_metrics": status.best_metrics if hasattr(status, 'best_metrics') else {}
    }

    return {
        "success": True,
        "config": config
    }


# ==================== 项目管理 ====================

class CreateProjectRequest(BaseModel):
    """创建项目请求模型"""
    name: str
    description: str = ""
    cover_image: str = None
    task_type: str = "detect"
    settings: dict = None


@router.post("/projects")
async def create_project(request: CreateProjectRequest):
    """
    创建新项目接口

    Args:
        request: 创建项目请求

    Returns:
        创建结果
    """
    logger.info(f"[训练] 创建项目: {request.name}, task_type: {request.task_type}")

    result = project_service.create_project(
        request.name,
        request.description,
        request.cover_image,
        request.task_type,
        request.settings
    )
    if result["success"]:
        logger.info(f"[训练] 项目创建成功: {result.get('project', {}).get('id')}")
        return result

    logger.error(f"[训练] 项目创建失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/projects")
async def list_projects(include_deleted: bool = False):
    """
    列出所有项目接口

    Args:
        include_deleted: 是否包含已删除项目

    Returns:
        项目列表
    """
    logger.debug(f"[训练] 列出项目: include_deleted={include_deleted}")

    return project_service.list_projects(include_deleted)


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """
    获取项目详情接口

    Args:
        project_id: 项目 ID

    Returns:
        项目详情
    """
    logger.debug(f"[训练] 获取项目: {project_id}")

    result = project_service.get_project(project_id)
    if result["success"]:
        return result

    logger.warning(f"[训练] 项目不存在: {project_id}")
    raise HTTPException(status_code=404, detail=result.get("message"))


@router.put("/projects/{project_id}")
async def update_project(
    project_id: str,
    name: str = None,
    description: str = None,
    cover_image: str = None,
    settings: Dict = None
):
    """
    更新项目接口

    Args:
        project_id: 项目 ID
        name: 新名称
        description: 新描述
        cover_image: 新封面
        settings: 新设置

    Returns:
        更新结果
    """
    logger.info(f"[训练] 更新项目: {project_id}")

    result = project_service.update_project(
        project_id, name, description, cover_image, settings
    )
    if result["success"]:
        logger.info(f"[训练] 项目更新成功: {project_id}")
        return result

    logger.error(f"[训练] 项目更新失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, permanent: bool = False):
    """
    删除项目接口

    Args:
        project_id: 项目 ID
        permanent: 是否永久删除

    Returns:
        删除结果
    """
    logger.info(f"[训练] 删除项目: {project_id}, permanent={permanent}")

    result = project_service.delete_project(project_id, permanent)
    if result["success"]:
        logger.info(f"[训练] 项目已删除: {project_id}")
        return result

    logger.error(f"[训练] 删除失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.post("/projects/{project_id}/restore")
async def restore_project(project_id: str):
    """
    从回收站恢复项目接口

    Args:
        project_id: 项目 ID

    Returns:
        恢复结果
    """
    logger.info(f"[训练] 恢复项目: {project_id}")

    result = project_service.restore_project(project_id)
    if result["success"]:
        logger.info(f"[训练] 项目已恢复: {project_id}")
        return result

    logger.error(f"[训练] 恢复失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/projects/recycle-bin")
async def get_recycle_bin():
    """
    获取回收站接口

    Returns:
        回收站项目列表
    """
    logger.debug("[训练] 获取回收站")

    return project_service.get_recycle_bin()


@router.post("/projects/recycle-bin/empty")
async def empty_recycle_bin():
    """
    清空回收站接口

    Returns:
        清空结果
    """
    logger.info("[训练] 清空回收站")

    return project_service.empty_recycle_bin()


# ==================== 模型管理 ====================

@router.post("/projects/{project_id}/models")
async def add_model(
    project_id: str,
    model_path: str,
    model_type: str = "yolo",
    metrics: str = None  # JSON string
):
    """
    添加模型到项目接口

    Args:
        project_id: 项目 ID
        model_path: 模型路径
        model_type: 模型类型
        metrics: 模型指标（JSON 字符串）

    Returns:
        添加结果
    """
    logger.info(f"[训练] 添加模型到项目: {project_id}, path={model_path}")

    model_metrics = json.loads(metrics) if metrics else None
    result = project_service.add_model(project_id, model_path, model_type, model_metrics)
    if result["success"]:
        logger.info(f"[训练] 模型添加成功")
        return result

    logger.error(f"[训练] 模型添加失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/projects/{project_id}/models")
async def get_project_models(project_id: str):
    """
    获取项目模型列表接口

    Args:
        project_id: 项目 ID

    Returns:
        模型列表
    """
    logger.debug(f"[训练] 获取项目模型: {project_id}")

    return project_service.get_models(project_id)


@router.delete("/projects/{project_id}/models/{model_id}")
async def delete_project_model(project_id: str, model_id: str):
    """
    删除模型接口

    Args:
        project_id: 项目 ID
        model_id: 模型 ID

    Returns:
        删除结果
    """
    logger.info(f"[训练] 删除模型: {project_id}/{model_id}")

    result = project_service.delete_model(project_id, model_id)
    if result["success"]:
        logger.info(f"[训练] 模型已删除")
        return result

    logger.error(f"[训练] 删除失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.post("/projects/{project_id}/models/{model_id}/migrate")
async def migrate_model(
    project_id: str,
    model_id: str,
    target_project_id: str
):
    """
    迁移模型到另一个项目接口

    Args:
        project_id: 源项目 ID
        model_id: 模型 ID
        target_project_id: 目标项目 ID

    Returns:
        迁移结果
    """
    logger.info(f"[训练] 迁移模型: {project_id}/{model_id} -> {target_project_id}")

    result = project_service.migrate_model(project_id, model_id, target_project_id)
    if result["success"]:
        logger.info(f"[训练] 模型迁移成功")
        return result

    logger.error(f"[训练] 迁移失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


# ==================== 活动日志 ====================

@router.get("/projects/{project_id}/activity")
async def get_activity_log(project_id: str, limit: int = 50):
    """
    获取项目活动日志接口

    Args:
        project_id: 项目 ID
        limit: 返回条数限制

    Returns:
        活动日志列表
    """
    logger.debug(f"[训练] 获取活动日志: {project_id}, limit={limit}")

    return project_service.get_activity_log(project_id, limit)


# ==================== 模型比较 ====================

@router.get("/projects/{project_id}/compare")
async def compare_models(project_id: str, model_ids: str = None):
    """
    比较模型性能接口

    Args:
        project_id: 项目 ID
        model_ids: 模型 ID 列表（JSON 字符串）

    Returns:
        比较结果
    """
    logger.debug(f"[训练] 比较模型: {project_id}")

    model_id_list = json.loads(model_ids) if model_ids else None
    return project_service.compare_models(project_id, model_id_list)


# ==================== 模型上传与管理 ====================

@router.post("/models/upload")
async def upload_model(
    file,
    project_id: str = None,
    name: str = None,
    description: str = ""
):
    """
    上传模型文件接口

    Args:
        file: 模型文件
        project_id: 项目 ID
        name: 模型名称
        description: 模型描述

    Returns:
        上传结果
    """
    logger.info(f"[训练] 上传模型: {file.filename}, project={project_id}")

    from fastapi import UploadFile

    result = model_service.upload_model(
        file=file,
        project_id=project_id,
        name=name,
        description=description
    )

    if result["success"]:
        logger.info(f"[训练] 模型上传成功: {result.get('model', {}).get('id')}")
        return result

    logger.error(f"[训练] 上传失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/models")
async def list_models(project_id: str = None):
    """
    列出所有模型接口

    Args:
        project_id: 项目 ID（可选）

    Returns:
        模型列表
    """
    logger.debug(f"[训练] 列出模型: project_id={project_id}")

    return model_service.list_models(project_id)


@router.get("/models")
async def list_user_models():
    """
    获取用户模型列表（上传的模型 + 训练项目的模型）

    Returns:
        用户模型列表
    """
    logger.debug("[训练] 获取用户模型列表")
    return model_service.list_models()


@router.get("/models/list")
async def list_models_alias(project_id: str = None):
    """
    列出所有模型接口（兼容前端）

    Args:
        project_id: 项目 ID（可选）

    Returns:
        模型列表
    """
    return model_service.list_models(project_id)


@router.get("/models/pretrained")
async def list_pretrained_models():
    """
    列出可用的预训练基础模型

    Returns:
        预训练模型列表
    """
    from pathlib import Path

    # 获取模型搜索路径
    from backend.core.config import settings

    pretrained_models = []
    seen = set()

    for root in settings.model_search_paths:
        if not root.exists():
            continue
        for path in root.glob("*.pt"):
            if path.name in seen:
                continue
            seen.add(path.name)

            # 检查是否为预训练模型（简单判断：文件名是模型名如 yolo11n.pt）
            if path.stat().st_size > 1000000:  # 大于1MB
                pretrained_models.append({
                    "name": path.stem,
                    "path": str(path),
                    "size": path.stat().st_size
                })

    # 添加 Ultralytics 官方预训练模型列表（如果未安装，会在训练时自动下载）
    official_models = [
        {"name": "yolo11n", "display": "YOLO11n (nano) - 最快"},
        {"name": "yolo11s", "display": "YOLO11s (small) - 快速"},
        {"name": "yolo11m", "display": "YOLO11m (medium) - 平衡"},
        {"name": "yolo11l", "display": "YOLO11l (large) - 高精度"},
        {"name": "yolo11x", "display": "YOLO11x (xlarge) - 最高精度"},
        {"name": "yolo26n", "display": "YOLO26n (nano) - 最新最快"},
        {"name": "yolo26s", "display": "YOLO26s (small) - 最新快速"},
        {"name": "yolo26m", "display": "YOLO26m (medium) - 最新平衡"},
        {"name": "yolo26l", "display": "YOLO26l (large) - 最新高精度"},
        {"name": "yolo26x", "display": "YOLO26x (xlarge) - 最新最高精度"},
    ]

    # 标记已安装的模型
    installed_names = {m["name"] for m in pretrained_models}
    for model in official_models:
        model["installed"] = model["name"] in installed_names
        model["path"] = next((m["path"] for m in pretrained_models if m["name"] == model["name"]), None)

    return {
        "success": True,
        "models": official_models,
        "installed": [m["name"] for m in pretrained_models]
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """
    获取模型详情接口

    Args:
        model_id: 模型 ID

    Returns:
        模型详情
    """
    logger.debug(f"[训练] 获取模型: {model_id}")

    result = model_service.get_model(model_id)
    if result["success"]:
        return result

    logger.warning(f"[训练] 模型不存在: {model_id}")
    raise HTTPException(status_code=404, detail=result.get("message"))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    删除模型接口

    Args:
        model_id: 模型 ID

    Returns:
        删除结果
    """
    logger.info(f"[训练] 删除模型: {model_id}")

    result = model_service.delete_model(model_id)
    if result["success"]:
        logger.info(f"[训练] 模型已删除: {model_id}")
        return result

    logger.error(f"[训练] 删除失败: {result.get('message')}")
    raise HTTPException(status_code=404, detail=result.get("message"))


# ==================== 模型指标 ====================

@router.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """
    获取模型训练指标接口

    Args:
        model_id: 模型 ID

    Returns:
        训练指标数据
    """
    logger.debug(f"[训练] 获取模型指标: {model_id}")

    return model_service.get_training_metrics(model_id)


@router.get("/models/{model_id}/charts")
async def get_validation_charts(model_id: str):
    """
    获取验证图表数据接口

    Args:
        model_id: 模型 ID

    Returns:
        图表数据
    """
    logger.debug(f"[训练] 获取验证图表: {model_id}")

    return model_service.get_validation_charts(model_id)


# ==================== 模型导出 ====================

@router.post("/models/{model_id}/export")
async def export_model(
    model_id: str,
    format: str = "onnx",
    img_size: int = 640,
    half: bool = False,
    simplify: bool = True
):
    """
    导出模型接口

    Args:
        model_id: 模型 ID
        format: 导出格式
        img_size: 输入图片尺寸
        half: 是否使用 FP16 量化
        simplify: 是否简化模型

    Returns:
        导出结果
    """
    logger.info(f"[训练] 导出模型: {model_id}, format={format}")

    result = model_service.export_model(
        model_id=model_id,
        format=format,
        img_size=img_size,
        half=half,
        simplify=simplify
    )
    if result["success"]:
        logger.info(f"[训练] 导出成功")
        return result

    logger.error(f"[训练] 导出失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/models/export/formats")
async def get_export_formats():
    """
    获取支持的导出格式接口

    Returns:
        导出格式列表
    """
    logger.debug("[训练] 获取导出格式")

    return model_service.get_export_formats()


@router.get("/models/{model_id}/exports")
async def get_model_exports(model_id: str):
    """
    获取模型的导出历史接口

    Args:
        model_id: 模型 ID

    Returns:
        导出历史列表
    """
    logger.debug(f"[训练] 获取导出历史: {model_id}")

    return model_service.get_export_status(model_id)


# ==================== 模型推理测试 ====================

@router.post("/models/{model_id}/infer")
async def test_model_inference(
    model_id: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    device: str = "auto",
    half: bool = False,
    file: UploadFile = None
):
    """
    测试模型推理接口

    Args:
        model_id: 模型 ID
        conf_threshold: 置信度阈值
        iou_threshold: IOU 阈值
        img_size: 输入图片尺寸
        device: 设备
        half: 是否使用 FP16
        file: 测试图片

    Returns:
        推理结果
    """
    logger.info(f"[训练] 测试推理: {model_id}")

    image_data = await file.read() if file else None

    result = model_service.test_inference(
        model_id=model_id,
        source=image_data,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        img_size=img_size,
        device=device,
        half=half
    )

    if result["success"]:
        logger.info(f"[训练] 推理测试成功: {result.get('num_detections', 0)} 个检测")
        return result

    logger.error(f"[训练] 推理测试失败: {result.get('message')}")
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.post("/models/{model_id}/batch-infer")
async def batch_inference(
    model_id: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    device: str = "auto",
    half: bool = False,
    files: List[UploadFile] = []
):
    """
    批量推理接口

    支持同时上传多张图片进行批量推理，返回每张图片的检测结果

    Args:
        model_id: 模型 ID
        conf_threshold: 置信度阈值
        iou_threshold: IOU 阈值
        img_size: 输入图片尺寸
        device: 设备
        half: 是否使用 FP16
        files: 测试图片列表

    Returns:
        批量推理结果
    """
    logger.info(f"[训练] 批量推理: {model_id}, 文件数={len(files)}")

    if not files:
        logger.warning("[训练] 没有上传图片")
        raise HTTPException(status_code=400, detail="请上传至少一张图片")

    # 保存临时文件
    temp_paths = []
    for f in files:
        temp_path = f"/tmp/infer_{int(time.time())}_{f.filename}"
        with open(temp_path, 'wb') as tmp:
            tmp.write(await f.read())
        temp_paths.append(temp_path)

    try:
        result = model_service.batch_inference(
            model_id=model_id,
            image_paths=temp_paths,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            img_size=img_size,
            device=device,
            half=half
        )
        logger.info(f"[训练] 批量推理完成: {len(temp_paths)} 张图片")
        return result
    finally:
        # 清理临时文件
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass


@router.get("/models/{model_id}/export/history")
async def get_export_history(model_id: str):
    """
    获取模型导出历史接口

    Args:
        model_id: 模型 ID

    Returns:
        导出历史
    """
    logger.debug(f"[训练] 获取导出历史: {model_id}")

    return model_service.get_export_status(model_id)
