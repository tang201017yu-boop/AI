"""
训练模块路由 - Training Routes
"""
from fastapi import APIRouter, HTTPException, UploadFile
from typing import Optional, Dict, Any
import json

from backend.core.config import settings
from backend.modules.training.training_service import training_service, export_service
from backend.modules.training.project_service import project_service
from backend.modules.training.model_service import model_service

router = APIRouter()


# ==================== 训练管理 ====================

@router.post("/training/start")
async def start_training(
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
    close_mosaic_epochs: int = 10,
    **kwargs
):
    """
    开始训练

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
    """
    result = training_service.start_training(
        project_name=project_name,
        dataset_path=dataset_path,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        optimizer=optimizer,
        amp=amp,
        workers=workers,
        lr0=lr0,
        lrf=lrf,
        warmup_epochs=warmup_epochs,
        warmup_bias_lr=warmup_bias_lr,
        mosaic=mosaic,
        close_mosaic_epochs=close_mosaic_epochs,
        **kwargs
    )
    if result["success"]:
        return result
    raise HTTPException(status_code=500, detail=result["message"])


@router.post("/training/resume")
async def resume_training(
    checkpoint_path: str,
    dataset_path: str = None
):
    """
    从检查点恢复训练

    Args:
        checkpoint_path: 检查点文件路径 (.pt)
        dataset_path: 数据集路径（可选）
    """
    from backend.core.yolo_engine import yolo_engine

    try:
        task_id = yolo_engine.resume_training(checkpoint_path, dataset_path)
        return {
            "success": True,
            "message": "恢复训练已开始",
            "task_id": task_id
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复训练失败: {str(e)}")


@router.get("/training/status/{task_id}")
async def get_training_status(task_id: str):
    """获取训练状态"""
    result = training_service.get_training_status(task_id)
    if result["success"]:
        return result
    raise HTTPException(status_code=404, detail=result.get("message"))


@router.get("/training/tasks")
async def list_training_tasks():
    """列出所有训练任务"""
    return {"success": True, "tasks": training_service.list_training_tasks()}


@router.post("/training/cancel/{task_id}")
async def cancel_training(task_id: str):
    """取消训练"""
    result = training_service.cancel_training(task_id)
    if result["success"]:
        return result
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
    验证模型性能

    返回详细的评估指标：
    - mAP@0.5, mAP@0.5:0.95
    - Precision, Recall
    - F1 score
    - 各类别 AP
    - 推理速度统计
    """
    from backend.core.yolo_engine import yolo_engine

    try:
        # 加载模型
        model = yolo_engine.load_model(model_path)

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

        return {
            "success": True,
            "model_path": model_path,
            "metrics": metrics,
            "message": "验证完成"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")


@router.get("/training/metrics/{task_id}")
async def get_training_metrics(task_id: str):
    """
    获取训练任务的详细指标

    返回：
    - 损失曲线 (box_loss, cls_loss, dfl_loss)
    - 精度指标 (mAP50, mAP50-95, precision, recall)
    - F1 曲线
    """
    from backend.core.yolo_engine import yolo_engine

    status = yolo_engine.get_training_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在")

    chart_data = status.get_chart_data()

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
    """获取所有实验"""
    return {
        "success": True,
        "experiments": list(training_service.experiments.values())
    }


@router.post("/experiments/compare")
async def compare_experiments(experiment_ids: str):  # JSON string
    """比较实验"""
    ids = json.loads(experiment_ids)
    return training_service.compare_experiments(ids)


@router.get("/models/compare")
async def get_model_comparison():
    """获取模型对比"""
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
    """导出模型"""
    result = export_service.export_model(
        model_path=model_path,
        format=format,
        img_size=img_size,
        half=half,
        simplify=simplify
    )
    if result["success"]:
        return result
    raise HTTPException(status_code=500, detail=result["message"])


@router.get("/export/formats")
async def get_export_formats():
    """获取支持的导出格式"""
    return export_service.get_export_formats()


@router.get("/export/recommended")
async def get_recommended_format(target: str):
    """获取推荐格式"""
    fmt = export_service.get_recommended_format(target)
    return {"success": True, "format": fmt, "format_info": export_service.EXPORT_FORMATS.get(fmt)}


# ==================== 训练监控 ====================

@router.get("/training/{task_id}/metrics")
async def get_training_metrics(task_id: str):
    """
    获取训练指标历史（用于图表）

    返回:
    - losses: 损失曲线历史 (box_loss, cls_loss, dfl_loss)
    - metrics: 性能指标历史 (mAP50, mAP50-95, precision, recall)
    - best_metrics: 最佳指标
    """
    from backend.core.yolo_engine import yolo_engine

    status = yolo_engine.get_training_status(task_id)
    if not status:
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
    """获取训练检查点列表"""
    from backend.core.yolo_engine import yolo_engine
    from pathlib import Path

    status = yolo_engine.get_training_status(task_id)
    if not status:
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

    return {
        "success": True,
        "checkpoints": sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)
    }


@router.get("/training/{task_id}/system-stats")
async def get_system_stats(task_id: str):
    """获取实时系统统计"""
    from backend.core.yolo_engine import yolo_engine
    import psutil

    status = yolo_engine.get_training_status(task_id)
    if not status:
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


# ==================== 项目管理 ====================

@router.post("/projects")
async def create_project(
    name: str,
    description: str = "",
    cover_image: str = None
):
    """创建新项目"""
    result = project_service.create_project(name, description, cover_image)
    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/projects")
async def list_projects(include_deleted: bool = False):
    """列出所有项目"""
    return project_service.list_projects(include_deleted)


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """获取项目详情"""
    result = project_service.get_project(project_id)
    if result["success"]:
        return result
    raise HTTPException(status_code=404, detail=result.get("message"))


@router.put("/projects/{project_id}")
async def update_project(
    project_id: str,
    name: str = None,
    description: str = None,
    cover_image: str = None,
    settings: Dict = None
):
    """更新项目"""
    result = project_service.update_project(
        project_id, name, description, cover_image, settings
    )
    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, permanent: bool = False):
    """删除项目"""
    result = project_service.delete_project(project_id, permanent)
    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.post("/projects/{project_id}/restore")
async def restore_project(project_id: str):
    """从回收站恢复项目"""
    result = project_service.restore_project(project_id)
    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/projects/recycle-bin")
async def get_recycle_bin():
    """获取回收站"""
    return project_service.get_recycle_bin()


@router.post("/projects/recycle-bin/empty")
async def empty_recycle_bin():
    """清空回收站"""
    return project_service.empty_recycle_bin()


# ==================== 模型管理 ====================

@router.post("/projects/{project_id}/models")
async def add_model(
    project_id: str,
    model_path: str,
    model_type: str = "yolo",
    metrics: str = None  # JSON string
):
    """添加模型到项目"""
    model_metrics = json.loads(metrics) if metrics else None
    result = project_service.add_model(project_id, model_path, model_type, model_metrics)
    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/projects/{project_id}/models")
async def get_project_models(project_id: str):
    """获取项目模型列表"""
    return project_service.get_models(project_id)


@router.delete("/projects/{project_id}/models/{model_id}")
async def delete_project_model(project_id: str, model_id: str):
    """删除模型"""
    result = project_service.delete_model(project_id, model_id)
    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.post("/projects/{project_id}/models/{model_id}/migrate")
async def migrate_model(
    project_id: str,
    model_id: str,
    target_project_id: str
):
    """迁移模型到另一个项目"""
    result = project_service.migrate_model(project_id, model_id, target_project_id)
    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))


# ==================== 活动日志 ====================

@router.get("/projects/{project_id}/activity")
async def get_activity_log(project_id: str, limit: int = 50):
    """获取项目活动日志"""
    return project_service.get_activity_log(project_id, limit)


# ==================== 模型比较 ====================

@router.get("/projects/{project_id}/compare")
async def compare_models(project_id: str, model_ids: str = None):
    """比较模型性能"""
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
    """上传模型文件"""
    from fastapi import UploadFile

    result = model_service.upload_model(
        file=file,
        project_id=project_id,
        name=name,
        description=description
    )

    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/models")
async def list_models(project_id: str = None):
    """列出所有模型"""
    return model_service.list_models(project_id)


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """获取模型详情"""
    result = model_service.get_model(model_id)
    if result["success"]:
        return result
    raise HTTPException(status_code=404, detail=result.get("message"))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """删除模型"""
    result = model_service.delete_model(model_id)
    if result["success"]:
        return result
    raise HTTPException(status_code=404, detail=result.get("message"))


# ==================== 模型指标 ====================

@router.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """获取模型训练指标"""
    return model_service.get_training_metrics(model_id)


@router.get("/models/{model_id}/charts")
async def get_validation_charts(model_id: str):
    """获取验证图表数据"""
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
    """导出模型"""
    result = model_service.export_model(
        model_id=model_id,
        format=format,
        img_size=img_size,
        half=half,
        simplify=simplify
    )
    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))


@router.get("/models/export/formats")
async def get_export_formats():
    """获取支持的导出格式"""
    return model_service.get_export_formats()


@router.get("/models/{model_id}/exports")
async def get_model_exports(model_id: str):
    """获取模型的导出历史"""
    return model_service.get_export_status(model_id)


# ==================== 模型推理测试 ====================

@router.post("/models/{model_id}/infer")
async def test_model_inference(
    model_id: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    file: UploadFile = None
):
    """测试模型推理"""
    image_data = await file.read() if file else None

    result = model_service.test_inference(
        model_id=model_id,
        image_data=image_data,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    if result["success"]:
        return result
    raise HTTPException(status_code=400, detail=result.get("message"))
