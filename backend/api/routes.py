"""
API 路由定义
"""
import sys
import re
import os
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.core.config import settings
from backend.models.schemas import (
    InferenceRequest, InferenceResponse, TrainingConfig,
    TrainingStatus, ModelInfo, DatasetInfo, ExportConfig,
    SystemInfo,
    ObjectCountingRequest, HeatmapRequest, SpeedEstimationRequest,
    DistanceCalculationRequest, ObjectBlurRequest, ObjectCropRequest,
    QueueManagementRequest, SolutionResponse
)
from backend.services.yolo_service import yolo_service
from backend.core.yolo_engine import yolo_engine
from backend.services import annotation_service, dataset_service, solutions_service
from backend.services.supervision_service import supervision_service
from backend.modules.training.training_service import training_service
from backend.modules.training.project_service import project_service
from backend.modules.training import routes as training_routes
from backend.utils.file_utils import allowed_file, save_uploaded_file, get_unique_filename

router = APIRouter()


class DatasetProjectCreate(BaseModel):
    name: str = Field(..., min_length=1)


class DatasetProjectRename(BaseModel):
    new_name: str = Field(..., min_length=1)


# 挂载训练模块路由
router.include_router(training_routes.router, prefix="/training", tags=["训练"])


# ==================== 系统信息 ====================
@router.get("/system/info", response_model=SystemInfo)
async def get_system_info():
    """获取系统信息"""
    import platform
    
    # 获取模型和数据集数量
    models = yolo_service.list_models() if yolo_service else []
    datasets = dataset_service.list_datasets()
    
    # 获取 GPU 信息
    gpu_available, gpu_info = (False, None)
    if yolo_service:
        gpu_available, gpu_info = yolo_service.get_device_info()
    
    # 获取 ultralytics 版本
    ultralytics_version = "N/A"
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except:
        pass
    
    return SystemInfo(
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        python_version=platform.python_version(),
        ultralytics_version=ultralytics_version,
        total_models=len(models),
        total_datasets=len(datasets),
        gpu_available=gpu_available,
        gpu_info=gpu_info
    )


@router.get("/system/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "yolo_service": yolo_service is not None,
        "supervision_available": True
    }


# ==================== 推理相关 ====================
@router.post("/inference/image")  # 移除 response_model 以支持 annotated_image
async def infer_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    model_path: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    img_size: Optional[int] = Form(None),
    draw_results: bool = Form(True)
):
    """图像推理"""
    if not yolo_service:
        raise HTTPException(status_code=500, detail="YOLO service not available")

    # 验证文件类型
    if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "mp4", "avi", "mov", "mkv", "flv", "wmv"]):
        raise HTTPException(status_code=400, detail="不支持的文件类型，请上传图片或视频")

    try:
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))

        # 执行推理
        result = yolo_service.infer(
            image_path=str(file_path),
            model_identifier=model_path or model_name,
            confidence=confidence,
            iou_threshold=iou_threshold,
            img_size=img_size
        )

        # 转换为字典（因为 yolo_service.infer 返回的是 Pydantic 模型）
        result_dict = result.model_dump() if hasattr(result, 'model_dump') else dict(result)

        # 如果需要绘制检测结果
        if draw_results and result_dict.get("success"):
            try:
                from backend.modules.inference.inference_service import inference_service
                detections = result_dict.get("detections", [])
                if detections:
                    annotated_image = inference_service._draw_detections(str(file_path), detections)
                    result_dict["annotated_image"] = annotated_image
                    print(f"[推理] 绘制标注图片成功，长度: {len(annotated_image)}")
            except Exception as e:
                import traceback
                print(f"[推理] 绘制标注图片失败: {e}")
                traceback.print_exc()

        return result_dict

    except Exception as e:
        import traceback
        print(f"[推理] 推理失败: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/batch")
async def infer_batch(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Form(None),
    model_path: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None)
):
    """批量推理"""
    if not yolo_service:
        raise HTTPException(status_code=500, detail="YOLO service not available")
    
    results = []
    for file in files:
        if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp"]):
            continue
        
        try:
            filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])
            file_path = settings.UPLOADS_DIR / filename
            save_uploaded_file(file, str(file_path))
            
            result = yolo_service.infer(
                image_path=str(file_path),
                model_identifier=model_path or model_name,
                confidence=confidence
            )
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}


# ==================== 训练相关 ====================
@router.post("/training/start")
async def start_training(config: TrainingConfig):
    """开始训练"""
    try:
        # 使用 training_service 来处理训练启动（包含数据集路径解析）
        result = training_service.start_training(
            project_name=config.project_name,
            dataset_path=config.dataset_path,
            model_type=config.model_type,
            epochs=config.epochs,
            batch_size=config.batch_size,
            img_size=config.img_size,
            device=config.device,
            optimizer=getattr(config, 'optimizer', 'auto'),
            amp=getattr(config, 'amp', True),
            workers=getattr(config, 'workers', 8),
            lr0=getattr(config, 'lr0', 0.01),
            lrf=getattr(config, 'lrf', 0.01),
            warmup_epochs=getattr(config, 'warmup_epochs', 3.0),
            warmup_bias_lr=getattr(config, 'warmup_bias_lr', 0.1),
            mosaic=getattr(config, 'mosaic', 1.0),
            close_mosaic_epochs=getattr(config, 'close_mosaic', 10)
        )
        if result.get("success"):
            return {
                "success": True,
                "task_id": result.get("task_id"),
                "message": "训练已开始"
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("message", "启动训练失败"))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status/{task_id}")
async def get_training_status(task_id: str):
    """获取训练状态"""
    # 优先从 training_service 获取（支持历史任务）
    from backend.modules.training.training_service import training_service
    result = training_service.get_training_status(task_id)
    if result.get("success"):
        return clean_nan_values(result["status"])

    # 如果 training_service 找不到，尝试从 yolo_engine 获取
    if not yolo_engine:
        raise HTTPException(status_code=500, detail="YOLO engine not available")

    try:
        status = yolo_engine.get_training_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        status_dict = status_to_dict(status)
        return clean_nan_values(status_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] 获取训练状态失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def clean_nan_values(obj):
    """递归清理字典/列表中的 NaN 值"""
    import math
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def status_to_dict(status) -> dict:
    """将 TrainingStatus 转换为字典，处理各种 Pydantic 版本"""
    # 方法1: 尝试 model_dump (Pydantic v2)
    if hasattr(status, 'model_dump'):
        try:
            return status.model_dump()
        except Exception:
            pass
    # 方法2: 尝试 model_dump(mode='json') (Pydantic v2)
    if hasattr(status, 'model_dump'):
        try:
            return status.model_dump(mode='json')
        except Exception:
            pass
    # 方法3: 尝试 dict() (Pydantic v1)
    if hasattr(status, 'dict'):
        try:
            return status.dict()
        except Exception:
            pass
    # 方法4: 手动构建字典
    return {
        'task_id': status.task_id,
        'status': status.status,
        'progress': status.progress,
        'current_epoch': status.current_epoch,
        'total_epochs': status.total_epochs,
        'metrics': status.metrics,
        'project_name': getattr(status, 'project_name', None),
        'created_at': status.created_at.isoformat() if hasattr(status.created_at, 'isoformat') else str(status.created_at),
        'updated_at': status.updated_at.isoformat() if hasattr(status.updated_at, 'isoformat') else str(status.updated_at),
        'error_message': status.error_message,
        'gpu_memory': status.gpu_memory,
    }


@router.get("/training/tasks")
async def list_training_tasks():
    """列出所有训练任务"""
    if not yolo_engine:
        raise HTTPException(status_code=500, detail="YOLO engine not available")

    try:
        tasks = yolo_engine.list_training_statuses()
        # 清理 NaN 值
        tasks_data = []
        for task in tasks:
            task_dict = status_to_dict(task)
            tasks_data.append(clean_nan_values(task_dict))
        return {"tasks": tasks_data}
    except Exception as e:
        logger.error(f"[API] 获取训练任务列表失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"tasks": []}
        import traceback
        traceback.print_exc()
        return {"tasks": []}


# ==================== 项目管理 ====================
@router.post("/training/projects")
async def create_project(name: str, description: str = "", cover_image: str = None):
    """创建新项目"""
    logger.debug(f"[API] 创建项目: name={name}")
    result = project_service.create_project(name, description, cover_image)
    return result


@router.get("/training/projects")
async def list_projects(include_deleted: bool = False):
    """列出所有项目"""
    logger.debug(f"[API] 列出项目: include_deleted={include_deleted}")
    result = project_service.list_projects(include_deleted)
    return result


@router.get("/training/projects/{project_id}")
async def get_project(project_id: str):
    """获取项目详情"""
    logger.debug(f"[API] 获取项目: project_id={project_id}")
    result = project_service.get_project(project_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@router.put("/training/projects/{project_id}")
async def update_project(project_id: str, data: dict):
    """更新项目"""
    logger.debug(f"[API] 更新项目: project_id={project_id}")
    result = project_service.update_project(
        project_id,
        name=data.get("name"),
        description=data.get("description"),
        cover_image=data.get("cover_image"),
        settings=data.get("settings")
    )
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@router.delete("/training/projects/{project_id}")
async def delete_project(project_id: str):
    """删除项目"""
    logger.debug(f"[API] 删除项目: project_id={project_id}")
    result = project_service.delete_project(project_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@router.post("/training/projects/{project_id}/restore")
async def restore_project(project_id: str):
    """恢复项目"""
    logger.debug(f"[API] 恢复项目: project_id={project_id}")
    result = project_service.restore_project(project_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@router.get("/training/projects/recycle-bin")
async def get_recycle_bin():
    """获取回收站"""
    logger.debug("[API] 获取回收站")
    result = project_service.get_recycle_bin()
    return result


@router.post("/training/projects/recycle-bin/empty")
async def empty_recycle_bin():
    """清空回收站"""
    logger.debug("[API] 清空回收站")
    result = project_service.empty_recycle_bin()
    return result


@router.get("/training/projects/{project_id}/models")
async def get_project_models(project_id: str):
    """获取项目模型列表"""
    logger.debug(f"[API] 获取项目模型: project_id={project_id}")
    result = project_service.get_models(project_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@router.get("/training/projects/{project_id}/activity")
async def get_project_activity(project_id: str, limit: int = 50):
    """获取项目活动日志"""
    logger.debug(f"[API] 获取项目活动: project_id={project_id}")
    result = project_service.get_activity_log(project_id, limit)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


# ==================== 模型相关 ====================
@router.get("/models/list", response_model=List[ModelInfo])
async def list_models():
    """列出所有模型"""
    if not yolo_service:
        raise HTTPException(status_code=500, detail="YOLO service not available")

    return yolo_service.list_models()


@router.get("/models/loaded")
async def get_loaded_models():
    """获取已加载到内存的模型列表"""
    if not yolo_engine:
        raise HTTPException(status_code=500, detail="YOLO engine not available")

    try:
        models = yolo_engine.get_loaded_models()
        return {
            "success": True,
            "models": models,
            "total": len(models)
        }
    except Exception as e:
        logger.error(f"[API] 获取已加载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/load")
async def load_model_to_memory(
    model_name: str = Form(...),
    device: str = Form(None)
):
    """预加载模型到内存"""
    if not yolo_engine:
        raise HTTPException(status_code=500, detail="YOLO engine not available")

    try:
        # 自动检测设备
        if device is None or device == "":
            device = "auto"
        model = yolo_engine.load_model(model_name, device)
        # 获取实际使用的设备
        actual_device = yolo_engine.default_device if device == "auto" else device
        return {
            "success": True,
            "message": f"模型 {model_name} 已加载到内存（设备: {actual_device}）",
            "model_name": model_name,
            "device": actual_device
        }
    except Exception as e:
        logger.error(f"[API] 加载模型失败: {str(e)}")
        return {
            "success": False,
            "message": f"加载模型失败: {str(e)}"
        }


@router.post("/models/export")
async def export_model(config: ExportConfig):
    """导出模型"""
    if not yolo_service:
        raise HTTPException(status_code=500, detail="YOLO service not available")
    
    result = yolo_service.export_model(config)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result


@router.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """上传模型文件"""
    if not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Only .pt files are allowed")
    
    try:
        filename = get_unique_filename(str(settings.MODELS_DIR), file.filename)
        file_path = settings.MODELS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        return {
            "success": True,
            "message": "Model uploaded successfully",
            "filename": filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 数据集项目（挂载在基础 API，避免仅部分路由部署时 404）====================
@router.get("/dataset-projects")
async def list_dataset_projects():
    """列出数据集项目（项目内含数据集）"""
    return {"success": True, "projects": dataset_service.list_dataset_projects()}


@router.post("/dataset-projects")
async def create_dataset_project(body: DatasetProjectCreate):
    """创建数据集项目"""
    result = dataset_service.create_dataset_project(body.name.strip())
    if result.get("success"):
        return result
    raise HTTPException(status_code=400, detail=result.get("message", "创建失败"))


@router.put("/dataset-projects/{project_id}/rename")
async def rename_dataset_project(project_id: str, body: DatasetProjectRename):
    """重命名数据集项目"""
    result = dataset_service.rename_dataset_project(project_id, body.new_name.strip())
    if result.get("success"):
        return result
    raise HTTPException(status_code=400, detail=result.get("message", "重命名失败"))


@router.delete("/dataset-projects/{project_id}")
async def delete_dataset_project(project_id: str):
    """删除数据集项目"""
    result = dataset_service.delete_dataset_project(project_id)
    if result.get("success"):
        return result
    raise HTTPException(status_code=400, detail=result.get("message", "删除失败"))


@router.put("/dataset-projects/{project_id}/datasets/{dataset_name}")
async def assign_dataset_to_project(project_id: str, dataset_name: str):
    """把数据集移动到指定项目"""
    result = dataset_service.assign_dataset_to_project(dataset_name, project_id)
    if result.get("success"):
        return result
    raise HTTPException(status_code=400, detail=result.get("message", "移动失败"))


# ==================== 数据集相关 ====================
@router.get("/datasets/list")
async def list_datasets():
    """列出所有数据集"""
    datasets = dataset_service.list_datasets()
    # 支持字典或对象格式
    datasets_list = []
    for item in datasets:
        if hasattr(item, 'dict'):
            datasets_list.append(item.dict())
        else:
            datasets_list.append(item)
    return {
        "total": len(datasets_list),
        "datasets": datasets_list
    }


@router.post("/datasets/refresh")
async def refresh_datasets():
    """重新扫描数据集目录"""
    datasets = dataset_service.refresh()
    datasets_list = []
    for item in datasets:
        if hasattr(item, 'dict'):
            datasets_list.append(item.dict())
        else:
            datasets_list.append(item)
    return {
        "total": len(datasets_list),
        "datasets": datasets_list
    }


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """上传数据集（zip 格式）"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    try:
        import zipfile
        import shutil

        # 保存上传的 zip 文件
        zip_path = settings.UPLOADS_DIR / file.filename
        save_uploaded_file(file, str(zip_path))

        # 解压到临时目录
        dataset_name = file.filename.replace('.zip', '')
        temp_path = settings.UPLOADS_DIR / f"temp_{dataset_name}"

        # 清理旧目录
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            namelist = zip_ref.namelist()
            print(f"[数据集上传] ZIP 内部文件数量: {len(namelist)}")
            zip_ref.extractall(temp_path)

        # 删除 zip 文件
        zip_path.unlink()

        # 处理嵌套目录：如果解压后只有一个子目录，把内容提上来
        dataset_path = settings.DATASETS_DIR / dataset_name

        if temp_path.exists():
            contents = list(temp_path.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                # ZIP 内部有嵌套目录，移动内容上去
                nested_dir = contents[0]
                print(f"[数据集上传] 检测到嵌套目录，移动内容: {nested_dir.name}")

                if dataset_path.exists():
                    shutil.rmtree(dataset_path)
                shutil.move(str(nested_dir), str(dataset_path))
                shutil.rmtree(temp_path)
            else:
                # 直接移动整个 temp 目录
                if dataset_path.exists():
                    shutil.rmtree(dataset_path)
                shutil.move(str(temp_path), str(dataset_path))

        # 统计文件
        extracted_files = list(dataset_path.rglob("*")) if dataset_path.exists() else []
        print(f"[数据集上传] 最终文件数量: {len(extracted_files)}")

        return {
            "success": True,
            "message": "Dataset uploaded successfully",
            "dataset_name": dataset_name,
            "path": str(dataset_path),
            "files_count": len(extracted_files)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 本地标注相关 ====================
@router.get("/annotation/projects")
async def list_annotation_projects():
    """列出所有标注项目"""
    try:
        projects = annotation_service.list_projects()
        return projects
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/annotation/projects")
async def create_annotation_project(data: dict):
    """创建标注项目"""
    try:
        name = data.get("name")
        description = data.get("description", "")
        
        if not name:
            raise HTTPException(status_code=400, detail="Project name is required")
        
        project = annotation_service.create_project(name, description)
        return project
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotation/projects/{project_id}")
async def get_annotation_project(project_id: str):
    """获取标注项目详情"""
    try:
        project = annotation_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/annotation/upload")
async def upload_annotation_images(
    project_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """上传标注图片"""
    try:
        result = annotation_service.upload_images(project_id, files)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/annotation/save")
async def save_annotation_data(data: dict):
    """保存标注数据"""
    try:
        project_id = data.get("project_id")
        annotations = data.get("annotations", {})
        classes = data.get("classes", [])
        
        if not project_id:
            raise HTTPException(status_code=400, detail="Project ID is required")
        
        result = annotation_service.save_annotations(project_id, annotations, classes)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotation/export/{project_id}")
async def export_annotation_project(project_id: str):
    """导出标注项目为YOLO格式"""
    try:
        zip_path = annotation_service.export_to_yolo(project_id)
        if not zip_path or not zip_path.exists():
            raise HTTPException(status_code=500, detail="Failed to export project")
        
        return FileResponse(
            path=str(zip_path),
            filename=zip_path.name,
            media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotation/image/{project_id}/{image_name}")
async def get_annotation_image(project_id: str, image_name: str):
    """获取标注图片"""
    try:
        image_path = annotation_service.get_image_path(project_id, image_name)
        if not image_path or not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(path=str(image_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/annotation/projects/{project_id}")
async def delete_annotation_project(project_id: str):
    """删除标注项目"""
    try:
        result = annotation_service.delete_project(project_id)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/annotation/auto-annotate/{project_id}")
async def auto_annotate_project(
    project_id: str,
    model_name: Optional[str] = Form("yolov8n.pt"),
    confidence: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    filter_classes: Optional[str] = Form(None),
    merge_mode: str = Form("replace")
):
    """使用YOLO模型自动标注项目 (增强版)"""
    try:
        # 解析类别过滤
        filter_classes_list = None
        if filter_classes:
            import json
            filter_classes_list = json.loads(filter_classes)
        
        result = annotation_service.auto_annotate_with_model(
            project_id=project_id,
            model_path=str(settings.MODELS_DIR / model_name),
            confidence=confidence,
            iou_threshold=iou_threshold,
            filter_classes=filter_classes_list,
            merge_mode=merge_mode
        )
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/annotation/batch-annotate/{project_id}")
async def batch_auto_annotate(
    project_id: str,
    data: dict
):
    """批量自动标注指定图片"""
    try:
        image_names = data.get("image_names", [])
        model_name = data.get("model_name", "yolov8n.pt")
        confidence = data.get("confidence", 0.25)
        iou_threshold = data.get("iou_threshold", 0.45)
        
        if not image_names:
            raise HTTPException(status_code=400, detail="image_names is required")
        
        result = annotation_service.batch_auto_annotate(
            project_id=project_id,
            image_names=image_names,
            model_path=str(settings.MODELS_DIR / model_name),
            confidence=confidence,
            iou_threshold=iou_threshold
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotation/statistics/{project_id}")
async def get_annotation_statistics(project_id: str):
    """获取标注统计信息"""
    try:
        stats = annotation_service.get_annotation_statistics(project_id)
        if not stats.get("success", False):
            raise HTTPException(status_code=404, detail=stats.get("message", "Project not found"))
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/annotation/visualize/{project_id}/{image_name}")
async def visualize_annotations(project_id: str, image_name: str):
    """可视化标注结果"""
    try:
        output_path = annotation_service.visualize_annotations(project_id, image_name)
        if not output_path or not output_path.exists():
            raise HTTPException(status_code=500, detail="Failed to visualize annotations")
        
        return FileResponse(path=str(output_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ==================== Ultralytics Solutions ====================

@router.post("/solutions/object-counting", response_model=SolutionResponse)
async def solution_object_counting(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    region_points: Optional[str] = Form(None),  # JSON string
    show_in: bool = Form(True),
    show_out: bool = Form(True),
    classes: Optional[str] = Form(None),  # JSON string
    conf: float = Form(0.25)
):
    """对象计数 - 统计进出区域的对象数量"""
    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")
    
    if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "mp4", "avi", "mov", "mkv", "flv", "wmv"]):
        raise HTTPException(status_code=400, detail="不支持的文件类型，请上传图片或视频")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        region = None
        if region_points:
            try:
                region = json.loads(region_points)
            except (json.JSONDecodeError, ValueError):
                pass  # 忽略无效的JSON

        class_list = None
        if classes:
            try:
                class_list = json.loads(classes)
            except (json.JSONDecodeError, ValueError):
                pass  # 忽略无效的JSON
        
        # 设置输出路径
        output_path = str(settings.UPLOADS_DIR / f"counted_{filename}")
        
        # 执行对象计数
        result = solutions_service.object_counting(
            source=str(file_path),
            model_name=model_name,
            region_points=region,
            show_in=show_in,
            show_out=show_out,
            classes=class_list,
            conf=conf,
            output_path=output_path
        )

        # 转换为相对路径
        if result.get("output_path"):
            result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

        return SolutionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solutions/heatmap", response_model=SolutionResponse)
async def solution_heatmap(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    colormap: str = Form("COLORMAP_JET"),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """热图生成 - 可视化检测密度"""
    import cv2
    import logging
    logger = logging.getLogger(__name__)

    # 转换 colormap 名称为 OpenCV 整数常量
    colormap_value = getattr(cv2, colormap, cv2.COLORMAP_JET)

    logger.info(f"[heatmap] 收到请求: filename={file.filename}, model={model_name}, conf={conf}")

    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")

    # 扩展支持的文件类型（图片 + 视频），让 cv2 自行验证能否打开
    IMAGE_EXTS = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"]
    VIDEO_EXTS = ["mp4", "avi", "mov", "mkv", "flv", "wmv", "m4v", "ts"]
    ALLOWED_EXTS = IMAGE_EXTS + VIDEO_EXTS

    fname = file.filename or ""
    if fname and not allowed_file(fname, ALLOWED_EXTS):
        logger.warning(f"[heatmap] 文件类型不支持: {fname}")
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {fname}，请上传图片或视频文件")

    try:
        import json
        import re

        # 保存上传文件
        # 清理文件名，移除 = & , 等 URL 特殊字符，防止静态文件 URL 解析出错
        def _sanitize(name):
            ext = name.rsplit('.', 1)[-1].lower() if '.' in name else 'jpg'
            safe = re.sub(r'[^A-Za-z0-9_\-]', '_', name.rsplit('.', 1)[0]) + '.' + ext
            return safe[:80]
        clean_fname = _sanitize(fname) if fname else 'upload.jpg'
        logger.info(f"[heatmap] 文件名清理: {fname!r} -> {clean_fname!r}")
        filename = get_unique_filename(str(settings.UPLOADS_DIR), clean_fname)
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))

        # 解析参数
        class_list = None
        if classes:
            try:
                class_list = json.loads(classes)
            except (json.JSONDecodeError, ValueError):
                pass

        # 设置输出路径
        output_path = str(settings.UPLOADS_DIR / f"heatmap_{filename}")

        # 生成热图
        result = solutions_service.generate_heatmap(
            source=str(file_path),
            model_name=model_name,
            colormap=colormap_value,
            classes=class_list,
            conf=conf,
            output_path=output_path
        )

        logger.info(f"[heatmap] 处理结果: success={result.get('success')}")

        # 转换为相对路径
        if result.get("output_path"):
            result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

        return SolutionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[heatmap] 处理异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solutions/speed-estimation", response_model=SolutionResponse)
async def solution_speed_estimation(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    region_points: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25),
    pixel_to_meter: float = Form(10),
    line_width: int = Form(2)
):
    """速度估算 - 计算对象移动速度"""
    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")

    # 速度估算：严格只允许 MP4 视频
    # 如果浏览器上传的文件名没有扩展名（例如是 blob/空名），就用 content_type 做兜底判断。
    file_name = file.filename or ""
    content_type = file.content_type or ""
    logger.info(f"[speed-estimation] 收到文件 filename='{file_name}', content_type='{content_type}'")
    is_mp4 = file_name.lower().endswith(".mp4") or content_type == "video/mp4"
    if not is_mp4:
        raise HTTPException(status_code=400, detail="速度估算只支持 MP4 视频文件，请上传 .mp4 文件")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        region = None
        if region_points:
            try:
                region = json.loads(region_points)
            except (json.JSONDecodeError, ValueError):
                pass  # 忽略无效的JSON

        class_list = None
        if classes:
            try:
                class_list = json.loads(classes)
            except (json.JSONDecodeError, ValueError):
                pass  # 忽略无效的JSON
        
        # 设置输出路径
        output_path = str(settings.UPLOADS_DIR / f"speed_{filename}")
        
        # 估算速度
        result = solutions_service.estimate_speed(
            source=str(file_path),
            model_name=model_name,
            region_points=region,
            classes=class_list,
            conf=conf,
            pixel_to_meter=pixel_to_meter,
            line_width=line_width,
            output_path=output_path
        )

        # 转换为相对路径
        if result.get("output_path"):
            result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

        return SolutionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solutions/distance-calculation", response_model=SolutionResponse)
async def solution_distance_calculation(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """距离计算 - 测量对象之间的距离"""
    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")
    
    if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp"]):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        class_list = None
        if classes:
            class_list = json.loads(classes)
        
        # 计算距离
        result = solutions_service.calculate_distance(
            image_path=str(file_path),
            model_name=model_name,
            classes=class_list,
            conf=conf
        )

        # 转换为相对路径
        output_image = result.get("output_image")
        if output_image:
            output_image = f"/uploads/{Path(output_image).name}"

        return SolutionResponse(
            success=result["success"],
            message=result["message"],
            results={"distances": result.get("distances", [])},
            output_path=output_image
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solutions/object-blur", response_model=SolutionResponse)
async def solution_object_blur(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25),
    blur_ratio: float = Form(50)
):
    """对象模糊 - 隐私保护"""
    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")
    
    if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "mp4", "avi", "mov", "mkv", "flv", "wmv"]):
        raise HTTPException(status_code=400, detail="不支持的文件类型，请上传图片或视频")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        class_list = None
        if classes:
            class_list = json.loads(classes)
        
        # 设置输出路径
        output_path = str(settings.UPLOADS_DIR / f"blurred_{filename}")
        
        # 模糊对象
        result = solutions_service.blur_objects(
            source=str(file_path),
            model_name=model_name,
            classes=class_list,
            conf=conf,
            blur_ratio=blur_ratio,
            output_path=output_path
        )

        # 转换为相对路径
        if result.get("output_path"):
            result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

        return SolutionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solutions/object-crop", response_model=SolutionResponse)
async def solution_object_crop(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """对象裁剪 - 提取检测到的对象"""
    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")
    
    if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp"]):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        class_list = None
        if classes:
            class_list = json.loads(classes)
        
        # 裁剪对象
        result = solutions_service.crop_objects(
            image_path=str(file_path),
            model_name=model_name,
            classes=class_list,
            conf=conf
        )

        # 更新裁剪图片路径为相对路径
        cropped_images = result.get("cropped_images", [])
        for img in cropped_images:
            if img.get("crop_path"):
                img["crop_path"] = f"/uploads/{Path(img['crop_path']).name}"

        return SolutionResponse(
            success=result["success"],
            message=result["message"],
            results={
                "total_crops": result.get("total_crops", 0),
                "cropped_images": cropped_images
            },
            output_path=f"/uploads/{Path(result.get('output_dir', '')).name}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solutions/queue-management", response_model=SolutionResponse)
async def solution_queue_management(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    region_points: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """队列管理 - 监控队列长度"""
    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")

    # 同 speed-estimation：content_type 兜底，避免文件名无扩展导致误判。
    file_name = file.filename or ""
    content_type = file.content_type or ""
    logger.info(f"[queue-management] 收到文件 filename='{file_name}', content_type='{content_type}'")
    is_image_or_video_by_type = content_type.startswith("image/") or content_type.startswith("video/")
    if not allowed_file(file_name, ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "mp4", "avi", "mov", "mkv", "flv", "wmv"]) and not is_image_or_video_by_type:
        raise HTTPException(status_code=400, detail="不支持的文件类型，请上传图片或视频")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        region = None
        if region_points:
            try:
                region = json.loads(region_points)
            except (json.JSONDecodeError, ValueError):
                pass  # 忽略无效的JSON

        class_list = None
        if classes:
            try:
                class_list = json.loads(classes)
            except (json.JSONDecodeError, ValueError):
                pass  # 忽略无效的JSON
        
        # 设置输出路径
        output_path = str(settings.UPLOADS_DIR / f"queue_{filename}")
        
        # 队列管理
        result = solutions_service.queue_management(
            source=str(file_path),
            model_name=model_name,
            region_points=region,
            classes=class_list,
            conf=conf,
            output_path=output_path
        )

        # 转换为相对路径
        if result.get("output_path"):
            result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

        return SolutionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solutions/parking-management", response_model=SolutionResponse)
async def solution_parking_management(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    parking_slots: Optional[str] = Form(None),  # JSON string: [[(x,y),...], ...]
    classes: Optional[str] = Form(None),        # JSON string
    conf: float = Form(0.25),
    line_width: int = Form(2)
):
    """停车管理 - 统计车位占用/空闲"""
    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")

    file_name = file.filename or ""
    content_type = file.content_type or ""
    is_image_or_video_by_type = content_type.startswith("image/") or content_type.startswith("video/")
    if not allowed_file(file_name, ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "mp4", "avi", "mov", "mkv", "flv", "wmv"]) and not is_image_or_video_by_type:
        raise HTTPException(status_code=400, detail="不支持的文件类型，请上传图片或视频")

    try:
        import json

        filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))

        slots = None
        if parking_slots:
            try:
                slots = json.loads(parking_slots)
            except (json.JSONDecodeError, ValueError):
                pass

        class_list = None
        if classes:
            try:
                class_list = json.loads(classes)
            except (json.JSONDecodeError, ValueError):
                pass

        output_path = str(settings.UPLOADS_DIR / f"parking_{filename}")
        result = solutions_service.parking_management(
            source=str(file_path),
            model_name=model_name,
            parking_slots=slots,
            classes=class_list,
            conf=conf,
            line_width=line_width,
            output_path=output_path
        )

        if result.get("output_path"):
            result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

        return SolutionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/solutions/list")
async def list_solutions():
    """列出所有可用的 Solutions 功能"""
    solutions_list = [
        {
            "name": "object-counting",
            "title": "对象计数",
            "description": "统计进出指定区域的对象数量，支持实时计数和分类统计",
            "input_types": ["image", "video"],
            "features": ["区域计数", "进出统计", "分类计数"]
        },
        {
            "name": "heatmap",
            "title": "热图生成",
            "description": "可视化检测密度，显示对象出现的热点区域",
            "input_types": ["image", "video"],
            "features": ["密度可视化", "热点分析", "轨迹追踪"]
        },
        {
            "name": "speed-estimation",
            "title": "速度估算",
            "description": "计算移动对象的速度，适用于交通监控等场景",
            "input_types": ["video"],
            "features": ["实时测速", "超速告警", "速度统计"]
        },
        {
            "name": "distance-calculation",
            "title": "距离计算",
            "description": "测量检测对象之间的像素距离",
            "input_types": ["image"],
            "features": ["对象间距", "空间分析", "距离标注"]
        },
        {
            "name": "object-blur",
            "title": "对象模糊",
            "description": "对检测到的对象进行模糊处理，保护隐私",
            "input_types": ["image", "video"],
            "features": ["隐私保护", "人脸模糊", "车牌模糊"]
        },
        {
            "name": "object-crop",
            "title": "对象裁剪",
            "description": "自动裁剪检测到的对象，提取感兴趣区域",
            "input_types": ["image"],
            "features": ["自动裁剪", "批量提取", "对象分离"]
        },
        {
            "name": "queue-management",
            "title": "队列管理",
            "description": "监控队列长度，分析排队情况",
            "input_types": ["video"],
            "features": ["队列计数", "等待时间", "流量分析"]
        }
    ]
    
    return {
        "total": len(solutions_list),
        "solutions": solutions_list
    }
