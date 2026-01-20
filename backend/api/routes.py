"""
API 路由定义
"""
import sys
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import settings
from backend.models.schemas import (
    InferenceRequest, InferenceResponse, TrainingConfig,
    TrainingStatus, ModelInfo, DatasetInfo, ExportConfig,
    SystemInfo,
    ObjectCountingRequest, HeatmapRequest, SpeedEstimationRequest,
    DistanceCalculationRequest, ObjectBlurRequest, ObjectCropRequest,
    QueueManagementRequest, SolutionResponse
)
from backend.services.yolo_service import yolo_service
from backend.services.annotation_service import annotation_service
from backend.services.dataset_service import dataset_service
from backend.services.solutions_service import solutions_service
from backend.services.supervision_service import supervision_service
from backend.utils.file_utils import allowed_file, save_uploaded_file, get_unique_filename

router = APIRouter()


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
@router.post("/inference/image", response_model=InferenceResponse)
async def infer_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    model_path: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    img_size: Optional[int] = Form(None)
):
    """图像推理"""
    if not yolo_service:
        raise HTTPException(status_code=500, detail="YOLO service not available")
    
    # 验证文件类型
    if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp"]):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
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
        
        return result
        
    except Exception as e:
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
            filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
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
    if not yolo_service:
        raise HTTPException(status_code=500, detail="YOLO service not available")
    
    try:
        task_id = yolo_service.train(config)
        return {
            "success": True,
            "task_id": task_id,
            "message": "Training started successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status/{task_id}", response_model=TrainingStatus)
async def get_training_status(task_id: str):
    """获取训练状态"""
    if not yolo_service:
        raise HTTPException(status_code=500, detail="YOLO service not available")
    
    status = yolo_service.get_training_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return status


@router.get("/training/tasks")
async def list_training_tasks():
    """列出所有训练任务"""
    if not yolo_service:
        raise HTTPException(status_code=500, detail="YOLO service not available")
    
    return {"tasks": yolo_service.list_training_statuses()}


# ==================== 模型相关 ====================
@router.get("/models/list", response_model=List[ModelInfo])
async def list_models():
    """列出所有模型"""
    if not yolo_service:
        raise HTTPException(status_code=500, detail="YOLO service not available")
    
    return yolo_service.list_models()


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


# ==================== 数据集相关 ====================
@router.get("/datasets/list")
async def list_datasets():
    """列出所有数据集"""
    datasets = dataset_service.list_datasets()
    return {
        "total": len(datasets),
        "datasets": [item.dict() for item in datasets]
    }


@router.post("/datasets/refresh")
async def refresh_datasets():
    """重新扫描数据集目录"""
    datasets = dataset_service.refresh()
    return {
        "total": len(datasets),
        "datasets": [item.dict() for item in datasets]
    }


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """上传数据集（zip 格式）"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")
    
    try:
        import zipfile
        
        # 保存上传的 zip 文件
        zip_path = settings.UPLOADS_DIR / file.filename
        save_uploaded_file(file, str(zip_path))
        
        # 解压到数据集目录
        dataset_name = file.filename.replace('.zip', '')
        dataset_path = settings.DATASETS_DIR / dataset_name
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # 删除 zip 文件
        zip_path.unlink()
        
        return {
            "success": True,
            "message": "Dataset uploaded successfully",
            "dataset_name": dataset_name,
            "path": str(dataset_path)
        }
    except Exception as e:
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
    
    if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"]):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        region = None
        if region_points:
            region = json.loads(region_points)
        
        class_list = None
        if classes:
            class_list = json.loads(classes)
        
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
    colormap: int = Form(2),  # cv2.COLORMAP_JET
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """热图生成 - 可视化检测密度"""
    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")
    
    if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"]):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        class_list = None
        if classes:
            class_list = json.loads(classes)
        
        # 设置输出路径
        output_path = str(settings.UPLOADS_DIR / f"heatmap_{filename}")
        
        # 生成热图
        result = solutions_service.generate_heatmap(
            source=str(file_path),
            model_name=model_name,
            colormap=colormap,
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


@router.post("/solutions/speed-estimation", response_model=SolutionResponse)
async def solution_speed_estimation(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    region_points: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """速度估算 - 计算对象移动速度"""
    if not solutions_service:
        raise HTTPException(status_code=500, detail="Solutions service not available")
    
    if not allowed_file(file.filename, ["mp4", "avi", "mov"]):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        region = None
        if region_points:
            region = json.loads(region_points)
        
        class_list = None
        if classes:
            class_list = json.loads(classes)
        
        # 设置输出路径
        output_path = str(settings.UPLOADS_DIR / f"speed_{filename}")
        
        # 估算速度
        result = solutions_service.estimate_speed(
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
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
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
    
    if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"]):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
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
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
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
    
    if not allowed_file(file.filename, ["mp4", "avi", "mov"]):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    try:
        import json
        
        # 保存上传文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        
        # 解析参数
        region = None
        if region_points:
            region = json.loads(region_points)
        
        class_list = None
        if classes:
            class_list = json.loads(classes)
        
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
