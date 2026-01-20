"""
解决方案模块路由 - Solutions Routes
独立模块路由
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from pathlib import Path
import json

from backend.core.config import settings
from backend.core.utils import allowed_file, save_uploaded_file, get_unique_filename
from backend.modules.solutions.solutions_service import solutions_service

router = APIRouter()


# ==================== 解决方案列表 ====================

@router.get("/solutions/list")
async def list_solutions():
    """列出所有解决方案"""
    return {"success": True, "solutions": solutions_service.list_solutions()}


# ==================== 对象计数 ====================

@router.post("/solutions/object-counting")
async def solution_object_counting(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    region_points: Optional[str] = Form(None),
    show_in: bool = Form(True),
    show_out: bool = Form(True),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """对象计数"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    region = json.loads(region_points) if region_points else None
    class_list = json.loads(classes) if classes else None

    output_path = str(settings.UPLOADS_DIR / f"counted_{filename}")

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

    if result.get("output_path"):
        result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

    return result


# ==================== 热图生成 ====================

@router.post("/solutions/heatmap")
async def solution_heatmap(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    colormap: int = Form(2),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """热图生成"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    class_list = json.loads(classes) if classes else None
    output_path = str(settings.UPLOADS_DIR / f"heatmap_{filename}")

    result = solutions_service.generate_heatmap(
        source=str(file_path),
        model_name=model_name,
        colormap=colormap,
        classes=class_list,
        conf=conf,
        output_path=output_path
    )

    if result.get("output_path"):
        result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

    return result


# ==================== 速度估算 ====================

@router.post("/solutions/speed-estimation")
async def solution_speed_estimation(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    region_points: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """速度估算"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    region = json.loads(region_points) if region_points else None
    class_list = json.loads(classes) if classes else None
    output_path = str(settings.UPLOADS_DIR / f"speed_{filename}")

    result = solutions_service.estimate_speed(
        source=str(file_path),
        model_name=model_name,
        region_points=region,
        classes=class_list,
        conf=conf,
        output_path=output_path
    )

    if result.get("output_path"):
        result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

    return result


# ==================== 距离计算 ====================

@router.post("/solutions/distance-calculation")
async def solution_distance_calculation(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """距离计算"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    class_list = json.loads(classes) if classes else None

    result = solutions_service.calculate_distance(
        image_path=str(file_path),
        model_name=model_name,
        classes=class_list,
        conf=conf
    )

    if result.get("output_image"):
        result["output_path"] = f"/uploads/{Path(result['output_image']).name}"

    return result


# ==================== 对象模糊 ====================

@router.post("/solutions/object-blur")
async def solution_object_blur(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25),
    blur_ratio: float = Form(50)
):
    """对象模糊"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    class_list = json.loads(classes) if classes else None
    output_path = str(settings.UPLOADS_DIR / f"blurred_{filename}")

    result = solutions_service.blur_objects(
        source=str(file_path),
        model_name=model_name,
        classes=class_list,
        conf=conf,
        blur_ratio=blur_ratio,
        output_path=output_path
    )

    if result.get("output_path"):
        result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

    return result


# ==================== 对象裁剪 ====================

@router.post("/solutions/object-crop")
async def solution_object_crop(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """对象裁剪"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    class_list = json.loads(classes) if classes else None

    result = solutions_service.crop_objects(
        image_path=str(file_path),
        model_name=model_name,
        classes=class_list,
        conf=conf
    )

    # 更新裁剪图片路径
    cropped_images = result.get("cropped_images", [])
    for img in cropped_images:
        if img.get("crop_path"):
            img["crop_path"] = f"/uploads/{Path(img['crop_path']).name}"

    return result


# ==================== 队列管理 ====================

@router.post("/solutions/queue-management")
async def solution_queue_management(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    region_points: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """队列管理"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    region = json.loads(region_points) if region_points else None
    class_list = json.loads(classes) if classes else None
    output_path = str(settings.UPLOADS_DIR / f"queue_{filename}")

    result = solutions_service.queue_management(
        source=str(file_path),
        model_name=model_name,
        region_points=region,
        classes=class_list,
        conf=conf,
        output_path=output_path
    )

    if result.get("output_path"):
        result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

    return result
