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
    region_type: str = Form("polygon"),
    region_points: Optional[str] = Form(None),
    show_in: bool = Form(True),
    show_out: bool = Form(True),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25),
    line_width: int = Form(2)
):
    """对象计数 - 支持区域计数和分类统计"""
    import traceback
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"[object-counting] Received request, region_points={region_points}, classes={classes}")
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        logger.info(f"[object-counting] File saved: {file_path}")

        # 解析区域坐标
        region = None
        if region_points:
            try:
                region = json.loads(region_points)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"[object-counting] Failed to parse region_points: {e}")

        # 解析类别列表
        class_list = None
        if classes:
            try:
                class_list = json.loads(classes)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"[object-counting] Failed to parse classes: {e}")

        output_path = str(settings.UPLOADS_DIR / f"counted_{filename}")

        result = solutions_service.object_counting(
            source=str(file_path),
            model_name=model_name,
            region_type=region_type,
            region_points=region,
            show_in=show_in,
            show_out=show_out,
            classes=class_list,
            conf=conf,
            line_width=line_width,
            output_path=output_path
        )

        if result.get("output_path"):
            result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

        return result
    except Exception as e:
        logger.error(f"[object-counting] Error: {traceback.format_exc()}")
        return {"success": False, "message": f"Error: {str(e)}"}


# ==================== 热图生成 ====================

# 存储正在进行的任务
heatmap_tasks = {}

@router.post("/solutions/heatmap")
async def solution_heatmap(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    colormap: str = Form("COLORMAP_JET"),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25)
):
    """热图生成（异步）"""
    import cv2
    import uuid
    import logging
    import traceback
    from concurrent.futures import ThreadPoolExecutor
    logger = logging.getLogger(__name__)

    try:
        # 转换 colormap 名称为 OpenCV 整数常量（修复：之前传字符串导致 cv2.applyColorMap 报错）
        colormap_int = getattr(cv2, colormap, cv2.COLORMAP_JET)

        # 生成任务 ID
        task_id = str(uuid.uuid4())[:8]

        # 保存上传的文件
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename or "upload.mp4")
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))

        output_path = str(settings.UPLOADS_DIR / f"heatmap_{task_id}_{filename}")

        # 初始化任务状态
        heatmap_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "正在处理...",
            "output_path": None
        }

        class_list = None
        if classes:
            try:
                class_list = json.loads(classes)
            except (json.JSONDecodeError, ValueError):
                pass

        # 在后台线程中执行
        def process_heatmap():
            try:
                def update_progress(progress, message):
                    heatmap_tasks[task_id]["progress"] = progress
                    heatmap_tasks[task_id]["message"] = message

                result = solutions_service.generate_heatmap(
                    source=str(file_path),
                    model_name=model_name,
                    colormap=colormap_int,   # 修复：传整数而非字符串
                    classes=class_list,
                    conf=conf,
                    output_path=output_path,
                    progress_callback=update_progress
                )
                heatmap_tasks[task_id]["status"] = "completed" if result.get("success") else "failed"
                heatmap_tasks[task_id]["message"] = result.get("message", "")
                heatmap_tasks[task_id]["progress"] = 100
                if result.get("output_path"):
                    heatmap_tasks[task_id]["output_path"] = f"/uploads/{Path(result['output_path']).name}"
            except Exception as e:
                logger.error(f"[heatmap] 后台处理失败: {traceback.format_exc()}")
                heatmap_tasks[task_id]["status"] = "failed"
                heatmap_tasks[task_id]["message"] = str(e)
                heatmap_tasks[task_id]["progress"] = 0

        executor = ThreadPoolExecutor(max_workers=2)
        executor.submit(process_heatmap)

        return {
            "success": True,
            "task_id": task_id,
            "message": "任务已提交，请轮询获取进度"
        }
    except Exception as e:
        logger.error(f"[heatmap] 请求处理失败: {traceback.format_exc()}")
        return {"success": False, "message": f"请求处理失败: {str(e)}"}


@router.get("/solutions/heatmap/status/{task_id}")
async def get_heatmap_status(task_id: str):
    """获取热图生成进度"""
    task = heatmap_tasks.get(task_id)
    if task:
        return task
    return {"status": "not_found", "message": "任务不存在"}


# ==================== 速度估算 ====================

@router.post("/solutions/speed-estimation")
async def solution_speed_estimation(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    region_points: Optional[str] = Form(None),
    classes: Optional[str] = Form(None),
    conf: float = Form(0.25),
    pixel_to_meter: float = Form(10),
    line_width: int = Form(2)
):
    """速度估算"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    region = None
    if region_points:
        try:
            region = json.loads(region_points)
        except (json.JSONDecodeError, ValueError):
            pass
    class_list = None
    if classes:
        try:
            class_list = json.loads(classes)
        except (json.JSONDecodeError, ValueError):
            pass
    output_path = str(settings.UPLOADS_DIR / f"speed_{filename}")

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

    class_list = None
    if classes:
        try:
            class_list = json.loads(classes)
        except (json.JSONDecodeError, ValueError):
            pass

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

    class_list = None
    if classes:
        try:
            class_list = json.loads(classes)
        except (json.JSONDecodeError, ValueError):
            pass
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

    class_list = None
    if classes:
        try:
            class_list = json.loads(classes)
        except (json.JSONDecodeError, ValueError):
            pass

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
    conf: float = Form(0.25),
    line_width: int = Form(2)
):
    """队列管理"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    region = None
    if region_points:
        try:
            region = json.loads(region_points)
        except (json.JSONDecodeError, ValueError):
            pass
    class_list = None
    if classes:
        try:
            class_list = json.loads(classes)
        except (json.JSONDecodeError, ValueError):
            pass
    output_path = str(settings.UPLOADS_DIR / f"queue_{filename}")

    result = solutions_service.queue_management(
        source=str(file_path),
        model_name=model_name,
        region_points=region,
        classes=class_list,
        conf=conf,
        line_width=line_width,
        output_path=output_path
    )

    if result.get("output_path"):
        result["output_path"] = f"/uploads/{Path(result['output_path']).name}"

    return result
