"""
推理模块路由 - Inference Routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional, List
import json

from backend.core.config import settings
from backend.core.utils import allowed_file, save_uploaded_file, get_unique_filename
from backend.modules.inference.inference_service import inference_service, monitor_service

router = APIRouter()


# ==================== 图片推理 ====================

@router.post("/inference/image")
async def infer_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25),
    iou_threshold: Optional[float] = Form(0.45),
    draw_results: bool = Form(True)
):
    """图片推理"""
    # 保存上传的图片
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    result = inference_service.infer_image(
        image_path=str(file_path),
        model_name=model_name,
        confidence=confidence,
        iou_threshold=iou_threshold,
        draw_results=draw_results
    )

    return result


@router.post("/inference/image/base64")
async def infer_image_base64(
    image_data: str = Form(...),  # base64 encoded
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25)
):
    """图片推理 (Base64)"""
    import base64

    try:
        image_bytes = base64.b64decode(image_data)
        result = inference_service.stream_infer(image_bytes, model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解码失败: {str(e)}")


# ==================== 视频推理 ====================

@router.post("/inference/video")
async def infer_video(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25),
    save_output: bool = Form(False)
):
    """视频推理"""
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))

    output_path = None
    if save_output:
        output_path = str(settings.UPLOADS_DIR / f"inferred_{filename}")

    result = inference_service.infer_video(
        video_path=str(file_path),
        model_name=model_name,
        confidence=confidence,
        output_path=output_path,
        save_output=save_output
    )

    return result


# ==================== 批量推理 ====================

@router.post("/inference/batch")
async def batch_infer(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25)
):
    """批量推理"""
    image_paths = []
    for file in files:
        filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))
        image_paths.append(str(file_path))

    result = inference_service.batch_infer(
        image_paths=image_paths,
        model_name=model_name,
        confidence=confidence
    )

    return result


# ==================== 监控 ====================

@router.post("/monitor/camera")
async def add_camera(
    name: str = Form(...),
    url: str = Form(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25)
):
    """添加监控摄像头"""
    return monitor_service.add_camera(name, url, model_name, confidence)


@router.get("/monitor/cameras")
async def list_cameras():
    """列出所有摄像头"""
    return {"success": True, "cameras": monitor_service.list_cameras()}


@router.get("/monitor/camera/{name}")
async def get_camera_status(name: str):
    """获取摄像头状态"""
    return monitor_service.get_camera_status(name)


@router.post("/monitor/camera/{name}/start")
async def start_monitoring(name: str):
    """开始监控"""
    return monitor_service.start_stream(name)
