# -*- coding: utf-8 -*-
"""
推理模块路由 - Inference Routes
提供图片、视频、摄像头的 YOLO 推理服务接口
"""
import logging
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional, List

from backend.core.config import settings
from backend.core.utils import allowed_file, save_uploaded_file, get_unique_filename
from backend.modules.inference.inference_service import inference_service, monitor_service

# 创建日志记录器
logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()


# =============================================================================
# 图片推理接口
# =============================================================================

@router.post("/inference/image")
async def infer_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25),
    iou_threshold: Optional[float] = Form(0.45),
    draw_results: bool = Form(True)
):
    """
    图片推理接口

    上传图片，使用 YOLO 模型进行目标检测

    参数:
        file: 上传的图像文件（jpg, png, bmp）
        model_name: 模型名称，默认使用配置中的 DEFAULT_MODEL
        confidence: 置信度阈值，低于此值的检测结果将被过滤
        iou_threshold: NMS IoU 阈值，用于去除重复检测框
        draw_results: 是否在原图上绘制检测结果

    返回:
        success: 是否成功
        detections: 检测结果列表
        annotated_image: 标注后的图片路径（base64 或路径）
        inference_time: 推理耗时（秒）

    示例:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/inference/image" \\
             -F "file=@test.jpg" \\
             -F "model_name=yolo11n.pt" \\
             -F "confidence=0.5"
        ```
    """
    # 记录推理请求
    # 使用 print 确保日志输出
    print(f"[DEBUG] 收到推理请求: model={model_name}, conf={confidence}, iou={iou_threshold}, draw_results={draw_results}")
    logger.info(f"[推理] 收到图片推理请求: model={model_name}, conf={confidence}, iou={iou_threshold}, draw_results={draw_results}")

    # 1. 验证文件类型
    if not allowed_file(file.filename, ['jpg', 'jpeg', 'png', 'bmp']):
        error_msg = f"不支持的文件类型: {file.filename}"
        logger.warning(f"[推理] {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    # 2. 生成唯一文件名并保存上传的图片
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename

    # 保存文件
    save_uploaded_file(file, str(file_path))
    logger.debug(f"[推理] 图片已保存至: {file_path}")

    # 3. 执行推理
    try:
        result = inference_service.infer_image(
            image_path=str(file_path),
            model_name=model_name,
            confidence=confidence,
            iou_threshold=iou_threshold,
            draw_results=draw_results
        )
    except Exception as e:
        error_msg = f"推理执行失败: {str(e)}"
        logger.error(f"[推理] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # 4. 返回结果
    if result.get("success"):
        num_detections = len(result.get("detections", []))
        inference_time = result.get("inference_time", 0)
        logger.info(f"[推理] 图片推理成功: {num_detections} 个检测结果, 耗时 {inference_time:.2f}s")
    else:
        error_msg = result.get("message", "推理失败")
        logger.error(f"[推理] 图片推理失败: {error_msg}")

    return result


@router.post("/inference/image/base64")
async def infer_image_base64(
    image_data: str = Form(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25)
):
    """
    图片推理接口（Base64 编码）

    接收 Base64 编码的图片数据，进行目标检测

    参数:
        image_data: Base64 编码的图片数据
        model_name: 模型名称
        confidence: 置信度阈值

    返回:
        success: 是否成功
        detections: 检测结果列表
        num_detections: 检测数量
    """
    logger.info(f"[推理] 收到 Base64 图片推理请求: model={model_name}, conf={confidence}")

    try:
        # 1. 解码 Base64 图片
        image_bytes = base64.b64decode(image_data)
        logger.debug(f"[推理] Base64 图片解码成功: {len(image_bytes)} bytes")

        # 2. 执行推理
        result = inference_service.stream_infer(image_bytes, model_name, confidence)

        # 3. 返回结果
        if result.get("success"):
            num_detections = result.get("num_detections", 0)
            inference_time = result.get("inference_time", 0)
            logger.info(f"[推理] Base64 图片推理成功: {num_detections} 个检测, 耗时 {inference_time:.3f}s")
        else:
            error_msg = result.get("message", "推理失败")
            logger.error(f"[推理] Base64 图片推理失败: {error_msg}")

        return result

    except Exception as e:
        error_msg = f"Base64 图片解码失败: {str(e)}"
        logger.error(f"[推理] {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)


# =============================================================================
# 视频推理接口
# =============================================================================

@router.post("/inference/video")
async def infer_video(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25),
    save_output: bool = Form(False)
):
    """
    视频推理接口

    上传视频文件，使用 YOLO 模型进行逐帧目标检测

    参数:
        file: 上传的视频文件（mp4, avi, mov）
        model_name: 模型名称
        confidence: 置信度阈值
        save_output: 是否保存标注后的视频

    返回:
        success: 是否成功
        total_frames: 处理的视频帧数
        total_detections: 总检测数
        avg_detections_per_frame: 平均每帧检测数
        output_path: 输出视频路径（如果 save_output=True）
        processing_time: 处理耗时（秒）
    """
    import uuid

    # 生成请求 ID 用于追踪
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[推理] 收到视频推理请求 (ID:{request_id}): model={model_name}, conf={confidence}")

    # 1. 保存上传的视频
    filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
    file_path = settings.UPLOADS_DIR / filename
    save_uploaded_file(file, str(file_path))
    logger.debug(f"[推理] 视频已保存至: {file_path}")

    # 2. 设置输出路径
    output_path = None
    if save_output:
        output_path = str(settings.UPLOADS_DIR / f"inferred_{request_id}_{filename}")
        logger.debug(f"[推理] 输出视频路径: {output_path}")

    # 3. 执行视频推理
    try:
        result = inference_service.infer_video(
            video_path=str(file_path),
            model_name=model_name,
            confidence=confidence,
            output_path=output_path,
            save_output=save_output
        )
    except Exception as e:
        error_msg = f"视频推理执行失败: {str(e)}"
        logger.error(f"[推理] (ID:{request_id}) {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # 4. 返回结果
    if result.get("success"):
        frames = result.get("total_frames", 0)
        detections = result.get("total_detections", 0)
        logger.info(f"[推理] 视频推理完成 (ID:{request_id}): {frames} 帧, {detections} 个检测")
    else:
        error_msg = result.get("message", "视频推理失败")
        logger.error(f"[推理] 视频推理失败 (ID:{request_id}): {error_msg}")

    return result


# =============================================================================
# 自动标注推理接口
# =============================================================================

@router.post("/inference/predict")
async def predict_for_annotation(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25)
):
    """
    自动标注推理接口

    用于数据标注页面的 YOLO 自动标注功能，返回标准化的检测结果

    参数:
        file: 标注页面传来的图片数据
        model_name: 使用的模型名称
        confidence: 置信度阈值

    返回:
        success: 是否成功
        detections: 检测结果列表，每项包含:
            - class_name: 类别名称
            - class_id: 类别 ID
            - confidence: 置信度
            - bbox: 边界框坐标 [x1, y1, x2, y2]
        num_detections: 检测数量
    """
    import uuid

    # 生成请求 ID
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[自动标注] 收到请求 (ID:{request_id}): model={model_name}, conf={confidence}")

    try:
        # 1. 读取图片数据
        image_data = await file.read()
        logger.debug(f"[自动标注] 图片大小: {len(image_data)} bytes")

        # 2. 执行推理
        result = inference_service.stream_infer(image_data, model_name, confidence)

        # 3. 返回结果
        if result.get("success"):
            num = result.get("num_detections", 0)
            logger.info(f"[自动标注] 推理完成 (ID:{request_id}): {num} 个检测")
        else:
            error_msg = result.get("message", "自动标注失败")
            logger.error(f"[自动标注] 推理失败 (ID:{request_id}): {error_msg}")

        return result

    except Exception as e:
        error_msg = f"自动标注异常: {str(e)}"
        logger.error(f"[自动标注] (ID:{request_id}) {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


# =============================================================================
# 批量推理接口
# =============================================================================

@router.post("/inference/batch")
async def batch_infer(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25)
):
    """
    批量推理接口

    一次处理多张图片，适用于批量测试场景

    参数:
        files: 多张图片文件列表（最多 32 张）
        model_name: 模型名称
        confidence: 置信度阈值

    返回:
        success: 是否成功
        total_images: 图片总数
        total_detections: 总检测数
        total_time: 总处理时间（秒）
        avg_time_per_image: 平均每张图处理时间
        results: 每张图片的检测结果
    """
    import uuid

    # 生成请求 ID
    request_id = str(uuid.uuid4())[:8]
    num_files = len(files)
    logger.info(f"[批量推理] 收到请求 (ID:{request_id}): {num_files} 张图片, model={model_name}")

    image_paths = []

    try:
        # 1. 保存所有上传的图片
        for file in files:
            filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)
            file_path = settings.UPLOADS_DIR / filename
            save_uploaded_file(file, str(file_path))
            image_paths.append(str(file_path))

        logger.debug(f"[批量推理] 保存了 {len(image_paths)} 张图片")

        # 2. 执行批量推理
        result = inference_service.batch_infer(
            image_paths=image_paths,
            model_name=model_name,
            confidence=confidence
        )

        # 3. 返回结果
        if result.get("success"):
            total = result.get("total_detections", 0)
            time = result.get("total_time", 0)
            logger.info(f"[批量推理] 完成 (ID:{request_id}): {num_files} 张, {total} 个检测, {time:.2f}s")
        else:
            error_msg = result.get("message", "批量推理失败")
            logger.error(f"[批量推理] 失败 (ID:{request_id}): {error_msg}")

        return result

    except Exception as e:
        error_msg = f"批量推理异常: {str(e)}"
        logger.error(f"[批量推理] (ID:{request_id}) {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


# =============================================================================
# 摄像头监控接口
# =============================================================================

@router.post("/monitor/camera")
async def add_camera(
    name: str = Form(...),
    url: str = Form(...),
    model_name: Optional[str] = Form(None),
    confidence: Optional[float] = Form(0.25)
):
    """
    添加监控摄像头

    参数:
        name: 摄像头名称（唯一标识）
        url: 视频流地址（rtsp, http, https）
        model_name: 使用的模型
        confidence: 置信度阈值

    返回:
        success: 是否成功
        camera: 摄像头信息
    """
    logger.info(f"[监控] 添加摄像头: {name}, URL: {url}")

    result = monitor_service.add_camera(name, url, model_name, confidence)

    if result.get("success"):
        logger.info(f"[监控] 摄像头添加成功: {name}")
    else:
        error_msg = result.get("message", "添加失败")
        logger.error(f"[监控] 摄像头添加失败: {error_msg}")

    return result


@router.get("/monitor/cameras")
async def list_cameras():
    """
    列出所有已添加的摄像头

    返回:
        success: 是否成功
        cameras: 摄像头列表
    """
    logger.debug("[监控] 查询所有摄像头")
    return {"success": True, "cameras": monitor_service.list_cameras()}


@router.get("/monitor/camera/{name}")
async def get_camera_status(name: str):
    """
    获取摄像头状态

    参数:
        name: 摄像头名称

    返回:
        摄像头详细信息和当前状态
    """
    logger.debug(f"[监控] 查询摄像头状态: {name}")
    return monitor_service.get_camera_status(name)


@router.post("/monitor/camera/{name}/start")
async def start_monitoring(name: str):
    """
    开始监控指定摄像头

    参数:
        name: 摄像头名称

    返回:
        启动结果
    """
    logger.info(f"[监控] 启动摄像头: {name}")
    return monitor_service.start_stream(name)
