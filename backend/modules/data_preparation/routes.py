"""
数据准备模块路由 - Data Preparation Routes
包括数据集管理、标注项目、SAM智能标注、智能存储、统计可视化
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import Optional, List
from pathlib import Path
import json

from backend.core.config import settings
from backend.core.utils import allowed_file, save_uploaded_file, extract_zip
from backend.modules.data_preparation.dataset_service import dataset_service
from backend.modules.data_preparation.annotation_service import annotation_service
from backend.modules.data_preparation.sam_service import sam_service
from backend.modules.data_preparation.storage_service import storage_service
from backend.modules.data_preparation.statistics_service import statistics_service

router = APIRouter()


# ==================== 数据集管理 ====================

@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    task_type: str = Form("detect"),
    use_smart_storage: bool = Form(True)
):
    """上传数据集（支持智能存储）"""
    return dataset_service.upload_dataset(file, name, task_type, use_smart_storage)


@router.get("/datasets/list")
async def list_datasets():
    """列出所有数据集"""
    return {"success": True, "datasets": dataset_service.list_datasets()}


@router.get("/datasets/{name}")
async def get_dataset(name: str):
    """获取数据集详情"""
    info = dataset_service.get_dataset_info(name)
    if info:
        return {"success": True, "dataset": info}
    raise HTTPException(status_code=404, detail="数据集不存在")


@router.delete("/datasets/{name}")
async def delete_dataset(name: str):
    """删除数据集"""
    result = dataset_service.delete_dataset(name)
    if result["success"]:
        return result
    raise HTTPException(status_code=404, detail=result["message"])


# ==================== 数据处理 ====================

@router.post("/datasets/{name}/process")
async def process_dataset(
    name: str,
    max_image_size: int = Query(4096),
    thumbnail_size: int = Query(256)
):
    """
    处理数据集

    处理步骤:
    1. 图像归一化 - 大图像调整大小（最大 4096 像素）
    2. 缩略图生成 - 生成 256 像素预览图
    3. 标签解析 - 提取 YOLO 格式标签
    4. 统计计算 - 计算类别分布
    """
    dataset_path = settings.DATASETS_DIR / name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    result = dataset_service.process_dataset(
        dataset_path,
        max_image_size=max_image_size,
        thumbnail_size=thumbnail_size
    )

    if result.get("success"):
        return result
    raise HTTPException(status_code=500, detail=result.get("message", "处理失败"))


@router.get("/datasets/{name}/thumbnails/{image_name}")
async def get_thumbnail(name: str, image_name: str):
    """获取图片缩略图"""
    thumb_path = dataset_service.get_thumbnail_path(name, image_name)
    if thumb_path and Path(thumb_path).exists():
        from fastapi.responses import FileResponse
        return FileResponse(thumb_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="缩略图不存在")


# ==================== 智能存储 API ====================

@router.post("/storage/upload")
async def smart_upload(
    file: UploadFile = File(...),
    verify_integrity: bool = Form(True)
):
    """智能上传 - 自动去重 + 完整性校验"""
    return dataset_service.upload_with_smart_storage(file, "uploads", verify_integrity)


@router.get("/storage/stats")
async def get_storage_stats():
    """获取存储统计信息"""
    stats = dataset_service.get_storage_stats()
    return {"success": True, "stats": stats}


@router.get("/storage/verify")
async def verify_storage_integrity(file_id: Optional[int] = None):
    """验证存储完整性"""
    result = storage_service.verify_integrity(file_id)
    return {"success": result.get("success", False), **result}


@router.get("/storage/similar")
async def find_similar_images(
    image_path: str = Query(...),
    threshold: float = Query(0.9)
):
    """查找相似图片"""
    similar = dataset_service.find_similar_images(image_path, threshold)
    return {"success": True, "similar_images": similar}


@router.post("/storage/cleanup")
async def cleanup_storage(min_references: int = Query(1)):
    """清理未使用的存储文件"""
    result = dataset_service.cleanup_storage(min_references)
    return result


# ==================== 标注项目 ====================

@router.post("/annotation/projects")
async def create_project(
    name: str = Form(...),
    task_type: str = Form("detect"),
    classes: Optional[str] = Form(None)
):
    """创建标注项目"""
    class_list = json.loads(classes) if classes else None
    return annotation_service.create_project(name, task_type, class_list)


@router.post("/annotation/projects/{project_name}/images")
async def add_images(project_name: str, files: List[UploadFile] = File(...)):
    """添加图片到项目"""
    return annotation_service.add_images(project_name, files)


@router.get("/annotation/projects/{project_name}/images")
async def get_project_images(project_name: str):
    """获取项目图片列表"""
    images = annotation_service.get_project_images(project_name)
    return {"success": True, "images": images}


@router.get("/annotation/projects/{project_name}/image/{image_name}")
async def get_image_annotation(project_name: str, image_name: str):
    """获取图片标注"""
    return annotation_service.get_annotation(project_name, image_name)


@router.post("/annotation/projects/{project_name}/image/{image_name}")
async def save_image_annotation(
    project_name: str,
    image_name: str,
    annotations: str = Form(...)
):
    """保存图片标注"""
    ann_list = json.loads(annotations)
    return annotation_service.save_annotation(project_name, image_name, ann_list)


@router.get("/annotation/tasks")
async def get_task_types():
    """获取支持的标注任务类型"""
    return {"success": True, "tasks": annotation_service.TASK_TYPES}


@router.post("/annotation/export/{project_name}")
async def export_dataset(project_name: str, format: str = "yolo"):
    """导出数据集"""
    return annotation_service.export_dataset(project_name, format)


# ==================== SAM 智能标注 ====================

@router.post("/sam/load")
async def load_sam_model(model_type: str = "vit_b"):
    """加载 SAM 模型"""
    return sam_service.load_model(model_type)


@router.get("/sam/status")
async def get_sam_status():
    """获取 SAM 状态"""
    return sam_service.get_model_status()


@router.post("/sam/set-image")
async def set_sam_image(image_path: str = Form(...)):
    """设置 SAM 图片"""
    return sam_service.set_image(image_path)


@router.post("/sam/predict")
async def sam_predict(
    points: str = Form(...),  # JSON string
    labels: str = Form(...)   # JSON string
):
    """SAM 预测"""
    points_list = json.loads(points)
    labels_list = json.loads(labels)
    return sam_service.predict(points_list, labels_list)


@router.post("/sam/auto-label")
async def auto_label(
    image_path: str = Form(...),
    class_names: str = Form(...),  # JSON string
    model_name: str = Form("yolo11n.pt")
):
    """自动标注"""
    class_list = json.loads(class_names)
    return sam_service.auto_label(image_path, class_list, model_name)


# ==================== 统计与可视化 ====================

@router.get("/datasets/{name}/statistics")
async def get_dataset_statistics(
    name: str,
    force_refresh: bool = Query(False)
):
    """
    获取数据集统计信息

    包括:
    - 摘要 (图片数、标注数、类别数)
    - 类别分布 (条形图数据)
    - 位置热图 (空间分布)
    - 维度分析 (宽度/高度分布)
    - 拆分明细 (训练/验证/测试)
    """
    dataset_path = settings.DATASETS_DIR / name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    result = statistics_service.get_dataset_statistics(
        str(dataset_path),
        force_refresh
    )

    if result.get("success"):
        return result
    raise HTTPException(status_code=500, detail=result.get("message", "统计失败"))


@router.get("/datasets/{name}/class-distribution")
async def get_class_distribution(name: str):
    """获取类别分布"""
    dataset_path = settings.DATASETS_DIR / name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    result = statistics_service.get_dataset_statistics(str(dataset_path))
    if result.get("success"):
        return {
            "success": True,
            "class_distribution": result.get("class_distribution", {})
        }
    raise HTTPException(status_code=500, detail="获取失败")


@router.get("/datasets/{name}/spatial-distribution")
async def get_spatial_distribution(name: str):
    """获取位置热图数据"""
    dataset_path = settings.DATASETS_DIR / name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    result = statistics_service.get_dataset_statistics(str(dataset_path))
    if result.get("success"):
        return {
            "success": True,
            "spatial_distribution": result.get("spatial_distribution", {})
        }
    raise HTTPException(status_code=500, detail="获取失败")


@router.get("/datasets/{name}/dimension-analysis")
async def get_dimension_analysis(name: str):
    """获取维度分析"""
    dataset_path = settings.DATASETS_DIR / name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    result = statistics_service.get_dataset_statistics(str(dataset_path))
    if result.get("success"):
        return {
            "success": True,
            "dimension_analysis": result.get("dimension_analysis", {})
        }
    raise HTTPException(status_code=500, detail="获取失败")


@router.get("/datasets/{name}/split-details")
async def get_split_details(name: str):
    """获取拆分明细"""
    dataset_path = settings.DATASETS_DIR / name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    result = statistics_service.get_dataset_statistics(str(dataset_path))
    if result.get("success"):
        return {
            "success": True,
            "split_details": result.get("split_details", {})
        }
    raise HTTPException(status_code=500, detail="获取失败")


@router.post("/datasets/{name}/statistics/clear-cache")
async def clear_statistics_cache(name: str):
    """清除统计缓存"""
    statistics_service.clear_cache()
    return {"success": True, "message": "缓存已清除"}

