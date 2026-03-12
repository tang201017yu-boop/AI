"""
数据准备模块路由 - Data Preparation Routes
包括数据集管理、标注项目、SAM智能标注、智能存储、统计可视化
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import Optional, List
from pathlib import Path
import json
import io
import logging

from backend.core.config import settings
from backend.core.utils import allowed_file, save_uploaded_file, extract_zip
from backend.modules.data_preparation.dataset_service import dataset_service
from backend.modules.data_preparation.annotation_service import annotation_service
from backend.modules.data_preparation.sam_service import sam_service
from backend.modules.data_preparation.smart_annotation_service import smart_annotation_service
from backend.modules.data_preparation.storage_service import storage_service
from backend.modules.data_preparation.statistics_service import statistics_service
from backend.modules.data_preparation.augmentation_service import augmentation_service, AugmentationConfig
from backend.modules.data_preparation.export_service import dataset_exporter

# 创建日志记录器
logger = logging.getLogger(__name__)

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


@router.get("/datasets/{name}/validate")
async def validate_dataset(
    name: str,
    min_size: int = Query(28, description="最小边尺寸"),
    max_size: int = Query(4096, description="最大边尺寸")
):
    """
    验证数据集图像

    检查:
    - 最小边尺寸 >= min_size (默认28像素)
    - 最大边尺寸 <= max_size (默认4096像素)
    - 支持的颜色模式 (RGB, L, P, RGBA, LA)
    - 文件完整性
    """
    dataset_path = settings.DATASETS_DIR / name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    images_dir = dataset_path / "images"
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail="数据集图像目录不存在")

    result = dataset_service.validate_images(images_dir, min_size, max_size)
    return result


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
    return {"success": True, "images": images, "total": len(images)}


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
    model_name: str = Form("yolo11n.pt"),
    confidence: float = Form(0.25)
):
    """
    自动标注 - YOLO 检测 + SAM 分割

    使用 YOLO 模型进行目标检测，然后使用 SAM 进行精确分割

    参数:
        image_path: 图片路径
        class_names: 类别名称列表 (JSON 字符串)
        model_name: YOLO 模型名称
        confidence: 置信度阈值

    返回:
        自动标注结果，包含检测框、分割掩码等
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"[API] 自动标注请求: image={image_path}, model={model_name}, conf={confidence}")

    class_list = json.loads(class_names)
    result = sam_service.auto_label(image_path, class_list, model_name, confidence)

    if result.get("success"):
        logger.info(f"[API] 自动标注完成: {result.get('total_annotations', 0)} 个标注")

    return result


@router.post("/sam/batch-auto-label")
async def batch_auto_label(
    image_paths: str = Form(...),  # JSON string array
    class_names: str = Form(...),  # JSON string
    model_name: str = Form("yolo11n.pt"),
    confidence: float = Form(0.25)
):
    """
    批量自动标注

    一次性对多张图片进行自动标注

    参数:
        image_paths: 图片路径列表 (JSON 字符串)
        class_names: 类别名称列表
        model_name: YOLO 模型名称
        confidence: 置信度阈值

    返回:
        批量标注结果
    """
    import logging
    logger = logging.getLogger(__name__)

    img_list = json.loads(image_paths)
    class_list = json.loads(class_names)

    logger.info(f"[API] 批量自动标注请求: {len(img_list)} 张图片")

    result = sam_service.batch_auto_label(img_list, class_list, model_name, confidence)

    return result


@router.post("/sam/batch-sam-label")
async def batch_sam_label(
    file: UploadFile = File(...),
    class_name: str = Form(...),
    model_name: str = Form("yolo11n.pt"),
    confidence: float = Form(0.25)
):
    """批量 SAM 标注 - 针对特定类别"""
    import tempfile
    import os

    # 保存上传的文件到临时目录
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        # 写入临时文件
        content = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(content)

        # 调用 SAM 服务
        result = sam_service.batch_sam_label(
            image_path=temp_path,
            class_name=class_name,
            model_name=model_name,
            confidence=confidence
        )

        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass

        return result
    except Exception as e:
        # 清理临时文件
        try:
            os.remove(temp_path)
        except:
            pass
        logger.error(f"[API] batch-sam-label error: {str(e)}")
        return {"success": False, "message": str(e)}


@router.post("/sam/export-yolo")
async def export_yolo(
    annotations: str = Form(...),  # JSON string
    output_dir: str = Form(...),
    image_path: str = Form(...)
):
    """
    导出为 YOLO 格式

    参数:
        annotations: 标注列表 (JSON 字符串)
        output_dir: 输出目录
        image_path: 原图路径

    Returns:
        导出结果
    """
    import logging
    logger = logging.getLogger(__name__)

    ann_list = json.loads(annotations)
    logger.info(f"[API] 导出 YOLO 格式: {output_dir}")

    result = sam_service.export_yolo_format(ann_list, output_dir, image_path)

    return result


@router.get("/sam/models")
async def get_sam_models():
    """
    获取支持的 SAM 模型列表

    Returns:
        模型列表
    """
    status = sam_service.get_model_status()
    return {
        "success": True,
        "models": status.get("supported_models", {}),
        "current_model": status.get("model_type"),
        "loaded": status.get("loaded")
    }


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
    from backend.modules.data_preparation.dataset_service import dataset_service

    dataset_path = dataset_service._get_dataset_path(name)
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
    from backend.modules.data_preparation.dataset_service import dataset_service
    dataset_path = dataset_service._get_dataset_path(name)
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


# ==================== 数据增强 ====================

@router.get("/augmentation/transforms")
async def get_available_transforms():
    """获取可用的增强变换"""
    try:
        transforms = augmentation_service.get_available_transforms()
        return {"success": True, "transforms": transforms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/augmentation/preview")
async def preview_augmentation(
    file: UploadFile = File(...),
    num_previews: int = Query(4)
):
    """
    预览增强效果

    上传一张图片，返回原图和增强后的图片（base64 编码）
    """
    try:
        # 保存临时文件
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file.filename.split('.')[-1] if '.' in file.filename else 'jpg'}"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # 生成预览
        results = augmentation_service.preview_augmentation(
            tmp_path,
            num_previews=num_previews
        )

        # 删除临时文件
        tmp_path.unlink()

        if results:
            return {"success": True, "previews": results}
        else:
            raise HTTPException(status_code=500, detail="预览生成失败")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{name}/augment")
async def augment_dataset(
    name: str,
    # 增强配置
    horizontal_flip: bool = Query(True),
    vertical_flip: bool = Query(False),
    rotate: bool = Query(True),
    scale: bool = Query(True),
    translate: bool = Query(False),
    brightness_contrast: bool = Query(True),
    hue_saturation: bool = Query(True),
    blur: bool = Query(False),
    noise: bool = Query(False),
    cutout: bool = Query(False),
    # 参数配置
    rotate_limit: int = Query(45),
    scale_range: float = Query(0.1),
    brightness_range: float = Query(0.2),
    contrast_range: float = Query(0.2),
    num_augmented: int = Query(5),
    output_format: str = Query("jpg")
):
    """
    增强数据集

    基于 Albumentations 库对数据集进行图像增强。

    支持的增强类型:
    - 几何变换: 翻转、旋转、缩放、平移
    - 光照变换: 亮度、对比度、色调、饱和度
    - 噪声变换: 模糊、高斯噪声、Cutout

    输出:
    - 增强后的图片保存到 {数据集名}_augmented/images/
    - 标签文件保存到 {数据集名}_augmented/labels/
    """
    try:
        # 验证数据集存在
        dataset_path = settings.DATASETS_DIR / name
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="数据集不存在")

        # 创建增强配置
        config = AugmentationConfig(
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rotate=rotate,
            scale=scale,
            translate=translate,
            brightness_contrast=brightness_contrast,
            hue_saturation=hue_saturation,
            blur=blur,
            noise=noise,
            cutout=cutout,
            rotate_limit=rotate_limit,
            scale_range=scale_range,
            brightness_range=brightness_range,
            contrast_range=contrast_range,
            num_augmented=num_augmented,
            output_format=output_format
        )

        # 生成输出目录
        output_name = f"{name}_augmented"
        output_dir = settings.DATASETS_DIR / output_name

        # 执行增强
        result = augmentation_service.augment_dataset(
            input_dir=dataset_path,
            output_dir=output_dir,
            config=config
        )

        if result.success:
            return {
                "success": True,
                "message": result.message,
                "data": {
                    "output_dataset": output_name,
                    "original_images": result.total_images,
                    "augmented_images": result.total_augmented,
                    "output_dir": result.output_dir,
                    "samples": result.samples[:5]  # 返回前5个样本
                }
            }
        else:
            raise HTTPException(status_code=500, detail=result.message)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{name}/augment/custom")
async def augment_dataset_custom(
    name: str,
    config_json: str = Form(...)
):
    """
    自定义配置增强数据集

    通过 JSON 格式的自定义配置进行增强。
    """
    try:
        import json as json_module

        # 解析配置
        config_dict = json_module.loads(config_json)

        # 验证数据集存在
        dataset_path = settings.DATASETS_DIR / name
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="数据集不存在")

        # 创建配置
        config = AugmentationConfig(
            horizontal_flip=config_dict.get('horizontal_flip', True),
            vertical_flip=config_dict.get('vertical_flip', False),
            rotate=config_dict.get('rotate', True),
            scale=config_dict.get('scale', True),
            translate=config_dict.get('translate', False),
            brightness_contrast=config_dict.get('brightness_contrast', True),
            hue_saturation=config_dict.get('hue_saturation', True),
            blur=config_dict.get('blur', False),
            noise=config_dict.get('noise', False),
            cutout=config_dict.get('cutout', False),
            rotate_limit=config_dict.get('rotate_limit', 45),
            scale_range=config_dict.get('scale_range', 0.1),
            brightness_range=config_dict.get('brightness_range', 0.2),
            contrast_range=config_dict.get('contrast_range', 0.2),
            num_augmented=config_dict.get('num_augmented', 5),
            output_format=config_dict.get('output_format', 'jpg')
        )

        # 生成输出目录
        output_name = f"{name}_augmented"
        output_dir = settings.DATASETS_DIR / output_name

        # 执行增强
        result = augmentation_service.augment_dataset(
            input_dir=dataset_path,
            output_dir=output_dir,
            config=config
        )

        if result.success:
            return {
                "success": True,
                "message": result.message,
                "data": {
                    "output_dataset": output_name,
                    "original_images": result.total_images,
                    "augmented_images": result.total_augmented
                }
            }
        else:
            raise HTTPException(status_code=500, detail=result.message)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON 格式错误")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/augmented/list")
async def list_augmented_datasets():
    """列出所有增强后的数据集"""
    try:
        augmented_datasets = []
        for ds_dir in settings.DATASETS_DIR.iterdir():
            if ds_dir.is_dir() and ds_dir.name.endswith('_augmented'):
                info_file = ds_dir / 'augmentation_info.json'
                if info_file.exists():
                    import json
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    augmented_datasets.append({
                        "name": ds_dir.name,
                        "original_images": info.get('original_images', 0),
                        "augmented_images": info.get('augmented_images', 0),
                        "created_at": info.get('created_at', '')
                    })

        return {"success": True, "datasets": augmented_datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 图片浏览 ====================

@router.get("/datasets/{name}/images")
async def list_dataset_images(
    name: str,
    split: str = Query(None, description="按拆分筛选: train, val, test"),
    view: str = Query("grid", description="视图类型: grid, compact, table"),
    sort: str = Query("name_asc", description="排序: name_asc, name_desc, date_new, date_old, size_asc, size_desc, labels_asc, labels_desc"),
    labeled: str = Query(None, description="筛选: labeled, unlabeled"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    """
    列出数据集中的图片

    支持:
    - 按拆分筛选 (train/val/test)
    - 不同视图 (grid/compact/table)
    - 排序 (name_asc, name_desc, date_new, date_old, size_asc, size_desc, labels_asc, labels_desc)
    - 标注筛选 (labeled, unlabeled)
    - 分页加载
    """
    from PIL import Image

    from backend.modules.data_preparation.dataset_service import dataset_service

    # 使用 dataset_service 的辅助函数获取正确的数据集路径
    dataset_path = dataset_service._get_dataset_path(name)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists():
        return {"success": True, "images": [], "total": 0}

    # 获取所有图片
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        image_files.extend(images_dir.rglob(ext))

    # 解析标签获取信息
    image_labels = {}
    if labels_dir.exists():
        for label_file in labels_dir.rglob("*.txt"):
            # 匹配标签和图片
            label_stem = label_file.stem
            for img_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                img_path = images_dir / f"{label_stem}{img_ext}"
                if img_path.exists():
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        labels = []
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                labels.append({
                                    "class_id": int(parts[0]),
                                    "bbox": [float(x) for x in parts[1:]] if len(parts) > 1 else []
                                })
                        image_labels[str(img_path)] = labels
                    except:
                        pass
                    break

    # 构建图片列表
    images = []
    for img_path in sorted(image_files):
        try:
            with Image.open(img_path) as img:
                width, height = img.size

            # 确定拆分（从路径推断或默认unknown）
            path_parts = str(img_path).split('/')
            if 'train' in path_parts:
                split_name = 'train'
            elif 'val' in path_parts:
                split_name = 'val'
            elif 'test' in path_parts:
                split_name = 'test'
            else:
                split_name = 'unknown'

            # 筛选
            if split and split != split_name:
                continue

            labels = image_labels.get(str(img_path), [])

            # 标注筛选
            is_labeled = len(labels) > 0
            if labeled == "labeled" and not is_labeled:
                continue
            if labeled == "unlabeled" and is_labeled:
                continue

            images.append({
                "filename": img_path.name,
                "path": f"/api/v1/datasets/{name}/images/{img_path.name}",
                "thumbnail": f"/api/v1/datasets/{name}/thumbnails/{img_path.name}",
                "width": width,
                "height": height,
                "size": img_path.stat().st_size,
                "modified": img_path.stat().st_mtime,
                "split": split_name,
                "label_count": len(labels),
                "labels": labels
            })
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    # 排序
    if sort:
        if sort == "name_asc":
            images.sort(key=lambda x: x["filename"])
        elif sort == "name_desc":
            images.sort(key=lambda x: x["filename"], reverse=True)
        elif sort == "date_new":
            images.sort(key=lambda x: x.get("modified", 0), reverse=True)
        elif sort == "date_old":
            images.sort(key=lambda x: x.get("modified", 0))
        elif sort == "size_asc":
            images.sort(key=lambda x: x.get("size", 0))
        elif sort == "size_desc":
            images.sort(key=lambda x: x.get("size", 0), reverse=True)
        elif sort == "labels_asc":
            images.sort(key=lambda x: x.get("label_count", 0))
        elif sort == "labels_desc":
            images.sort(key=lambda x: x.get("label_count", 0), reverse=True)

    # 分页
    total = len(images)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_images = images[start:end]

    return {
        "success": True,
        "images": paginated_images,
        "total": total,
        "page": page,
        "page_size": page_size,
        "view": view
    }


@router.get("/datasets/{name}/images/{filename}")
async def get_dataset_image(name: str, filename: str):
    """获取数据集图片"""
    from fastapi.responses import FileResponse

    # 使用 dataset_service 获取正确的数据集路径（支持嵌套结构）
    dataset_path = dataset_service._get_dataset_path(name)
    images_dir = dataset_path / "images"

    # 先尝试直接路径
    image_path = images_dir / filename
    if not image_path.exists():
        # 递归搜索子目录
        found = False
        for img_path in images_dir.rglob(filename):
            image_path = img_path
            found = True
            break
        if not found:
            raise HTTPException(status_code=404, detail="图片不存在")

    # 确定图片类型
    ext = filename.lower().split('.')[-1]
    media_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'bmp': 'image/bmp',
        'webp': 'image/webp'
    }

    return FileResponse(image_path, media_type=media_types.get(ext, 'image/jpeg'))


@router.get("/datasets/{name}/image-info/{filename}")
async def get_image_info(name: str, filename: str):
    """获取图片详细信息"""
    from PIL import Image

    dataset_path = settings.DATASETS_DIR / name
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    image_path = images_dir / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")

    # 获取图片尺寸
    width, height = 0, 0
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except:
        pass

    # 获取标签
    label_file = labels_dir / f"{Path(filename).stem}.txt"
    labels = []
    if label_file.exists():
        try:
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if parts:
                        labels.append({
                            "class_id": int(parts[0]),
                            "bbox": [float(x) for x in parts[1:]] if len(parts) > 1 else []
                        })
        except:
            pass

    # 确定拆分
    path_parts = str(image_path).split('/')
    if 'train' in path_parts:
        split = 'train'
    elif 'val' in path_parts:
        split = 'val'
    elif 'test' in path_parts:
        split = 'test'
    else:
        split = 'unknown'

    return {
        "success": True,
        "info": {
            "filename": filename,
            "width": width,
            "height": height,
            "split": split,
            "label_count": len(labels),
            "labels": labels,
            "image_url": f"/api/v1/datasets/{name}/images/{filename}",
            "thumbnail_url": f"/api/v1/datasets/{name}/thumbnails/{filename}"
        }
    }


@router.get("/datasets/{name}/export/ndjson")
async def export_dataset_ndjson(name: str):
    """
    导出数据集为 NDJSON 格式

    格式:
    {"filename": "img001.jpg", "split": "train", "labels": [...]}
    {"filename": "img002.jpg", "split": "train", "labels": [...]}
    """
    from fastapi.responses import StreamingResponse
    import io

    dataset_path = settings.DATASETS_DIR / name
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    # 收集标签信息
    image_labels = {}
    if labels_dir.exists():
        for label_file in labels_dir.rglob("*.txt"):
            label_stem = label_file.stem
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                labels = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        labels.append({
                            "class_id": int(parts[0]),
                            "bbox": [float(x) for x in parts[1:]] if len(parts) > 1 else []
                        })
                image_labels[label_stem] = labels
            except:
                pass

    # 生成 NDJSON
    def generate():
        for img_path in sorted(images_dir.rglob('*')):
            if not img_path.is_file():
                continue

            ext = img_path.suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                continue

            # 确定拆分
            path_parts = str(img_path).split('/')
            if 'train' in path_parts:
                split = 'train'
            elif 'val' in path_parts:
                split = 'val'
            elif 'test' in path_parts:
                split = 'test'
            else:
                split = 'unknown'

            # 获取标签
            labels = image_labels.get(img_path.stem, [])

            # 构建记录
            record = {
                "filename": img_path.name,
                "split": split,
                "labels": labels
            }

            yield json.dumps(record, ensure_ascii=False) + '\n'

    filename = f"{name}_export.ndjson"
    response = StreamingResponse(generate(), media_type="application/octet-stream")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@router.get("/datasets/{name}/export")
async def export_dataset(
    name: str,
    format: str = Query("yolo", description="导出格式: yolo, coco, ndjson"),
    splits: str = Query("train,val", description="包含的拆分，逗号分隔")
):
    """
    导出数据集为指定格式

    参数:
        - format: yolo, coco, ndjson
        - splits: train,val,test (逗号分隔)
    """
    from fastapi.responses import StreamingResponse

    # 解析拆分
    include_splits = [s.strip() for s in splits.split(',') if s.strip()]

    try:
        data, content_type, filename = dataset_exporter.generate_export_response(
            name, format, include_splits
        )

        response = StreamingResponse(
            io.BytesIO(data),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@router.get("/datasets/{name}/detect-format")
async def detect_dataset_format(name: str):
    """自动检测数据集格式"""
    dataset_path = settings.DATASETS_DIR / name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    format_type = dataset_service.detect_dataset_format(dataset_path)
    return {"success": True, "format": format_type}


@router.post("/datasets/{name}/import-coco")
async def import_coco_dataset(
    name: str,
    split_images: bool = Form(True),
    train_ratio: float = Form(0.8)
):
    """导入 COCO 格式数据集"""
    dataset_path = settings.DATASETS_DIR / name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")

    result = dataset_service.import_coco_dataset(dataset_path, name, split_images, train_ratio)
    return result


# ==================== 智能标注 (支持多种任务类型) ====================

@router.get("/annotation/tasks")
async def get_task_types():
    """
    获取支持的标注任务类型

    返回所有支持的 YOLO 任务类型：
    - detect: 目标检测
    - segment: 实例分割
    - pose: 姿势估计
    - obb: 旋转框检测
    - classify: 图像分类
    """
    logger.info("[API] 获取标注任务类型")
    return smart_annotation_service.get_task_types()


@router.post("/annotation/task-type")
async def set_task_type(
    task_type: str = Form(...)
):
    """
    设置当前任务类型

    参数:
        task_type: 任务类型 (detect/segment/pose/obb/classify)
    """
    logger.info(f"[API] 设置任务类型: {task_type}")
    return smart_annotation_service.set_task_type(task_type)


@router.post("/annotation/classes")
async def set_annotation_classes(
    classes: str = Form(...)  # JSON string array
):
    """
    设置类别列表

    参数:
        classes: 类别名称列表 (JSON 字符串)
    """
    class_list = json.loads(classes)
    logger.info(f"[API] 设置类别: {class_list}")
    return smart_annotation_service.set_classes(class_list)


@router.post("/annotation/load-image")
async def load_annotation_image(
    file: UploadFile = File(...)
):
    """
    加载要标注的图片

    参数:
        file: 图片文件
    """
    logger.info(f"[API] 加载标注图片: {file.filename}")

    # 验证文件类型
    if not allowed_file(file.filename, ['jpg', 'jpeg', 'png', 'bmp']):
        raise HTTPException(status_code=400, detail="不支持的文件类型")

    # 保存临时文件
    import tempfile
    from pathlib import Path

    temp_dir = Path(tempfile.gettempdir()) / "annotation"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename

    save_uploaded_file(file, str(temp_path))

    # 加载图片
    result = smart_annotation_service.load_image(str(temp_path))

    return result


@router.post("/annotation/auto-annotate")
async def auto_annotate(
    file: UploadFile = File(...),
    task: str = Form("detect"),
    model_name: str = Form(None),
    confidence: float = Form(0.25)
):
    """
    自动标注 - 支持多种任务类型

    参数:
        file: 图片文件
        task: 任务类型 (detect/segment/pose/obb/classify)
        model_name: 模型名称（可选，默认根据任务自动选择）
        confidence: 置信度阈值

    返回:
        标注结果，包含检测框、分割掩码、关键点等
    """
    logger.info(f"[API] 自动标注: task={task}, model={model_name}, conf={confidence}")

    # 验证文件类型
    if not allowed_file(file.filename, ['jpg', 'jpeg', 'png', 'bmp']):
        raise HTTPException(status_code=400, detail="不支持的文件类型")

    # 保存临时文件
    import tempfile
    from pathlib import Path

    temp_dir = Path(tempfile.gettempdir()) / "annotation"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename

    save_uploaded_file(file, str(temp_path))

    # 加载图片
    load_result = smart_annotation_service.load_image(str(temp_path))
    if not load_result.get("success"):
        return load_result

    # 执行自动标注
    result = smart_annotation_service.auto_annotate(task, model_name, confidence)

    if result.get("success"):
        logger.info(f"[API] 自动标注完成: {result.get('total', 0)} 个标注")

    return result


@router.post("/annotation/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model_name: str = Form("yolo11n.pt"),
    confidence: float = Form(0.25)
):
    """
    目标检测 - 边界框标注

    使用 YOLO 进行目标检测，生成矩形边界框
    """
    logger.info(f"[API] 目标检测: model={model_name}")

    # 保存并加载图片
    import tempfile
    from pathlib import Path
    temp_dir = Path(tempfile.gettempdir()) / "annotation"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename
    save_uploaded_file(file, str(temp_path))

    load_result = smart_annotation_service.load_image(str(temp_path))
    if not load_result.get("success"):
        return load_result

    return smart_annotation_service.detect_objects(model_name, confidence)


@router.post("/annotation/segment")
async def segment_objects(
    file: UploadFile = File(...),
    model_name: str = Form("yolo11n-seg.pt"),
    confidence: float = Form(0.25)
):
    """
    实例分割 - 多边形标注

    使用 YOLO 进行实例分割，生成像素级掩码
    """
    logger.info(f"[API] 实例分割: model={model_name}")

    # 保存并加载图片
    import tempfile
    from pathlib import Path
    temp_dir = Path(tempfile.gettempdir()) / "annotation"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename
    save_uploaded_file(file, str(temp_path))

    load_result = smart_annotation_service.load_image(str(temp_path))
    if not load_result.get("success"):
        return load_result

    return smart_annotation_service.segment_objects(model_name, confidence)


@router.post("/annotation/pose")
async def detect_pose(
    file: UploadFile = File(...),
    model_name: str = Form("yolo11n-pose.pt"),
    confidence: float = Form(0.25)
):
    """
    姿势估计 - 关键点标注

    使用 YOLO 进行人体姿势估计，生成17个COCO关键点
    """
    logger.info(f"[API] 姿势估计: model={model_name}")

    # 保存并加载图片
    import tempfile
    from pathlib import Path
    temp_dir = Path(tempfile.gettempdir()) / "annotation"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename
    save_uploaded_file(file, str(temp_path))

    load_result = smart_annotation_service.load_image(str(temp_path))
    if not load_result.get("success"):
        return load_result

    return smart_annotation_service.detect_pose(model_name, confidence)


@router.post("/annotation/obb")
async def detect_obb(
    file: UploadFile = File(...),
    model_name: str = Form("yolo11n-obb.pt"),
    confidence: float = Form(0.25)
):
    """
    旋转框检测 - 定向边界框

    使用 YOLO 进行旋转框检测，生成倾斜的边界框
    """
    logger.info(f"[API] 旋转框检测: model={model_name}")

    # 保存并加载图片
    import tempfile
    from pathlib import Path
    temp_dir = Path(tempfile.gettempdir()) / "annotation"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename
    save_uploaded_file(file, str(temp_path))

    load_result = smart_annotation_service.load_image(str(temp_path))
    if not load_result.get("success"):
        return load_result

    return smart_annotation_service.detect_obb(model_name, confidence)


@router.post("/annotation/classify")
async def classify_image(
    file: UploadFile = File(...),
    model_name: str = Form("yolo11n-cls.pt"),
    top_k: int = Form(5)
):
    """
    图像分类

    使用 YOLO 进行图像分类，返回图像级标签
    """
    logger.info(f"[API] 图像分类: model={model_name}")

    # 保存并加载图片
    import tempfile
    from pathlib import Path
    temp_dir = Path(tempfile.gettempdir()) / "annotation"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename
    save_uploaded_file(file, str(temp_path))

    load_result = smart_annotation_service.load_image(str(temp_path))
    if not load_result.get("success"):
        return load_result

    return smart_annotation_service.classify_image(model_name, top_k)


@router.post("/annotation/export")
async def export_annotations(
    annotations: str = Form(...),  # JSON string
    output_dir: str = Form(...),
    image_filename: str = Form(...)
):
    """
    导出标注为 YOLO 格式

    参数:
        annotations: 标注列表 (JSON 字符串)
        output_dir: 输出目录
        image_filename: 图片文件名
    """
    ann_list = json.loads(annotations)
    logger.info(f"[API] 导出标注: {output_dir}")

    return smart_annotation_service.export_yolo_format(ann_list, output_dir, image_filename)


@router.get("/annotation/keypoint-names")
async def get_keypoint_names():
    """
    获取 COCO 关键点名称列表

    返回17个COCO关键点定义
    """
    from backend.modules.data_preparation.smart_annotation_service import COCO_KEYPOINTS, COCO_SKELETON
    return {
        "success": True,
        "keypoints": COCO_KEYPOINTS,
        "skeleton": COCO_SKELETON
    }