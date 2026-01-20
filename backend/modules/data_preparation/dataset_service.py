"""
数据集服务 - Dataset Service
处理数据集上传、管理、统计等功能
集成智能存储: 重复数据删除、完整性校验、存储优化
数据处理: 归一化、缩略图生成、标签解析、统计计算
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import threading
from collections import defaultdict
import math

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.core.config import settings
from backend.core.utils import allowed_file, get_unique_filename, save_uploaded_file, extract_zip
from backend.modules.data_preparation.storage_service import storage_service


class DatasetService:
    """数据集服务"""

    def __init__(self):
        self.dataset_cache: Dict[str, Dict] = {}
        self.cache_ttl = settings.DATASET_CACHE_TTL
        self._processing_lock = threading.Lock()

    def upload_dataset(
        self,
        file,
        name: str = None,
        task_type: str = "detect",
        use_smart_storage: bool = True,
        auto_process: bool = True
    ) -> Dict[str, Any]:
        """
        上传数据集
        支持: 图片、视频、ZIP压缩包
        集成智能存储功能
        自动处理: 归一化、缩略图、标签解析、统计计算
        """
        filename = file.filename or "dataset.zip"
        original_name = Path(filename).stem

        # 生成唯一名称
        if not name:
            name = f"{original_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 创建数据集目录
        dataset_dir = settings.DATASETS_DIR / name
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        for dir_path in [images_dir, labels_dir, dataset_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 处理上传文件
        file_path = dataset_dir / filename

        if filename.endswith('.zip'):
            # 解压 ZIP
            temp_zip = dataset_dir / "temp.zip"
            save_uploaded_file(file, str(temp_zip))
            extract_zip(str(temp_zip), str(dataset_dir))
            temp_zip.unlink()

            # 重新组织结构
            self._reorganize_dataset(dataset_dir)

            # 如果启用智能存储，处理重复数据
            if use_smart_storage:
                storage_result = self._process_dataset_with_smart_storage(dataset_dir)
            else:
                storage_result = {"stored": 0, "deduplicated": 0, "saved_mb": 0}
        else:
            # 保存文件
            unique_name = get_unique_filename(str(images_dir), filename)
            saved_path = str(images_dir / unique_name)
            save_uploaded_file(file, saved_path)

            # 使用智能存储
            if use_smart_storage and self._is_image_file(filename):
                storage_result = storage_service.add_file(saved_path)
            else:
                storage_result = {"action": "stored", "saved_mb": 0}

        # 自动处理数据集（归一化、缩略图、标签解析、统计）
        processing_result = {}
        if auto_process:
            processing_result = self.process_dataset(dataset_dir)

        # 生成数据集信息
        info = self._generate_dataset_info(dataset_dir, name, task_type)

        # 添加存储统计信息
        info["storage"] = {
            "smart_storage_enabled": use_smart_storage,
            "stored_count": storage_result.get("stored", 0) if isinstance(storage_result, dict) else 0,
            "deduplicated_count": storage_result.get("deduplicated", 0) if isinstance(storage_result, dict) else 0,
            "saved_mb": storage_result.get("saved_mb", 0) if isinstance(storage_result, dict) else 0
        }

        # 添加处理统计信息
        info["processing"] = {
            "auto_processed": auto_process,
            "normalized": processing_result.get("normalized", {}),
            "thumbnails": processing_result.get("thumbnails", {}),
            "labels": processing_result.get("labels", {}),
            "statistics": processing_result.get("statistics", {})
        }

        self.dataset_cache[name] = {
            "info": info,
            "timestamp": datetime.now().timestamp()
        }

        return {
            "success": True,
            "message": "数据集上传成功",
            "dataset": info,
            "storage_info": info["storage"],
            "processing": info["processing"]
        }

    def upload_with_smart_storage(
        self,
        file,
        dataset_name: str,
        verify_integrity: bool = True
    ) -> Dict[str, Any]:
        """
        使用智能存储上传图片
        自动去重 + 完整性校验
        """
        filename = file.filename or "upload.jpg"
        file_path = settings.UPLOADS_DIR / filename
        save_uploaded_file(file, str(file_path))

        # 添加到智能存储
        result = storage_service.add_file(str(file_path))

        # 验证完整性
        integrity_check = None
        if verify_integrity and result.get("success"):
            integrity_result = storage_service.verify_integrity(result.get("file_id"))
            integrity_check = {
                "valid": integrity_result.get("valid", 0) == 1,
                "message": "校验通过" if integrity_result.get("success") else "文件损坏"
            }

        return {
            "success": result.get("success", False),
            "action": result.get("action"),
            "file_id": result.get("file_id"),
            "storage_path": result.get("storage_path"),
            "integrity": integrity_check,
            "message": result.get("message")
        }

    def _process_dataset_with_smart_storage(self, dataset_dir: Path) -> Dict[str, Any]:
        """使用智能存储处理数据集目录"""
        images_dir = dataset_dir / "images"

        if not images_dir.exists():
            return {"stored": 0, "deduplicated": 0, "saved_mb": 0}

        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            image_files.extend(images_dir.rglob(ext))

        if not image_files:
            return {"stored": 0, "deduplicated": 0, "saved_mb": 0}

        # 批量添加到智能存储
        result = storage_service.add_files_batch(
            [str(f) for f in image_files],
            progress_callback=None
        )

        return result

    def _is_image_file(self, filename: str) -> bool:
        """判断是否为图片文件"""
        ext = Path(filename).suffix.lower()
        return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    def verify_dataset_integrity(self, name: str) -> Dict[str, Any]:
        """
        验证数据集完整性
        返回校验结果
        """
        storage_stats = storage_service.verify_integrity()

        # 获取数据集详情
        dataset_info = self.get_dataset_info(name)
        if not dataset_info:
            return {"success": False, "message": "数据集不存在"}

        return {
            "success": storage_stats.get("success", True),
            "dataset": name,
            "storage_integrity": storage_stats,
            "verified_at": datetime.now().isoformat()
        }

    def find_similar_images(self, image_path: str, threshold: float = 0.9) -> List[Dict[str, Any]]:
        """查找相似图片"""
        return storage_service.find_similar(image_path, threshold)

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return storage_service.get_storage_stats()

    def cleanup_storage(self, min_references: int = 1) -> Dict[str, Any]:
        """清理未使用的存储文件"""
        return storage_service.cleanup_unused(min_references)

    # ==================== 数据处理 pipeline ====================

    def process_dataset(
        self,
        dataset_dir: Path,
        max_image_size: int = 4096,
        thumbnail_size: int = 256
    ) -> Dict[str, Any]:
        """
        完整的数据处理 pipeline

        处理步骤:
        1. 图像归一化 - 大图像调整大小（最大 4096 像素）
        2. 缩略图生成 - 生成 256 像素预览图
        3. 标签解析 - 提取 YOLO 格式标签
        4. 统计计算 - 计算类别分布

        Returns:
            处理结果统计信息
        """
        if not PIL_AVAILABLE:
            return {
                "success": False,
                "message": "Pillow 库未安装，无法处理图像"
            }

        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        thumbs_dir = dataset_dir / ".thumbnails"

        # 创建缩略图目录
        thumbs_dir.mkdir(exist_ok=True)

        result = {
            "success": True,
            "normalized": {"total": 0, "skipped": 0, "failed": 0},
            "thumbnails": {"total": 0, "skipped": 0, "failed": 0},
            "labels": {"total": 0, "classes_found": [], "class_counts": {}},
            "statistics": {}
        }

        # 1. 归一化图像
        normalize_result = self._normalize_images(images_dir, max_image_size)
        result["normalized"] = normalize_result

        # 2. 生成缩略图
        thumbnail_result = self._generate_thumbnails(images_dir, thumbs_dir, thumbnail_size)
        result["thumbnails"] = thumbnail_result

        # 3. 解析标签
        if labels_dir.exists():
            label_result = self._parse_labels(labels_dir)
            result["labels"] = label_result

            # 4. 计算统计信息
            statistics = self._calculate_class_distribution(label_result)
            result["statistics"] = statistics

        return result

    def _normalize_images(self, images_dir: Path, max_size: int = 4096) -> Dict[str, int]:
        """归一化图像 - 大图像调整大小"""
        total = 0
        skipped = 0
        failed = 0

        if not PIL_AVAILABLE:
            return {"total": total, "skipped": skipped, "failed": failed}

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            for img_path in images_dir.rglob(ext):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size

                        # 检查是否需要调整大小
                        if width > max_size or height > max_size:
                            # 保持宽高比调整大小
                            ratio = min(max_size / width, max_size / height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)

                            # 使用高质量重采样
                            resized_img = img.resize(
                                (new_width, new_height),
                                Image.LANCZOS
                            )

                            # 保存调整后的图像
                            output_path = img_path
                            if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                                resized_img.save(output_path, 'JPEG', quality=95)
                            elif img_path.suffix.lower() == '.png':
                                resized_img.save(output_path, 'PNG', compress_level=1)
                            else:
                                resized_img.save(output_path)

                            total += 1
                        else:
                            skipped += 1

                except Exception as e:
                    failed += 1
                    print(f"归一化失败 {img_path}: {e}")

        return {"total": total, "skipped": skipped, "failed": failed}

    def _generate_thumbnails(
        self,
        images_dir: Path,
        thumbs_dir: Path,
        thumb_size: int = 256
    ) -> Dict[str, int]:
        """生成缩略图 - 256 像素预览"""
        total = 0
        skipped = 0
        failed = 0

        if not PIL_AVAILABLE:
            return {"total": total, "skipped": skipped, "failed": failed}

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            for img_path in images_dir.rglob(ext):
                try:
                    # 生成缩略图文件名
                    thumb_name = f"{img_path.stem}_thumb{img_path.suffix}"
                    thumb_path = thumbs_dir / thumb_name

                    # 如果已存在且是最新的，跳过
                    if thumb_path.exists():
                        if thumb_path.stat().st_mtime >= img_path.stat().st_mtime:
                            skipped += 1
                            continue

                    with Image.open(img_path) as img:
                        # 创建缩略图（保持宽高比）
                        img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
                        img.save(thumb_path, 'JPEG', quality=85, optimize=True)
                        total += 1

                except Exception as e:
                    failed += 1
                    print(f"缩略图生成失败 {img_path}: {e}")

        return {"total": total, "skipped": skipped, "failed": failed}

    def _parse_labels(self, labels_dir: Path) -> Dict[str, Any]:
        """解析标签 - 提取 YOLO 格式标签信息"""
        total_annotations = 0
        class_counts: Dict[int, int] = defaultdict(int)
        class_id_to_name: Dict[int, str] = {}
        images_with_labels = 0

        # 首先尝试从 data.yaml 获取类别名称
        data_yaml = labels_dir.parent / "data.yaml" if labels_dir.parent.exists() else None
        if data_yaml and data_yaml.exists():
            with open(data_yaml, 'r') as f:
                content = f.read()
                if 'names:' in content:
                    names_start = content.find('names:') + 7
                    names_content = content[names_start:]
                    for line in names_content.split('\n'):
                        if ':' in line and not line.strip().startswith('#'):
                            try:
                                idx = int(line.split(':')[0].strip())
                                name = line.split(':')[1].strip()
                                class_id_to_name[idx] = name
                            except:
                                pass

        # 解析所有标签文件
        for label_file in labels_dir.rglob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    has_annotation = False
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_annotations += 1
                            has_annotation = True

                    if has_annotation:
                        images_with_labels += 1

            except Exception as e:
                print(f"解析标签失败 {label_file}: {e}")

        # 构建类名列表
        classes_found = []
        for class_id in sorted(class_counts.keys()):
            class_name = class_id_to_name.get(class_id, f"class_{class_id}")
            classes_found.append({
                "id": class_id,
                "name": class_name,
                "count": class_counts[class_id]
            })

        return {
            "total": total_annotations,
            "images_with_labels": images_with_labels,
            "classes_found": classes_found,
            "class_counts": dict(class_counts)
        }

    def _calculate_class_distribution(self, label_result: Dict[str, Any]) -> Dict[str, Any]:
        """计算类别分布统计"""
        class_counts = label_result.get("class_counts", {})
        classes_found = label_result.get("classes_found", [])

        if not class_counts:
            return {
                "total_annotations": 0,
                "total_classes": 0,
                "distribution": [],
                "chart_data": None
            }

        total = sum(class_counts.values())

        # 构建分布数据
        distribution = []
        for cls in classes_found:
            distribution.append({
                "class_id": cls["id"],
                "class_name": cls["name"],
                "count": cls["count"],
                "percentage": round(cls["count"] / total * 100, 2)
            })

        # Chart.js 格式数据
        chart_data = {
            "labels": [c["class_name"] for c in distribution],
            "datasets": [{
                "label": "标注数量",
                "data": [c["count"] for c in distribution],
                "backgroundColor": self._generate_colors(len(distribution))
            }]
        }

        return {
            "total_annotations": total,
            "total_classes": len(distribution),
            "distribution": distribution,
            "chart_data": chart_data
        }

    def _generate_colors(self, count: int) -> List[str]:
        """生成图表颜色"""
        base_colors = [
            "rgba(54, 162, 235, 0.8)",
            "rgba(255, 99, 132, 0.8)",
            "rgba(75, 192, 192, 0.8)",
            "rgba(255, 206, 86, 0.8)",
            "rgba(153, 102, 255, 0.8)",
            "rgba(255, 159, 64, 0.8)",
            "rgba(199, 199, 199, 0.8)",
            "rgba(83, 102, 255, 0.8)",
            "rgba(40, 159, 64, 0.8)",
            "rgba(210, 99, 132, 0.8)"
        ]

        colors = []
        for i in range(count):
            colors.append(base_colors[i % len(base_colors)])
        return colors

    def get_thumbnail_path(self, dataset_name: str, image_name: str) -> Optional[str]:
        """获取图片缩略图路径"""
        dataset_dir = settings.DATASETS_DIR / dataset_name
        thumbs_dir = dataset_dir / ".thumbnails"

        # 尝试不同的扩展名
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            thumb_name = f"{Path(image_name).stem}_thumb{ext}"
            thumb_path = thumbs_dir / thumb_name
            if thumb_path.exists():
                return str(thumb_path)

        return None

    def _reorganize_dataset(self, dataset_dir: Path):
        """重新组织数据集结构"""
        for item in dataset_dir.iterdir():
            if item.is_dir():
                if item.name in ["images", "labels"]:
                    continue
                elif item.name in ["train", "val", "test"]:
                    (dataset_dir / "images" / item.name).mkdir(exist_ok=True)
                    (dataset_dir / "labels" / item.name).mkdir(exist_ok=True)
                    for sub_item in (item / "images").iterdir() if (item / "images").exists() else []:
                        if sub_item.is_file():
                            sub_item.rename(dataset_dir / "images" / item.name / sub_item.name)
                    for sub_item in (item / "labels").iterdir() if (item / "labels").exists() else []:
                        if sub_item.is_file():
                            sub_item.rename(dataset_dir / "labels" / item.name / sub_item.name)

    def _generate_dataset_info(
        self,
        dataset_dir: Path,
        name: str,
        task_type: str
    ) -> Dict[str, Any]:
        """生成数据集信息"""
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        # 统计图片数量
        image_count = 0
        class_counts: Dict[str, int] = {}

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_count += len(list(images_dir.rglob(ext)))

        # 读取 YAML 配置获取类别
        yaml_path = dataset_dir / "data.yaml"
        classes = []
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    content = f.read()
                    if 'names:' in content:
                        names_start = content.find('names:') + 7
                        names_content = content[names_start:]
                        for line in names_content.split('\n'):
                            if ':' in line and not line.strip().startswith('#'):
                                try:
                                    idx = int(line.split(':')[0].strip())
                                    name = line.split(':')[1].strip()
                                    classes.append(name)
                                except:
                                    pass
            except:
                pass

        if not classes:
            classes = ["object"]

        # 计算分割信息
        split_info = {"total": image_count}

        return {
            "name": name,
            "path": str(dataset_dir),
            "task_type": task_type,
            "num_images": image_count,
            "num_classes": len(classes),
            "classes": classes,
            "split": split_info,
            "created_at": datetime.now().isoformat()
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """列出所有数据集"""
        datasets = []

        for dataset_dir in settings.DATASETS_DIR.iterdir():
            if not dataset_dir.is_dir():
                continue

            cache_entry = self.dataset_cache.get(dataset_dir.name)
            if cache_entry:
                if datetime.now().timestamp() - cache_entry["timestamp"] < self.cache_ttl:
                    datasets.append(cache_entry["info"])
                    continue

            info = self._generate_dataset_info(dataset_dir, dataset_dir.name, "detect")
            self.dataset_cache[dataset_dir.name] = {
                "info": info,
                "timestamp": datetime.now().timestamp()
            }
            datasets.append(info)

        return datasets

    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取数据集详细信息"""
        dataset_dir = settings.DATASETS_DIR / name
        if not dataset_dir.exists():
            return None

        cache_entry = self.dataset_cache.get(name)
        if cache_entry:
            return cache_entry["info"]

        return self._generate_dataset_info(dataset_dir, name, "detect")

    def delete_dataset(self, name: str) -> Dict[str, Any]:
        """删除数据集"""
        dataset_dir = settings.DATASETS_DIR / name
        if not dataset_dir.exists():
            return {"success": False, "message": "数据集不存在"}

        import shutil
        shutil.rmtree(dataset_dir)

        if name in self.dataset_cache:
            del self.dataset_cache[name]

        return {"success": True, "message": "数据集已删除"}


# 全局实例
dataset_service = DatasetService()
