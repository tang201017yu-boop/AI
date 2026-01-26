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

            # 使用改进的解压函数
            extract_result = extract_zip(str(temp_zip), str(dataset_dir))
            temp_zip.unlink()

            if not extract_result["success"]:
                return {
                    "success": False,
                    "message": f"解压失败: {', '.join(extract_result['errors'][:3])}"
                }

            print(f"ZIP解压完成: {extract_result['extracted_files']} 个文件, "
                  f"图片: {extract_result['image_files']}, 标签: {extract_result['label_files']}")

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
            # 确保 count 是整数（YAML 解析可能产生浮点数）
            count = int(class_counts[class_id]) if class_counts[class_id] else 0
            classes_found.append({
                "id": class_id,
                "name": class_name,
                "count": count
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
        """重新组织数据集结构 - 支持各种目录结构"""
        from pathlib import Path
        import shutil

        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        # 确保目标目录存在
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有图片和标签文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images_found = []
        labels_found = []
        yaml_file = None  # 存储 data.yaml 路径

        # 遍历所有子目录查找文件
        for item in dataset_dir.rglob('*'):
            if item.is_file():
                ext = item.suffix.lower()

                # 跳过临时文件和隐藏文件
                if item.name.startswith('.') or item.name.startswith('_'):
                    continue

                if ext in image_extensions:
                    # 检查是否在目标目录本身（避免重复）
                    if not str(item).startswith(str(images_dir)) and not str(item).startswith(str(labels_dir)):
                        images_found.append(item)
                elif ext == '.txt':
                    # 跳过data.yaml等配置文件
                    if item.name != 'data.yaml' and item.name != 'dataset.yaml':
                        if not str(item).startswith(str(images_dir)) and not str(item).startswith(str(labels_dir)):
                            labels_found.append(item)
                elif ext in ['.yaml', '.yml']:
                    # 找到 data.yaml 或 dataset.yaml
                    if item.name in ['data.yaml', 'dataset.yaml']:
                        yaml_file = item

        # 移动图片到 images 目录
        moved_images = 0
        for img_path in images_found:
            try:
                # 使用相对路径作为新文件名（避免重名）
                rel_path = img_path.relative_to(dataset_dir)
                new_name = str(rel_path).replace('/', '_').replace('\\', '_')
                target_path = images_dir / new_name

                # 如果目标文件已存在，添加序号
                counter = 1
                while target_path.exists():
                    stem = new_name.rsplit('.', 1)[0]
                    ext = new_name.rsplit('.', 1)[1] if '.' in new_name else ''
                    new_name = f"{stem}_{counter}.{ext}"
                    target_path = images_dir / new_name
                    counter += 1

                shutil.move(str(img_path), str(target_path))
                moved_images += 1

            except Exception as e:
                print(f"Error moving image {img_path}: {e}")

        # 移动标签到 labels 目录
        moved_labels = 0
        for label_path in labels_found:
            try:
                rel_path = label_path.relative_to(dataset_dir)
                new_name = str(rel_path).replace('/', '_').replace('\\', '_')
                target_path = labels_dir / new_name

                counter = 1
                while target_path.exists():
                    stem = new_name.rsplit('.', 1)[0]
                    ext = new_name.rsplit('.', 1)[1] if '.' in new_name else ''
                    new_name = f"{stem}_{counter}.{ext}"
                    target_path = labels_dir / new_name
                    counter += 1

                shutil.move(str(label_path), str(target_path))
                moved_labels += 1

            except Exception as e:
                print(f"Error moving label {label_path}: {e}")

        # 处理 data.yaml 文件
        if yaml_file and yaml_file.exists():
            try:
                target_yaml = dataset_dir / "data.yaml"
                # 如果目标文件已存在，先备份
                if target_yaml.exists():
                    target_yaml.rename(str(target_yaml) + ".backup")
                # 移动并更新 YAML 内容
                self._move_and_update_yaml(yaml_file, target_yaml, dataset_dir)
                print(f"data.yaml 已移动到: {target_yaml}")
            except Exception as e:
                print(f"Error moving data.yaml: {e}")

        # 清理空的子目录
        for item in dataset_dir.iterdir():
            if item.is_dir():
                try:
                    # 只删除我们自己创建的空目录（不删除用户原有的结构）
                    if item.name in ['images', 'labels', '.thumbnails']:
                        # 如果是空目录，删除
                        if not any(item.iterdir()):
                            item.rmdir()
                    elif item.name not in ['train', 'val', 'test', 'images', 'labels']:
                        # 删除其他空目录
                        if not any(item.iterdir()):
                            item.rmdir()
                        else:
                            # 递归删除空子目录
                            self._cleanup_empty_dirs(item)
                            if not any(item.iterdir()):
                                item.rmdir()
                except Exception:
                    pass

    def _move_and_update_yaml(self, source_yaml: Path, target_yaml: Path, dataset_dir: Path):
        """
        移动并更新 YAML 文件
        更新路径引用以确保正确指向 images 和 labels 目录
        """
        import shutil
        import yaml as pyyaml

        # 读取原始 YAML 内容
        with open(source_yaml, 'r', encoding='utf-8') as f:
            content = f.read()

        # 尝试解析 YAML 并更新路径
        try:
            data = pyyaml.safe_load(content)
            if data:
                updated = False
                source_parent = source_yaml.parent

                # 更新 train、val、test 路径
                for key in ['train', 'val', 'test']:
                    if key in data and data[key]:
                        old_path = str(data[key])

                        # 处理 ../ 开头的相对路径
                        if old_path.startswith('../'):
                            # 相对于 source_yaml.parent 计算正确路径
                            rel_path = old_path  # 保持 ../train/images 格式
                            # 检查目标是否存在
                            potential_path = source_parent / rel_path
                            if not potential_path.exists():
                                # 尝试去掉 ../
                                clean_path = old_path.replace('../', '')
                                potential_path = dataset_dir / clean_path
                                if potential_path.exists():
                                    rel_path = clean_path
                                    old_path = f"../{clean_path}"

                        # 直接移动文件
                        shutil.move(str(source_yaml), str(target_yaml))

                        # 修改 YAML 内容
                        content = content.replace('train: ../train/images', 'train: train/images')
                        content = content.replace('val: ../valid/images', 'val: valid/images')
                        content = content.replace('val: ../val/images', 'val: valid/images')
                        content = content.replace('test: ../test/images', 'test: test/images')

                        with open(target_yaml, 'w', encoding='utf-8') as f:
                            f.write(content)

                        print(f"  data.yaml 路径已更新")
                        updated = True
                else:
                    # 直接移动文件
                    shutil.move(str(source_yaml), str(target_yaml))
            else:
                # 直接移动文件
                shutil.move(str(source_yaml), str(target_yaml))
        except Exception as e:
            print(f"  解析 YAML 失败，直接移动: {e}")
            try:
                shutil.move(str(source_yaml), str(target_yaml))
            except:
                pass

        # 删除源文件（如果还在）
        if source_yaml.exists():
            try:
                source_yaml.unlink()
            except:
                pass

    def _cleanup_empty_dirs(self, directory: Path):
        """递归清理空目录"""
        for item in directory.iterdir():
            if item.is_dir():
                self._cleanup_empty_dirs(item)
                try:
                    if not any(item.iterdir()):
                        item.rmdir()
                except Exception:
                    pass

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
