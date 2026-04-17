"""
数据集服务 - Dataset Service
处理数据集上传、管理、统计等功能
集成智能存储: 重复数据删除、完整性校验、存储优化
数据处理: 归一化、缩略图生成、标签解析、统计计算
"""
import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import threading
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


def _invalidate_dataset_statistics_cache() -> None:
    """数据集增删改移后清除统计缓存，避免概览页显示过期数字。"""
    try:
        from backend.modules.data_preparation.statistics_service import statistics_service
        statistics_service.clear_cache()
    except Exception as e:
        logger.debug("清除统计缓存跳过: %s", e)

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
from backend.modules.data_preparation.coco_parser import coco_parser as CocoParser


class DatasetService:
    """数据集服务"""

    def __init__(self):
        self.dataset_cache: Dict[str, Dict] = {}
        self.cache_ttl = settings.DATASET_CACHE_TTL
        self._processing_lock = threading.Lock()
        self._projects_meta_file = settings.DATASETS_DIR / ".dataset_projects.json"
        self._default_project_name = "默认项目"

    def _load_projects_meta(self) -> Dict[str, Any]:
        """加载数据集项目元数据"""
        default_meta = {
            "projects": [{
                "id": "default",
                "name": self._default_project_name,
                "created_at": datetime.now().isoformat()
            }],
            "dataset_to_project": {}
        }
        if not self._projects_meta_file.exists():
            return default_meta
        try:
            with open(self._projects_meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("数据集项目元数据格式错误（非对象），已使用默认配置")
                return default_meta
            raw_projects = data.get("projects", [])
            if not isinstance(raw_projects, list):
                raw_projects = []
            projects = []
            for p in raw_projects:
                if not isinstance(p, dict):
                    continue
                pid = p.get("id")
                if not pid:
                    continue
                projects.append({
                    "id": str(pid),
                    "name": p.get("name") or str(pid),
                    "created_at": p.get("created_at") or datetime.now().isoformat(),
                })
            raw_map = data.get("dataset_to_project", {})
            if not isinstance(raw_map, dict):
                raw_map = {}
            dataset_to_project = {
                str(k): str(v) for k, v in raw_map.items()
                if k is not None and v is not None
            }
            data["projects"] = projects
            data["dataset_to_project"] = dataset_to_project
            if not any(p.get("id") == "default" for p in data["projects"]):
                data["projects"].insert(0, {
                    "id": "default",
                    "name": self._default_project_name,
                    "created_at": datetime.now().isoformat()
                })
            return data
        except Exception as e:
            logger.warning("读取数据集项目元数据失败，使用默认配置: %s", e)
            return default_meta

    def _save_projects_meta(self, data: Dict[str, Any]) -> None:
        """保存数据集项目元数据"""
        path = self._projects_meta_file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _dataset_entry_to_dict(self, d: Any) -> Optional[Dict[str, Any]]:
        """将 list_datasets 的单条结果规范为可 JSON 序列化的 dict"""
        if d is None:
            return None
        if isinstance(d, dict):
            return d
        if hasattr(d, "model_dump"):
            try:
                return d.model_dump()
            except Exception:
                return None
        if hasattr(d, "dict"):
            try:
                return d.dict()
            except Exception:
                return None
        return None

    def list_dataset_projects(self) -> List[Dict[str, Any]]:
        """列出数据集项目（含项目下的数据集）"""
        meta = self._load_projects_meta()
        datasets = self.list_datasets()
        ds_map = {}
        for d in datasets:
            row = self._dataset_entry_to_dict(d)
            name = row.get("name") if row else None
            if row and name:
                ds_map[name] = row

        projects = []
        for p in meta["projects"]:
            pid = p.get("id")
            project_datasets = []
            for ds_name, ds_project in meta["dataset_to_project"].items():
                if ds_project == pid and ds_name in ds_map:
                    project_datasets.append(ds_map[ds_name])
            projects.append({
                "id": pid,
                "name": p.get("name"),
                "created_at": p.get("created_at"),
                "datasets": sorted(project_datasets, key=lambda x: str(x.get("name", "")).lower()),
                "dataset_count": len(project_datasets)
            })

        # 未分配的数据集自动归到默认项目
        default_project = next((x for x in projects if x["id"] == "default"), None)
        if default_project is not None:
            assigned = set(meta["dataset_to_project"].keys())
            unassigned = [d for n, d in ds_map.items() if n not in assigned]
            default_project["datasets"].extend(sorted(unassigned, key=lambda x: str(x.get("name", "")).lower()))
            default_project["dataset_count"] = len(default_project["datasets"])

        # 默认项目置顶，其余按名称稳定排序，保证前端下拉与列表顺序一致
        def _project_sort_key(item: Dict[str, Any]) -> tuple:
            pid = item.get("id") or ""
            name = str(item.get("name") or "")
            return (0 if pid == "default" else 1, name.lower())

        projects.sort(key=_project_sort_key)
        return projects

    def create_dataset_project(self, name: str) -> Dict[str, Any]:
        """创建数据集项目"""
        name = (name or "").strip()
        if not name:
            return {"success": False, "message": "项目名称不能为空"}

        meta = self._load_projects_meta()
        if any(p.get("name") == name for p in meta["projects"]):
            return {"success": False, "message": "项目名称已存在"}

        project_id = f"project_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        project = {
            "id": project_id,
            "name": name,
            "created_at": datetime.now().isoformat()
        }
        meta["projects"].append(project)
        try:
            self._save_projects_meta(meta)
        except OSError as e:
            logger.exception("保存数据集项目元数据失败: %s", self._projects_meta_file)
            return {"success": False, "message": f"无法写入项目配置（检查 data/datasets 目录权限）: {e}"}
        return {"success": True, "project": project}

    def rename_dataset_project(self, project_id: str, new_name: str) -> Dict[str, Any]:
        """重命名数据集项目"""
        if project_id == "default":
            return {"success": False, "message": "默认项目不支持重命名"}
        new_name = (new_name or "").strip()
        if not new_name:
            return {"success": False, "message": "新名称不能为空"}

        meta = self._load_projects_meta()
        if any(p.get("name") == new_name and p.get("id") != project_id for p in meta["projects"]):
            return {"success": False, "message": "项目名称已存在"}

        for p in meta["projects"]:
            if p.get("id") == project_id:
                p["name"] = new_name
                self._save_projects_meta(meta)
                return {"success": True, "project": p}
        return {"success": False, "message": "项目不存在"}

    def delete_dataset_project(self, project_id: str) -> Dict[str, Any]:
        """删除数据集项目（项目下数据集回到默认项目）"""
        if project_id == "default":
            return {"success": False, "message": "默认项目不支持删除"}
        meta = self._load_projects_meta()
        if not any(p.get("id") == project_id for p in meta["projects"]):
            return {"success": False, "message": "项目不存在"}

        meta["projects"] = [p for p in meta["projects"] if p.get("id") != project_id]
        for ds_name, pid in list(meta["dataset_to_project"].items()):
            if pid == project_id:
                meta["dataset_to_project"].pop(ds_name, None)
        self._save_projects_meta(meta)
        return {"success": True, "message": "项目已删除"}

    def assign_dataset_to_project(self, dataset_name: str, project_id: str) -> Dict[str, Any]:
        """将数据集分配到项目"""
        dataset_dir = settings.DATASETS_DIR / dataset_name
        if not dataset_dir.exists():
            return {"success": False, "message": "数据集不存在"}

        meta = self._load_projects_meta()
        if project_id != "default" and not any(p.get("id") == project_id for p in meta["projects"]):
            return {"success": False, "message": "项目不存在"}

        if project_id == "default":
            meta["dataset_to_project"].pop(dataset_name, None)
        else:
            meta["dataset_to_project"][dataset_name] = project_id
        self._save_projects_meta(meta)
        _invalidate_dataset_statistics_cache()
        return {"success": True, "message": "数据集已移动"}

    def rename_dataset(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """重命名数据集目录并更新映射"""
        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()
        if not old_name or not new_name:
            return {"success": False, "message": "名称不能为空"}
        if old_name == new_name:
            return {"success": True, "message": "名称未变化"}

        old_dir = settings.DATASETS_DIR / old_name
        new_dir = settings.DATASETS_DIR / new_name
        if not old_dir.exists():
            return {"success": False, "message": "原数据集不存在"}
        if new_dir.exists():
            return {"success": False, "message": "目标名称已存在"}

        shutil.move(str(old_dir), str(new_dir))

        # 更新缓存
        if old_name in self.dataset_cache:
            entry = self.dataset_cache.pop(old_name)
            if "info" in entry:
                entry["info"]["name"] = new_name
                entry["info"]["path"] = str(new_dir)
            self.dataset_cache[new_name] = entry

        # 更新项目映射
        meta = self._load_projects_meta()
        if old_name in meta["dataset_to_project"]:
            meta["dataset_to_project"][new_name] = meta["dataset_to_project"].pop(old_name)
        self._save_projects_meta(meta)

        _invalidate_dataset_statistics_cache()
        return {"success": True, "message": "数据集重命名成功", "new_name": new_name}

    def _get_dataset_path(self, name: str) -> Path:
        """
        获取数据集实际根目录，兼容多种解压结构：
        1. DATASETS_DIR/name/images/           (标准平铺)
        2. DATASETS_DIR/name/images/train/     (images 下拆分)
        3. DATASETS_DIR/name/train/images/     (Roboflow 根级拆分)
        4. DATASETS_DIR/name/name/images/      (ZIP 同名嵌套)
        5. DATASETS_DIR/name/name/train/images (ZIP 同名嵌套 + Roboflow)
        """
        base_path = settings.DATASETS_DIR / name
        nested_path = base_path / name
        SPLITS = ['train', 'val', 'valid', 'test']

        def _has_any(root: Path) -> bool:
            """检查该目录是否有 images/ 或 split/images/ 结构"""
            if (root / "images").exists():
                return True
            return any((root / sp / "images").exists() for sp in SPLITS)

        # 优先检查基础路径
        if _has_any(base_path):
            return base_path

        # 再检查同名嵌套路径（ZIP 内含同名顶级文件夹）
        if nested_path.exists() and _has_any(nested_path):
            return nested_path

        # 兜底：扫描一级子目录，找第一个含 images 的
        if base_path.exists():
            for child in base_path.iterdir():
                if child.is_dir() and _has_any(child):
                    return child

        return base_path

    def collect_dataset_image_entries(
        self, dataset_path: Path
    ) -> List[Tuple[Path, str, Optional[Path]]]:
        """
        与 GET /datasets/{name}/images 相同的扫描规则。
        返回 (图片路径, split 名, 标签查找根目录)。
        """
        IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        SPLIT_DIRS = ["train", "val", "valid", "test"]
        root_images_dir = dataset_path / "images"
        root_labels_dir = (
            dataset_path / "labels"
            if (dataset_path / "labels").exists()
            else dataset_path / "annotation"
        )

        has_root_images = (
            root_images_dir.exists()
            and any(
                p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                for p in root_images_dir.rglob("*")
            )
        )

        split_image_dirs: Dict[str, Path] = {}
        split_label_dirs: Dict[str, Path] = {}
        for sp in SPLIT_DIRS:
            sp_images = dataset_path / sp / "images"
            sp_labels = dataset_path / sp / "labels"
            if sp_images.exists() and any(
                p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                for p in sp_images.rglob("*")
            ):
                canonical = "val" if sp == "valid" else sp
                split_image_dirs[canonical] = sp_images
                if sp_labels.exists():
                    split_label_dirs[canonical] = sp_labels

        use_split_structure = not has_root_images and bool(split_image_dirs)
        out: List[Tuple[Path, str, Optional[Path]]] = []

        if not has_root_images and not use_split_structure:
            return out

        if use_split_structure:
            for sp_name, sp_dir in split_image_dirs.items():
                for img_path in sp_dir.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in IMAGE_SUFFIXES:
                        out.append((img_path, sp_name, split_label_dirs.get(sp_name)))
        else:
            for img_path in root_images_dir.rglob("*"):
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                path_parts = img_path.parts
                if "train" in path_parts:
                    sp_name = "train"
                elif "val" in path_parts:
                    sp_name = "val"
                elif "test" in path_parts:
                    sp_name = "test"
                else:
                    sp_name = "unknown"
                lbl_root = root_labels_dir if root_labels_dir.exists() else None
                out.append((img_path, sp_name, lbl_root))

        return out

    def count_label_lines_for_image(self, img_path: Path, label_root: Optional[Path]) -> int:
        """与图片列表接口一致：在 label_root 下按文件名主干查找 YOLO txt 并统计非空行数。"""
        if not label_root or not label_root.exists():
            return 0
        stem = img_path.stem
        for lf in label_root.rglob(f"{stem}.txt"):
            if lf.is_file():
                try:
                    return sum(1 for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
                except Exception:
                    return 0
        return 0

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

        # 兜底校验：压缩包里没有任何图片时，返回明确错误，避免前端看到“上传成功但图片为0”
        if int(info.get("num_images", 0) or 0) <= 0:
            try:
                shutil.rmtree(dataset_dir, ignore_errors=True)
            except Exception:
                pass
            return {
                "success": False,
                "message": "未检测到可用图片文件（支持 jpg/jpeg/png/bmp/webp），请检查 ZIP 内容与目录结构",
            }

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

        _invalidate_dataset_statistics_cache()

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
        thumbnail_size: int = 256,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        完整的数据处理 pipeline

        处理步骤:
        1. 图像验证 - 检查尺寸、格式、完整性
        2. 图像归一化 - 大图像调整大小（最大 4096 像素）
        3. 缩略图生成 - 生成 256 像素预览图
        4. 标签解析 - 提取 YOLO 格式标签
        5. 统计计算 - 计算类别分布

        Args:
            dataset_dir: 数据集目录
            max_image_size: 最大图像尺寸
            thumbnail_size: 缩略图尺寸
            validate: 是否进行图像验证

        Returns:
            处理结果统计信息
        """
        if not PIL_AVAILABLE:
            return {
                "success": False,
                "message": "Pillow 库未安装，无法处理图像"
            }

        # 获取正确的数据集路径（支持嵌套结构）
        actual_dataset_dir = self._get_dataset_path(dataset_dir.name) if dataset_dir.name else dataset_dir
        if dataset_dir.name and not (actual_dataset_dir / "images").exists():
            actual_dataset_dir = dataset_dir

        images_dir = actual_dataset_dir / "images"
        labels_dir = actual_dataset_dir / "labels"
        thumbs_dir = actual_dataset_dir / ".thumbnails"

        # 创建缩略图目录
        thumbs_dir.mkdir(exist_ok=True)

        result = {
            "success": True,
            "validation": {},
            "normalized": {"total": 0, "skipped": 0, "failed": 0},
            "thumbnails": {"total": 0, "skipped": 0, "failed": 0},
            "labels": {"total": 0, "classes_found": [], "class_counts": {}},
            "statistics": {}
        }

        # 0. 验证图像
        if validate and images_dir.exists():
            validation_result = self.validate_images(images_dir, min_size=28, max_size=max_image_size)
            result["validation"] = validation_result

            # 如果有无效图像（过小），记录警告
            if validation_result.get("invalid", 0) > 0:
                result["success"] = True
                result["warning"] = f"发现 {validation_result['invalid']} 个无效图像"

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

    # ==================== 图像验证 ====================

    def validate_images(
        self,
        images_dir: Path,
        min_size: int = 28,
        max_size: int = 4096
    ) -> Dict[str, Any]:
        """
        验证图像文件

        检查:
        - 最小边尺寸 >= min_size (默认28像素)
        - 最大边尺寸 <= max_size (默认4096像素，会自动调整)
        - 支持的颜色模式 (RGB, L, P)
        - 文件完整性

        Args:
            images_dir: 图像目录
            min_size: 最小边尺寸
            max_size: 最大边尺寸

        Returns:
            验证结果统计
        """
        if not PIL_AVAILABLE:
            return {
                "success": False,
                "message": "Pillow 库未安装",
                "valid": 0,
                "invalid": 0,
                "errors": []
            }

        valid_count = 0
        invalid_count = 0
        errors = []
        needs_resize = []

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            for img_path in images_dir.rglob(ext):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size

                        # 检查最小尺寸
                        min_dim = min(width, height)
                        if min_dim < min_size:
                            invalid_count += 1
                            errors.append({
                                "file": img_path.name,
                                "error": f"图像过小: {width}x{height} (最小边需>= {min_size}像素)",
                                "type": "too_small"
                            })
                            continue

                        # 检查是否需要调整大小
                        if width > max_size or height > max_size:
                            needs_resize.append({
                                "file": img_path.name,
                                "original": f"{width}x{height}",
                                "action": "will_resize"
                            })
                            valid_count += 1
                            continue

                        # 检查颜色模式
                        mode = img.mode
                        if mode not in ['RGB', 'L', 'P', 'RGBA', 'LA']:
                            invalid_count += 1
                            errors.append({
                                "file": img_path.name,
                                "error": f"不支持的颜色模式: {mode}",
                                "type": "unsupported_mode"
                            })
                            continue

                        valid_count += 1

                except Exception as e:
                    invalid_count += 1
                    errors.append({
                        "file": img_path.name,
                        "error": f"文件损坏或无法读取: {str(e)}",
                        "type": "corrupted"
                    })

        return {
            "success": True,
            "valid": valid_count,
            "invalid": invalid_count,
            "needs_resize": len(needs_resize),
            "resize_details": needs_resize[:10],  # 只返回前10个
            "errors": errors[:50],  # 限制错误数量
            "total_checked": valid_count + invalid_count
        }

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
        # 获取正确的数据集路径（支持嵌套结构）
        dataset_dir = self._get_dataset_path(dataset_name)
        thumbs_dir = dataset_dir / ".thumbnails"

        # 尝试不同的扩展名
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            thumb_name = f"{Path(image_name).stem}_thumb{ext}"
            thumb_path = thumbs_dir / thumb_name
            if thumb_path.exists():
                return str(thumb_path)

        return None

    def _reorganize_dataset(self, dataset_dir: Path):
        """
        重新组织数据集结构，兼容多种 YOLO 目录格式:
        - 标准格式: images/train/, images/val/, labels/train/, labels/val/
        - Roboflow 格式: train/images/, valid/images/, train/labels/, valid/labels/
        - 平铺格式: images/*.jpg, labels/*.txt
        - 嵌套格式: <root>/<dataset_name>/images/ ...

        策略: 优先保留已有 images/ 结构；对于 Roboflow 格式，
        将 train/images/ → images/train/，valid/images/ → images/val/ 等，
        保持原始文件名不变，使标签匹配继续有效。
        """
        import shutil

        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        # ---- 检测已有结构 ----
        # 如果 images/ 下已经有图片（含子目录），直接使用，不移动
        def _has_images(d: Path) -> bool:
            if not d.exists():
                return False
            return any(
                f.suffix.lower() in image_extensions
                for f in d.rglob('*')
                if f.is_file()
            )

        if _has_images(images_dir):
            # images/ 已有图片，只需处理 data.yaml 位置
            self._ensure_yaml_at_root(dataset_dir)
            return

        # ---- 检测 Roboflow / 标准 YOLO 分割结构 ----
        # train/images/, valid/images/, test/images/ 等
        SPLIT_MAP = {
            'train': 'train',
            'valid': 'val',   # Roboflow 用 valid，标准化为 val
            'val':   'val',
            'test':  'test',
        }

        split_found = {}  # canonical_name -> src_images_dir
        label_split_found = {}  # canonical_name -> src_labels_dir
        for src_name, canonical in SPLIT_MAP.items():
            src_img = dataset_dir / src_name / "images"
            src_lbl = dataset_dir / src_name / "labels"
            if _has_images(src_img):
                # 如果已有同名 canonical（valid/val 均映射 val），只取第一个
                if canonical not in split_found:
                    split_found[canonical] = src_img
                    if src_lbl.exists():
                        label_split_found[canonical] = src_lbl

        if split_found:
            # 将 train/images/ → images/train/，valid/images/ → images/val/ 等
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            for canonical, src_img_dir in split_found.items():
                dst_img_dir = images_dir / canonical
                dst_img_dir.mkdir(parents=True, exist_ok=True)

                for img_file in src_img_dir.rglob('*'):
                    if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                        target = dst_img_dir / img_file.name
                        counter = 1
                        while target.exists():
                            target = dst_img_dir / f"{img_file.stem}_{counter}{img_file.suffix}"
                            counter += 1
                        try:
                            shutil.move(str(img_file), str(target))
                        except Exception as e:
                            print(f"Error moving image {img_file}: {e}")

            for canonical, src_lbl_dir in label_split_found.items():
                dst_lbl_dir = labels_dir / canonical
                dst_lbl_dir.mkdir(parents=True, exist_ok=True)

                for lbl_file in src_lbl_dir.rglob('*.txt'):
                    target = dst_lbl_dir / lbl_file.name
                    counter = 1
                    while target.exists():
                        target = dst_lbl_dir / f"{lbl_file.stem}_{counter}{lbl_file.suffix}"
                        counter += 1
                    try:
                        shutil.move(str(lbl_file), str(target))
                    except Exception as e:
                        print(f"Error moving label {lbl_file}: {e}")

            # 清理已空的源目录
            for src_name in list(SPLIT_MAP.keys()):
                src_dir = dataset_dir / src_name
                if src_dir.exists():
                    self._cleanup_empty_dirs(src_dir)
                    try:
                        if not any(src_dir.iterdir()):
                            src_dir.rmdir()
                    except Exception:
                        pass

            self._ensure_yaml_at_root(dataset_dir)
            return

        # ---- 回退：扫描全部子目录，把散落的图片收入 images/ ----
        yaml_file = None
        images_found = []
        labels_found = []

        for item in dataset_dir.rglob('*'):
            if not item.is_file():
                continue
            if item.name.startswith('.') or item.name.startswith('_'):
                continue
            ext = item.suffix.lower()
            # 已在目标目录内的跳过
            try:
                item.relative_to(images_dir)
                continue
            except ValueError:
                pass
            try:
                item.relative_to(labels_dir)
                continue
            except ValueError:
                pass

            if ext in image_extensions:
                images_found.append(item)
            elif ext == '.txt':
                labels_found.append(item)
            elif ext in ('.yaml', '.yml') and item.name in ('data.yaml', 'dataset.yaml'):
                yaml_file = item

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images_found:
            target = images_dir / img_path.name
            counter = 1
            while target.exists():
                target = images_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                counter += 1
            try:
                shutil.move(str(img_path), str(target))
            except Exception as e:
                print(f"Error moving image {img_path}: {e}")

        for lbl_path in labels_found:
            target = labels_dir / lbl_path.name
            counter = 1
            while target.exists():
                target = labels_dir / f"{lbl_path.stem}_{counter}{lbl_path.suffix}"
                counter += 1
            try:
                shutil.move(str(lbl_path), str(target))
            except Exception as e:
                print(f"Error moving label {lbl_path}: {e}")

        if yaml_file and yaml_file.exists():
            self._ensure_yaml_at_root(dataset_dir, yaml_file)

        # 清理空目录
        for item in list(dataset_dir.iterdir()):
            if item.is_dir() and item.name not in ('images', 'labels', '.thumbnails'):
                self._cleanup_empty_dirs(item)
                try:
                    if not any(item.iterdir()):
                        item.rmdir()
                except Exception:
                    pass

    def _ensure_yaml_at_root(self, dataset_dir: Path, yaml_source: Path = None):
        """确保 data.yaml 位于数据集根目录"""
        import shutil
        target_yaml = dataset_dir / "data.yaml"

        if yaml_source and yaml_source.exists() and yaml_source != target_yaml:
            try:
                if target_yaml.exists():
                    target_yaml.rename(str(target_yaml) + ".backup")
                self._move_and_update_yaml(yaml_source, target_yaml, dataset_dir)
            except Exception as e:
                print(f"Error moving data.yaml: {e}")
            return

        # 在子目录中搜索 data.yaml
        if not target_yaml.exists():
            for found in dataset_dir.rglob("data.yaml"):
                if found != target_yaml:
                    try:
                        self._move_and_update_yaml(found, target_yaml, dataset_dir)
                        break
                    except Exception as e:
                        print(f"Error moving data.yaml: {e}")

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
        # 支持嵌套目录结构：dataset_name/images/train 和 dataset_name/images/val
        images_dir = dataset_dir / "images"

        # 如果顶层没有 images，尝试在嵌套目录中查找（使用数据集名称作为子目录）
        if not images_dir.exists():
            nested_images_dir = dataset_dir / name / "images"
            if nested_images_dir.exists():
                images_dir = nested_images_dir

        labels_dir = dataset_dir / "labels"

        # 如果顶层没有 labels，尝试在嵌套目录中查找
        if not labels_dir.exists():
            nested_labels_dir = dataset_dir / name / "labels"
            if nested_labels_dir.exists():
                labels_dir = nested_labels_dir

        # 统计图片数量 - 支持 train/val 子目录及 Roboflow 结构
        image_count = 0
        class_counts: Dict[str, int] = {}

        image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        if images_dir.exists():
            image_count += sum(
                1 for p in images_dir.rglob('*')
                if p.is_file() and p.suffix.lower() in image_suffixes
            )

        # 兼容 Roboflow 结构: train/images/, valid/images/, test/images/
        if image_count == 0:
            for sp in ['train', 'val', 'valid', 'test']:
                sp_dir = dataset_dir / sp / "images"
                if sp_dir.exists():
                    image_count += sum(
                        1 for p in sp_dir.rglob('*')
                        if p.is_file() and p.suffix.lower() in image_suffixes
                    )

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
        datasets: List[Dict[str, Any]] = []
        root = settings.DATASETS_DIR
        if not root.exists() or not root.is_dir():
            logger.warning("数据集根目录不可用: %s", root)
            return datasets

        try:
            entries = list(root.iterdir())
        except OSError as e:
            logger.exception("无法扫描数据集目录 %s: %s", root, e)
            return datasets

        for dataset_dir in entries:
            if not dataset_dir.is_dir():
                continue
            name = dataset_dir.name
            if name.startswith("."):
                continue

            try:
                cache_entry = self.dataset_cache.get(name)
                if cache_entry:
                    if datetime.now().timestamp() - cache_entry["timestamp"] < self.cache_ttl:
                        info = cache_entry["info"]
                        if isinstance(info, dict):
                            datasets.append(info)
                            continue
                        self.dataset_cache.pop(name, None)

                info = self._generate_dataset_info(dataset_dir, name, "detect")
                self.dataset_cache[name] = {
                    "info": info,
                    "timestamp": datetime.now().timestamp()
                }
                datasets.append(info)
            except Exception as e:
                logger.warning("跳过数据集目录 %s（扫描失败）: %s", name, e)
                continue

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

    def resolve_dataset_image_path(self, dataset_name: str, filename: str) -> Optional[Path]:
        """定位数据集中某张图片的绝对路径（与图片 GET 路由搜索顺序一致）。"""
        dataset_path = self._get_dataset_path(dataset_name)
        if not dataset_path.exists():
            return None

        search_dirs: List[Path] = []
        if (dataset_path / "images").exists():
            search_dirs.append(dataset_path / "images")
        for sp in ["train", "val", "valid", "test"]:
            sp_dir = dataset_path / sp / "images"
            if sp_dir.exists():
                search_dirs.append(sp_dir)
        if not search_dirs:
            search_dirs = [dataset_path]

        for search_dir in search_dirs:
            candidate = search_dir / filename
            if candidate.is_file():
                return candidate
            for img_path in search_dir.rglob(filename):
                if img_path.is_file():
                    return img_path
        return None

    def _resolve_label_path_for_image(self, dataset_path: Path, image_path: Path) -> Optional[Path]:
        """根据图片路径查找对应 YOLO txt 标签路径。"""
        stem = image_path.stem
        try:
            rel = image_path.relative_to(dataset_path)
        except ValueError:
            rel = None
        if rel is not None:
            parts = list(rel.parts)
            # Roboflow: train/images/foo.jpg -> train/labels/foo.txt
            if len(parts) >= 3 and parts[1] == "images":
                split = parts[0]
                lbl = dataset_path / split / "labels" / f"{stem}.txt"
                if lbl.is_file():
                    return lbl
            # 标准: images/train/foo.jpg -> labels/train/foo.txt
            if len(parts) >= 3 and parts[0] == "images":
                sub = parts[1]
                if sub in ("train", "val", "test", "valid"):
                    canon = "val" if sub == "valid" else sub
                    lbl = dataset_path / "labels" / canon / f"{stem}.txt"
                    if lbl.is_file():
                        return lbl
        flat = dataset_path / "labels" / f"{stem}.txt"
        if flat.is_file():
            return flat
        labels_dir = dataset_path / "labels"
        if labels_dir.exists():
            for p in labels_dir.rglob(f"{stem}.txt"):
                if p.is_file():
                    return p
        return None

    def delete_dataset_image(self, dataset_name: str, filename: str) -> Dict[str, Any]:
        """删除数据集中单张图片及对应标签、缩略图缓存。"""
        dataset_path = self._get_dataset_path(dataset_name)
        if not dataset_path.exists():
            return {"success": False, "message": "数据集不存在"}

        image_path = self.resolve_dataset_image_path(dataset_name, filename)
        if not image_path or not image_path.is_file():
            return {"success": False, "message": "图片不存在"}

        try:
            image_path.unlink()
        except OSError as e:
            return {"success": False, "message": str(e)}

        stem = image_path.stem
        suffix = image_path.suffix

        label_path = self._resolve_label_path_for_image(dataset_path, image_path)
        if label_path and label_path.is_file():
            try:
                label_path.unlink()
            except OSError:
                pass

        thumbs_dir = dataset_path / ".thumbnails"
        if thumbs_dir.exists():
            for ext in (suffix, ".jpg", ".jpeg", ".png", ".webp"):
                t = thumbs_dir / f"{stem}_thumb{ext}"
                if t.exists():
                    try:
                        t.unlink()
                    except OSError:
                        pass

        if dataset_name in self.dataset_cache:
            del self.dataset_cache[dataset_name]
        _invalidate_dataset_statistics_cache()
        return {"success": True, "message": "图片已删除", "filename": filename}

    def delete_dataset(self, name: str) -> Dict[str, Any]:
        """删除数据集"""
        dataset_dir = settings.DATASETS_DIR / name
        if not dataset_dir.exists():
            return {"success": False, "message": "数据集不存在"}

        import shutil
        shutil.rmtree(dataset_dir)

        if name in self.dataset_cache:
            del self.dataset_cache[name]

        meta = self._load_projects_meta()
        if name in meta["dataset_to_project"]:
            meta["dataset_to_project"].pop(name, None)
            self._save_projects_meta(meta)

        _invalidate_dataset_statistics_cache()
        return {"success": True, "message": "数据集已删除"}

    # ==================== 格式检测 ====================

    def detect_dataset_format(self, dataset_dir: Path) -> str:
        """
        自动检测数据集格式

        Returns:
            "yolo" - YOLO 格式
            "coco" - COCO 格式
            "raw" - 原始格式（仅有图像，无标注）
        """
        # 检查 data.yaml（YOLO 格式特征）
        data_yaml = dataset_dir / "data.yaml"
        if data_yaml.exists():
            try:
                content = data_yaml.read_text()
                # YOLO 格式的 data.yaml 包含 names, train, val 等键
                if 'names:' in content and ('train:' in content or 'val:' in content):
                    return "yolo"
            except:
                pass

        # 检查 COCO JSON 文件
        for json_file in dataset_dir.rglob("*.json"):
            try:
                content = json_file.read_text()
                data = json.loads(content)
                # COCO 格式包含 images, annotations, categories 数组
                if 'images' in data and 'annotations' in data and 'categories' in data:
                    return "coco"
            except:
                continue

        # 检查是否有 YOLO 标签文件
        labels_dir = dataset_dir / "labels"
        if labels_dir.exists():
            label_files = list(labels_dir.rglob("*.txt"))
            if label_files:
                return "yolo"

        # 默认为原始格式
        return "raw"

    # ==================== COCO 导入 ====================

    def import_coco_dataset(
        self,
        dataset_dir: Path,
        dataset_name: str = None,
        split_images: bool = True,
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        导入 COCO 格式数据集

        Args:
            dataset_dir: 数据集目录
            dataset_name: 数据集名称
            split_images: 是否拆分训练/验证集
            train_ratio: 训练集比例

        Returns:
            导入结果
        """
        # 查找 COCO JSON 文件
        coco_json_file = None
        for json_file in dataset_dir.rglob("instances_*.json"):
            coco_json_file = json_file
            break

        if not coco_json_file:
            # 尝试查找任何包含 images/annotations/categories 的 JSON
            for json_file in dataset_dir.rglob("*.json"):
                try:
                    content = json_file.read_text()
                    data = json.loads(content)
                    if 'images' in data and 'annotations' in data and 'categories' in data:
                        coco_json_file = json_file
                        break
                except:
                    continue

        if not coco_json_file:
            return {
                "success": False,
                "message": "未找到 COCO JSON 文件"
            }

        # 加载 COCO 数据
        parser = CocoParser()
        if not parser.load_coco_json(coco_json_file):
            return {
                "success": False,
                "message": "COCO JSON 解析失败"
            }

        # 查找图像目录
        images_dir = None
        # 可能的图像目录
        possible_image_dirs = [
            dataset_dir / "images",
            dataset_dir / "train2017",
            dataset_dir / "val2017",
            dataset_dir / "train",
            dataset_dir / "val",
        ]

        for img_dir in possible_image_dirs:
            if img_dir.exists() and any(img_dir.iterdir()):
                images_dir = img_dir
                break

        if not images_dir:
            # 尝试从 JSON 中获取图像路径
            images_dir = dataset_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

        # 构建 image_id -> filename 映射
        image_id_to_filename = {}
        for img_id, img_info in parser.images.items():
            file_name = img_info.get('file_name', '')
            image_id_to_filename[img_id] = Path(file_name).name

        # 创建标签目录
        labels_dir = dataset_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 生成 YOLO 标签
        result = parser.generate_yolo_labels(
            images_dir,
            labels_dir,
            image_id_to_filename
        )

        if not result.get('success'):
            return result

        # 生成 data.yaml
        parser.generate_data_yaml(dataset_name or dataset_dir.name, dataset_dir)

        # 如果需要拆分 train/val
        if split_images:
            self._split_train_val(dataset_dir, train_ratio)

        # 处理数据集（归一化、缩略图等）
        processing_result = self.process_dataset(dataset_dir)

        return {
            "success": True,
            "message": f"COCO 数据集导入成功",
            "format": "coco",
            "task_type": result.get('task_type', 'detect'),
            "total_images": result.get('total_images', 0),
            "total_annotations": result.get('total_annotations', 0),
            "classes": result.get('classes', []),
            "processing": processing_result
        }

    def _split_train_val(self, dataset_dir: Path, train_ratio: float = 0.8):
        """拆分训练集和验证集"""
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        if not images_dir.exists():
            return

        # 获取所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            image_files.extend(list(images_dir.glob(f"**/*{ext}")))
            image_files.extend(list(images_dir.glob(f"**/*{ext.upper()}")))

        if not image_files:
            return

        # 随机打乱
        import random
        random.shuffle(image_files)

        # 计算分割点
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        # 创建子目录
        train_images_dir = images_dir / "train"
        val_images_dir = images_dir / "val"
        train_labels_dir = labels_dir / "train"
        val_labels_dir = labels_dir / "val"

        train_images_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        # 移动文件
        for img_file in train_files:
            dest = train_images_dir / img_file.name
            if not dest.exists():
                img_file.rename(dest)

            # 移动对应标签
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                dest_label = train_labels_dir / label_file.name
                if not dest_label.exists():
                    label_file.rename(dest_label)

        for img_file in val_files:
            dest = val_images_dir / img_file.name
            if not dest.exists():
                img_file.rename(dest)

            # 移动对应标签
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                dest_label = val_labels_dir / label_file.name
                if not dest_label.exists():
                    label_file.rename(dest_label)

        # 删除空的原始目录
        for subdir in [images_dir, labels_dir]:
            if subdir.exists():
                # 删除空文件
                for f in subdir.iterdir():
                    if f.is_file():
                        f.unlink()
                # 尝试删除空目录
                try:
                    subdir.rmdir()
                except:
                    pass


# 全局实例
dataset_service = DatasetService()
