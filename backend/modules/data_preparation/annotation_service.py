"""
标注服务 - Annotation Service
提供5种YOLO任务类型的标注功能
"""
import cv2
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import quote


def _public_annotation_image_url(project_name: str, file_name: str) -> str:
    """静态服务 /annotation-images 下文件的可访问 URL（对路径段编码，支持中文/空格文件名）。"""
    p = quote(str(project_name), safe="")
    f = quote(str(file_name), safe="")
    return f"/annotation-images/{p}/images/{f}"

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.core.config import settings
from backend.core.utils import save_uploaded_file, get_unique_filename


class AnnotationService:
    """标注服务"""

    # 支持的任务类型
    TASK_TYPES = {
        "detect": {
            "name": "目标检测",
            "format": "YOLO Detection",
            "label_format": "class_id x_center y_center width height (normalized)"
        },
        "segment": {
            "name": "实例分割",
            "format": "YOLO Segmentation",
            "label_format": "class_id x1 y1 x2 y2 ... (normalized polygons)"
        },
        "pose": {
            "name": "姿态估计",
            "format": "YOLO Pose",
            "label_format": "class_id x y visibility ... (keypoints)"
        },
        "obb": {
            "name": "旋转框检测",
            "format": "YOLO OBB",
            "label_format": "class_id x1 y1 x2 y2 x3 y3 x4 y4 (rotated boxes)"
        },
        "classify": {
            "name": "图像分类",
            "format": "YOLO Classification",
            "label_format": "class_id (folder-based)"
        }
    }

    def __init__(self):
        self.projects: Dict[str, Dict] = {}

    def create_project(
        self,
        name: str,
        task_type: str = "detect",
        classes: List[str] = None
    ) -> Dict[str, Any]:
        """创建标注项目"""
        if task_type not in self.TASK_TYPES:
            return {"success": False, "message": f"不支持的任务类型: {task_type}"}

        project_dir = settings.ANNOTATION_PROJECTS_DIR / name
        project_dir.mkdir(parents=True, exist_ok=True)

        project = {
            "name": name,
            "task_type": task_type,
            "classes": classes or ["object"],
            "created_at": datetime.now().isoformat(),
            "images": [],
            "annotations": {}
        }

        # 保存项目配置
        with open(project_dir / "project.json", 'w') as f:
            json.dump(project, f, indent=2, ensure_ascii=False)

        self.projects[name] = project

        return {
            "success": True,
            "message": "项目创建成功",
            "project": project
        }

    def add_images(self, project_name: str, files) -> Dict[str, Any]:
        """添加图片到项目"""
        project_dir = settings.ANNOTATION_PROJECTS_DIR / project_name
        if not project_dir.exists():
            return {"success": False, "message": "项目不存在"}

        images_dir = project_dir / "images"
        images_dir.mkdir(exist_ok=True)

        added_count = 0
        # 记录原文件名 -> 实际保存文件名的映射
        name_map = {}
        for file in files:
            if file.filename:
                unique_name = get_unique_filename(str(images_dir), file.filename)
                save_uploaded_file(file, str(images_dir / unique_name))
                name_map[file.filename] = unique_name
                added_count += 1

        # 更新项目 - 使用原文件名
        project = self._load_project(project_name)
        if project:
            # 确保 images 键存在
            if "images" not in project:
                project["images"] = []
            # 用 name_map 中的原文件名添加到列表
            for original_name in name_map.keys():
                if original_name not in project["images"]:
                    project["images"].append(original_name)
            self._save_project(project_name, project)

        return {
            "success": True,
            "message": f"添加了 {added_count} 张图片",
            "images_dir": str(images_dir),
            "name_map": name_map,
            "files": [{"original": original, "saved": saved} for original, saved in name_map.items()]
        }

    def save_annotation(
        self,
        project_name: str,
        image_name: str,
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """保存标注"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[标注] save_annotation: project={project_name}, image={image_name}, annotations_count={len(annotations)}")

        project_dir = settings.ANNOTATION_PROJECTS_DIR / project_name
        if not project_dir.exists():
            return {"success": False, "message": "项目不存在"}

        project = self._load_project(project_name)
        if not project:
            return {"success": False, "message": "项目不存在"}

        # 确保必要的键存在
        if "task_type" not in project:
            project["task_type"] = "detect"
        if "classes" not in project:
            project["classes"] = []
        if "annotations" not in project:
            project["annotations"] = {}
        if "images" not in project:
            project["images"] = []

        task_type = project["task_type"]
        labels_dir = project_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        images_dir = project_dir / "images"

        # 兜底：若任一标注没有提供 image_size，则从磁盘读图片真实尺寸塞回去，
        # 避免后端按默认 640×640 归一化（这样切换图片再回来反归一化时 bbox 会漂移）
        annotations = self._ensure_image_size_on_annotations(
            annotations, images_dir, image_name
        )

        # 转换标注为 YOLO 格式
        yolo_labels = self._convert_to_yolo_format(annotations, task_type, project["classes"])

        # 保存标签文件
        label_path = labels_dir / Path(image_name).with_suffix('.txt').name
        with open(label_path, 'w') as f:
            for label in yolo_labels:
                f.write(label + '\n')

        # 更新项目
        project["annotations"][image_name] = {
            "annotations": annotations,
            "updated_at": datetime.now().isoformat()
        }
        self._save_project(project_name, project)

        return {
            "success": True,
            "message": "标注已保存",
            "label_path": str(label_path)
        }

    def _convert_to_yolo_format(
        self,
        annotations: List[Dict[str, Any]],
        task_type: str,
        classes: List[str]
    ) -> List[str]:
        """转换标注为 YOLO 格式"""
        lines = []

        for ann in annotations:
            class_name = ann.get("class", "object")
            if class_name not in classes:
                if class_name.isdigit():
                    idx = int(class_name)
                    if idx < len(classes):
                        class_name = classes[idx]
                else:
                    classes.append(class_name)

            class_id = classes.index(class_name) if class_name in classes else 0

            if task_type == "detect":
                # 检测框: class_id x_center y_center width height (归一化)
                bbox = ann.get("bbox", [0, 0, 0, 0])
                if bbox is None or len(bbox) < 4 or any(v is None for v in bbox):
                    print(f"[WARN] 跳过无效 bbox: {bbox}")
                    continue
                x1, y1, x2, y2 = bbox
                img_w, img_h = ann.get("image_size", [640, 640])
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            elif task_type == "segment":
                # 分割: class_id x1 y1 x2 y2 ... (归一化多边形)
                points = ann.get("points", [])
                if points:
                    img_w, img_h = ann.get("image_size", [640, 640])
                    normalized = [f"{p[0]/img_w:.6f} {p[1]/img_h:.6f}" for p in points]
                    lines.append(f"{class_id} " + " ".join(normalized))

            elif task_type == "pose":
                # 关键点: class_id x y visibility ...
                bbox = ann.get("bbox", [0, 0, 0, 0])
                if bbox is None or len(bbox) < 4 or any(v is None for v in bbox):
                    print(f"[WARN] 跳过无效 bbox: {bbox}")
                    continue
                x1, y1, x2, y2 = bbox
                img_w, img_h = ann.get("image_size", [640, 640])
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                keypoints = ann.get("keypoints", [])
                kp_line = " ".join([f"{k[0]/img_w:.6f} {k[1]/img_h:.6f} {k[2] if len(k) > 2 else 1}" for k in keypoints])
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + kp_line)

            elif task_type == "obb":
                # 旋转框: class_id x1 y1 x2 y2 x3 y3 x4 y4
                points = ann.get("points", [])
                if len(points) >= 4:
                    img_w, img_h = ann.get("image_size", [640, 640])
                    normalized = " ".join([f"{p[0]/img_w:.6f} {p[1]/img_h:.6f}" for p in points[:4]])
                    lines.append(f"{class_id} " + normalized)

        return lines

    def list_projects(self) -> List[Dict[str, Any]]:
        """列出所有标注项目"""
        projects_dir = settings.ANNOTATION_PROJECTS_DIR
        if not projects_dir.exists():
            return []

        projects = []
        for project_path in projects_dir.iterdir():
            if project_path.is_dir():
                project_file = project_path / "project.json"
                if project_file.exists():
                    try:
                        with open(project_file, 'r') as f:
                            project = json.load(f)
                            # 生成唯一ID
                            project['id'] = project_path.name
                            # 计算图片数量
                            images_dir = project_path / "images"
                            if images_dir.exists():
                                project['image_count'] = len([f for f in images_dir.iterdir()
                                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                            else:
                                project['image_count'] = 0
                            projects.append(project)
                    except Exception as e:
                        print(f"Error loading project {project_path.name}: {e}")

        # 按创建时间排序
        projects.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return projects

    def delete_project(self, project_name: str) -> Dict[str, Any]:
        """删除标注项目"""
        import shutil

        project_dir = settings.ANNOTATION_PROJECTS_DIR / project_name
        if not project_dir.exists():
            return {"success": False, "message": "项目不存在"}

        try:
            # 删除项目目录
            shutil.rmtree(project_dir)
            return {"success": True, "message": "项目删除成功"}
        except Exception as e:
            return {"success": False, "message": f"删除失败: {str(e)}"}

    def get_project_images(self, project_name: str) -> List[Dict[str, str]]:
        """获取项目图片列表"""
        project = self._load_project(project_name)
        if not project:
            return []

        # 使用 project.json 中存储的原文件名
        stored_images = project.get("images", [])
        project_dir = settings.ANNOTATION_PROJECTS_DIR / project_name
        images_dir = project_dir / "images"

        if not images_dir.exists():
            return []

        # 构建磁盘文件名到 URL 的映射
        disk_to_url = {}
        disk_names_set = set()
        for f in images_dir.iterdir():
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                url = _public_annotation_image_url(project_name, f.name)
                disk_to_url[f.name] = url
                disk_names_set.add(f.name)

        # 使用存储的原文件名，查找对应的 URL
        images = []
        for img_name in stored_images:
            url = disk_to_url.get(img_name)
            if url:
                # 原文件名存在于磁盘上
                images.append({"name": img_name, "url": url, "original_url": url})
            else:
                # 原文件名不在磁盘上（可能被重命名了）
                # 尝试模糊匹配：在 disk_names 中找包含原文件名的
                matched = False
                for disk_name in disk_names_set:
                    if disk_name.startswith(img_name.rsplit('.', 1)[0]) or img_name.startswith(disk_name.rsplit('.', 1)[0]):
                        u = disk_to_url[disk_name]
                        images.append({"name": disk_name, "url": u, "original_url": u})
                        matched = True
                        break
                if not matched:
                    # 最后尝试：直接用原文件名作为 name，构造 URL
                    # (服务器应该能处理这种情况)
                    u = _public_annotation_image_url(project_name, img_name)
                    images.append({"name": img_name, "url": u, "original_url": u})

        return images

    def get_annotation(self, project_name: str, image_name: str) -> Dict[str, Any]:
        """获取图片标注"""
        project = self._load_project(project_name)
        if not project:
            return {"success": False, "message": "项目不存在"}

        project_dir = settings.ANNOTATION_PROJECTS_DIR / project_name
        labels_dir = project_dir / "labels"
        images_dir = project_dir / "images"

        # 尝试精确匹配
        label_path = labels_dir / Path(image_name).with_suffix('.txt').name
        if not label_path.exists():
            # 尝试模糊匹配：找 labels 目录中以 image_name 开头的 .txt 文件
            name_prefix = image_name.rsplit('.', 1)[0]
            for f in labels_dir.iterdir():
                if f.suffix == '.txt' and (f.stem.startswith(name_prefix) or name_prefix.startswith(f.stem)):
                    label_path = f
                    break

        print(f"[标注DEBUG] get_annotation: project={project_name}, image={image_name}, label_path={label_path}, exists={label_path.exists()}")

        if not label_path.exists():
            return {"success": True, "annotations": []}

        # 解析标签前先取一次原图尺寸，让 bbox / 多边形按真实分辨率还原（避免按固定 640 显示错位）
        img_w, img_h = self._read_image_size(images_dir, image_name, label_path)

        with open(label_path, 'r') as f:
            content = f.read()

        return {
            "success": True,
            "annotations": self._parse_yolo_labels(content, project, img_w, img_h),
            "image_size": [img_w, img_h],
        }

    def _ensure_image_size_on_annotations(
        self,
        annotations: List[Dict[str, Any]],
        images_dir: Path,
        image_name: str,
    ) -> List[Dict[str, Any]]:
        """对没有 image_size 的标注，统一补上图片真实 (W, H)；找不到原图则保留默认 [640,640]。"""
        if not annotations:
            return annotations

        # 是否所有标注已有合法 image_size
        def _ok(sz: Any) -> bool:
            return (
                isinstance(sz, (list, tuple))
                and len(sz) >= 2
                and all(isinstance(v, (int, float)) and v > 0 for v in sz[:2])
            )

        if all(_ok(a.get("image_size")) for a in annotations):
            return annotations

        # 通过 stem 模糊匹配（避免大小写后缀差异）解析图片真实尺寸
        stem = Path(image_name).stem
        candidates: List[Path] = []
        direct = images_dir / image_name
        if direct.exists():
            candidates.append(direct)
        else:
            for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                p = images_dir / f"{stem}{ext}"
                if p.exists():
                    candidates.append(p)
                    break
            if not candidates and images_dir.exists():
                for p in images_dir.iterdir():
                    if p.is_file() and p.stem == stem:
                        candidates.append(p)
                        break

        size_pair = None
        for cand in candidates:
            try:
                img = cv2.imread(str(cand))
                if img is not None:
                    h, w = img.shape[:2]
                    if w > 0 and h > 0:
                        size_pair = [int(w), int(h)]
                        break
            except Exception:
                continue

        if size_pair is None:
            return annotations

        patched: List[Dict[str, Any]] = []
        for ann in annotations:
            if _ok(ann.get("image_size")):
                patched.append(ann)
            else:
                new_ann = dict(ann)
                new_ann["image_size"] = size_pair
                patched.append(new_ann)
        return patched

    @staticmethod
    def _read_image_size(images_dir: Path, image_name: str, label_path: Path) -> tuple:
        """根据图片名（必要时通过 stem 模糊匹配）读出真实 (width, height)。失败时回退 640x640。"""
        candidates: List[Path] = []
        direct = images_dir / image_name
        if direct.exists():
            candidates.append(direct)
        else:
            stem = label_path.stem
            for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                p = images_dir / f"{stem}{ext}"
                if p.exists():
                    candidates.append(p)
                    break
            if not candidates and images_dir.exists():
                # 兜底：在 images_dir 找 stem 为前缀的文件
                for p in images_dir.iterdir():
                    if p.is_file() and p.stem.startswith(label_path.stem):
                        candidates.append(p)
                        break

        for cand in candidates:
            try:
                img = cv2.imread(str(cand))
                if img is not None:
                    h, w = img.shape[:2]
                    if w > 0 and h > 0:
                        return int(w), int(h)
            except Exception:
                continue
        return 640, 640

    def _load_project(self, project_name: str) -> Optional[Dict]:
        """加载项目"""
        project_path = settings.ANNOTATION_PROJECTS_DIR / project_name / "project.json"
        if not project_path.exists():
            return None

        with open(project_path, 'r') as f:
            return json.load(f)

    def _save_project(self, project_name: str, project: Dict):
        """保存项目"""
        project_path = settings.ANNOTATION_PROJECTS_DIR / project_name / "project.json"
        with open(project_path, 'w') as f:
            json.dump(project, f, indent=2, ensure_ascii=False)

    def _parse_yolo_labels(
        self,
        content: str,
        project: Dict,
        img_w: int = 640,
        img_h: int = 640,
    ) -> List[Dict]:
        """解析 YOLO 标签文件，使用真实图片宽高反归一化坐标。"""
        annotations: List[Dict] = []
        classes = project.get("classes", [])
        task_type = project.get("task_type", "detect")
        W = max(1, int(img_w))
        H = max(1, int(img_h))

        for line in content.strip().split('\n'):
            if not line.strip():
                continue

            parts = line.split()
            try:
                class_id = int(float(parts[0]))
            except (ValueError, IndexError):
                continue
            class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"

            if task_type == "detect":
                if len(parts) < 5:
                    continue
                x_center, y_center, width, height = map(float, parts[1:5])
                x1 = int(round((x_center - width / 2) * W))
                y1 = int(round((y_center - height / 2) * H))
                x2 = int(round((x_center + width / 2) * W))
                y2 = int(round((y_center + height / 2) * H))
                annotations.append({
                    "class": class_name,
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                })

            elif task_type in ["segment", "obb"]:
                points = []
                for i in range(1, len(parts) - 1, 2):
                    try:
                        x = float(parts[i]) * W
                        y = float(parts[i + 1]) * H
                        points.append([x, y])
                    except ValueError:
                        continue
                annotations.append({
                    "class": class_name,
                    "class_id": class_id,
                    "points": points,
                })

            elif task_type == "pose":
                if len(parts) < 5:
                    continue
                x_center, y_center, width, height = map(float, parts[1:5])
                keypoints = []
                for i in range(5, len(parts), 3):
                    try:
                        kp = [float(parts[i]) * W, float(parts[i + 1]) * H]
                    except (ValueError, IndexError):
                        continue
                    if i + 2 < len(parts):
                        try:
                            kp.append(float(parts[i + 2]))
                        except ValueError:
                            kp.append(1)
                    keypoints.append(kp)
                annotations.append({
                    "class": class_name,
                    "class_id": class_id,
                    "bbox": [
                        (x_center - width / 2) * W,
                        (y_center - height / 2) * H,
                        (x_center + width / 2) * W,
                        (y_center + height / 2) * H,
                    ],
                    "keypoints": keypoints,
                })

        return annotations

    def export_dataset(
        self,
        project_name: str,
        format: str = "yolo",
        split: str = "all"
    ) -> Dict[str, Any]:
        """导出数据集（兼容旧接口）：内部调用 generate_version 默认参数。"""
        return self.generate_version(project_name=project_name)

    # ==================== 生成数据集版本（Generate New Version） ====================

    IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def list_versions(self, project_name: str) -> List[Dict[str, Any]]:
        """列出某项目已生成的所有数据集版本（最新在前）。"""
        project = self._load_project(project_name)
        if not project:
            return []
        versions = list(project.get("versions", []) or [])
        # 校验每个版本对应的数据集目录是否仍然存在
        for v in versions:
            ds_name = v.get("dataset_name")
            if ds_name:
                v["exists"] = (settings.DATASETS_DIR / ds_name).exists()
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return versions

    def generate_version(
        self,
        project_name: str,
        dataset_name: Optional[str] = None,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        preprocessing: Optional[Dict[str, Any]] = None,
        augmentation: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """从标注项目派生一个可训练的数据集版本（Roboflow 风格）。

        步骤：
        1. 配对 images/<x> 与 labels/<x>.txt（仅纳入已标注的图片）
        2. 按 (1 - val - test, val, test) 打乱切分
        3. 复制到 DATASETS_DIR/<dataset_name>/{images,labels}/{train,val,test}
        4. 可选预处理（resize 长边到 size，YOLO 归一化标签无需改写）
        5. 可选数据增强（仅作用于 train 切分）
        6. 写入 data.yaml（path / train / val / test / nc / names）
        7. 在 project.json 的 versions 数组追加版本元数据

        Args:
            preprocessing: {"resize": True, "size": 640}
            augmentation: {"enabled": True, "horizontal_flip": True, "brightness_contrast": True, "num_augmented": 2, ...}
        """
        import shutil
        import random
        import logging

        logger = logging.getLogger(__name__)

        project = self._load_project(project_name)
        if not project:
            return {"success": False, "message": f"项目不存在: {project_name}"}

        project_dir = settings.ANNOTATION_PROJECTS_DIR / project_name
        images_dir = project_dir / "images"
        labels_dir = project_dir / "labels"

        if not images_dir.exists():
            return {"success": False, "message": "项目尚未上传任何图片"}
        if not labels_dir.exists():
            return {"success": False, "message": "项目尚未保存任何标注"}

        # 1) 配对 image/label，跳过没有标注或标注为空的
        pairs: List[tuple] = []
        skipped_unannotated: List[str] = []
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in self.IMAGE_SUFFIXES:
                continue
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists() and label_path.stat().st_size > 0:
                pairs.append((img_path, label_path))
            else:
                skipped_unannotated.append(img_path.name)

        if not pairs:
            return {
                "success": False,
                "message": "项目内尚无已标注图片，无法生成数据集版本",
            }

        # 2) 校验比例
        try:
            val_ratio = max(0.0, min(0.9, float(val_ratio or 0)))
            test_ratio = max(0.0, min(0.9, float(test_ratio or 0)))
        except (TypeError, ValueError):
            return {"success": False, "message": "val_ratio/test_ratio 需要为数值"}
        if val_ratio + test_ratio >= 1.0:
            return {"success": False, "message": "val_ratio + test_ratio 必须小于 1"}

        # 3) 决定数据集名（自动 v1, v2, ...）
        existing_versions: List[Dict[str, Any]] = list(project.get("versions", []) or [])
        next_index = len(existing_versions) + 1
        if not dataset_name:
            base_name = self._slugify_dataset_name(project_name)
            candidate = f"{base_name}_v{next_index}"
            while (settings.DATASETS_DIR / candidate).exists():
                next_index += 1
                candidate = f"{base_name}_v{next_index}"
            dataset_name = candidate
        else:
            # 用户显式传入：只清洗特殊字符，不强制 v{N}
            dataset_name = self._slugify_dataset_name(dataset_name)

        dataset_dir = settings.DATASETS_DIR / dataset_name
        if dataset_dir.exists():
            if not overwrite:
                return {
                    "success": False,
                    "message": f"数据集已存在: {dataset_name}（设置 overwrite=true 可覆盖）",
                }
            shutil.rmtree(dataset_dir, ignore_errors=True)

        # 4) 创建目录结构
        splits = ["train", "val"]
        if test_ratio > 0:
            splits.append("test")
        for split in splits:
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # 5) 切分
        rng = random.Random(int(seed) if seed is not None else 42)
        pairs_shuffled = list(pairs)
        rng.shuffle(pairs_shuffled)

        n = len(pairs_shuffled)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
        # 至少留 1 张给 train
        n_train = max(1, n - n_val - n_test)
        # 防止上面 round 之后超过总量
        n_val = min(n_val, n - n_train)
        n_test = max(0, n - n_train - n_val)

        train_pairs = pairs_shuffled[:n_train]
        val_pairs = pairs_shuffled[n_train:n_train + n_val]
        test_pairs = pairs_shuffled[n_train + n_val:n_train + n_val + n_test]

        split_pairs = {"train": train_pairs, "val": val_pairs}
        if test_ratio > 0:
            split_pairs["test"] = test_pairs

        # 6) 预处理
        resize_enabled = bool(preprocessing and preprocessing.get("resize"))
        target_size = 0
        if resize_enabled:
            try:
                target_size = int((preprocessing or {}).get("size") or 640)
                target_size = max(64, min(2048, target_size))
            except Exception:
                target_size = 640

        # 7) 写出图像与标签
        counts = {"train": 0, "val": 0, "test": 0}
        for split, items in split_pairs.items():
            img_out = dataset_dir / "images" / split
            lab_out = dataset_dir / "labels" / split
            for img_path, label_path in items:
                dst_img = img_out / img_path.name
                dst_lab = lab_out / f"{img_path.stem}.txt"

                wrote_img = False
                if resize_enabled:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            if max(h, w) != target_size:
                                scale = target_size / float(max(h, w))
                                new_w = max(1, int(round(w * scale)))
                                new_h = max(1, int(round(h * scale)))
                                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                                img = cv2.resize(img, (new_w, new_h), interpolation=interp)
                            cv2.imwrite(str(dst_img), img)
                            wrote_img = True
                    except Exception as e:
                        logger.warning(f"resize 失败，回退到直接复制 {img_path.name}: {e}")

                if not wrote_img:
                    shutil.copy2(img_path, dst_img)

                # YOLO 标签是归一化坐标，等比 / 任意 resize 都不需要改写
                shutil.copy2(label_path, dst_lab)
                counts[split] += 1

        # 8) 数据增强（仅作用于 train 切分）
        augmented_count = 0
        if augmentation and augmentation.get("enabled"):
            try:
                augmented_count = self._augment_train_split(dataset_dir, augmentation)
            except Exception as e:
                logger.warning(f"数据增强失败（已跳过）: {e}")

        # 9) 生成 data.yaml
        classes = list(project.get("classes") or ["object"])
        if not classes:
            classes = ["object"]
        yaml_lines: List[str] = [
            f"# 由「生成新版本」自动生成 - {datetime.now().isoformat()}",
            f"path: {dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
        ]
        if test_ratio > 0:
            yaml_lines.append("test: images/test")
        yaml_lines.append("")
        yaml_lines.append(f"nc: {len(classes)}")
        yaml_lines.append("names:")
        for i, cname in enumerate(classes):
            safe_name = str(cname).replace("\n", " ").replace(":", "_")
            yaml_lines.append(f"  {i}: {safe_name}")

        (dataset_dir / "data.yaml").write_text(
            "\n".join(yaml_lines) + "\n", encoding="utf-8"
        )

        # 10) 元数据写回 project.json
        version_label = f"v{next_index}"
        version_info = {
            "version": version_label,
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_dir),
            "created_at": datetime.now().isoformat(),
            "task_type": project.get("task_type", "detect"),
            "classes": classes,
            "split_counts": {
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
            },
            "ratios": {
                "train": round(1 - val_ratio - test_ratio, 4),
                "val": val_ratio,
                "test": test_ratio,
            },
            "seed": int(seed) if seed is not None else 42,
            "preprocessing": dict(preprocessing or {}),
            "augmentation": {
                **dict(augmentation or {}),
                "augmented_images": augmented_count,
            },
            "skipped_unannotated": len(skipped_unannotated),
        }
        existing_versions.append(version_info)
        project["versions"] = existing_versions
        self._save_project(project_name, project)

        return {
            "success": True,
            "message": (
                f"已生成 {version_label}: 训练 {counts['train']}, 验证 {counts['val']}, "
                f"测试 {counts['test']}（增强 {augmented_count} 张）"
            ),
            "version": version_info,
            "dataset_path": str(dataset_dir),
            "skipped_unannotated_examples": skipped_unannotated[:10],
        }

    @staticmethod
    def _slugify_dataset_name(name: str) -> str:
        """清洗数据集目录名：保留中英文/数字/_-，其余替换成 _"""
        import re
        cleaned = re.sub(r"[^\w\u4e00-\u9fa5\-]+", "_", str(name).strip())
        cleaned = cleaned.strip("_") or "dataset"
        return cleaned[:120]

    def _augment_train_split(
        self,
        dataset_dir: Path,
        augmentation_options: Dict[str, Any],
    ) -> int:
        """对 dataset_dir/images/train 应用数据增强，输出回写到同一 train 目录。

        通过创建临时输入/输出目录复用 AugmentationService（其期望 input_dir/{images,labels}）。
        """
        import shutil
        import tempfile

        from backend.modules.data_preparation.augmentation_service import (
            augmentation_service,
            AugmentationConfig,
            ALBUMENTATIONS_AVAILABLE,
        )

        if not ALBUMENTATIONS_AVAILABLE:
            raise RuntimeError("Albumentations 未安装，无法进行数据增强")

        train_img = dataset_dir / "images" / "train"
        train_lab = dataset_dir / "labels" / "train"

        cfg = AugmentationConfig(
            horizontal_flip=bool(augmentation_options.get("horizontal_flip", True)),
            vertical_flip=bool(augmentation_options.get("vertical_flip", False)),
            rotate=bool(augmentation_options.get("rotate", False)),
            scale=bool(augmentation_options.get("scale", False)),
            translate=bool(augmentation_options.get("translate", False)),
            brightness_contrast=bool(augmentation_options.get("brightness_contrast", True)),
            hue_saturation=bool(augmentation_options.get("hue_saturation", False)),
            blur=bool(augmentation_options.get("blur", False)),
            noise=bool(augmentation_options.get("noise", False)),
            cutout=bool(augmentation_options.get("cutout", False)),
            num_augmented=max(1, int(augmentation_options.get("num_augmented", 2))),
            output_format=str(augmentation_options.get("output_format", "jpg")),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "in"
            out_dir = tmp_path / "out"
            (in_dir / "images").mkdir(parents=True, exist_ok=True)
            (in_dir / "labels").mkdir(parents=True, exist_ok=True)

            for f in train_img.iterdir():
                if f.is_file():
                    shutil.copy2(f, in_dir / "images" / f.name)
            for f in train_lab.iterdir():
                if f.is_file():
                    shutil.copy2(f, in_dir / "labels" / f.name)

            result = augmentation_service.augment_dataset(in_dir, out_dir, cfg)
            if not getattr(result, "success", False):
                return 0

            moved = 0
            out_img_dir = out_dir / "images"
            out_lab_dir = out_dir / "labels"
            if out_img_dir.exists():
                for f in out_img_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in self.IMAGE_SUFFIXES:
                        shutil.copy2(f, train_img / f.name)
                        moved += 1
            if out_lab_dir.exists():
                for f in out_lab_dir.iterdir():
                    if f.is_file() and f.suffix.lower() == ".txt":
                        shutil.copy2(f, train_lab / f.name)

            return moved


# 全局实例
annotation_service = AnnotationService()
