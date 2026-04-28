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

        with open(label_path, 'r') as f:
            content = f.read()

        return {
            "success": True,
            "annotations": self._parse_yolo_labels(content, project)
        }

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

    def _parse_yolo_labels(self, content: str, project: Dict) -> List[Dict]:
        """解析 YOLO 标签文件"""
        annotations = []
        classes = project.get("classes", [])
        task_type = project.get("task_type", "detect")

        for line in content.strip().split('\n'):
            if not line.strip():
                continue

            parts = line.split()
            class_id = int(parts[0])
            class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"

            if task_type == "detect":
                x_center, y_center, width, height = map(float, parts[1:5])
                x1 = int((x_center - width/2) * 640)
                y1 = int((y_center - height/2) * 640)
                x2 = int((x_center + width/2) * 640)
                y2 = int((y_center + height/2) * 640)
                annotations.append({
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2]
                })

            elif task_type in ["segment", "obb"]:
                points = []
                for i in range(1, len(parts), 2):
                    x = float(parts[i]) * 640
                    y = float(parts[i+1]) * 640
                    points.append([x, y])
                annotations.append({
                    "class": class_name,
                    "points": points
                })

            elif task_type == "pose":
                x_center, y_center, width, height = map(float, parts[1:5])
                keypoints = []
                for i in range(5, len(parts), 3):
                    kp = [float(parts[i]) * 640, float(parts[i+1]) * 640]
                    if i+2 < len(parts):
                        kp.append(float(parts[i+2]))
                    keypoints.append(kp)
                annotations.append({
                    "class": class_name,
                    "bbox": [
                        (x_center - width/2) * 640,
                        (y_center - height/2) * 640,
                        (x_center + width/2) * 640,
                        (y_center + height/2) * 640
                    ],
                    "keypoints": keypoints
                })

        return annotations

    def export_dataset(
        self,
        project_name: str,
        format: str = "yolo",
        split: str = "all"
    ) -> Dict[str, Any]:
        """导出数据集"""
        project = self._load_project(project_name)
        if not project:
            return {"success": False, "message": "项目不存在"}

        # TODO: 实现数据集导出功能
        return {
            "success": True,
            "message": "数据集导出功能开发中",
            "project": project_name
        }


# 全局实例
annotation_service = AnnotationService()
