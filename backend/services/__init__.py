# Services package - Re-export from modules for backward compatibility
from backend.modules.data_preparation.dataset_service import dataset_service

# 导入 annotation service 并添加兼容方法
from backend.modules.data_preparation.annotation_service import AnnotationService as _AnnotationService
from backend.modules.solutions.solutions_service import solutions_service as _solutions_service
from backend.services.arbitration_service import arbitration_service
from backend.services.supervision_service import supervision_service
from backend.services.yolo_service import yolo_service

# 创建 annotation_service 实例
_annotation_service_instance = _AnnotationService()

# 添加兼容方法 (从原 backend/services/annotation_service.py)
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class AnnotationServiceCompat:
    """Annotation Service 兼容层 - 添加原 services/annotation_service.py 的方法"""

    def __init__(self):
        from backend.core.config import settings
        self._inner = _AnnotationService()
        self.projects_dir = Path(settings.DATA_DIR) / "annotation_projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.projects_file = self.projects_dir / "projects.json"
        if not self.projects_file.exists():
            self._save_projects_index([])

    def _load_projects_index(self) -> List[Dict[str, Any]]:
        try:
            with open(self.projects_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def _save_projects_index(self, projects: List[Dict[str, Any]]):
        try:
            with open(self.projects_file, 'w', encoding='utf-8') as f:
                json.dump(projects, f, ensure_ascii=False, indent=2)
        except:
            pass

    def list_projects(self) -> List[Dict[str, Any]]:
        """列出所有项目"""
        return self._load_projects_index()

    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """创建新项目"""
        project_id = str(uuid.uuid4())
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "images").mkdir(exist_ok=True)
        (project_dir / "annotations").mkdir(exist_ok=True)

        project = {
            "id": project_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "path": str(project_dir)
        }

        project_config_file = project_dir / "project.json"
        with open(project_config_file, 'w', encoding='utf-8') as f:
            json.dump(project, f, ensure_ascii=False, indent=2)

        projects = self._load_projects_index()
        projects.append(project)
        self._save_projects_index(projects)
        return project

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """获取项目详情"""
        project_dir = self.projects_dir / project_id
        project_config_file = project_dir / "project.json"
        if not project_config_file.exists():
            return None
        try:
            with open(project_config_file, 'r', encoding='utf-8') as f:
                project = json.load(f)
            images_dir = project_dir / "images"
            images = []
            for img_path in images_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    images.append({
                        "name": img_path.name,
                        "url": f"/api/v1/annotation/image/{project_id}/{img_path.name}",
                        "path": str(img_path)
                    })
            annotations_file = project_dir / "annotations" / "annotations.json"
            annotations = {}
            if annotations_file.exists():
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
            classes_file = project_dir / "annotations" / "classes.json"
            classes = ['person', 'car', 'dog', 'cat']
            if classes_file.exists():
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes = json.load(f)
            project['images'] = images
            project['annotations'] = annotations
            project['classes'] = classes
            return project
        except:
            return None

    def upload_images(self, project_id: str, files: List[Any]) -> Dict[str, Any]:
        """上传图片"""
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            return {"success": False, "message": "Project not found"}
        images_dir = project_dir / "images"
        uploaded = 0
        try:
            for file in files:
                file_path = images_dir / file.filename
                with open(file_path, 'wb') as f:
                    f.write(file.file.read())
                uploaded += 1
            return {"success": True, "uploaded": uploaded, "message": f"Uploaded {uploaded} images"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def save_annotations(self, project_id: str, annotations: Dict[str, List[Dict]], classes: List[str]) -> Dict[str, Any]:
        """保存标注"""
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            return {"success": False, "message": "Project not found"}
        try:
            annotations_dir = project_dir / "annotations"
            annotations_dir.mkdir(exist_ok=True)
            annotations_file = annotations_dir / "annotations.json"
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
            classes_file = annotations_dir / "classes.json"
            with open(classes_file, 'w', encoding='utf-8') as f:
                json.dump(classes, f, ensure_ascii=False, indent=2)
            return {"success": True, "message": "Annotations saved"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def export_to_yolo(self, project_id: str):
        """导出为 YOLO 格式"""
        return None  # 需要实现

    def get_image_path(self, project_id: str, image_name: str) -> Optional[Path]:
        """获取图片路径"""
        project_dir = self.projects_dir / project_id
        image_path = project_dir / "images" / image_name
        if image_path.exists():
            return image_path
        return None

    def delete_project(self, project_id: str) -> Dict[str, Any]:
        """删除项目"""
        import shutil
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            return {"success": False, "message": "Project not found"}
        try:
            shutil.rmtree(project_dir)
            projects = self._load_projects_index()
            projects = [p for p in projects if p['id'] != project_id]
            self._save_projects_index(projects)
            return {"success": True, "message": "Project deleted"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def auto_annotate_with_model(self, project_id: str, model_path: str, confidence: float = 0.25, iou_threshold: float = 0.45, filter_classes: Optional[List[str]] = None, merge_mode: str = "replace") -> Dict[str, Any]:
        """自动标注"""
        return {"success": False, "message": "Not implemented"}

    def batch_auto_annotate(self, project_id: str, image_names: List[str], model_path: str, confidence: float = 0.25, iou_threshold: float = 0.45) -> Dict[str, Any]:
        """批量自动标注"""
        return {"success": False, "message": "Not implemented"}

    def get_annotation_statistics(self, project_id: str) -> Dict[str, Any]:
        """获取标注统计"""
        return {"success": False, "message": "Not implemented"}

    def visualize_annotations(self, project_id: str, image_name: str, output_dir: Optional[str] = None):
        """可视化标注"""
        return None

annotation_service = AnnotationServiceCompat()
solutions_service = _solutions_service

__all__ = [
    'dataset_service',
    'annotation_service',
    'arbitration_service',
    'solutions_service',
    'supervision_service',
    'yolo_service'
]
