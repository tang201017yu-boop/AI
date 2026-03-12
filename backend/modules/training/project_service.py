"""
项目管理服务 - Project Management Service
提供项目创建、编辑、删除、模型管理、活动日志等功能
"""
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid

from backend.core.config import settings


class ProjectStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ActivityType(Enum):
    PROJECT_CREATED = "project_created"
    PROJECT_UPDATED = "project_updated"
    PROJECT_DELETED = "project_deleted"
    MODEL_UPLOADED = "model_uploaded"
    MODEL_TRAINED = "model_trained"
    MODEL_EXPORTED = "model_exported"
    MODEL_DOWNLOADED = "model_downloaded"
    MODEL_MIGRATED = "model_migrated"
    SETTINGS_CHANGED = "settings_changed"


class ProjectService:
    """
    项目管理服务

    功能:
    - 项目创建、编辑、删除
    - 模型管理
    - 活动日志
    - 模型比较
    - 回收站
    """

    def __init__(self):
        self.projects_dir = settings.MODELS_DIR / "projects"
        self.recycle_bin_dir = settings.MODELS_DIR / ".recycle_bin"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.recycle_bin_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 项目 CRUD ====================

    def create_project(
        self,
        name: str,
        description: str = "",
        cover_image: str = None,
        task_type: str = "detect",
        settings: dict = None
    ) -> Dict[str, Any]:
        """
        创建新项目

        Args:
            name: 项目名称
            description: 项目描述
            cover_image: 封面图片路径
            task_type: 任务类型
            settings: 项目设置

        Returns:
            创建的项目信息
        """
        # 生成项目ID
        project_id = str(uuid.uuid4())[:8]
        project_slug = self._slugify(name)
        project_dir = self.projects_dir / f"{project_slug}_{project_id}"

        # 如果目录已存在，添加序号
        counter = 1
        while project_dir.exists():
            project_dir = self.projects_dir / f"{project_slug}_{project_id}_{counter}"
            counter += 1

        # 创建项目结构
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "models").mkdir(exist_ok=True)
        (project_dir / "exports").mkdir(exist_ok=True)
        (project_dir / "activity").mkdir(exist_ok=True)

        # 合并设置
        project_settings = self._default_settings()
        if settings:
            project_settings.update(settings)

        # 保存项目信息
        project_info = {
            "id": project_id,
            "name": name,
            "slug": project_slug,
            "description": description,
            "cover_image": cover_image,
            "task_type": task_type,
            "status": ProjectStatus.ACTIVE.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "settings": project_settings
        }

        self._save_project_json(project_dir, project_info)

        # 记录活动
        self._log_activity(project_dir, ActivityType.PROJECT_CREATED, {
            "message": f"项目 '{name}' 已创建"
        })

        return {
            "success": True,
            "project": self._format_project(project_dir, project_info)
        }

    def get_project(self, project_id: str) -> Dict[str, Any]:
        """获取项目详情"""
        project_dir, project_info = self._find_project(project_id)
        if not project_dir:
            return {"success": False, "message": "项目不存在"}

        return {
            "success": True,
            "project": self._format_project(project_dir, project_info)
        }

    def list_projects(self, include_deleted: bool = False) -> Dict[str, Any]:
        """
        列出所有项目

        Args:
            include_deleted: 是否包含已删除的项目

        Returns:
            项目列表
        """
        projects = []

        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_info = self._load_project_json(project_dir)
            if not project_info:
                continue

            # 检查状态
            status = project_info.get("status", ProjectStatus.ACTIVE.value)
            if status == ProjectStatus.DELETED.value and not include_deleted:
                continue

            # 获取模型数量
            models_count = len(list((project_dir / "models").glob("*.pt")))

            project = self._format_project(project_dir, project_info)
            project["models_count"] = models_count
            projects.append(project)

        # 按创建时间排序（最新的在前）
        projects.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {
            "success": True,
            "projects": projects,
            "total": len(projects)
        }

    def update_project(
        self,
        project_id: str,
        name: str = None,
        description: str = None,
        cover_image: str = None,
        settings: Dict = None
    ) -> Dict[str, Any]:
        """
        更新项目信息

        Args:
            project_id: 项目ID
            name: 新名称
            description: 新描述
            cover_image: 新封面
            settings: 新设置

        Returns:
            更新结果
        """
        project_dir, project_info = self._find_project(project_id)
        if not project_dir:
            return {"success": False, "message": "项目不存在"}

        changes = []

        if name and name != project_info["name"]:
            old_name = project_info["name"]
            project_info["name"] = name
            project_info["slug"] = self._slugify(name)
            changes.append(f"名称: '{old_name}' → '{name}'")

        if description is not None:
            project_info["description"] = description
            changes.append("描述已更新")

        if cover_image is not None:
            project_info["cover_image"] = cover_image
            changes.append("封面图片已更新")

        if settings:
            project_info["settings"].update(settings)
            changes.append("设置已更新")

        project_info["updated_at"] = datetime.now().isoformat()
        self._save_project_json(project_dir, project_info)

        # 记录活动
        if changes:
            self._log_activity(project_dir, ActivityType.PROJECT_UPDATED, {
                "message": "项目已更新",
                "changes": changes
            })

        return {
            "success": True,
            "project": self._format_project(project_dir, project_info)
        }

    def delete_project(self, project_id: str, permanent: bool = False) -> Dict[str, Any]:
        """
        删除项目

        Args:
            project_id: 项目ID
            permanent: 是否永久删除（不进入回收站）

        Returns:
            删除结果
        """
        project_dir, project_info = self._find_project(project_id)
        if not project_dir:
            return {"success": False, "message": "项目不存在"}

        if permanent:
            # 永久删除
            shutil.rmtree(project_dir)
            return {
                "success": True,
                "message": "项目已永久删除"
            }
        else:
            # 移动到回收站
            deleted_at = datetime.now().isoformat()
            expires_at = datetime.now().timestamp() + 30 * 24 * 60 * 60  # 30天后过期

            # 更新状态
            project_info["status"] = ProjectStatus.DELETED.value
            project_info["deleted_at"] = deleted_at
            project_info["expires_at"] = datetime.fromtimestamp(expires_at).isoformat()
            self._save_project_json(project_dir, project_info)

            # 移动到回收站
            target_dir = self.recycle_bin_dir / project_dir.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(project_dir), str(target_dir))

            # 记录活动
            self._log_activity(target_dir, ActivityType.PROJECT_DELETED, {
                "message": f"项目 '{project_info['name']}' 已移至回收站",
                "expires_at": project_info["expires_at"]
            })

            return {
                "success": True,
                "message": "项目已移至回收站，30天内可恢复"
            }

    def restore_project(self, project_id: str) -> Dict[str, Any]:
        """从回收站恢复项目"""
        # 在回收站中查找
        for item in self.recycle_bin_dir.iterdir():
            if not item.is_dir():
                continue

            project_info = self._load_project_json(item)
            if not project_info:
                continue

            if project_info.get("id") == project_id:
                # 恢复项目
                project_info["status"] = ProjectStatus.ACTIVE.value
                project_info["deleted_at"] = None
                project_info["expires_at"] = None
                project_info["updated_at"] = datetime.now().isoformat()
                self._save_project_json(item, project_info)

                # 移回项目目录
                target_dir = self.projects_dir / item.name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(item), str(target_dir))

                # 记录活动
                self._log_activity(target_dir, ActivityType.PROJECT_UPDATED, {
                    "message": f"项目 '{project_info['name']}' 已从回收站恢复"
                })

                return {
                    "success": True,
                    "project": self._format_project(target_dir, project_info)
                }

        return {"success": False, "message": "项目不在回收站中或已过期"}

    def get_recycle_bin(self) -> Dict[str, Any]:
        """获取回收站中的项目"""
        expired = []
        valid = []

        for item in self.recycle_bin_dir.iterdir():
            if not item.is_dir():
                continue

            project_info = self._load_project_json(item)
            if not project_info:
                continue

            expires_at = project_info.get("expires_at")
            is_expired = False

            if expires_at:
                try:
                    exp_date = datetime.fromisoformat(expires_at)
                    if datetime.now() > exp_date:
                        is_expired = True
                except:
                    pass

            project = self._format_project(item, project_info)

            if is_expired:
                expired.append(project)
            else:
                valid.append(project)

        return {
            "success": True,
            "items": valid,
            "expired": expired,
            "total": len(valid) + len(expired)
        }

    def empty_recycle_bin(self) -> Dict[str, Any]:
        """清空回收站"""
        count = 0

        for item in self.recycle_bin_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                count += 1

        return {
            "success": True,
            "message": f"已永久删除 {count} 个项目"
        }

    # ==================== 模型管理 ====================

    def add_model(
        self,
        project_id: str,
        model_path: str,
        model_type: str = "yolo",
        metrics: Dict = None
    ) -> Dict[str, Any]:
        """
        添加模型到项目

        Args:
            project_id: 项目ID
            model_path: 模型文件路径
            model_type: 模型类型
            metrics: 模型指标

        Returns:
            添加结果
        """
        project_dir, project_info = self._find_project(project_id)
        if not project_dir:
            return {"success": False, "message": "项目不存在"}

        model_name = Path(model_path).name
        target_path = project_dir / "models" / model_name

        # 复制模型文件
        try:
            if Path(model_path).exists():
                shutil.copy2(model_path, target_path)
        except Exception as e:
            return {"success": False, "message": f"复制模型失败: {str(e)}"}

        # 创建模型记录
        model_record = {
            "id": str(uuid.uuid4())[:8],
            "name": model_name,
            "path": str(target_path),
            "type": model_type,
            "metrics": metrics or {},
            "uploaded_at": datetime.now().isoformat()
        }

        # 保存模型记录
        models_index = project_dir / "models" / "index.json"
        models = self._load_json(models_index) if models_index.exists() else []
        models.append(model_record)
        self._save_json(models_index, models)

        # 记录活动
        self._log_activity(project_dir, ActivityType.MODEL_UPLOADED, {
            "message": f"模型 '{model_name}' 已上传",
            "model_id": model_record["id"],
            "metrics": metrics
        })

        return {
            "success": True,
            "model": model_record
        }

    def get_models(self, project_id: str) -> Dict[str, Any]:
        """获取项目的所有模型"""
        project_dir, project_info = self._find_project(project_id)
        if not project_dir:
            return {"success": False, "message": "项目不存在"}

        models_index = project_dir / "models" / "index.json"
        models = self._load_json(models_index) if models_index.exists() else []

        return {
            "success": True,
            "models": models,
            "total": len(models)
        }

    def migrate_model(
        self,
        project_id: str,
        model_id: str,
        target_project_id: str
    ) -> Dict[str, Any]:
        """
        迁移模型到另一个项目

        Args:
            project_id: 源项目ID
            model_id: 模型ID
            target_project_id: 目标项目ID

        Returns:
            迁移结果
        """
        source_dir, source_info = self._find_project(project_id)
        if not source_dir:
            return {"success": False, "message": "源项目不存在"}

        target_dir, target_info = self._find_project(target_project_id)
        if not target_dir:
            return {"success": False, "message": "目标项目不存在"}

        # 查找模型
        models_index = source_dir / "models" / "index.json"
        models = self._load_json(models_index) if models_index.exists() else []

        model = None
        for m in models:
            if m["id"] == model_id:
                model = m
                break

        if not model:
            return {"success": False, "message": "模型不存在"}

        # 复制模型文件
        source_model_path = Path(model["path"])
        target_model_path = target_dir / "models" / source_model_path.name

        try:
            shutil.copy2(source_model_path, target_model_path)
        except Exception as e:
            return {"success": False, "message": f"复制模型失败: {str(e)}"}

        # 添加到目标项目
        new_model = dict(model)
        new_model["id"] = str(uuid.uuid4())[:8]
        new_model["path"] = str(target_model_path)
        new_model["migrated_at"] = datetime.now().isoformat()
        new_model["source_project"] = project_id

        target_models_index = target_dir / "models" / "index.json"
        target_models = self._load_json(target_models_index) if target_models_index.exists() else []
        target_models.append(new_model)
        self._save_json(target_models_index, target_models)

        # 从源项目移除
        models = [m for m in models if m["id"] != model_id]
        self._save_json(models_index, models)

        # 删除源模型文件
        if source_model_path.exists():
            source_model_path.unlink()

        # 记录活动
        self._log_activity(target_dir, ActivityType.MODEL_MIGRATED, {
            "message": f"模型 '{model['name']}' 已从项目 '{source_info['name']}' 迁移到此项目",
            "model_id": new_model["id"],
            "source_project": source_info["name"]
        })

        return {
            "success": True,
            "model": new_model
        }

    def delete_model(self, project_id: str, model_id: str) -> Dict[str, Any]:
        """删除模型"""
        project_dir, project_info = self._find_project(project_id)
        if not project_dir:
            return {"success": False, "message": "项目不存在"}

        models_index = project_dir / "models" / "index.json"
        models = self._load_json(models_index) if models_index.exists() else []

        model = None
        for m in models:
            if m["id"] == model_id:
                model = m
                break

        if not model:
            return {"success": False, "message": "模型不存在"}

        # 删除模型文件
        model_path = Path(model["path"])
        if model_path.exists():
            model_path.unlink()

        # 移除记录
        models = [m for m in models if m["id"] != model_id]
        self._save_json(models_index, models)

        return {
            "success": True,
            "message": "模型已删除"
        }

    # ==================== 活动日志 ====================

    def get_activity_log(
        self,
        project_id: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        获取项目活动日志

        Args:
            project_id: 项目ID
            limit: 返回条数限制

        Returns:
            活动日志列表
        """
        project_dir, project_info = self._find_project(project_id)
        if not project_dir:
            return {"success": False, "message": "项目不存在"}

        activity_dir = project_dir / "activity"
        activities = []

        for log_file in sorted(activity_dir.glob("*.json"), reverse=True)[:limit]:
            activity = self._load_json(log_file)
            if activity:
                activities.append(activity)

        return {
            "success": True,
            "activities": activities,
            "total": len(activities)
        }

    # ==================== 模型比较 ====================

    def compare_models(
        self,
        project_id: str,
        model_ids: List[str] = None
    ) -> Dict[str, Any]:
        """
        比较模型性能

        Args:
            project_id: 项目ID
            model_ids: 要比较的模型ID列表（None表示所有模型）

        Returns:
            比较数据（用于图表）
        """
        project_dir, project_info = self._find_project(project_id)
        if not project_dir:
            return {"success": False, "message": "项目不存在"}

        models_index = project_dir / "models" / "index.json"
        models = self._load_json(models_index) if models_index.exists() else []

        if model_ids:
            models = [m for m in models if m["id"] in model_ids]

        # 构建比较数据
        comparison = {
            "models": [],
            "loss_data": {
                "labels": [],
                "datasets": []
            },
            "metrics_data": {
                "labels": [],
                "datasets": []
            }
        }

        # 添加每个模型的最新指标
        for model in models:
            metrics = model.get("metrics", {})
            comparison["models"].append({
                "id": model["id"],
                "name": model["name"],
                "metrics": metrics
            })

        # 收集所有唯一的epoch/批次
        epochs = set()
        for model in models:
            for key in metrics.keys():
                if "epoch" in key.lower():
                    epoch = key.split("/")[0].replace("epoch", "")
                    try:
                        epochs.add(int(epoch))
                    except:
                        pass

        comparison["loss_data"]["labels"] = sorted(epochs)
        comparison["metrics_data"]["labels"] = sorted(epochs)

        # 为每个模型创建数据集
        colors = [
            "rgba(255, 99, 132, 1)",
            "rgba(54, 162, 235, 1)",
            "rgba(255, 206, 86, 1)",
            "rgba(75, 192, 192, 1)",
            "rgba(153, 102, 255, 1)"
        ]

        for idx, model in enumerate(models):
            metrics = model.get("metrics", {})
            color = colors[idx % len(colors)]

            # 损失数据
            loss_data = []
            for epoch in comparison["loss_data"]["labels"]:
                key = f"epoch_{epoch}/box_loss"
                if key in metrics:
                    loss_data.append(metrics[key])
                else:
                    loss_data.append(None)

            comparison["loss_data"]["datasets"].append({
                "label": model["name"],
                "data": loss_data,
                "borderColor": color,
                "backgroundColor": color.replace("1)", "0.1)"),
                "fill": False,
                "tension": 0.4
            })

            # 指标数据 (mAP50)
            map_data = []
            for epoch in comparison["metrics_data"]["labels"]:
                key = f"epoch_{epoch}/metrics/mAP50(B)"
                if key in metrics:
                    map_data.append(metrics[key])
                else:
                    map_data.append(None)

            comparison["metrics_data"]["datasets"].append({
                "label": model["name"],
                "data": map_data,
                "borderColor": color,
                "backgroundColor": color.replace("1)", "0.1)"),
                "fill": False,
                "tension": 0.4
            })

        return {
            "success": True,
            "comparison": comparison
        }

    # ==================== 辅助方法 ====================

    def _find_project(self, project_id: str):
        """查找项目"""
        # 先在活动项目目录查找
        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_info = self._load_project_json(project_dir)
            if project_info and project_info.get("id") == project_id:
                return project_dir, project_info

        # 在回收站查找
        for item in self.recycle_bin_dir.iterdir():
            if not item.is_dir():
                continue

            project_info = self._load_project_json(item)
            if project_info and project_info.get("id") == project_id:
                return item, project_info

        return None, None

    def _slugify(self, text: str) -> str:
        """生成URL友好的slug"""
        import re
        slug = re.sub(r'[^\w\s-]', '', text).strip().lower()
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug

    def _default_settings(self) -> Dict:
        """默认项目设置"""
        return {
            "default_model_type": "yolo11n",
            "default_epochs": 100,
            "default_batch_size": 32,
            "default_img_size": 640,
            "auto_save_checkpoints": True,
            "checkpoint_interval": 10,
            "notification_enabled": True
        }

    def _format_project(self, project_dir: Path, project_info: Dict) -> Dict:
        """格式化项目信息"""
        return {
            "id": project_info.get("id"),
            "name": project_info.get("name"),
            "description": project_info.get("description", ""),
            "cover_image": project_info.get("cover_image"),
            "status": project_info.get("status"),
            "created_at": project_info.get("created_at"),
            "updated_at": project_info.get("updated_at"),
            "settings": project_info.get("settings", {})
        }

    def _load_project_json(self, project_dir: Path) -> Dict:
        """加载项目JSON"""
        json_path = project_dir / "project.json"
        if not json_path.exists():
            return None
        return self._load_json(json_path)

    def _save_project_json(self, project_dir: Path, data: Dict):
        """保存项目JSON"""
        json_path = project_dir / "project.json"
        self._save_json(json_path, data)

    def _load_json(self, path: Path) -> Any:
        """加载JSON文件"""
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, path: Path, data: Any):
        """保存JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _log_activity(self, project_dir: Path, activity_type: ActivityType, details: Dict):
        """记录活动"""
        activity_dir = project_dir / "activity"
        activity_dir.mkdir(parents=True, exist_ok=True)

        activity = {
            "id": str(uuid.uuid4())[:8],
            "type": activity_type.value,
            "timestamp": datetime.now().isoformat(),
            **details
        }

        filename = f"{activity_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        self._save_json(activity_dir / filename, activity)


# 全局实例
project_service = ProjectService()
