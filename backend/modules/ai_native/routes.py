"""
AI原生应用路由 - 自然语言生成检测系统
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

# 导入服务
from backend.modules.ai_native.ai_native_service import ai_native_service


class CreateProjectRequest(BaseModel):
    """创建项目请求"""
    requirement: str
    project_name: Optional[str] = None


@router.get("/ai-native/understand")
async def understand_requirement(text: str):
    """
    理解自然语言需求
    示例：我要检测隧洞裂缝，超过5mm且有渗水的要报警
    """
    try:
        result = ai_native_service.understand_requirement(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai-native/create")
async def create_detection_system(request: CreateProjectRequest):
    """
    一键创建检测系统
    工程师描述需求，AI自动生成完整方案
    """
    try:
        result = ai_native_service.create_detection_system(
            requirement=request.requirement,
            project_name=request.project_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai-native/projects")
async def list_ai_projects():
    """列出所有AI原生项目"""
    try:
        projects = ai_native_service.list_projects()
        return {"total": len(projects), "projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai-native/project/{project_id}")
async def get_ai_project(project_id: str):
    """获取AI项目详情"""
    try:
        project = ai_native_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai-native/scenes")
async def list_scenes():
    """列出支持的检测场景"""
    from backend.modules.ai_native.ai_native_service import SCENE_KNOWLEDGE

    scenes = []
    for name, info in SCENE_KNOWLEDGE.items():
        scenes.append({
            "name": name,
            "description": info["description"],
            "keywords": info["keywords"][:5],
            "default_threshold": info["default_threshold"]
        })

    return {
        "total": len(scenes),
        "scenes": scenes
    }
