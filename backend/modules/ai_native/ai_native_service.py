"""
AI原生应用服务 - 让普通人也能使用AI
水利工程师自然语言需求 -> 自动生成检测系统
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid

# 内置知识库：场景 -> 模型映射
SCENE_KNOWLEDGE = {
    # 水利场景
    "裂缝": {
        "models": ["crack", "fissure", "defect"],
        "keywords": ["裂缝", "裂纹", "裂痕", "破损", "crack", "fissure"],
        "default_threshold": 0.5,
        "description": "混凝土结构裂缝检测"
    },
    "渗水": {
        "models": ["water", "leak", "seepage", "wet"],
        "keywords": ["渗水", "漏水", "湿润", "水渍", "渗漏", "leak", "seepage", "wet"],
        "default_threshold": 0.6,
        "description": "渗水和水渍检测"
    },
    "沉降": {
        "models": ["subsidence", "deformation", "settlement"],
        "keywords": ["沉降", "下沉", "变形", "位移", "subsidence", "settlement"],
        "default_threshold": 0.5,
        "description": "结构沉降检测"
    },
    "滑坡": {
        "models": ["landslide", "collapse", "rockfall"],
        "keywords": ["滑坡", "塌方", "落石", "landslide", "collapse"],
        "default_threshold": 0.6,
        "description": "边坡滑坡检测"
    },
    "积水": {
        "models": ["water", "pooling", "flood"],
        "keywords": ["积水", "水淹", "内涝", "water", "pooling", "flood"],
        "default_threshold": 0.5,
        "description": "积水检测"
    },
    "异物": {
        "models": ["object", "intrusion", "foreign"],
        "keywords": ["异物", "入侵", "遗留", "object", "intrusion", "foreign"],
        "default_threshold": 0.5,
        "description": "异常物体检测"
    },
    "人员": {
        "models": ["person", "worker", "helmet", "vest"],
        "keywords": ["人", "人员", "工人", "person", "worker"],
        "default_threshold": 0.5,
        "description": "人员检测"
    },
    "安全帽": {
        "models": ["helmet", "hardhat", "safety"],
        "keywords": ["安全帽", "头盔", "helmet", "hardhat"],
        "default_threshold": 0.6,
        "description": "安全帽佩戴检测"
    },
    "防护服": {
        "models": ["vest", "clothes", "ppe"],
        "keywords": ["防护服", "工作服", "vest", "ppe"],
        "default_threshold": 0.6,
        "description": "防护装备检测"
    },
    # 通用场景
    "烟火": {
        "models": ["fire", "smoke", "flame"],
        "keywords": ["火", "烟火", "火灾", "fire", "smoke", "flame"],
        "default_threshold": 0.7,
        "description": "火焰和烟雾检测"
    },
    "拥堵": {
        "models": ["crowd", "queue", "congestion"],
        "keywords": ["拥堵", "拥挤", "排队", "crowd", "congestion"],
        "default_threshold": 0.5,
        "description": "人群拥堵检测"
    }
}

# 阈值提取模式
THRESHOLD_PATTERNS = [
    r"超过?(\d+(?:\.\d+)?)\s*(?:mm|厘米|cm|像素|px|%)",
    r"大于?(\d+(?:\.\d+)?)\s*(?:mm|厘米|cm|像素|px|%)",
    r"高于?(\d+(?:\.\d+)?)\s*(?:mm|厘米|cm|像素|px|%)",
    r"达到?(\d+(?:\.\d+)?)\s*(?:mm|厘米|cm|像素|px|%)",
    r"阈值\s*[为:]?\s*(\d+(?:\.\d+)?)",
    r"置信度?\s*[为:]?\s*(\d+(?:\.\d+)?)\s*%?",
    r"(\d+(?:\.\d+)?)\s*%",
]

# 报警关键词
ALARM_KEYWORDS = ["报警", "告警", "警报", "提醒", "通知", "alarm", "alert", "warn"]


class AINativeService:
    """AI原生服务 - 自然语言生成检测系统"""

    def __init__(self):
        self.projects_dir = Path("data/ai_projects")
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.projects_index = self.projects_dir / "projects.json"
        if not self.projects_index.exists():
            self._save_index([])

    def _load_index(self) -> List[Dict]:
        try:
            with open(self.projects_index, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def _save_index(self, projects: List[Dict]):
        with open(self.projects_index, 'w', encoding='utf-8') as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)

    def understand_requirement(self, text: str) -> Dict[str, Any]:
        """
        理解自然语言需求
        核心：把"人话"翻译成"技术参数"
        """
        result = {
            "success": True,
            "scenes": [],           # 检测场景
            "threshold": 0.5,       # 阈值
            "threshold_unit": None,  # 阈值单位
            "alarm": False,          # 是否需要报警
            "alarm_type": None,      # 报警类型
            "models": [],           # 需要的模型
            "config": {},           # 生成的配置
            "explanation": "",       # AI解释
            "raw_text": text
        }

        text_lower = text.lower()

        # 1. 场景识别
        scenes = self._extract_scenes(text)
        result["scenes"] = scenes

        # 2. 阈值提取
        threshold, unit = self._extract_threshold(text)
        if threshold is not None:
            result["threshold"] = threshold
            result["threshold_unit"] = unit

        # 3. 报警需求识别
        alarm_info = self._extract_alarm(text)
        result["alarm"] = alarm_info["need_alarm"]
        result["alarm_type"] = alarm_info["type"]

        # 4. 生成模型列表
        models = []
        for scene in scenes:
            if scene in SCENE_KNOWLEDGE:
                models.extend(SCENE_KNOWLEDGE[scene]["models"])
        result["models"] = list(set(models)) if models else ["object"]

        # 5. 生成配置
        result["config"] = self._generate_config(result)

        # 6. 生成解释
        result["explanation"] = self._generate_explanation(result)

        return result

    def _extract_scenes(self, text: str) -> List[str]:
        """提取检测场景"""
        found_scenes = []
        text_lower = text.lower()

        for scene, info in SCENE_KNOWLEDGE.items():
            # 检查关键词
            for keyword in info["keywords"]:
                if keyword.lower() in text_lower:
                    if scene not in found_scenes:
                        found_scenes.append(scene)
                    break

        # 智能推断：如果提到"检测"但没有具体场景，默认通用检测
        if not found_scenes and "检测" in text:
            found_scenes.append("异物")

        return found_scenes

    def _extract_threshold(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """提取阈值"""
        for pattern in THRESHOLD_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # 判断单位
                if "mm" in text.lower():
                    return value, "mm"
                elif "厘米" in text.lower() or "cm" in text.lower():
                    return value, "cm"
                elif "像素" in text.lower() or "px" in text.lower():
                    return value, "px"
                elif "%" in text:
                    return value / 100, "percent"
                else:
                    # 默认置信度
                    if value <= 1:
                        return value, "confidence"
                    else:
                        return value / 100, "confidence"

        return None, None

    def _extract_alarm(self, text: str) -> Dict[str, Any]:
        """提取报警需求"""
        text_lower = text.lower()
        need_alarm = any(kw in text_lower for kw in ALARM_KEYWORDS)

        alarm_type = "popup"
        if "声音" in text or "语音" in text or "sound" in text_lower:
            alarm_type = "sound"
        elif "短信" in text or "sms" in text_lower:
            alarm_type = "sms"
        elif "邮件" in text or "email" in text_lower:
            alarm_type = "email"
        elif "webhook" in text_lower:
            alarm_type = "webhook"

        return {
            "need_alarm": need_alarm,
            "type": alarm_type
        }

    def _generate_config(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """生成检测配置"""
        scenes = parsed.get("scenes", [])
        threshold = parsed.get("threshold", 0.5)
        alarm = parsed.get("alarm", False)
        alarm_type = parsed.get("alarm_type", "popup")
        models = parsed.get("models", ["object"])

        config = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "detection": {
                "scenes": scenes,
                "models": models,
                "threshold": threshold,
                "iou_threshold": 0.45,
            },
            "alarm": {
                "enabled": alarm,
                "type": alarm_type,
                "conditions": []
            },
            "post_processing": {
                "min_area": None,
                "max_area": None,
                "tracking": False
            }
        }

        # 为每个场景添加报警条件
        for scene in scenes:
            if scene in SCENE_KNOWLEDGE:
                condition = {
                    "scene": scene,
                    "threshold": threshold,
                    "action": alarm_type if alarm else "none"
                }
                config["alarm"]["conditions"].append(condition)

        return config

    def _generate_explanation(self, parsed: Dict[str, Any]) -> str:
        """生成人类可读的解释"""
        scenes = parsed.get("scenes", [])
        threshold = parsed.get("threshold", 0.5)
        threshold_unit = parsed.get("threshold_unit")
        alarm = parsed.get("alarm", False)
        models = parsed.get("models", [])

        explanation = "我已经理解您的需求：\n\n"

        if scenes:
            explanation += f"📍 检测场景：{', '.join(scenes)}\n"

        explanation += f"🎯 检测模型：{', '.join(models) if models else '通用目标检测'}\n"

        if threshold_unit:
            if threshold_unit == "confidence":
                explanation += f"⚡ 置信度阈值：{threshold:.0%}\n"
            else:
                explanation += f"⚡ 阈值：{threshold}{threshold_unit}\n"
        else:
            explanation += f"⚡ 置信度阈值：{threshold:.0%}（默认值）\n"

        if alarm:
            explanation += f"🔔 报警：已启用（{parsed.get('alarm_type')}）\n"
        else:
            explanation += "🔔 报警：未启用\n"

        explanation += "\n💡 系统将自动完成以下工作：\n"
        explanation += "   1. 选择合适的AI模型\n"
        explanation += "   2. 配置检测参数\n"
        if alarm:
            explanation += "   3. 设置报警规则\n"
        explanation += "   4. 生成可部署的检测系统\n"

        return explanation

    def create_detection_system(self, requirement: str, project_name: str = None) -> Dict[str, Any]:
        """
        一键创建检测系统
        工程师只需要描述需求，系统自动生成完整方案
        """
        if not project_name:
            project_name = f"project_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        project_id = str(uuid.uuid4())
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # 1. 理解需求
        understanding = self.understand_requirement(requirement)

        if not understanding["scenes"]:
            return {
                "success": False,
                "message": "无法理解您的需求，请尝试描述具体的检测场景，如'裂缝'、'渗水'等"
            }

        # 2. 保存项目
        project = {
            "id": project_id,
            "name": project_name,
            "requirement": requirement,
            "understanding": understanding,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "path": str(project_dir)
        }

        # 保存配置文件
        config_file = project_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(understanding["config"], f, ensure_ascii=False, indent=2)

        # 生成部署脚本
        deploy_script = self._generate_deploy_script(project, understanding)
        deploy_file = project_dir / "deploy.sh"
        with open(deploy_file, 'w', encoding='utf-8') as f:
            f.write(deploy_script)

        # 保存项目信息
        project_file = project_dir / "project.json"
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project, f, ensure_ascii=False, indent=2)

        # 更新索引
        projects = self._load_index()
        projects.append({
            "id": project_id,
            "name": project_name,
            "created_at": project["created_at"],
            "status": project["status"]
        })
        self._save_index(projects)

        return {
            "success": True,
            "project_id": project_id,
            "project_name": project_name,
            "message": "检测系统创建成功！",
            "understanding": understanding,
            "config_path": str(config_file),
            "deploy_path": str(deploy_file)
        }

    def _generate_deploy_script(self, project: Dict, understanding: Dict) -> str:
        """生成部署脚本"""
        scenes = understanding.get("scenes", [])
        models = understanding.get("models", [])
        threshold = understanding.get("threshold", 0.5)
        alarm = understanding.get("alarm", False)

        script = f'''#!/bin/bash
# ========================================
# AI检测系统 - 一键部署脚本
# 项目：{project["name"]}
# 创建时间：{project["created_at"]}
# ========================================

echo "正在部署AI检测系统..."
echo "检测场景：{', '.join(scenes)}"
echo "模型：{', '.join(models)}"
echo "阈值：{threshold}"
echo ""

# 检查环境
if ! command -v python3 &> /dev/null; then
    echo "错误：需要 Python 3.8+"
    exit 1
fi

# 创建工作目录
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

# 安装依赖
echo "安装依赖..."
pip install -q ultralytics opencv-python pillow

# 启动检测服务
echo "启动检测服务..."
python3 -c "
import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 检测函数
def detect(frame):
    results = model.predict(frame, conf={threshold}, verbose=False)
    return results

print('检测服务已启动，按 Ctrl+C 停止')
# 这里可以添加视频流处理逻辑
"

echo "部署完成！"
'''

        return script

    def list_projects(self) -> List[Dict]:
        """列出所有AI项目"""
        return self._load_index()

    def get_project(self, project_id: str) -> Optional[Dict]:
        """获取项目详情"""
        project_dir = self.projects_dir / project_id
        project_file = project_dir / "project.json"

        if not project_file.exists():
            return None

        with open(project_file, 'r', encoding='utf-8') as f:
            return json.load(f)


# 全局实例
ai_native_service = AINativeService()
