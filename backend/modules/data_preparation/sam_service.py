"""
SAM 智能标注服务 - Segment Anything Model Service
提供基于点击的智能标注功能
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment-anything not installed. Install with: pip install segment-anything")


class SAMService:
    """SAM 智能标注服务"""

    def __init__(self):
        self.predictor = None
        self.model_loaded = False
        self.model_type = "vit_b"  # 默认模型

    def load_model(self, model_type: str = "vit_b") -> Dict[str, Any]:
        """加载 SAM 模型"""
        if not SAM_AVAILABLE:
            return {
                "success": False,
                "message": "SAM 未安装，请运行: pip install segment-anything"
            }

        try:
            if self.predictor is not None:
                return {"success": True, "message": "模型已加载", "model_type": self.model_type}

            # 模型路径
            model_paths = {
                "vit_b": "data/models/sam_vit_b.pth",
                "vit_l": "data/models/sam_vit_l.pth",
                "vit_h": "data/models/sam_vit_h.pth"
            }

            model_path = model_paths.get(model_type, model_paths["vit_b"])

            # 检查模型是否存在
            from pathlib import Path
            if not Path(model_path).exists():
                return {
                    "success": False,
                    "message": f"模型文件不存在: {model_path}",
                    "download_url": "https://github.com/facebookresearch/segment-anything#model-checkpoints"
                }

            sam = sam_model_registry[model_type](checkpoint=model_path)
            self.predictor = SamPredictor(sam)
            self.model_type = model_type
            self.model_loaded = True

            return {
                "success": True,
                "message": "模型加载成功",
                "model_type": model_type
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"模型加载失败: {str(e)}"
            }

    def set_image(self, image_path: str) -> Dict[str, Any]:
        """设置要标注的图片"""
        if not self.model_loaded:
            return {"success": False, "message": "SAM 模型未加载"}

        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "message": "无法读取图片"}

            # 转换 BGR 到 RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)

            return {
                "success": True,
                "message": "图片已加载",
                "image_size": image.shape[:2]
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"设置图片失败: {str(e)}"
            }

    def predict(
        self,
        points: List[List[float]],
        labels: List[int],
        mask_input: bool = True
    ) -> Dict[str, Any]:
        """
        预测分割掩码

        Args:
            points: 点击点坐标 [[x, y], ...]
            labels: 点标签 [1=前景, 0=背景]
            mask_input: 是否使用掩码输入
        """
        if not self.model_loaded:
            return {"success": False, "message": "SAM 模型未加载"}

        try:
            # 预测
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                mask_input=mask_input if hasattr(self, '_mask_input') else None
            )

            # 找到最佳掩码
            best_idx = scores.argmax()
            best_mask = masks[best_idx]
            best_score = scores[best_idx]

            # 转换掩码为多边形
            polygons = self._mask_to_polygon(best_mask)

            # 保存掩码用于后续迭代
            self._mask_input = logits[best_idx:best_idx+1]

            return {
                "success": True,
                "masks": polygons,
                "mask_image": best_mask.tolist(),
                "score": float(best_score)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"预测失败: {str(e)}"
            }

    def clear_mask(self):
        """清除当前的掩码输入"""
        if hasattr(self, '_mask_input'):
            del self._mask_input

    def _mask_to_polygon(self, mask) -> List[List[float]]:
        """将掩码转换为多边形"""
        import cv2
        import numpy as np

        # 查找轮廓
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            # 简化多边形
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 转换为列表
            polygon = approx.flatten().tolist()
            if len(polygon) >= 6:  # 至少3个点
                # 转换为 [x1, y1, x2, y2, ...] 格式
                points = []
                for i in range(0, len(polygon), 2):
                    points.append([polygon[i], polygon[i+1]])
                polygons.append(points)

        return polygons

    def auto_label(
        self,
        image_path: str,
        class_names: List[str],
        model_name: str = "yolo11n.pt"
    ) -> Dict[str, Any]:
        """
        自动标注 - 使用 YOLO 模型预标注

        Args:
            image_path: 图片路径
            class_names: 类别名称列表
            model_name: YOLO 模型名称
        """
        if not SAM_AVAILABLE:
            return {"success": False, "message": "SAM 未安装"}

        try:
            from backend.core.yolo_engine import yolo_engine

            # YOLO 推理获取检测框
            result = yolo_engine.infer(
                image_path=image_path,
                model_identifier=model_name,
                confidence=0.25
            )

            if not result.get("success"):
                return result

            # 设置图片
            self.set_image(image_path)

            annotations = []
            for det in result.get("detections", []):
                bbox = det["bbox"]
                class_id = det["class_id"]
                class_name = det["class_name"]

                # 获取中心点
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                # 使用 SAM 分割
                sam_result = self.predict(
                    points=[[cx, cy]],
                    labels=[1]
                )

                if sam_result.get("success"):
                    annotations.append({
                        "class": class_name,
                        "class_id": class_id,
                        "bbox": bbox,
                        "segmentation": sam_result["masks"][0] if sam_result["masks"] else None,
                        "confidence": det["confidence"]
                    })

            return {
                "success": True,
                "message": f"自动标注完成，找到 {len(annotations)} 个对象",
                "annotations": annotations
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"自动标注失败: {str(e)}"
            }

    def get_model_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            "available": SAM_AVAILABLE,
            "loaded": self.model_loaded,
            "model_type": self.model_type if self.model_loaded else None
        }


# 全局实例
sam_service = SAMService()
