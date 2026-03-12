# -*- coding: utf-8 -*-
"""
SAM 智能标注服务 - Segment Anything Model Service
提供基于 SAM 的智能分割标注和 YOLO 预标注功能
参考 Ultralytics 文档: https://docs.ultralytics.com/zh/platform/data/annotation/
"""
import os
import cv2
import logging
import base64
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# 创建日志记录器
logger = logging.getLogger(__name__)

# 尝试导入 SAM
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("Warning: segment-anything not installed. Install with: pip install segment-anything")


class SAMService:
    """
    SAM 智能标注服务类
    提供基于 SAM 模型的智能分割和 YOLO 预标注功能
    """

    # 支持的 SAM 模型类型
    MODEL_TYPES = {
        "vit_b": {"name": "ViT-B", "speed": "快速", "size": "~375MB", "description": "速度最快，精度较低"},
        "vit_l": {"name": "ViT-L", "speed": "中等", "size": "~1.2GB", "description": "平衡速度和精度"},
        "vit_h": {"name": "ViT-H", "speed": "较慢", "size": "~2.5GB", "description": "精度最高，速度较慢"}
    }

    def __init__(self):
        """初始化 SAM 服务"""
        logger.info("[SAM] 初始化智能标注服务")
        self.predictor = None
        self.model_loaded = False
        self.model_type = "vit_b"  # 默认模型
        self.current_image = None
        self.current_image_path = None
        self._mask_input = None  # 用于迭代优化

    def load_model(self, model_type: str = "vit_b") -> Dict[str, Any]:
        """
        加载 SAM 模型
        Args:
            model_type: 模型类型 (vit_b/vit_l/vit_h)
        Returns:
            Dict: 加载结果
        """
        logger.info(f"[SAM] 加载模型: {model_type}")

        if not SAM_AVAILABLE:
            error_msg = "SAM 未安装，请运行: pip install segment-anything"
            logger.error(f"[SAM] {error_msg}")
            return {
                "success": False,
                "message": error_msg
            }

        try:
            # 如果模型已加载，直接返回
            if self.model_loaded and self.model_type == model_type:
                logger.info("[SAM] 模型已加载")
                return {"success": True, "message": "模型已加载", "model_type": self.model_type}

            # 模型路径配置
            model_paths = {
                "vit_b": "data/models/sam_vit_b.pth",
                "vit_l": "data/models/sam_vit_l.pth",
                "vit_h": "data/models/sam_vit_h.pth"
            }

            model_path = model_paths.get(model_type, model_paths["vit_b"])

            # 检查模型文件是否存在
            if not Path(model_path).exists():
                error_msg = f"模型文件不存在: {model_path}"
                logger.error(f"[SAM] {error_msg}")
                return {
                    "success": False,
                    "message": error_msg,
                    "download_url": "https://github.com/facebookresearch/segment-anything#model-checkpoints"
                }

            # 加载模型
            logger.info(f"[SAM] 加载 SAM 模型: {model_path}")
            sam = sam_model_registry[model_type](checkpoint=model_path)
            self.predictor = SamPredictor(sam)
            self.model_type = model_type
            self.model_loaded = True
            self._mask_input = None

            logger.info(f"[SAM] 模型加载成功: {model_type}")
            return {
                "success": True,
                "message": "模型加载成功",
                "model_type": model_type,
                "model_info": self.MODEL_TYPES.get(model_type)
            }

        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            logger.error(f"[SAM] {error_msg}")
            return {
                "success": False,
                "message": error_msg
            }

    def get_model_status(self) -> Dict[str, Any]:
        """
        获取模型状态
        Returns:
            Dict: 模型状态信息
        """
        return {
            "available": SAM_AVAILABLE,
            "loaded": self.model_loaded,
            "model_type": self.model_type if self.model_loaded else None,
            "supported_models": self.MODEL_TYPES
        }

    def set_image(self, image_path: str) -> Dict[str, Any]:
        """
        设置要标注的图片
        Args:
            image_path: 图片路径
        Returns:
            Dict: 设置结果
        """
        logger.info(f"[SAM] 设置图片: {image_path}")

        if not self.model_loaded:
            error_msg = "SAM 模型未加载"
            logger.warning(f"[SAM] {error_msg}")
            return {"success": False, "message": error_msg}

        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                error_msg = f"无法读取图片: {image_path}"
                logger.error(f"[SAM] {error_msg}")
                return {"success": False, "message": error_msg}

            # 保存图片信息
            self.current_image = image
            self.current_image_path = image_path

            # 转换 BGR 到 RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 设置图片到 predictor
            self.predictor.set_image(image_rgb)

            # 清除之前的掩码
            self._mask_input = None

            logger.info(f"[SAM] 图片已加载: {image.shape[1]}x{image.shape[0]}")

            return {
                "success": True,
                "message": "图片已加载",
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            }

        except Exception as e:
            error_msg = f"设置图片失败: {str(e)}"
            logger.error(f"[SAM] {error_msg}")
            return {
                "success": False,
                "message": error_msg
            }

    def predict(
        self,
        points: List[List[float]],
        labels: List[int],
        mask_input: bool = True
    ) -> Dict[str, Any]:
        """
        预测分割掩码（基于点击）
        Args:
            points: 点击点坐标 [[x, y], ...]
            labels: 点标签 [1=前景, 0=背景]
            mask_input: 是否使用掩码输入（用于迭代优化）
        Returns:
            Dict: 预测结果
        """
        logger.debug(f"[SAM] 预测分割: {len(points)} 个点")

        if not self.model_loaded:
            error_msg = "SAM 模型未加载"
            logger.warning(f"[SAM] {error_msg}")
            return {"success": False, "message": error_msg}

        try:
            # 转换为 numpy 数组
            point_coords = np.array(points) if points else None
            point_labels = np.array(labels) if labels else None

            # 预测分割掩码
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=self._mask_input if mask_input else None,
                multimask_output=True
            )

            # 找到最佳掩码（根据分数）
            best_idx = scores.argmax()
            best_mask = masks[best_idx]
            best_score = scores[best_idx]

            # 转换掩码为多边形
            polygons = self._mask_to_polygon(best_mask)

            # 保存掩码用于后续迭代优化
            self._mask_input = logits[best_idx:best_idx+1]

            # 生成掩码图片（base64）
            mask_image = self._generate_mask_image(best_mask)

            logger.info(f"[SAM] 分割完成: {len(polygons)} 个对象, 分数: {best_score:.3f}")

            return {
                "success": True,
                "masks": polygons,
                "mask_image": mask_image,
                "score": float(best_score),
                "all_scores": scores.tolist()
            }

        except Exception as e:
            error_msg = f"预测失败: {str(e)}"
            logger.error(f"[SAM] {error_msg}")
            return {
                "success": False,
                "message": error_msg
            }

    def predict_box(
        self,
        bbox: List[float]
    ) -> Dict[str, Any]:
        """
        基于边界框预测分割掩码
        Args:
            bbox: 边界框坐标 [x1, y1, x2, y2]
        Returns:
            Dict: 预测结果
        """
        logger.debug(f"[SAM] 基于边界框分割: {bbox}")

        if not self.model_loaded:
            return {"success": False, "message": "SAM 模型未加载"}

        try:
            # 从边界框生成点（使用四个角点）
            x1, y1, x2, y2 = bbox
            point_coords = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ])
            point_labels = np.array([1, 1, 1, 1])  # 前景点

            # 预测
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            # 选择最佳掩码
            best_idx = scores.argmax()
            best_mask = masks[best_idx]
            best_score = scores[best_idx]

            # 转换掩码为多边形
            polygons = self._mask_to_polygon(best_mask)

            logger.info(f"[SAM] 框选分割完成: {len(polygons)} 个对象")

            return {
                "success": True,
                "masks": polygons,
                "score": float(best_score)
            }

        except Exception as e:
            logger.error(f"[SAM] 框选分割失败: {e}")
            return {"success": False, "message": str(e)}

    def clear_mask(self):
        """清除当前的掩码输入"""
        self._mask_input = None
        logger.debug("[SAM] 已清除掩码")

    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """
        将二值掩码转换为多边形坐标列表
        Args:
            mask: 二值掩码数组
        Returns:
            List: 多边形坐标列表
        """
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

            # 转换为 [x1, y1, x2, y2, ...] 格式
            if len(approx) >= 3:  # 至少3个点
                polygon = approx.flatten().tolist()
                polygons.append(polygon)

        return polygons

    def _generate_mask_image(self, mask: np.ndarray) -> str:
        """
        生成掩码图片（带透明度）
        Args:
            mask: 二值掩码数组
        Returns:
            str: Base64 编码的图片
        """
        if self.current_image is None:
            return ""

        try:
            # 创建彩色掩码（绿色）
            h, w = mask.shape
            color_mask = np.zeros((h, w, 4), dtype=np.uint8)
            color_mask[mask, 0] = 0     # B
            color_mask[mask, 1] = 255   # G
            color_mask[mask, 2] = 0     # R
            color_mask[mask, 3] = 128   # A (半透明)

            # 叠加到原图
            overlay = self.current_image.copy()
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGBA)
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

            # 转换回 BGR 并编码
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR)
            _, buffer = cv2.imencode('.png', overlay)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"[SAM] 生成掩码图片失败: {e}")
            return ""

    def auto_label(
        self,
        image_path: str,
        class_names: List[str],
        model_name: str = "yolo11n.pt",
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        自动标注 - 使用 YOLO 检测 + SAM 分割
        参考 Ultralytics 文档: YOLO 预标注 + SAM 分割

        Args:
            image_path: 图片路径
            class_names: 类别名称列表
            model_name: YOLO 模型名称
            confidence: 置信度阈值

        Returns:
            Dict: 自动标注结果
        """
        logger.info(f"[SAM] 开始自动标注: {image_path}, 模型: {model_name}")

        if not SAM_AVAILABLE:
            return {"success": False, "message": "SAM 未安装"}

        try:
            # 导入 YOLO 引擎
            from backend.core.yolo_engine import yolo_engine

            # 设置图片
            self.set_image(image_path)

            # 使用 YOLO 进行检测
            logger.debug(f"[SAM] YOLO 检测: conf={confidence}")
            result = yolo_engine.infer(
                image_path=image_path,
                model_identifier=model_name,
                confidence=confidence
            )

            if not result.get("success"):
                return result

            detections = result.get("detections", [])
            logger.info(f"[SAM] YOLO 检测到 {len(detections)} 个对象")

            # 对每个检测结果进行 SAM 分割
            annotations = []
            for i, det in enumerate(detections):
                bbox = det["bbox"]
                class_id = det["class_id"]
                class_name = det["class_name"]
                conf = det["confidence"]

                logger.debug(f"[SAM] 处理对象 {i+1}: {class_name}")

                # 使用 SAM 分割
                seg_result = self.predict_box(bbox)

                if seg_result.get("success") and seg_result.get("masks"):
                    # 转换多边形坐标为 YOLO 格式
                    yolo_segmentation = self._polygons_to_yolo_format(
                        seg_result["masks"],
                        self.current_image.shape[1],  # width
                        self.current_image.shape[0]   # height
                    )

                    annotations.append({
                        "class": class_name,
                        "class_id": class_id,
                        "bbox": bbox,
                        "segmentation": yolo_segmentation,
                        "confidence": conf,
                        "segment_score": seg_result.get("score", 0)
                    })

            # 生成标注预览图
            preview_image = self._generate_annotation_preview(annotations)

            logger.info(f"[SAM] 自动标注完成: {len(annotations)} 个对象")

            return {
                "success": True,
                "message": f"自动标注完成，找到 {len(annotations)} 个对象",
                "annotations": annotations,
                "preview_image": preview_image,
                "total_detections": len(detections),
                "total_annotations": len(annotations)
            }

        except Exception as e:
            error_msg = f"自动标注失败: {str(e)}"
            logger.error(f"[SAM] {error_msg}")
            import traceback
            logger.error(f"[SAM] {traceback.format_exc()}")
            return {
                "success": False,
                "message": error_msg
            }

    def batch_auto_label(
        self,
        image_paths: List[str],
        class_names: List[str],
        model_name: str = "yolo11n.pt",
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        批量自动标注

        Args:
            image_paths: 图片路径列表
            class_names: 类别名称列表
            model_name: YOLO 模型名称
            confidence: 置信度阈值

        Returns:
            Dict: 批量标注结果
        """
        logger.info(f"[SAM] 开始批量自动标注: {len(image_paths)} 张图片")

        results = []
        total_annotations = 0

        for i, img_path in enumerate(image_paths):
            logger.info(f"[SAM] 处理图片 {i+1}/{len(image_paths)}: {img_path}")

            result = self.auto_label(img_path, class_names, model_name, confidence)

            if result.get("success"):
                results.append({
                    "image_path": img_path,
                    "annotations": result.get("annotations", []),
                    "total": len(result.get("annotations", []))
                })
                total_annotations += len(result.get("annotations", []))

        logger.info(f"[SAM] 批量标注完成: {len(results)} 张图片, {total_annotations} 个标注")

        return {
            "success": True,
            "message": f"批量标注完成: {len(results)} 张图片, {total_annotations} 个标注",
            "results": results,
            "total_images": len(image_paths),
            "processed_images": len(results),
            "total_annotations": total_annotations
        }

    def batch_sam_label(
        self,
        image_path: str,
        class_name: str,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        批量 SAM 标注 - 针对特定类别的所有对象

        Args:
            image_path: 图片路径
            class_name: 类别名称
            model_name: YOLO 模型
            confidence: 置信度阈值

        Returns:
            Dict: 标注结果
        """
        logger.info(f"[SAM] 批量标注类别: {class_name}, 图片: {image_path}")

        # 检查 SAM 模型是否加载
        if not self.model_loaded:
            return {"success": False, "message": "SAM 模型未加载，请先加载模型"}

        # 设置图片
        set_result = self.set_image(image_path)
        if not set_result.get("success"):
            return set_result

        # 检查图片是否设置成功
        if self.current_image is None:
            return {"success": False, "message": "图片加载失败"}

        # 使用 YOLO 检测
        try:
            from backend.core.yolo_engine import yolo_engine
            result = yolo_engine.infer(
                image_path=image_path,
                model_identifier=model_name,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"[SAM] YOLO 检测失败: {str(e)}")
            return {"success": False, "message": f"YOLO 检测失败: {str(e)}"}

        if not result.get("success"):
            return result

        detections = result.get("detections", [])

        # 过滤特定类别
        filtered = [d for d in detections if d["class_name"].lower() == class_name.lower()]

        logger.info(f"[SAM] 找到 {len(filtered)} 个 {class_name} 对象，共 {len(detections)} 个检测结果")

        # 对每个检测结果进行 SAM 分割
        annotations = []
        for det in filtered:
            bbox = det["bbox"]
            seg_result = self.predict_box(bbox)

            if seg_result.get("success") and seg_result.get("masks"):
                yolo_seg = self._polygons_to_yolo_format(
                    seg_result["masks"],
                    self.current_image.shape[1],
                    self.current_image.shape[0]
                )

                annotations.append({
                    "class": det["class_name"],
                    "class_id": det["class_id"],
                    "bbox": bbox,
                    "segmentation": yolo_seg,
                    "confidence": det["confidence"],
                    "segment_score": seg_result.get("score", 0)
                })

        preview = self._generate_annotation_preview(annotations)

        return {
            "success": True,
            "message": f"找到 {len(annotations)} 个 {class_name} 对象",
            "class_name": class_name,
            "annotations": annotations,
            "preview_image": preview,
            "total": len(annotations)
        }

    def export_yolo_format(
        self,
        annotations: List[Dict],
        output_dir: str,
        image_path: str
    ) -> Dict[str, Any]:
        """
        导出为 YOLO 格式

        Args:
            annotations: 标注列表
            output_dir: 输出目录
            image_path: 原图路径

        Returns:
            Dict: 导出结果
        """
        logger.info(f"[SAM] 导出 YOLO 格式: {output_dir}")

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 获取图片文件名
            img_name = Path(image_path).stem

            # 转换标注为 YOLO 格式
            if self.current_image is None:
                # 重新读取图片获取尺寸
                img = cv2.imread(image_path)
                if img is None:
                    return {"success": False, "message": "无法读取图片"}
                h, w = img.shape[:2]
            else:
                h, w = self.current_image.shape[:2]

            # 写入标注文件
            label_file = output_path / f"{img_name}.txt"

            with open(label_file, 'w') as f:
                for ann in annotations:
                    class_id = ann.get("class_id", 0)
                    segmentation = ann.get("segmentation", "")

                    # 写入 YOLO 格式: class_id x1 y1 x2 y2 ... (归一化)
                    if segmentation:
                        f.write(f"{class_id} {segmentation}\n")

            # 复制图片
            import shutil
            img_ext = Path(image_path).suffix
            dest_img = output_path / f"{img_name}{img_ext}"
            shutil.copy(image_path, dest_img)

            logger.info(f"[SAM] 导出成功: {label_file}")

            return {
                "success": True,
                "message": "导出成功",
                "label_file": str(label_file),
                "image_file": str(dest_img)
            }

        except Exception as e:
            error_msg = f"导出失败: {str(e)}"
            logger.error(f"[SAM] {error_msg}")
            return {"success": False, "message": error_msg}

    def _polygons_to_yolo_format(
        self,
        polygons: List,
        img_width: int,
        img_height: int
    ) -> str:
        """
        将多边形转换为 YOLO 格式（归一化坐标）
        Args:
            polygons: 多边形坐标列表
            img_width: 图片宽度
            img_height: 图片高度
        Returns:
            str: YOLO 格式的分割标注
        """
        yolo_parts = []

        for polygon in polygons:
            # polygon 是 [x1, y1, x2, y2, ...] 格式
            normalized = []
            for i in range(0, len(polygon), 2):
                x = polygon[i] / img_width
                y = polygon[i + 1] / img_height
                normalized.append(f"{x:.6f} {y:.6f}")

            yolo_parts.append(" ".join(normalized))

        return " ".join(yolo_parts)

    def _generate_annotation_preview(self, annotations: List[Dict]) -> str:
        """
        生成标注预览图
        Args:
            annotations: 标注列表
        Returns:
            str: Base64 编码的预览图
        """
        if self.current_image is None:
            return ""

        try:
            # 复制原图
            preview = self.current_image.copy()

            # 随机颜色
            np.random.seed(42)
            colors = np.random.randint(0, 255, (len(annotations), 3))

            # 绘制每个标注
            for i, ann in enumerate(annotations):
                color = tuple(map(int, colors[i]))
                bbox = ann.get("bbox", [])
                class_name = ann.get("class", "object")
                conf = ann.get("confidence", 0)

                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)

                    # 绘制边界框
                    cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(preview, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 编码为 base64
            _, buffer = cv2.imencode('.jpg', preview)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return f"data:image/jpeg;base64,{img_base64}"

        except Exception as e:
            logger.error(f"[SAM] 生成预览图失败: {e}")
            return ""


# 创建全局 SAM 服务实例
sam_service = SAMService()

logger.info("[SAM] 智能标注服务模块加载完成")
