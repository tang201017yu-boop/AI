# -*- coding: utf-8 -*-
"""
智能标注服务 - Smart Annotation Service
支持多种 YOLO 任务类型的智能标注
参考 Ultralytics 文档: https://docs.ultralytics.com/zh/platform/data/annotation/
"""
import os
import cv2
import logging
import base64
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# 创建日志记录器
logger = logging.getLogger(__name__)

# 任务类型定义
TASK_TYPES = {
    "detect": {
        "name": "目标检测",
        "label_format": "边界框 (x, y, 宽度, 高度)",
        "description": "通过轴对齐边界框识别物体及其位置",
        "applications": ["库存盘点", "交通监控", "野生动物监测", "安防系统"]
    },
    "segment": {
        "name": "实例分割",
        "label_format": "多边形顶点坐标",
        "description": "为每个对象实例创建像素级精确的掩码",
        "applications": ["医学影像", "自动驾驶", "照片编辑", "农业分析"]
    },
    "pose": {
        "name": "姿势估计",
        "label_format": "17个关键点 (COCO骨架)",
        "description": "识别人体姿势和关键点位置",
        "applications": ["运动分析", "健身追踪", "动作识别", "人机交互"]
    },
    "obb": {
        "name": "旋转框检测",
        "label_format": "4个角点坐标",
        "description": "为倾斜对象绘制旋转边界框",
        "applications": ["航拍图像", "遥感卫星", "工业检测", "文字识别"]
    },
    "classify": {
        "name": "图像分类",
        "label_format": "图像级标签",
        "description": "为整个图像分配类别标签",
        "applications": ["内容审核", "场景识别", "质量检测", "产品分类"]
    }
}

# COCO 关键点定义
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO 骨架连接
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13],  # 腿
    [12, 13], [6, 12], [7, 13], [6, 7],  # 身体
    [6, 8], [7, 9], [8, 10], [9, 11],  # 手臂
    [2, 3], [1, 2], [1, 3], [1, 4], [1, 5], [4, 6], [5, 7]  # 头
]


class SmartAnnotationService:
    """
    智能标注服务类
    支持检测、分割、姿势估计、旋转框、分类等多种任务
    """

    def __init__(self):
        """初始化智能标注服务"""
        logger.info("[标注] 初始化智能标注服务")
        self.current_task = "detect"  # 默认任务类型
        self.current_classes = []     # 当前类别列表
        self.current_image = None      # 当前图片
        self.current_image_path = None # 当前图片路径

    def get_task_types(self) -> Dict[str, Any]:
        """
        获取支持的标注任务类型
        Returns:
            Dict: 任务类型信息
        """
        return {
            "success": True,
            "tasks": TASK_TYPES,
            "current_task": self.current_task
        }

    def set_task_type(self, task_type: str) -> Dict[str, Any]:
        """
        设置当前任务类型
        Args:
            task_type: 任务类型 (detect/segment/pose/obb/classify)
        Returns:
            Dict: 设置结果
        """
        if task_type not in TASK_TYPES:
            return {
                "success": False,
                "message": f"不支持的任务类型: {task_type}"
            }

        self.current_task = task_type
        logger.info(f"[标注] 设置任务类型: {task_type}")

        return {
            "success": True,
            "message": f"任务类型已设置为: {TASK_TYPES[task_type]['name']}",
            "task": TASK_TYPES[task_type]
        }

    def set_classes(self, classes: List[str]) -> Dict[str, Any]:
        """
        设置类别列表
        Args:
            classes: 类别名称列表
        Returns:
            Dict: 设置结果
        """
        self.current_classes = classes
        logger.info(f"[标注] 设置类别: {classes}")

        return {
            "success": True,
            "message": f"已设置 {len(classes)} 个类别",
            "classes": classes
        }

    def load_image(self, image_path: str) -> Dict[str, Any]:
        """
        加载图片
        Args:
            image_path: 图片路径
        Returns:
            Dict: 加载结果
        """
        logger.info(f"[标注] 加载图片: {image_path}")

        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "message": "无法读取图片"}

            self.current_image = image
            self.current_image_path = image_path

            h, w = image.shape[:2]

            logger.info(f"[标注] 图片加载成功: {w}x{h}")

            return {
                "success": True,
                "message": "图片加载成功",
                "image_size": {"width": w, "height": h},
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            }

        except Exception as e:
            logger.error(f"[标注] 加载图片失败: {e}")
            return {"success": False, "message": str(e)}

    def detect_objects(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        目标检测 - 使用 YOLO 检测物体
        Args:
            model_name: YOLO 模型名称
            confidence: 置信度阈值
        Returns:
            Dict: 检测结果
        """
        logger.info(f"[标注] 目标检测: model={model_name}, conf={confidence}")

        if self.current_image is None:
            return {"success": False, "message": "请先加载图片"}

        try:
            from backend.core.yolo_engine import yolo_engine

            # 保存临时图片
            import tempfile
            temp_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(temp_path, self.current_image)

            # 执行推理
            result = yolo_engine.infer(
                image_path=temp_path,
                model_identifier=model_name,
                confidence=confidence
            )

            if not result.get("success"):
                return result

            detections = result.get("detections", [])

            # 生成预览图
            preview = self._draw_detections(detections)

            logger.info(f"[标注] 检测到 {len(detections)} 个对象")

            return {
                "success": True,
                "message": f"检测到 {len(detections)} 个对象",
                "task": "detect",
                "detections": detections,
                "preview_image": preview,
                "total": len(detections)
            }

        except Exception as e:
            logger.error(f"[标注] 检测失败: {e}")
            return {"success": False, "message": str(e)}

    def segment_objects(
        self,
        model_name: str = "yolo11n-seg.pt",
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        实例分割 - 使用 YOLO 进行分割
        Args:
            model_name: YOLO分割模型
            confidence: 置信度阈值
        Returns:
            Dict: 分割结果
        """
        logger.info(f"[标注] 实例分割: model={model_name}, conf={confidence}")

        if self.current_image is None:
            return {"success": False, "message": "请先加载图片"}

        try:
            from backend.core.yolo_engine import yolo_engine

            # 使用分割模型
            model = yolo_engine.load_model(model_name)

            # 保存临时图片
            import tempfile
            temp_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(temp_path, self.current_image)

            # 执行分割推理
            results = model.predict(
                source=temp_path,
                conf=confidence,
                verbose=False
            )

            if not results or len(results) == 0:
                return {"success": False, "message": "分割失败"}

            result = results[0]

            # 提取分割结果
            masks = result.masks
            boxes = result.boxes
            annotations = []

            if masks is not None:
                for i in range(len(masks)):
                    # 获取类别信息
                    cls_id = int(boxes[i].cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(boxes[i].conf[0])

                    # 获取边界框
                    bbox = boxes[i].xyxy[0].tolist()

                    # 获取分割掩码
                    mask = masks[i].data[0].cpu().numpy()
                    polygons = self._mask_to_polygons(mask)

                    annotations.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": bbox,
                        "segmentation": polygons
                    })

            # 生成预览图
            preview = self._draw_segmentations(annotations)

            logger.info(f"[标注] 分割完成: {len(annotations)} 个对象")

            return {
                "success": True,
                "message": f"分割完成: {len(annotations)} 个对象",
                "task": "segment",
                "annotations": annotations,
                "preview_image": preview,
                "total": len(annotations)
            }

        except Exception as e:
            logger.error(f"[标注] 分割失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": str(e)}

    def detect_pose(
        self,
        model_name: str = "yolo11n-pose.pt",
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        姿势估计 - 检测人体关键点
        Args:
            model_name: YOLO姿势模型
            confidence: 置信度阈值
        Returns:
            Dict: 姿势估计结果
        """
        logger.info(f"[标注] 姿势估计: model={model_name}, conf={confidence}")

        if self.current_image is None:
            return {"success": False, "message": "请先加载图片"}

        try:
            from backend.core.yolo_engine import yolo_engine

            # 使用姿势模型
            model = yolo_engine.load_model(model_name)

            # 保存临时图片
            import tempfile
            temp_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(temp_path, self.current_image)

            # 执行姿势估计
            results = model.predict(
                source=temp_path,
                conf=confidence,
                verbose=False,
                kptiline=True  # 绘制骨架
            )

            if not results or len(results) == 0:
                return {"success": False, "message": "姿势估计失败"}

            result = results[0]
            keypoints = result.keypoints
            boxes = result.boxes

            annotations = []

            if keypoints is not None:
                for i in range(len(keypoints)):
                    # 获取类别信息
                    cls_id = int(boxes[i].cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(boxes[i].conf[0])

                    # 获取关键点数据
                    kp_data = keypoints.data[i].cpu().numpy()

                    # 格式化关键点
                    keypoint_list = []
                    for j, kp_name in enumerate(COCO_KEYPOINTS):
                        if j < len(kp_data):
                            x, y, v = kp_data[j]
                            keypoint_list.append({
                                "name": kp_name,
                                "x": float(x),
                                "y": float(y),
                                "visibility": int(v)  # 0=未标记, 1=被遮挡, 2=可见
                            })

                    annotations.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "keypoints": keypoint_list,
                        "num_keypoints": len([k for k in keypoint_list if k["visibility"] > 0])
                    })

            # 生成预览图
            preview = self._draw_pose_preview(annotations)

            logger.info(f"[标注] 姿势估计完成: {len(annotations)} 个人")

            return {
                "success": True,
                "message": f"检测到 {len(annotations)} 个人",
                "task": "pose",
                "keypoint_names": COCO_KEYPOINTS,
                "skeleton": COCO_SKELETON,
                "annotations": annotations,
                "preview_image": preview,
                "total": len(annotations)
            }

        except Exception as e:
            logger.error(f"[标注] 姿势估计失败: {e}")
            return {"success": False, "message": str(e)}

    def detect_obb(
        self,
        model_name: str = "yolo11n-obb.pt",
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        旋转框检测 - 检测倾斜对象
        Args:
            model_name: YOLO OBB模型
            confidence: 置信度阈值
        Returns:
            Dict: 旋转框检测结果
        """
        logger.info(f"[标注] 旋转框检测: model={model_name}, conf={confidence}")

        if self.current_image is None:
            return {"success": False, "message": "请先加载图片"}

        try:
            from backend.core.yolo_engine import yolo_engine

            # 使用 OBB 模型
            model = yolo_engine.load_model(model_name)

            # 保存临时图片
            import tempfile
            temp_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(temp_path, self.current_image)

            # 执行 OBB 检测
            results = model.predict(
                source=temp_path,
                conf=confidence,
                verbose=False,
                obb=True
            )

            if not results or len(results) == 0:
                return {"success": False, "message": "旋转框检测失败"}

            result = results[0]
            obb = result.obb
            annotations = []

            if obb is not None:
                for i in range(len(obb)):
                    cls_id = int(obb.cls[i])
                    cls_name = result.names[cls_id]
                    conf = float(obb.conf[i])

                    # 获取旋转边界框角点 (8个值: x1,y1,x2,y2,x3,y3,x4,y4)
                    rotated_box = obb.xyxyxyxy[i].cpu().numpy().tolist()

                    annotations.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "rotated_bbox": rotated_box
                    })

            # 生成预览图
            preview = self._draw_obb_preview(annotations)

            logger.info(f"[标注] 旋转框检测完成: {len(annotations)} 个对象")

            return {
                "success": True,
                "message": f"检测到 {len(annotations)} 个对象",
                "task": "obb",
                "annotations": annotations,
                "preview_image": preview,
                "total": len(annotations)
            }

        except Exception as e:
            logger.error(f"[标注] 旋转框检测失败: {e}")
            return {"success": False, "message": str(e)}

    def classify_image(
        self,
        model_name: str = "yolo11n-cls.pt",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        图像分类
        Args:
            model_name: YOLO分类模型
            top_k: 返回前k个结果
        Returns:
            Dict: 分类结果
        """
        logger.info(f"[标注] 图像分类: model={model_name}")

        if self.current_image is None:
            return {"success": False, "message": "请先加载图片"}

        try:
            from backend.core.yolo_engine import yolo_engine

            # 使用分类模型
            model = yolo_engine.load_model(model_name)

            # 保存临时图片
            import tempfile
            temp_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(temp_path, self.current_image)

            # 执行分类
            results = model.predict(
                source=temp_path,
                verbose=False
            )

            if not results or len(results) == 0:
                return {"success": False, "message": "分类失败"}

            result = results[0]

            # 获取分类结果
            probs = result.probs
            top_indices = probs.top5[:top_k]
            top_conf = probs.top5conf[:top_k].tolist()

            # 获取类别名称
            names = result.names if hasattr(result, 'names') else {}

            predictions = []
            for idx, conf in zip(top_indices, top_conf):
                predictions.append({
                    "class_id": int(idx),
                    "class_name": names.get(int(idx), f"class_{idx}"),
                    "confidence": float(conf)
                })

            # 获取第一张图的预测结果
            preview_class = predictions[0]["class_name"] if predictions else "未知"

            logger.info(f"[标注] 图像分类完成: {preview_class}")

            return {
                "success": True,
                "message": f"图像分类完成",
                "task": "classify",
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None,
                "total": len(predictions)
            }

        except Exception as e:
            logger.error(f"[标注] 图像分类失败: {e}")
            return {"success": False, "message": str(e)}

    def auto_annotate(
        self,
        task: str = "detect",
        model_name: str = None,
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        自动标注 - 根据任务类型自动选择模型
        Args:
            task: 任务类型
            model_name: 模型名称（可选）
            confidence: 置信度阈值
        Returns:
            Dict: 自动标注结果
        """
        logger.info(f"[标注] 自动标注: task={task}, model={model_name}, conf={confidence}")

        if self.current_image is None:
            return {"success": False, "message": "请先加载图片"}

        # 根据任务选择默认模型
        if model_name is None:
            model_map = {
                "detect": "yolo11n.pt",
                "segment": "yolo11n-seg.pt",
                "pose": "yolo11n-pose.pt",
                "obb": "yolo11n-obb.pt",
                "classify": "yolo11n-cls.pt"
            }
            model_name = model_map.get(task, "yolo11n.pt")

        # 根据任务调用相应的方法
        if task == "detect":
            return self.detect_objects(model_name, confidence)
        elif task == "segment":
            return self.segment_objects(model_name, confidence)
        elif task == "pose":
            return self.detect_pose(model_name, confidence)
        elif task == "obb":
            return self.detect_obb(model_name, confidence)
        elif task == "classify":
            return self.classify_image(model_name, 5)
        else:
            return {"success": False, "message": f"不支持的任务类型: {task}"}

    def export_yolo_format(
        self,
        annotations: List[Dict],
        output_dir: str,
        image_filename: str
    ) -> Dict[str, Any]:
        """
        导出为 YOLO 格式
        Args:
            annotations: 标注列表
            output_dir: 输出目录
            image_filename: 图片文件名
        Returns:
            Dict: 导出结果
        """
        logger.info(f"[标注] 导出 YOLO 格式: {output_dir}")

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if self.current_image is None:
                return {"success": False, "message": "没有图片"}

            h, w = self.current_image.shape[:2]

            # 获取文件名（不含扩展名）
            stem = Path(image_filename).stem

            # 根据任务类型导出
            if self.current_task == "detect":
                label_file = output_path / f"{stem}.txt"
                with open(label_file, 'w') as f:
                    for ann in annotations:
                        cls_id = ann.get("class_id", 0)
                        bbox = ann.get("bbox", [])
                        if len(bbox) == 4:
                            # 转换为 YOLO 格式: class x_center y_center width height
                            x1, y1, x2, y2 = bbox
                            x_center = ((x1 + x2) / 2) / w
                            y_center = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            elif self.current_task == "segment":
                label_file = output_path / f"{stem}.txt"
                with open(label_file, 'w') as f:
                    for ann in annotations:
                        cls_id = ann.get("class_id", 0)
                        segmentation = ann.get("segmentation", [])
                        # 分割是归一化的多边形坐标
                        if segmentation:
                            # 已经是归一化格式，直接写入
                            f.write(f"{cls_id} {segmentation}\n")

            elif self.current_task == "pose":
                label_file = output_path / f"{stem}.txt"
                with open(label_file, 'w') as f:
                    for ann in annotations:
                        cls_id = ann.get("class_id", 0)
                        keypoints = ann.get("keypoints", [])
                        if keypoints:
                            # 格式: class x1 y1 v1 x2 y2 v2 ...
                            kp_str = " ".join([
                                f"{kp['x']/w:.6f} {kp['y']/h:.6f} {kp['visibility']}"
                                for kp in keypoints
                            ])
                            f.write(f"{cls_id} {kp_str}\n")

            elif self.current_task == "obb":
                label_file = output_path / f"{stem}.txt"
                with open(label_file, 'w') as f:
                    for ann in annotations:
                        cls_id = ann.get("class_id", 0)
                        rotated_bbox = ann.get("rotated_bbox", [])
                        if len(rotated_bbox) == 8:
                            # 归一化角点
                            coords = []
                            for i in range(0, 8, 2):
                                coords.append(f"{rotated_bbox[i]/w:.6f}")
                                coords.append(f"{rotated_bbox[i+1]/h:.6f}")
                            f.write(f"{cls_id} {' '.join(coords)}\n")

            # 复制图片
            import shutil
            img_ext = Path(image_filename).suffix
            dest_img = output_path / f"{stem}{img_ext}"
            cv2.imwrite(str(dest_img), self.current_image)

            logger.info(f"[标注] 导出成功: {label_file}")

            return {
                "success": True,
                "message": "导出成功",
                "label_file": str(label_file),
                "image_file": str(dest_img)
            }

        except Exception as e:
            logger.error(f"[标注] 导出失败: {e}")
            return {"success": False, "message": str(e)}

    # ==================== 辅助方法 ====================

    def _mask_to_polygons(self, mask: np.ndarray) -> str:
        """将掩码转换为 YOLO 格式的多边形"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if self.current_image is None:
            return ""

        h, w = self.current_image.shape[:2]
        polygons = []

        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) >= 3:
                # 归一化坐标
                normalized = []
                for point in approx:
                    x, y = point[0]
                    normalized.append(f"{x/w:.6f} {y/h:.6f}")
                polygons.append(" ".join(normalized))

        return " ".join(polygons)

    def _draw_detections(self, detections: List[Dict]) -> str:
        """绘制检测框"""
        if self.current_image is None:
            return ""

        preview = self.current_image.copy()
        h, w = preview.shape[:2]

        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(detections), 3))

        for i, det in enumerate(detections):
            color = tuple(map(int, colors[i]))
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

                label = f"{det.get('class_name', 'object')}: {det.get('confidence', 0):.2f}"
                cv2.putText(preview, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return self._encode_image(preview)

    def _draw_segmentations(self, annotations: List[Dict]) -> str:
        """绘制分割结果"""
        if self.current_image is None:
            return ""

        preview = self.current_image.copy()

        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(annotations), 3))

        for i, ann in enumerate(annotations):
            color = tuple(map(int, colors[i]))
            bbox = ann.get("bbox", [])
            class_name = ann.get("class_name", "object")
            conf = ann.get("confidence", 0)

            # 绘制边界框
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name}: {conf:.2f}"
                cv2.putText(preview, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return self._encode_image(preview)

    def _draw_pose_preview(self, annotations: List[Dict]) -> str:
        """绘制姿势估计结果"""
        if self.current_image is None:
            return ""

        preview = self.current_image.copy()

        # 绘制关键点和骨架
        for ann in annotations:
            keypoints = ann.get("keypoints", [])
            class_name = ann.get("class_name", "person")

            # 绘制关键点
            for kp in keypoints:
                if kp["visibility"] > 0:
                    x, y = int(kp["x"]), int(kp["y"])
                    cv2.circle(preview, (x, y), 3, (0, 255, 0), -1)

            # 绘制骨架
            for pair in COCO_SKELETON:
                if pair[0] <= len(keypoints) and pair[1] <= len(keypoints):
                    kp1 = keypoints[pair[0] - 1]
                    kp2 = keypoints[pair[1] - 1]
                    if kp1["visibility"] > 0 and kp2["visibility"] > 0:
                        x1, y1 = int(kp1["x"]), int(kp1["y"])
                        x2, y2 = int(kp2["x"]), int(kp2["y"])
                        cv2.line(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x1, y1 = int(bbox[0]), int(bbox[1])
                label = f"{class_name}: {ann.get('confidence', 0):.2f}"
                cv2.putText(preview, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return self._encode_image(preview)

    def _draw_obb_preview(self, annotations: List[Dict]) -> str:
        """绘制旋转框结果"""
        if self.current_image is None:
            return ""

        preview = self.current_image.copy()

        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(annotations), 3))

        for i, ann in enumerate(annotations):
            color = tuple(map(int, colors[i]))
            rotated_bbox = ann.get("rotated_bbox", [])
            class_name = ann.get("class_name", "object")
            conf = ann.get("confidence", 0)

            if len(rotated_bbox) == 8:
                points = []
                for j in range(0, 8, 2):
                    points.append((int(rotated_bbox[j]), int(rotated_bbox[j + 1])))

                # 绘制旋转框
                for j in range(4):
                    cv2.line(preview, points[j], points[(j + 1) % 4], color, 2)

                # 绘制中心点
                cx = sum(p[0] for p in points) / 4
                cy = sum(p[1] for p in points) / 4
                cv2.circle(preview, (int(cx), int(cy)), 3, color, -1)

                # 绘制标签
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(preview, label, (int(cx), int(cy) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return self._encode_image(preview)

    def _encode_image(self, image) -> str:
        """将图片编码为 base64"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            logger.error(f"[标注] 编码图片失败: {e}")
            return ""


# 创建全局服务实例
smart_annotation_service = SmartAnnotationService()


class MultiObjectAutoAnnotator:
    """
    多对象自动标注器
    支持批量多对象标注，适用于复杂场景
    """

    def __init__(self):
        """初始化多对象标注器"""
        logger.info("[多对象标注] 初始化多对象自动标注器")
        self.current_service = smart_annotation_service
        self.results_cache = []  # 缓存标注结果
        self.current_image = None
        self.current_image_path = None

    def annotate_single_image(
        self,
        image_path: str,
        task: str = "detect",
        model_name: str = None,
        confidence: float = 0.25,
        class_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        对单张图片进行多对象标注
        Args:
            image_path: 图片路径
            task: 任务类型
            model_name: 模型名称
            confidence: 置信度阈值
            class_names: 类别名称列表
        Returns:
            Dict: 标注结果
        """
        logger.info(f"[多对象标注] 处理图片: {image_path}")

        try:
            # 加载图片
            load_result = self.current_service.load_image(image_path)
            if not load_result.get("success"):
                return load_result

            self.current_image_path = image_path
            self.current_image = self.current_service.current_image

            # 设置类别
            if class_names:
                self.current_service.set_classes(class_names)

            # 执行自动标注
            result = self.current_service.auto_annotate(task, model_name, confidence)

            # 缓存结果
            if result.get("success"):
                self.results_cache = result.get("annotations", []) or result.get("detections", [])
                result["image_path"] = image_path

            return result

        except Exception as e:
            logger.error(f"[多对象标注] 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": str(e)}

    def annotate_batch(
        self,
        image_paths: List[str],
        task: str = "detect",
        model_name: str = None,
        confidence: float = 0.25,
        class_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        批量标注多张图片
        Args:
            image_paths: 图片路径列表
            task: 任务类型
            model_name: 模型名称
            confidence: 置信度阈值
            class_names: 类别名称列表
        Returns:
            Dict: 批量标注结果
        """
        logger.info(f"[多对象标注] 批量处理 {len(image_paths)} 张图片")

        results = []
        success_count = 0
        failed_count = 0

        for img_path in image_paths:
            try:
                result = self.annotate_single_image(
                    img_path, task, model_name, confidence, class_names
                )

                if result.get("success"):
                    success_count += 1
                else:
                    failed_count += 1

                results.append({
                    "image_path": img_path,
                    "success": result.get("success", False),
                    "result": result
                })

            except Exception as e:
                logger.error(f"[多对象标注] 处理图片失败 {img_path}: {e}")
                failed_count += 1
                results.append({
                    "image_path": img_path,
                    "success": False,
                    "error": str(e)
                })

        return {
            "success": True,
            "message": f"批量处理完成: 成功 {success_count}, 失败 {failed_count}",
            "total": len(image_paths),
            "success_count": success_count,
            "failed_count": failed_count,
            "results": results
        }

    def upload_and_annotate(
        self,
        image_data: bytes,
        filename: str,
        task: str = "detect",
        model_name: str = None,
        confidence: float = 0.25,
        class_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        上传图片并自动标注
        Args:
            image_data: 图片二进制数据
            filename: 文件名
            task: 任务类型
            model_name: 模型名称
            confidence: 置信度阈值
            class_names: 类别名称列表
        Returns:
            Dict: 标注结果
        """
        logger.info(f"[多对象标注] 上传并标注: {filename}")

        try:
            # 保存临时文件
            import tempfile
            from pathlib import Path

            temp_dir = Path(tempfile.gettempdir()) / "annotation_upload"
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / filename

            with open(temp_path, 'wb') as f:
                f.write(image_data)

            # 执行标注
            return self.annotate_single_image(
                str(temp_path), task, model_name, confidence, class_names
            )

        except Exception as e:
            logger.error(f"[多对象标注] 上传标注失败: {e}")
            return {"success": False, "message": str(e)}

    def export_to_yolo(
        self,
        output_dir: str,
        image_filename: str = None
    ) -> Dict[str, Any]:
        """
        导出标注结果为YOLO格式
        Args:
            output_dir: 输出目录
            image_filename: 图片文件名（可选）
        Returns:
            Dict: 导出结果
        """
        logger.info(f"[多对象标注] 导出YOLO格式: {output_dir}")

        if not self.results_cache:
            return {"success": False, "message": "没有标注结果可导出"}

        try:
            # 使用 smart_annotation_service 导出
            filename = image_filename or "image.jpg"
            return self.current_service.export_yolo_format(
                self.results_cache,
                output_dir,
                filename
            )
        except Exception as e:
            logger.error(f"[多对象标注] 导出失败: {e}")
            return {"success": False, "message": str(e)}

    def get_available_models(self) -> Dict[str, Any]:
        """
        获取可用的模型列表
        Returns:
            Dict: 模型列表
        """
        models = {
            "detect": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo26n.pt", "yolo26s.pt", "yolo26m.pt"],
            "segment": ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt"],
            "pose": ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt"],
            "obb": ["yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt"],
            "classify": ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt"]
        }

        return {
            "success": True,
            "models": models,
            "default_models": {
                "detect": "yolo11n.pt",
                "segment": "yolo11n-seg.pt",
                "pose": "yolo11n-pose.pt",
                "obb": "yolo11n-obb.pt",
                "classify": "yolo11n-cls.pt"
            }
        }


# 创建全局多对象标注器实例
multi_object_annotator = MultiObjectAutoAnnotator()

logger.info("[标注] 多对象自动标注器加载完成")
logger.info("[标注] 智能标注服务模块加载完成")
