"""
推理服务 - Inference Service
提供图片/视频推理、批量推理、API 推理等功能
"""
import cv2
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import base64
import io

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.core.config import settings
from backend.core.utils import save_uploaded_file, get_unique_filename


class InferenceService:
    """推理服务"""

    def __init__(self):
        self.model_cache: Dict[str, YOLO] = {}

    def infer_image(
        self,
        image_path: str,
        model_name: str = None,
        confidence: float = None,
        iou_threshold: float = None,
        img_size: int = None,
        draw_results: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        图片推理

        Args:
            image_path: 图片路径
            model_name: 模型名称
            confidence: 置信度阈值
            iou_threshold: IOU 阈值
            img_size: 输入尺寸
            draw_results: 是否绘制结果
        """
        from backend.core.yolo_engine import yolo_engine

        start_time = time.time()

        result = yolo_engine.infer(
            image_path=image_path,
            model_identifier=model_name,
            confidence=confidence,
            iou_threshold=iou_threshold,
            img_size=img_size
        )

        inference_time = time.time() - start_time

        if draw_results and result.get("success"):
            # 绘制检测框
            annotated_image = self._draw_detections(image_path, result["detections"])
            result["annotated_image"] = annotated_image

        result["inference_time"] = inference_time

        return result

    def infer_video(
        self,
        video_path: str,
        model_name: str = None,
        confidence: float = None,
        output_path: str = None,
        save_output: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        视频推理

        Returns:
            每帧检测结果和统计信息
        """
        from backend.core.yolo_engine import yolo_engine

        model = yolo_engine.load_model(model_name)
        confidence = confidence or settings.CONFIDENCE_THRESHOLD

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "message": "无法打开视频"}

        # 输出视频设置
        writer = None
        if save_output and output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        results_per_frame = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            # 推理
            result = model.predict(frame, conf=confidence, verbose=False)

            if result and len(result) > 0:
                boxes = result[0].boxes
                detections = []
                for i in range(len(boxes)):
                    detections.append({
                        "class_id": int(boxes[i].cls[0]),
                        "class_name": result[0].names[int(boxes[i].cls[0])],
                        "confidence": float(boxes[i].conf[0]),
                        "bbox": boxes[i].xyxy[0].tolist()
                    })

                total_detections += len(detections)
                frame_count += 1

                # 绘制
                annotated_frame = result[0].plot()
                if writer:
                    writer.write(annotated_frame)

                results_per_frame.append({
                    "frame": frame_count,
                    "detections": detections
                })

        cap.release()
        if writer:
            writer.release()

        return {
            "success": True,
            "message": "视频推理完成",
            "total_frames": frame_count,
            "total_detections": total_detections,
            "avg_detections_per_frame": total_detections / max(frame_count, 1),
            "output_path": output_path if save_output else None,
            "results": results_per_frame[:100]  # 限制返回数量
        }

    def batch_infer(
        self,
        image_paths: List[str],
        model_name: str = None,
        confidence: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """批量推理"""
        from backend.core.yolo_engine import yolo_engine

        model = yolo_engine.load_model(model_name)
        confidence = confidence or settings.CONFIDENCE_THRESHOLD

        results = []
        total_time = 0
        total_detections = 0

        for img_path in image_paths:
            start = time.time()
            result = model.predict(img_path, conf=confidence, verbose=False)
            elapsed = time.time() - start

            detections = []
            if result and len(result) > 0:
                boxes = result[0].boxes
                for i in range(len(boxes)):
                    detections.append({
                        "class_id": int(boxes[i].cls[0]),
                        "class_name": result[0].names[int(boxes[i].cls[0])],
                        "confidence": float(boxes[i].conf[0])
                    })

            results.append({
                "image": img_path,
                "detections": detections,
                "time": elapsed
            })

            total_time += elapsed
            total_detections += len(detections)

        return {
            "success": True,
            "message": f"处理了 {len(image_paths)} 张图片",
            "total_images": len(image_paths),
            "total_time": total_time,
            "avg_time_per_image": total_time / len(image_paths),
            "total_detections": total_detections,
            "results": results
        }

    def _draw_detections(
        self,
        image_path: str,
        detections: List[Dict]
    ) -> str:
        """绘制检测结果并返回 base64 编码的图片"""
        img = cv2.imread(image_path)
        if img is None:
            return ""

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            class_name = det["class_name"]
            confidence = det["confidence"]

            # 绘制框
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 转换为 base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return f"data:image/jpeg;base64,{img_base64}"

    def stream_infer(self, image_data: bytes, model_name: str = None):
        """
        流式推理（用于实时推理 API）

        Args:
            image_data: 图片二进制数据
            model_name: 模型名称

        Returns:
            检测结果
        """
        from backend.core.yolo_engine import yolo_engine
        import numpy as np

        # 解码图片
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"success": False, "message": "图片解码失败"}

        model = yolo_engine.load_model(model_name)
        confidence = settings.CONFIDENCE_THRESHOLD

        result = model.predict(image, conf=confidence, verbose=False)

        detections = []
        if result and len(result) > 0:
            boxes = result[0].boxes
            for i in range(len(boxes)):
                detections.append({
                    "class_id": int(boxes[i].cls[0]),
                    "class_name": result[0].names[int(boxes[i].cls[0])],
                    "confidence": float(boxes[i].conf[0]),
                    "bbox": boxes[i].xyxy[0].tolist()
                })

        return {
            "success": True,
            "detections": detections,
            "num_detections": len(detections)
        }


class MonitorService:
    """监控服务"""

    def __init__(self):
        self.cameras: Dict[str, Dict] = {}
        self.streams: Dict[str, Any] = {}

    def add_camera(
        self,
        name: str,
        url: str,
        model_name: str = None,
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """添加监控摄像头"""
        self.cameras[name] = {
            "url": url,
            "model_name": model_name,
            "confidence": confidence,
            "status": "offline",
            "added_at": datetime.now().isoformat()
        }

        return {"success": True, "message": "摄像头已添加", "camera": self.cameras[name]}

    def start_stream(self, camera_name: str) -> Dict[str, Any]:
        """开始视频流推理"""
        if camera_name not in self.cameras:
            return {"success": False, "message": "摄像头不存在"}

        # TODO: 实现视频流推理
        return {
            "success": True,
            "message": "视频流推理功能开发中",
            "camera": camera_name
        }

    def get_camera_status(self, camera_name: str) -> Dict[str, Any]:
        """获取摄像头状态"""
        if camera_name not in self.cameras:
            return {"success": False, "message": "摄像头不存在"}

        return {
            "success": True,
            "camera": self.cameras[camera_name]
        }

    def list_cameras(self) -> List[Dict[str, Any]]:
        """列出所有摄像头"""
        return [{"name": k, **v} for k, v in self.cameras.items()]


# 全局实例
inference_service = InferenceService()
monitor_service = MonitorService()
