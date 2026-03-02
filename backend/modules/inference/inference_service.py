# -*- coding: utf-8 -*-
"""
推理服务模块 - Inference Service
提供图片推理、视频推理、批量推理、流式推理等功能
"""
import cv2
import time
import logging
import base64
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 尝试导入 YOLO 模型
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    ULTRALYTICS_AVAILABLE = False
    logging.warning(f"[推理] Ultralytics 未安装: {e}")

# 导入配置和工具函数
from backend.core.config import settings
from backend.core.utils import save_uploaded_file, get_unique_filename

# 创建日志记录器
logger = logging.getLogger(__name__)


class InferenceService:
    """
    推理服务类
    提供统一的推理接口，支持图片、视频、批量推理
    """

    def __init__(self):
        """初始化推理服务"""
        logger.info("[推理] 初始化推理服务")
        self.model_cache: Dict[str, YOLO] = {}  # 模型缓存字典

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
        单张图片推理
        Args:
            image_path: 图片文件路径
            model_name: 模型名称（可选）
            confidence: 置信度阈值（可选）
            iou_threshold: IOU 阈值（可选）
            img_size: 输入图片尺寸（可选）
            draw_results: 是否绘制检测结果
        Returns:
            Dict: 推理结果，包含检测框、置信度等信息
        """
        # 导入 YOLO 引擎
        from backend.core.yolo_engine import yolo_engine

        logger.info(f"[推理] 开始图片推理: {image_path}, 模型: {model_name}, 绘制结果: {draw_results}")
        start_time = time.time()  # 记录推理开始时间

        # 调用 YOLO 引擎进行推理
        logger.debug(f"[推理] 调用 YOLO 引擎进行推理")
        result = yolo_engine.infer(
            image_path=image_path,
            model_identifier=model_name,
            confidence=confidence,
            iou_threshold=iou_threshold,
            img_size=img_size
        )

        # 计算推理耗时
        inference_time = time.time() - start_time

        # 如果需要绘制检测结果
        print(f"[推理] draw_results={draw_results}, result.success={result.get('success')}")  # 调试
        logger.info(f"[推理] draw_results={draw_results}, result.success={result.get('success')}")
        if draw_results and result.get("success"):
            print(f"[推理] 开始绘制检测结果图片, detections数量: {len(result.get('detections', []))}")  # 调试
            logger.info(f"[推理] 开始绘制检测结果图片, detections数量: {len(result.get('detections', []))}")

            # 使用 ultralytics 自带的 plot 方法来生成标注图片
            try:
                import numpy as np
                # 重新加载模型进行推理并获取带标注的图片
                model = yolo_engine.load_model(model_name)
                img = cv2.imread(image_path)
                if img is not None:
                    # 执行推理
                    results = model.predict(img, conf=confidence, verbose=False)
                    if results and len(results) > 0:
                        # 使用 plot 方法生成标注图片
                        annotated_img = results[0].plot()
                        # 转换为 base64
                        _, buffer = cv2.imencode('.jpg', annotated_img)
                        annotated_image = base64.b64encode(buffer).decode('utf-8')
                        annotated_image = f"data:image/jpeg;base64,{annotated_image}"
                        print(f"[推理] 绘制完成, annotated_image长度: {len(annotated_image)}")  # 调试
                        logger.info(f"[推理] 绘制完成, annotated_image长度: {len(annotated_image)}")
                        result["annotated_image"] = annotated_image
                    else:
                        print("[推理] 没有检测结果")  # 调试
                else:
                    print(f"[推理] 无法读取图片: {image_path}")  # 调试
            except Exception as e:
                print(f"[推理] 绘制失败: {e}")  # 调试
                logger.error(f"[推理] 绘制标注图片失败: {e}")
                # 如果绘制失败，使用原来的方法
                annotated_image = self._draw_detections(image_path, result["detections"])
                result["annotated_image"] = annotated_image
        else:
            print(f"[推理] 跳过绘制: draw_results={draw_results}, success={result.get('success')}")  # 调试
            logger.warning(f"[推理] 跳过绘制: draw_results={draw_results}, success={result.get('success')}")

        # 添加推理时间到结果
        result["inference_time"] = inference_time

        # 记录推理完成日志
        num_detections = len(result.get("detections", []))
        logger.info(f"[推理] 图片推理完成: {num_detections} 个检测, 耗时 {inference_time:.2f}s")

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
        视频推理（逐帧处理）
        Args:
            video_path: 视频文件路径
            model_name: 模型名称
            confidence: 置信度阈值
            output_path: 输出视频路径（可选）
            save_output: 是否保存标注后的视频
        Returns:
            Dict: 推理统计信息
        """
        # 导入 YOLO 引擎
        from backend.core.yolo_engine import yolo_engine

        logger.info(f"[推理] 开始视频推理: {video_path}")

        # 加载模型
        model = yolo_engine.load_model(model_name)

        # 使用默认置信度
        confidence = confidence or settings.CONFIDENCE_THRESHOLD

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            error_msg = f"无法打开视频: {video_path}"
            logger.error(f"[推理] {error_msg}")
            return {"success": False, "message": error_msg}

        # 获取视频信息
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"[推理] 视频信息: {video_width}x{video_height}, {video_fps} FPS")

        # 初始化视频写入器（如果需要保存输出）
        writer = None
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
            writer = cv2.VideoWriter(output_path, fourcc, video_fps, (video_width, video_height))
            logger.info(f"[推理] 输出视频将保存至: {output_path}")

        # 初始化统计变量
        frame_count = 0
        total_detections = 0
        results_per_frame = []

        # 逐帧读取视频
        while True:
            success, frame = cap.read()
            if not success:
                break  # 视频结束

            # 使用模型进行推理
            result = model.predict(frame, conf=confidence, verbose=False)

            # 处理检测结果
            if result and len(result) > 0:
                boxes = result[0].boxes
                detections = []

                # 遍历所有检测框
                for i in range(len(boxes)):
                    detections.append({
                        "class_id": int(boxes[i].cls[0]),  # 类别 ID
                        "class_name": result[0].names[int(boxes[i].cls[0])],  # 类别名称
                        "confidence": float(boxes[i].conf[0]),  # 置信度
                        "bbox": boxes[i].xyxy[0].tolist()  # 边界框坐标
                    })

                total_detections += len(detections)
                frame_count += 1

                # 绘制检测框到当前帧
                annotated_frame = result[0].plot()

                # 如果需要保存视频，写入当前帧
                if writer:
                    writer.write(annotated_frame)

                # 保存每帧的检测结果（限制返回数量）
                if len(results_per_frame) < 100:
                    results_per_frame.append({
                        "frame": frame_count,
                        "detections": detections
                    })

        # 释放资源
        cap.release()
        if writer:
            writer.release()
            logger.info(f"[推理] 输出视频已保存: {output_path}")

        # 返回统计结果
        logger.info(f"[推理] 视频推理完成: {frame_count} 帧, {total_detections} 个检测")

        return {
            "success": True,
            "message": "视频推理完成",
            "total_frames": frame_count,
            "total_detections": total_detections,
            "avg_detections_per_frame": total_detections / max(frame_count, 1),
            "output_path": output_path if save_output else None,
            "results": results_per_frame
        }

    def batch_infer(
        self,
        image_paths: List[str],
        model_name: str = None,
        confidence: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        批量推理（多张图片）
        Args:
            image_paths: 图片路径列表
            model_name: 模型名称
            confidence: 置信度阈值
        Returns:
            Dict: 批量推理结果
        """
        # 导入 YOLO 引擎
        from backend.core.yolo_engine import yolo_engine

        num_images = len(image_paths)
        logger.info(f"[推理] 开始批量推理: {num_images} 张图片")

        # 加载模型
        model = yolo_engine.load_model(model_name)

        # 使用默认置信度
        confidence = confidence or settings.CONFIDENCE_THRESHOLD

        # 初始化结果列表
        results = []
        total_time = 0
        total_detections = 0

        # 遍历每张图片进行推理
        for img_path in image_paths:
            start = time.time()

            # 执行推理
            result = model.predict(img_path, conf=confidence, verbose=False)
            elapsed = time.time() - start

            # 处理检测结果
            detections = []
            if result and len(result) > 0:
                boxes = result[0].boxes
                for i in range(len(boxes)):
                    detections.append({
                        "class_id": int(boxes[i].cls[0]),
                        "class_name": result[0].names[int(boxes[i].cls[0])],
                        "confidence": float(boxes[i].conf[0])
                    })

            # 保存单张图片的结果
            results.append({
                "image": img_path,
                "detections": detections,
                "time": elapsed
            })

            total_time += elapsed
            total_detections += len(detections)

        # 返回批量推理统计结果
        logger.info(f"[推理] 批量推理完成: {num_images} 张图片, "
                   f"{total_detections} 个检测, 总耗时 {total_time:.2f}s")

        return {
            "success": True,
            "message": f"处理了 {num_images} 张图片",
            "total_images": num_images,
            "total_time": total_time,
            "avg_time_per_image": total_time / num_images,
            "total_detections": total_detections,
            "results": results
        }

    def _draw_detections(
        self,
        image_path: str,
        detections: List[Dict]
    ) -> str:
        """
        在图片上绘制检测框（私有方法）
        Args:
            image_path: 图片路径
            detections: 检测结果列表
        Returns:
            str: Base64 编码的标注图片
        """
        # 读取原始图片
        img = cv2.imread(image_path)

        if img is None:
            logger.warning(f"[推理] 无法读取图片: {image_path}")
            return ""

        logger.info(f"[推理] 绘制检测框: {len(detections)} 个, 图片尺寸: {img.shape}")

        # 遍历所有检测结果绘制框和标签
        for det in detections:
            # 获取边界框坐标
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # 获取类别名称和置信度
            class_name = det["class_name"]
            confidence = det["confidence"]

            # 绘制绿色边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签（类别名 + 置信度）
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 将图片转换为 Base64 编码
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        logger.debug(f"[推理] 绘制完成: {len(detections)} 个检测框")

        # 返回 Data URL 格式的 Base64 图片
        return f"data:image/jpeg;base64,{img_base64}"

    def stream_infer(self, image_data: bytes, model_name: str = None, confidence: float = None):
        """
        流式推理（用于实时 API 调用）
        Args:
            image_data: 图片二进制数据
            model_name: 模型名称
            confidence: 置信度阈值
        Returns:
            Dict: 检测结果
        """
        # 导入 YOLO 引擎和 numpy
        from backend.core.yolo_engine import yolo_engine
        import numpy as np

        logger.debug(f"[推理] 流式推理开始, 数据大小: {len(image_data)} bytes")

        # 将二进制数据解码为图片
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 检查图片解码是否成功
        if image is None:
            error_msg = "图片解码失败"
            logger.error(f"[推理] {error_msg}")
            return {"success": False, "message": error_msg}

        # 加载模型
        model = yolo_engine.load_model(model_name)

        # 使用默认置信度
        conf = confidence if confidence is not None else settings.CONFIDENCE_THRESHOLD

        # 记录推理开始时间
        start_time = time.time()

        # 执行推理
        result = model.predict(image, conf=conf, verbose=False)

        # 计算推理时间
        inference_time = time.time() - start_time

        # 处理检测结果
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

        # 获取图片尺寸
        image_shape = image.shape if len(image.shape) == 3 else [image.shape[0], image.shape[1], 1]

        # 记录完成日志
        logger.info(f"[推理] 流式推理完成: {len(detections)} 个检测, 耗时 {inference_time:.3f}s")

        return {
            "success": True,
            "detections": detections,
            "num_detections": len(detections),
            "inference_time": inference_time,
            "image_shape": image_shape
        }


class MonitorService:
    """
    监控服务类
    管理摄像头监控相关的功能
    """

    def __init__(self):
        """初始化监控服务"""
        logger.info("[监控] 初始化监控服务")
        self.cameras: Dict[str, Dict] = {}  # 摄像头配置字典
        self.streams: Dict[str, Any] = {}  # 视频流字典

    def add_camera(
        self,
        name: str,
        url: str,
        model_name: str = None,
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        添加监控摄像头
        Args:
            name: 摄像头名称
            url: 视频流地址
            model_name: 使用的模型
            confidence: 置信度阈值
        Returns:
            Dict: 操作结果
        """
        # 保存摄像头配置
        self.cameras[name] = {
            "url": url,
            "model_name": model_name,
            "confidence": confidence,
            "status": "offline",  # 初始状态为离线
            "added_at": datetime.now().isoformat()
        }

        logger.info(f"[监控] 添加摄像头: {name}, URL: {url}")

        return {"success": True, "message": "摄像头已添加", "camera": self.cameras[name]}

    def start_stream(self, camera_name: str) -> Dict[str, Any]:
        """
        开始视频流推理
        Args:
            camera_name: 摄像头名称
        Returns:
            Dict: 操作结果
        """
        # 检查摄像头是否存在
        if camera_name not in self.cameras:
            error_msg = f"摄像头不存在: {camera_name}"
            logger.error(f"[监控] {error_msg}")
            return {"success": False, "message": error_msg}

        # TODO: 实现视频流推理
        logger.warning(f"[监控] 视频流推理功能开发中: {camera_name}")

        return {
            "success": True,
            "message": "视频流推理功能开发中",
            "camera": camera_name
        }

    def get_camera_status(self, camera_name: str) -> Dict[str, Any]:
        """
        获取摄像头状态
        Args:
            camera_name: 摄像头名称
        Returns:
            Dict: 摄像头信息
        """
        if camera_name not in self.cameras:
            return {"success": False, "message": "摄像头不存在"}

        return {
            "success": True,
            "camera": self.cameras[camera_name]
        }

    def list_cameras(self) -> List[Dict[str, Any]]:
        """
        列出所有摄像头
        Returns:
            List[Dict]: 摄像头列表
        """
        return [{"name": k, **v} for k, v in self.cameras.items()]


# 创建全局推理服务实例
inference_service = InferenceService()
monitor_service = MonitorService()

logger.info("[推理] 推理服务模块加载完成")
