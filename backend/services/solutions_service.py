"""
Ultralytics Solutions Service
 集成所有 Ultralytics YOLO 解决方案功能
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import time
import torch

try:
    from ultralytics import YOLO, solutions
    from ultralytics.utils.plotting import Annotator, colors
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

from config.config import settings


def get_model_path(model: 'YOLO') -> str:
    """获取模型的正确路径，用于 Solutions API"""
    # 尝试多种方式获取模型路径
    if hasattr(model, 'ckpt_path') and model.ckpt_path:
        return str(model.ckpt_path)
    if hasattr(model, 'model') and hasattr(model.model, 'yaml_path'):
        return str(model.model.yaml_path)
    # 默认返回模型名称
    return getattr(model, 'model_name', 'yolo11n.pt')


class SolutionsService:
    """Ultralytics Solutions 服务类"""
    
    def __init__(self):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics YOLO is not installed")
        
        self.models: Dict[str, YOLO] = {}
        
    def load_model(self, model_name: str = None) -> 'YOLO':
        """加载 YOLO 模型"""
        model_name = model_name or settings.DEFAULT_MODEL
        
        if model_name in self.models:
            return self.models[model_name]
        
        model_path = settings.MODELS_DIR / model_name
        
        if not model_path.exists():
            print(f"Downloading pretrained model: {model_name}")
            model = YOLO(model_name)
        else:
            model = YOLO(str(model_path))
        
        self.models[model_name] = model
        return model
    
    # ==================== 对象计数 (Object Counting) ====================
    def object_counting(
        self,
        source: str,
        model_name: str = None,
        region_points: List[Tuple[int, int]] = None,
        show_in: bool = True,
        show_out: bool = True,
        classes: List[int] = None,
        conf: float = 0.25,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        对象计数功能
        
        Args:
            source: 视频/图片路径
            model_name: 模型名称
            region_points: 计数区域的点列表 [(x1,y1), (x2,y2), ...]
            show_in: 是否显示进入计数
            show_out: 是否显示离开计数
            classes: 要计数的类别列表
            conf: 置信度阈值
            output_path: 输出路径
        
        Returns:
            计数结果和统计信息
        """
        try:
            model = self.load_model(model_name)
            
            # 默认区域（如果未指定）
            if region_points is None:
                region_points = [(20, 400), (1260, 400), (1260, 360), (20, 360)]
            
            # 初始化对象计数器
            model_path = get_model_path(model)
            counter = solutions.ObjectCounter(
                show=False,
                region=region_points,
                model=model_path,
                classes=classes,
                show_in=show_in,
                show_out=show_out,
                line_width=2
            )
            
            # 处理视频/图片
            cap = cv2.VideoCapture(source)
            
            # 准备输出
            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            results_data = {
                "in_count": 0,
                "out_count": 0,
                "classwise_count": {},
                "total_frames": 0
            }
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # 执行计数
                result = counter(frame)
                
                # 更新统计
                if hasattr(result, 'in_count'):
                    results_data["in_count"] = result.in_count
                if hasattr(result, 'out_count'):
                    results_data["out_count"] = result.out_count
                if hasattr(result, 'classwise_count'):
                    results_data["classwise_count"] = result.classwise_count
                
                results_data["total_frames"] += 1
                
                # 保存输出
                if output_path and hasattr(result, 'plot_im'):
                    out.write(result.plot_im)
            
            cap.release()
            if output_path:
                out.release()
            
            return {
                "success": True,
                "message": "Object counting completed",
                "results": results_data,
                "output_path": output_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Object counting failed: {str(e)}",
                "results": None
            }
    
    # ==================== 热图 (Heatmap) ====================
    def generate_heatmap(
        self,
        source: str,
        model_name: str = None,
        colormap: int = cv2.COLORMAP_JET,
        classes: List[int] = None,
        conf: float = 0.25,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        生成检测热图
        
        Args:
            source: 视频/图片路径
            model_name: 模型名称
            colormap: OpenCV 颜色映射
            classes: 要检测的类别
            conf: 置信度阈值
            output_path: 输出路径
        
        Returns:
            热图生成结果
        """
        try:
            model = self.load_model(model_name)
            
            # 初始化热图生成器
            model_path = get_model_path(model)
            heatmap = solutions.Heatmap(
                show=False,
                model=model_path,
                colormap=colormap,
                classes=classes,
                line_width=2
            )
            
            # 处理视频/图片
            cap = cv2.VideoCapture(source)
            
            # 准备输出
            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # 生成热图
                result = heatmap(frame)
                
                frame_count += 1
                
                # 保存输出
                if output_path and hasattr(result, 'plot_im'):
                    out.write(result.plot_im)
            
            cap.release()
            if output_path:
                out.release()
            
            return {
                "success": True,
                "message": "Heatmap generation completed",
                "total_frames": frame_count,
                "output_path": output_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Heatmap generation failed: {str(e)}",
                "total_frames": 0
            }
    
    # ==================== 速度估算 (Speed Estimation) ====================
    def estimate_speed(
        self,
        source: str,
        model_name: str = None,
        region_points: List[Tuple[int, int]] = None,
        classes: List[int] = None,
        conf: float = 0.25,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        估算对象速度
        
        Args:
            source: 视频路径
            model_name: 模型名称
            region_points: 速度检测区域
            classes: 要检测的类别
            conf: 置信度阈值
            output_path: 输出路径
        
        Returns:
            速度估算结果
        """
        try:
            model = self.load_model(model_name)
            
            # 默认区域
            if region_points is None:
                region_points = [(20, 400), (1260, 400)]
            
            # 初始化速度估算器
            model_path = get_model_path(model)
            speed_estimator = solutions.SpeedEstimator(
                show=False,
                model=model_path,
                region=region_points,
                classes=classes,
                line_width=2
            )
            
            # 处理视频
            cap = cv2.VideoCapture(source)
            
            # 准备输出
            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            speed_data = {
                "speeds": [],
                "frame_count": 0
            }
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # 估算速度
                result = speed_estimator(frame)
                
                # 收集速度数据
                if hasattr(result, 'speed_dict'):
                    speed_data["speeds"].append(result.speed_dict)
                
                speed_data["frame_count"] += 1
                
                # 保存输出
                if output_path and hasattr(result, 'plot_im'):
                    out.write(result.plot_im)
            
            cap.release()
            if output_path:
                out.release()
            
            return {
                "success": True,
                "message": "Speed estimation completed",
                "results": speed_data,
                "output_path": output_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Speed estimation failed: {str(e)}",
                "results": None
            }
    
    # ==================== 距离计算 (Distance Calculation) ====================
    def calculate_distance(
        self,
        image_path: str,
        model_name: str = None,
        classes: List[int] = None,
        conf: float = 0.25
    ) -> Dict[str, Any]:
        """
        计算对象之间的距离
        
        Args:
            image_path: 图片路径
            model_name: 模型名称
            classes: 要检测的类别
            conf: 置信度阈值
        
        Returns:
            距离计算结果
        """
        try:
            model = self.load_model(model_name)
            
            # 读取图像
            img = cv2.imread(image_path)
            
            # 执行检测
            results = model.predict(source=img, conf=conf, classes=classes, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) < 2:
                return {
                    "success": False,
                    "message": "需要至少检测到2个对象才能计算距离",
                    "distances": []
                }
            
            # 获取边界框中心点
            boxes = results[0].boxes
            centroids = []
            
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cx = int((xyxy[0] + xyxy[2]) / 2)
                cy = int((xyxy[1] + xyxy[3]) / 2)
                centroids.append((cx, cy))
            
            # 计算所有对象之间的距离
            distances = []
            annotator = Annotator(img, line_width=2)
            
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    p1 = centroids[i]
                    p2 = centroids[j]
                    
                    # 计算欧氏距离（像素）
                    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    
                    distances.append({
                        "object1_index": i,
                        "object2_index": j,
                        "pixel_distance": float(distance),
                        "centroid1": p1,
                        "centroid2": p2
                    })
                    
                    # 绘制距离线
                    cv2.line(img, p1, p2, (0, 255, 0), 2)
                    mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    cv2.putText(img, f"{distance:.1f}px", mid_point, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 保存结果图像
            output_path = str(settings.UPLOADS_DIR / f"distance_{Path(image_path).name}")
            cv2.imwrite(output_path, img)
            
            return {
                "success": True,
                "message": "Distance calculation completed",
                "distances": distances,
                "output_image": output_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Distance calculation failed: {str(e)}",
                "distances": []
            }
    
    # ==================== 对象模糊 (Object Blurring) ====================
    def blur_objects(
        self,
        source: str,
        model_name: str = None,
        classes: List[int] = None,
        conf: float = 0.25,
        blur_ratio: float = 50,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        对检测到的对象进行模糊处理（隐私保护）
        
        Args:
            source: 视频/图片路径
            model_name: 模型名称
            classes: 要模糊的类别
            conf: 置信度阈值
            blur_ratio: 模糊强度
            output_path: 输出路径
        
        Returns:
            模糊处理结果
        """
        try:
            model = self.load_model(model_name)
            
            # 初始化对象模糊器
            model_path = get_model_path(model)
            blur = solutions.ObjectBlur(
                show=False,
                model=model_path,
                classes=classes,
                blur_ratio=int(blur_ratio)
            )
            
            # 处理视频/图片
            cap = cv2.VideoCapture(source)
            
            # 准备输出
            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            blurred_objects = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # 模糊对象
                result = blur(frame)
                
                frame_count += 1
                
                # 保存输出
                if output_path and hasattr(result, 'plot_im'):
                    out.write(result.plot_im)
            
            cap.release()
            if output_path:
                out.release()
            
            return {
                "success": True,
                "message": "Object blurring completed",
                "total_frames": frame_count,
                "output_path": output_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Object blurring failed: {str(e)}",
                "total_frames": 0
            }
    
    # ==================== 对象裁剪 (Object Cropping) ====================
    def crop_objects(
        self,
        image_path: str,
        model_name: str = None,
        classes: List[int] = None,
        conf: float = 0.25,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        裁剪检测到的对象
        
        Args:
            image_path: 图片路径
            model_name: 模型名称
            classes: 要裁剪的类别
            conf: 置信度阈值
            output_dir: 输出目录
        
        Returns:
            裁剪结果
        """
        try:
            model = self.load_model(model_name)
            
            # 设置输出目录
            if output_dir is None:
                output_dir = str(settings.UPLOADS_DIR / "cropped-objects")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # 读取图像
            img = cv2.imread(image_path)
            
            # 执行检测
            results = model.predict(source=img, conf=conf, classes=classes, verbose=False)
            
            cropped_images = []
            
            if len(results) > 0:
                boxes = results[0].boxes
                
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    
                    # 裁剪对象
                    cropped = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    
                    # 保存裁剪图像
                    crop_filename = f"{class_name}_{i}_{Path(image_path).stem}.jpg"
                    crop_path = str(Path(output_dir) / crop_filename)
                    cv2.imwrite(crop_path, cropped)
                    
                    cropped_images.append({
                        "class_name": class_name,
                        "class_id": cls_id,
                        "bbox": xyxy.tolist(),
                        "crop_path": crop_path,
                        "crop_size": cropped.shape[:2]
                    })
            
            return {
                "success": True,
                "message": "Object cropping completed",
                "total_crops": len(cropped_images),
                "cropped_images": cropped_images,
                "output_dir": output_dir
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Object cropping failed: {str(e)}",
                "cropped_images": []
            }
    
    # ==================== 队列管理 (Queue Management) ====================
    def queue_management(
        self,
        source: str,
        model_name: str = None,
        region_points: List[Tuple[int, int]] = None,
        classes: List[int] = None,
        conf: float = 0.25,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        队列管理功能
        
        Args:
            source: 视频路径
            model_name: 模型名称
            region_points: 队列区域
            classes: 要计数的类别
            conf: 置信度阈值
            output_path: 输出路径
        
        Returns:
            队列管理结果
        """
        try:
            model = self.load_model(model_name)
            
            # 默认队列区域
            if region_points is None:
                region_points = [(20, 400), (1260, 400), (1260, 360), (20, 360)]
            
            # 初始化队列管理器
            model_path = get_model_path(model)
            queue = solutions.QueueManager(
                show=False,
                model=model_path,
                region=region_points,
                classes=classes,
                line_width=2
            )
            
            # 处理视频
            cap = cv2.VideoCapture(source)
            
            # 准备输出
            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            queue_data = {
                "max_queue_count": 0,
                "frame_counts": [],
                "total_frames": 0
            }
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # 队列管理
                result = queue(frame)
                
                # 收集队列数据
                if hasattr(result, 'queue_count'):
                    queue_count = result.queue_count
                    queue_data["frame_counts"].append(queue_count)
                    queue_data["max_queue_count"] = max(queue_data["max_queue_count"], queue_count)
                
                queue_data["total_frames"] += 1
                
                # 保存输出
                if output_path and hasattr(result, 'plot_im'):
                    out.write(result.plot_im)
            
            cap.release()
            if output_path:
                out.release()
            
            # 计算平均队列长度
            if queue_data["frame_counts"]:
                queue_data["avg_queue_count"] = sum(queue_data["frame_counts"]) / len(queue_data["frame_counts"])
            else:
                queue_data["avg_queue_count"] = 0
            
            return {
                "success": True,
                "message": "Queue management completed",
                "results": queue_data,
                "output_path": output_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Queue management failed: {str(e)}",
                "results": None
            }


# 全局服务实例
solutions_service = SolutionsService() if ULTRALYTICS_AVAILABLE else None
