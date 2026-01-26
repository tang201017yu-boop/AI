"""
解决方案服务 - Solutions Service
提供 Ultralytics Solutions 功能（独立模块）
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from ultralytics import YOLO, solutions
    from ultralytics.utils.plotting import Annotator, colors
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed")

from backend.core.config import settings


def _to_numpy(frame):
    """将帧转换为 numpy uint8 数组（兼容 MPS/tensor）"""
    if frame is None:
        return None

    # 如果已经是 numpy 数组
    if isinstance(frame, np.ndarray):
        # 确保是 uint8
        if frame.dtype != np.uint8:
            if frame.dtype in [np.float32, np.float64]:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        return frame

    # PyTorch tensor 或 MPS array
    if hasattr(frame, 'cpu'):
        frame = frame.cpu().numpy()
    elif hasattr(frame, 'numpy'):
        frame = frame.numpy()
    else:
        try:
            frame = np.array(frame)
        except:
            return None

    # 确保是 uint8
    if frame.dtype != np.uint8:
        if frame.dtype in [np.float32, np.float64]:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    return frame


def get_model_path(model: 'YOLO') -> str:
    """获取模型路径"""
    if hasattr(model, 'ckpt_path') and model.ckpt_path:
        return str(model.ckpt_path)
    return getattr(model, 'model_name', 'yolo11n.pt')


class SolutionsService:
    """Ultralytics Solutions 服务类 - 独立模块"""

    # 解决方案列表
    SOLUTIONS = {
        "object-counting": {
            "name": "对象计数",
            "description": "统计进出指定区域的对象数量",
            "input_types": ["image", "video"],
            "features": ["区域计数", "进出统计", "分类计数"]
        },
        "heatmap": {
            "name": "热图生成",
            "description": "可视化检测密度，显示热点区域",
            "input_types": ["image", "video"],
            "features": ["密度可视化", "热点分析"]
        },
        "speed-estimation": {
            "name": "速度估算",
            "description": "计算移动对象的速度",
            "input_types": ["video"],
            "features": ["实时测速", "速度统计"]
        },
        "distance-calculation": {
            "name": "距离计算",
            "description": "测量对象之间的像素距离",
            "input_types": ["image"],
            "features": ["对象间距", "距离标注"]
        },
        "object-blur": {
            "name": "对象模糊",
            "description": "对检测对象进行模糊处理",
            "input_types": ["image", "video"],
            "features": ["隐私保护", "人脸模糊"]
        },
        "object-crop": {
            "name": "对象裁剪",
            "description": "提取并裁剪检测对象",
            "input_types": ["image"],
            "features": ["自动裁剪", "批量提取"]
        },
        "queue-management": {
            "name": "队列管理",
            "description": "监控队列长度和等待时间",
            "input_types": ["video"],
            "features": ["队列计数", "流量分析"]
        }
    }

    def __init__(self):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics YOLO is not installed")

        self.models: Dict[str, YOLO] = {}

    def load_model(self, model_name: str = None) -> 'YOLO':
        """加载模型"""
        from backend.core.yolo_engine import yolo_engine
        return yolo_engine.load_model(model_name)

    def list_solutions(self) -> List[Dict[str, Any]]:
        """列出所有解决方案"""
        solutions_list = []
        for name, info in self.SOLUTIONS.items():
            solutions_list.append({
                "name": name,
                "title": info["name"],
                "description": info["description"],
                "input_types": info["input_types"],
                "features": info["features"]
            })
        return solutions_list

    # ==================== 对象计数 ====================
    def object_counting(
        self,
        source: str,
        model_name: str = None,
        region_points: List[Tuple] = None,
        show_in: bool = True,
        show_out: bool = True,
        classes: List[int] = None,
        conf: float = 0.25,
        output_path: str = None
    ) -> Dict[str, Any]:
        """对象计数"""
        try:
            model = self.load_model(model_name)
            model_path = get_model_path(model)

            if region_points is None:
                region_points = [(20, 400), (1260, 400), (1260, 360), (20, 360)]

            counter = solutions.ObjectCounter(
                show=False,
                region=region_points,
                model=model_path,
                classes=classes,
                show_in=show_in,
                show_out=show_out,
                line_width=2
            )

            cap = cv2.VideoCapture(source)
            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            results_data = {"in_count": 0, "out_count": 0, "total_frames": 0}

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                result = counter(frame)
                results_data["total_frames"] += 1

                if hasattr(result, 'in_count'):
                    results_data["in_count"] = result.in_count
                if hasattr(result, 'out_count'):
                    results_data["out_count"] = result.out_count

                if output_path and hasattr(result, 'plot_im'):
                    out.write(result.plot_im)

            cap.release()
            if output_path:
                out.release()

            return {
                "success": True,
                "message": "对象计数完成",
                "results": results_data,
                "output_path": output_path
            }
        except Exception as e:
            return {"success": False, "message": f"对象计数失败: {str(e)}"}

    # ==================== 热图生成 ====================
    def generate_heatmap(
        self,
        source: str,
        model_name: str = None,
        colormap: int = cv2.COLORMAP_JET,
        classes: List[int] = None,
        conf: float = 0.25,
        output_path: str = None,
        progress_callback: callable = None
    ) -> Dict[str, Any]:
        """生成热图"""
        try:
            import scipy.ndimage as ndi

            model = self.load_model(model_name)
            model_path = get_model_path(model)

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return {"success": False, "message": "无法打开视频文件"}

            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 100  # 默认估计值

            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            # 初始化热图累计器（2D 数组）
            heatmap_2d = np.zeros((height, width), dtype=np.float32)

            frame_count = 0
            while True:
                success, orig_frame = cap.read()
                if not success:
                    break

                # 确保帧是 numpy uint8
                frame = _to_numpy(orig_frame)
                if frame is None:
                    continue

                # YOLO 推理获取检测结果
                results = model.predict(frame, conf=conf, classes=classes, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []

                # 累加热图（2D）
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    # 在检测框区域添加高斯热点
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    # 创建单个检测的热图
                    single_heatmap = np.zeros((height, width), dtype=np.float32)
                    single_heatmap[y1:y2, x1:x2] = 1.0

                    # 高斯模糊
                    single_heatmap = ndi.gaussian_filter(single_heatmap, sigma=15)

                    heatmap_2d += single_heatmap

                # 归一化并应用颜色映射
                if frame_count >= 0 and np.max(heatmap_2d) > 0:
                    heatmap_normalized = (heatmap_2d / np.max(heatmap_2d)).clip(0, 1)
                    heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), colormap)

                    # 混合原始帧和热图
                    alpha = 0.5
                    annotated_frame = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
                else:
                    annotated_frame = frame

                frame_count += 1

                # 更新进度
                if progress_callback:
                    progress = int((frame_count / total_frames) * 100)
                    progress_callback(progress, f"正在处理第 {frame_count}/{total_frames} 帧")

                if output_path and out is not None:
                    out.write(annotated_frame)

            cap.release()
            if output_path and out is not None:
                out.release()

            return {
                "success": True,
                "message": f"热图生成完成，共处理 {frame_count} 帧",
                "total_frames": frame_count,
                "output_path": output_path
            }
        except Exception as e:
            import traceback
            return {"success": False, "message": f"热图生成失败: {str(e)}\n{traceback.format_exc()}"}

    # ==================== 速度估算 ====================
    def estimate_speed(
        self,
        source: str,
        model_name: str = None,
        region_points: List[Tuple] = None,
        classes: List[int] = None,
        conf: float = 0.25,
        output_path: str = None
    ) -> Dict[str, Any]:
        """速度估算"""
        try:
            model = self.load_model(model_name)
            model_path = get_model_path(model)

            if region_points is None:
                region_points = [(20, 400), (1260, 400)]

            speed_estimator = solutions.SpeedEstimator(
                show=False,
                model=model_path,
                region=region_points,
                classes=classes,
                line_width=2
            )

            cap = cv2.VideoCapture(source)
            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            frame_count = 0
            speeds = []
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                result = speed_estimator(frame)
                frame_count += 1

                if hasattr(result, 'speed_dict'):
                    speeds.append(result.speed_dict)

                if output_path:
                    out.write(frame)

            cap.release()
            if output_path:
                out.release()

            return {
                "success": True,
                "message": "速度估算完成",
                "results": {"speeds": speeds, "frame_count": frame_count},
                "output_path": output_path
            }
        except Exception as e:
            return {"success": False, "message": f"速度估算失败: {str(e)}"}

    # ==================== 距离计算 ====================
    def calculate_distance(
        self,
        image_path: str,
        model_name: str = None,
        classes: List[int] = None,
        conf: float = 0.25
    ) -> Dict[str, Any]:
        """距离计算"""
        try:
            model = self.load_model(model_name)
            img = cv2.imread(image_path)

            results = model.predict(source=img, conf=conf, classes=classes, verbose=False)

            if len(results) == 0 or len(results[0].boxes) < 2:
                return {"success": False, "message": "需要至少检测到2个对象", "distances": []}

            boxes = results[0].boxes
            centroids = []
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cx = int((xyxy[0] + xyxy[2]) / 2)
                cy = int((xyxy[1] + xyxy[3]) / 2)
                centroids.append((cx, cy))

            distances = []
            annotator = Annotator(img, line_width=2)

            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    p1, p2 = centroids[i], centroids[j]
                    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    distances.append({
                        "object1_index": i,
                        "object2_index": j,
                        "pixel_distance": float(distance)
                    })
                    cv2.line(img, p1, p2, (0, 255, 0), 2)
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    cv2.putText(img, f"{distance:.1f}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            output_path = str(settings.UPLOADS_DIR / f"distance_{Path(image_path).name}")
            cv2.imwrite(output_path, img)

            return {
                "success": True,
                "message": "距离计算完成",
                "distances": distances,
                "output_image": output_path
            }
        except Exception as e:
            return {"success": False, "message": f"距离计算失败: {str(e)}"}

    # ==================== 对象模糊 ====================
    def blur_objects(
        self,
        source: str,
        model_name: str = None,
        classes: List[int] = None,
        conf: float = 0.25,
        blur_ratio: float = 50,
        output_path: str = None
    ) -> Dict[str, Any]:
        """对象模糊"""
        try:
            model = self.load_model(model_name)
            model_path = get_model_path(model)

            blur = solutions.ObjectBlur(
                show=False,
                model=model_path,
                classes=classes,
                blur_ratio=int(blur_ratio)
            )

            cap = cv2.VideoCapture(source)
            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                blur(frame)
                frame_count += 1

                if output_path:
                    out.write(frame)

            cap.release()
            if output_path:
                out.release()

            return {
                "success": True,
                "message": "对象模糊完成",
                "total_frames": frame_count,
                "output_path": output_path
            }
        except Exception as e:
            return {"success": False, "message": f"对象模糊失败: {str(e)}"}

    # ==================== 对象裁剪 ====================
    def crop_objects(
        self,
        image_path: str,
        model_name: str = None,
        classes: List[int] = None,
        conf: float = 0.25,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """对象裁剪"""
        try:
            model = self.load_model(model_name)
            img = cv2.imread(image_path)

            if output_dir is None:
                output_dir = str(settings.UPLOADS_DIR / "cropped-objects")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            results = model.predict(source=img, conf=conf, classes=classes, verbose=False)

            cropped_images = []
            if len(results) > 0:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    cropped = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    crop_filename = f"{class_name}_{i}_{Path(image_path).stem}.jpg"
                    crop_path = str(Path(output_dir) / crop_filename)
                    cv2.imwrite(crop_path, cropped)

                    cropped_images.append({
                        "class_name": class_name,
                        "class_id": cls_id,
                        "crop_path": crop_path,
                        "crop_size": list(cropped.shape[:2])
                    })

            return {
                "success": True,
                "message": "对象裁剪完成",
                "total_crops": len(cropped_images),
                "cropped_images": cropped_images,
                "output_dir": output_dir
            }
        except Exception as e:
            return {"success": False, "message": f"对象裁剪失败: {str(e)}"}

    # ==================== 队列管理 ====================
    def queue_management(
        self,
        source: str,
        model_name: str = None,
        region_points: List[Tuple] = None,
        classes: List[int] = None,
        conf: float = 0.25,
        output_path: str = None
    ) -> Dict[str, Any]:
        """队列管理"""
        try:
            model = self.load_model(model_name)
            model_path = get_model_path(model)

            if region_points is None:
                region_points = [(20, 400), (1260, 400), (1260, 360), (20, 360)]

            queue = solutions.QueueManager(
                show=False,
                model=model_path,
                region=region_points,
                classes=classes,
                line_width=2
            )

            cap = cv2.VideoCapture(source)
            if output_path:
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            queue_data = {"max_queue_count": 0, "frame_counts": [], "total_frames": 0}

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                result = queue(frame)
                queue_data["total_frames"] += 1

                if hasattr(result, 'queue_count'):
                    qc = result.queue_count
                    queue_data["frame_counts"].append(qc)
                    queue_data["max_queue_count"] = max(queue_data["max_queue_count"], qc)

                if output_path:
                    out.write(frame)

            cap.release()
            if output_path:
                out.release()

            if queue_data["frame_counts"]:
                queue_data["avg_queue_count"] = sum(queue_data["frame_counts"]) / len(queue_data["frame_counts"])

            return {
                "success": True,
                "message": "队列管理完成",
                "results": queue_data,
                "output_path": output_path
            }
        except Exception as e:
            return {"success": False, "message": f"队列管理失败: {str(e)}"}


# 全局实例
solutions_service = SolutionsService() if ULTRALYTICS_AVAILABLE else None
