"""
解决方案服务 - Solutions Service
提供 Ultralytics Solutions 功能（独立模块）
"""
import os
import re
import shutil
import subprocess
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 创建日志记录器
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO, solutions
    from ultralytics.utils.plotting import Annotator, colors
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed")

from backend.core.config import settings


def _open_solution_mp4_writer(path: str, fps: float, width: int, height: int):
    """
    创建 MP4 写入器。优先尝试浏览器可解码的 H.264 fourcc，否则回退 mp4v。
    说明：多数浏览器无法播放 OpenCV 默认的 MPEG-4 Part 2（mp4v），需 H.264 或事后 remux。
    """
    w, h = int(width), int(height)
    fps_f = float(fps) if fps and float(fps) > 0 else 30.0
    p = str(path)
    for tag in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        vw = cv2.VideoWriter(p, fourcc, fps_f, (w, h))
        if vw.isOpened():
            if tag == "mp4v":
                logger.warning(
                    "[video] OpenCV 仅能以 mp4v 打开写入器，浏览器内可能无法播放；"
                    "请安装 ffmpeg 以便自动转码为 H.264，或换用带 FFmpeg 的 OpenCV 构建"
                )
            else:
                logger.info("[video] VideoWriter fourcc=%s path=%s", tag, Path(p).name)
            return vw
        vw.release()
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps_f, (w, h))
    if not vw.isOpened():
        logger.error("[video] 无法为 %s 创建 VideoWriter", Path(p).name)
    return vw


def _resolve_ffmpeg_executable() -> Optional[str]:
    """
    解析用于转码的 ffmpeg 可执行文件路径：
    1) 环境变量 FFMPEG_PATH
    2) PATH 中的 ffmpeg
    3) imageio-ffmpeg 自带的二进制（无需在服务器上单独安装 ffmpeg）
    """
    env_p = os.environ.get("FFMPEG_PATH", "").strip()
    if env_p and Path(env_p).is_file():
        return env_p
    which = shutil.which("ffmpeg")
    if which:
        return which
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception as e:
        logger.debug("[video] imageio-ffmpeg 不可用: %s", e)
    return None


def _remux_mp4_for_html5_video(path: Optional[str]) -> None:
    """
    将已写好的 MP4 用 ffmpeg 转为 H.264 + yuv420p + faststart，便于 <video> 标签播放。
    若既无系统 ffmpeg 也未安装 imageio-ffmpeg，则静默跳过。
    """
    if not path or Path(path).suffix.lower() != ".mp4":
        return
    ffmpeg = _resolve_ffmpeg_executable()
    if not ffmpeg:
        logger.warning(
            "[video] 未找到 ffmpeg（请 pip install imageio-ffmpeg 或安装系统 ffmpeg），"
            "输出可能无法在浏览器内播放: %s",
            Path(path).name,
        )
        return
    src = Path(path)
    if not src.is_file() or src.stat().st_size == 0:
        return
    tmp = src.parent / f".{src.stem}.html5_tmp{src.suffix}"
    try:
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp),
        ]
        subprocess.run(cmd, check=True, timeout=7200)
        os.replace(str(tmp), str(src))
        logger.info("[video] ffmpeg remux OK: %s", src.name)
    except Exception as e:
        logger.warning("[video] ffmpeg remux failed for %s: %s", path, e)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


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
        },
        "parking-management": {
            "name": "停车管理",
            "description": "统计停车位占用与空闲数量",
            "input_types": ["image", "video"],
            "features": ["车位占用", "空闲统计", "区域告警"]
        }
    }

    def __init__(self):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics YOLO is not installed")

        self.models: Dict[str, YOLO] = {}
        # 检查 GPU 兼容性（实际测试而非版本号判断）
        import torch
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                self.default_device = "0"
            except Exception:
                self.default_device = "cpu"
                logger.warning("[解决方案] GPU 不兼容当前 PyTorch，使用 CPU")
        else:
            self.default_device = "cpu"
        logger.info(f"[解决方案] 默认设备: {self.default_device}")

    def load_model(self, model_name: str = None) -> 'YOLO':
        """加载模型，优先使用 GPU"""
        from backend.core.yolo_engine import yolo_engine
        # 优先使用 GPU
        return yolo_engine.load_model(model_name, device=self.default_device)

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

    @staticmethod
    def _normalize_region_points(region_points) -> Optional[List[Tuple[int, int]]]:
        """将 JSON 解析后的区域转为 (x,y) 元组列表。"""
        if not region_points:
            return None
        out: List[Tuple[int, int]] = []
        for p in region_points:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((int(float(p[0])), int(float(p[1]))))
        return out if out else None

    @staticmethod
    def _default_counting_region(width: int, height: int, region_type: str) -> List[Tuple[int, int]]:
        """按画面尺寸生成默认计数区域（避免固定 1280×720 导致小图/竖图完全错位）。"""
        margin_x = max(2, int(width * 0.02))
        margin_y = max(2, int(height * 0.02))
        if region_type == "line":
            y = max(margin_y, int(height * 0.55))
            return [(margin_x, y), (width - margin_x, y)]
        return [
            (margin_x, int(height * 0.45)),
            (width - margin_x, int(height * 0.45)),
            (width - margin_x, height - margin_y),
            (margin_x, height - margin_y),
        ]

    @staticmethod
    def _class_counts_from_counter(counter) -> Dict[str, int]:
        """当前帧各类检测数量（与 Ultralytics 终端摘要一致，按类名排序输出）。"""
        cls_counts: Dict[str, int] = {}
        for cls_id in getattr(counter, "clss", []) or []:
            try:
                idx = int(cls_id)
                names = getattr(counter, "names", None)
                if isinstance(names, (list, tuple)) and idx < len(names):
                    cls_name = str(names[idx])
                elif isinstance(names, dict):
                    cls_name = str(names.get(idx, idx))
                else:
                    cls_name = str(idx)
            except Exception:
                cls_name = str(cls_id)
            cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1
        return cls_counts

    @staticmethod
    def _format_class_summary_line(cls_counts: Dict[str, int]) -> str:
        """形如: 1 bus, 6 car, 1 motorcycle, 1 person（按类名字母序）。"""
        if not cls_counts:
            return "no objects"
        parts = [f"{cls_counts[k]} {k}" for k in sorted(cls_counts.keys())]
        return ", ".join(parts)

    # ==================== 对象计数 ====================
    def object_counting(
        self,
        source: str,
        model_name: str = None,
        region_type: str = "polygon",
        region_points: List[Tuple] = None,
        show_in: bool = True,
        show_out: bool = True,
        classes: List[int] = None,
        conf: float = 0.25,
        line_width: int = 2,
        output_path: str = None
    ) -> Dict[str, Any]:
        """对象计数 - 支持区域计数和分类统计。

        说明：Ultralytics ObjectCounter 的进出计数依赖连续帧轨迹；单张图片上 in/out 通常为 0。
        对图片会额外返回 detected_objects（当前画面跟踪到的目标数），便于静态图查看检测效果。
        """
        try:
            model = self.load_model(model_name)
            model_path = get_model_path(model)

            region_points = self._normalize_region_points(region_points)
            src_path = Path(source)
            suffix = src_path.suffix.lower()
            is_image = suffix in (
                ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
            )

            # ---------- 图片：imread，单帧处理，结果图用 imwrite ----------
            if is_image:
                frame = cv2.imread(str(source))
                if frame is None:
                    return {"success": False, "message": "无法读取图片文件，请确认格式正确"}
                h, w = frame.shape[:2]
                if region_points is None:
                    region_points = self._default_counting_region(w, h, region_type)

                counter = solutions.ObjectCounter(
                    show=False,
                    region=region_points,
                    model=model_path,
                    classes=classes,
                    show_in=show_in,
                    show_out=show_out,
                    line_width=line_width,
                    device=self.default_device,
                    conf=conf,
                )
                result = counter(frame)
                in_c = getattr(result, "in_count", 0)
                out_c = getattr(result, "out_count", 0)
                total_tracks = getattr(result, "total_tracks", 0)
                cls_counts = self._class_counts_from_counter(counter)
                cls_summary = self._format_class_summary_line(cls_counts)
                speed_data = getattr(result, "speed", {}) or {}
                sol_ms = float(speed_data.get("solution", 0.0))
                tr_ms = float(speed_data.get("track", 0.0))
                results_data = {
                    "in_count": in_c,
                    "out_count": out_c,
                    "total_frames": 1,
                    "detected_objects": total_tracks,
                    "objects_by_class": cls_counts,
                    "recent_inference_logs": [
                        f"1: {h}x{w} {sol_ms:.1f}ms, {cls_summary} | track {tr_ms:.1f}ms"
                    ],
                }
                if output_path and getattr(result, "plot_im", None) is not None:
                    cv2.imwrite(output_path, result.plot_im)

                return {
                    "success": True,
                    "message": "对象计数完成（静态图仅显示检测与区域；进出计数需视频连续帧）",
                    "results": results_data,
                    "output_path": output_path,
                }

            # ---------- 视频：首帧确定默认区域，再顺序处理（避免 VideoCapture 无法回绕） ----------
            cap = cv2.VideoCapture(source)
            ok, first = cap.read()
            if not ok or first is None:
                cap.release()
                return {"success": False, "message": "无法打开视频或读取首帧"}

            fh, fw = first.shape[:2]
            if region_points is None:
                region_points = self._default_counting_region(fw, fh, region_type)

            counter = solutions.ObjectCounter(
                show=False,
                region=region_points,
                model=model_path,
                classes=classes,
                show_in=show_in,
                show_out=show_out,
                line_width=line_width,
                device=self.default_device,
                conf=conf,
            )

            out = None
            if output_path:
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or fw
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or fh
                out = _open_solution_mp4_writer(output_path, fps, width, height)

            results_data = {"in_count": 0, "out_count": 0, "total_frames": 0}
            last_tracks = 0
            last_class_counts: Dict[str, int] = {}
            recent_logs: List[str] = []

            def _process_frame(frame_bgr):
                nonlocal last_tracks, last_class_counts
                r = counter(frame_bgr)
                results_data["total_frames"] += 1
                if hasattr(r, "in_count"):
                    results_data["in_count"] = r.in_count
                if hasattr(r, "out_count"):
                    results_data["out_count"] = r.out_count
                if hasattr(r, "total_tracks"):
                    last_tracks = r.total_tracks
                if output_path and out is not None and out.isOpened() and getattr(r, "plot_im", None) is not None:
                    out.write(r.plot_im)

                # 组织前端可展示的推理摘要（与 Ultralytics 终端: 帧号 分辨率 solution_ms, 各类数量）
                fh_i, fw_i = frame_bgr.shape[:2]
                cls_counts = self._class_counts_from_counter(counter)
                last_class_counts = dict(cls_counts)
                cls_summary = self._format_class_summary_line(cls_counts)
                speed_data = getattr(r, "speed", {}) or {}
                sol_ms = float(speed_data.get("solution", 0.0))
                tr_ms = float(speed_data.get("track", 0.0))
                log_line = (
                    f"{results_data['total_frames']}: {fh_i}x{fw_i} {sol_ms:.1f}ms, {cls_summary} | track {tr_ms:.1f}ms"
                )
                recent_logs.append(log_line)
                if len(recent_logs) > 20:
                    recent_logs.pop(0)
                return r

            _process_frame(first)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                _process_frame(frame)

            cap.release()
            if out is not None:
                out.release()
            if output_path:
                _remux_mp4_for_html5_video(output_path)

            results_data["detected_objects"] = last_tracks
            results_data["objects_by_class"] = last_class_counts
            results_data["recent_inference_logs"] = recent_logs

            return {
                "success": True,
                "message": "对象计数完成",
                "results": results_data,
                "output_path": output_path,
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
            src_path = Path(source)
            suffix = src_path.suffix.lower()
            is_image = suffix in (
                ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
            )

            # ---------- 图片：单帧生成热力图，输出图片 ----------
            if is_image:
                frame = cv2.imread(str(source))
                if frame is None:
                    return {"success": False, "message": "无法读取图片文件，请确认格式正确"}

                height, width = frame.shape[:2]
                heatmap_2d = np.zeros((height, width), dtype=np.float32)

                # 单帧检测，累加检测框区域的热点
                results = model.predict(frame, conf=conf, classes=classes, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    single_heatmap = np.zeros((height, width), dtype=np.float32)
                    single_heatmap[y1:y2, x1:x2] = 1.0
                    single_heatmap = ndi.gaussian_filter(single_heatmap, sigma=15)
                    heatmap_2d += single_heatmap

                if np.max(heatmap_2d) > 0:
                    heatmap_normalized = (heatmap_2d / np.max(heatmap_2d)).clip(0, 1)
                    heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), colormap)
                    alpha = 0.5
                    annotated_frame = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
                else:
                    annotated_frame = frame

                if output_path:
                    cv2.imwrite(output_path, annotated_frame)

                return {
                    "success": True,
                    "message": "热图生成完成，共处理 1 帧",
                    "total_frames": 1,
                    "output_path": output_path
                }

            # ---------- 视频：逐帧累积热力图，输出视频 ----------
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return {"success": False, "message": "无法打开视频文件"}

            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 100  # 默认估计值

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            out = None
            if output_path:
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                out = _open_solution_mp4_writer(output_path, fps, width, height)

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

                if output_path and out is not None and out.isOpened():
                    out.write(annotated_frame)

            cap.release()
            if output_path and out is not None:
                out.release()
                _remux_mp4_for_html5_video(output_path)

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
        pixel_to_meter: float = 10,
        line_width: int = 2,
        output_path: str = None
    ) -> Dict[str, Any]:
        """速度估算 - 参考 Ultralytics 官方文档"""
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
                line_width=2,
                device=self.default_device
            )

            cap = cv2.VideoCapture(source)
            out = None
            if output_path:
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = _open_solution_mp4_writer(output_path, fps, width, height)

            frame_count = 0
            speeds = []
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                result = speed_estimator(frame)
                frame_count += 1

                # speed 是整体速度，speed_dict 是每个跟踪对象的速度
                if hasattr(result, 'speed'):
                    speeds.append(result.speed)

                if output_path and out is not None and out.isOpened():
                    out.write(frame)

            cap.release()
            if output_path and out is not None:
                out.release()
                _remux_mp4_for_html5_video(output_path)

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
        conf: float = 0.25,
        max_connections: int = 5  # 最多显示的连接数
    ) -> Dict[str, Any]:
        """距离计算"""
        try:
            model = self.load_model(model_name)
            img = cv2.imread(image_path)

            # 获取图片尺寸，用于自适应调整绘制参数
            img_height, img_width = img.shape[:2]
            # 根据图片尺寸计算线条粗细和字体大小
            scale_factor = max(img_width, img_height) / 1000  # 基准尺寸
            line_width = max(2, int(2 * scale_factor))  # 线条粗细
            font_scale = max(0.6, 0.8 * scale_factor)  # 字体大小
            text_thickness = max(2, int(2 * scale_factor))  # 文字线条粗细

            results = model.predict(source=img, conf=conf, classes=classes, verbose=False)

            if len(results) == 0 or len(results[0].boxes) < 2:
                return {"success": False, "message": "需要至少检测到2个对象", "distances": []}

            boxes = results[0].boxes
            num_objects = len(boxes)

            # 创建带透明通道的图片用于绘制半透明线条
            img_overlay = img.copy()

            centroids = []
            # 绘制每个检测对象的中心点和边框
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                cx = int((xyxy[0] + xyxy[2]) / 2)
                cy = int((xyxy[1] + xyxy[3]) / 2)
                centroids.append((cx, cy))
                # 绘制检测框 - 黄色
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), line_width)
                # 绘制中心点 - 黄色实心圆
                cv2.circle(img, (cx, cy), max(4, int(6 * scale_factor)), (0, 255, 255), -1)
                # 在中心点旁边标注序号
                cv2.putText(img, str(idx + 1), (cx + 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale * 0.7, (0, 255, 255), text_thickness)

            # 计算所有距离对
            all_distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    p1, p2 = centroids[i], centroids[j]
                    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    all_distances.append({
                        "object1_index": i + 1,
                        "object2_index": j + 1,
                        "pixel_distance": float(distance),
                        "p1": p1,
                        "p2": p2
                    })

            # 按距离排序，只显示最近的几对
            all_distances.sort(key=lambda x: x["pixel_distance"])
            connections_to_show = all_distances[:max_connections]

            # 使用半透明线条绘制连接线
            for conn in connections_to_show:
                p1, p2 = conn["p1"], conn["p2"]
                # 绘制半透明粗线作为底色
                cv2.line(img_overlay, p1, p2, (255, 0, 255), line_width * 3)
                # 绘制实线
                cv2.line(img_overlay, p1, p2, (180, 0, 180), line_width)

            # 混合透明层
            alpha = 0.6
            img = cv2.addWeighted(img_overlay, alpha, img, 1 - alpha, 0)

            # 在原图上绘制文字标签（不旋转）
            for conn in connections_to_show:
                p1, p2 = conn["p1"], conn["p2"]
                distance = conn["pixel_distance"]
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

                # 绘制距离文字背景（半透明黑色矩形）
                text = f"{int(distance)}px"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                bg_x1 = mid[0] - text_w // 2 - 8
                bg_y1 = mid[1] - text_h - 8
                bg_x2 = mid[0] + text_w // 2 + 8
                bg_y2 = mid[1] + 8
                # 确保背景框在图像范围内
                bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
                bg_x2, bg_y2 = min(img_width, bg_x2), min(img_height, bg_y2)
                # 绘制半透明背景
                overlay = img.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
                # 绘制文字 - 亮黄色
                cv2.putText(img, text, (bg_x1 + 5, bg_y1 + text_h + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), text_thickness)

            output_path = str(settings.UPLOADS_DIR / f"distance_{Path(image_path).name}")
            cv2.imwrite(output_path, img)

            return {
                "success": True,
                "message": "距离计算完成",
                "distances": all_distances,
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
                blur_ratio=int(blur_ratio),
                device=self.default_device
            )

            cap = cv2.VideoCapture(source)
            out = None
            if output_path:
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = _open_solution_mp4_writer(output_path, fps, width, height)

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                blur(frame)
                frame_count += 1

                if output_path and out is not None and out.isOpened():
                    out.write(frame)

            cap.release()
            if output_path and out is not None:
                out.release()
                _remux_mp4_for_html5_video(output_path)

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

            # 与其它方案一致：直接写入 uploads 根目录，避免子目录在部分部署上未就绪导致静态 URL 404
            if output_dir is None:
                output_dir = str(settings.UPLOADS_DIR)
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
                    safe_cls = re.sub(r"[^\w.\-]+", "_", str(class_name))[:40].strip("_") or str(cls_id)
                    crop_filename = f"crop_{safe_cls}_{i}_{Path(image_path).stem}.jpg"
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
        line_width: int = 2,
        output_path: str = None
    ) -> Dict[str, Any]:
        """队列管理 - 参考 Ultralytics 官方文档"""
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
                line_width=2,
                device=self.default_device
            )

            cap = cv2.VideoCapture(source)
            out = None
            if output_path:
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = _open_solution_mp4_writer(output_path, fps, width, height)

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

                if output_path and out is not None and out.isOpened():
                    out.write(frame)

            cap.release()
            if output_path and out is not None:
                out.release()
                _remux_mp4_for_html5_video(output_path)

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

    # ==================== 停车管理 ====================
    def parking_management(
        self,
        source: str,
        model_name: str = None,
        parking_slots: List[List[Tuple[int, int]]] = None,
        classes: List[int] = None,
        conf: float = 0.25,
        line_width: int = 2,
        output_path: str = None
    ) -> Dict[str, Any]:
        """停车管理：基于车位多边形统计占用情况（图片/视频）"""
        try:
            model = self.load_model(model_name)

            if parking_slots is None:
                parking_slots = [
                    [(80, 420), (260, 420), (260, 600), (80, 600)],
                    [(280, 420), (460, 420), (460, 600), (280, 600)],
                    [(480, 420), (660, 420), (660, 600), (480, 600)],
                ]

            slot_polys = [np.array(slot, dtype=np.int32) for slot in parking_slots if len(slot) >= 3]
            if not slot_polys:
                return {"success": False, "message": "停车位区域为空，请提供 parking_slots"}

            vehicle_centers_per_frame: List[List[Tuple[int, int]]] = []
            occupied_per_frame: List[int] = []
            total_frames = 0

            def process_frame(frame: np.ndarray) -> np.ndarray:
                nonlocal total_frames
                total_frames += 1

                results = model.predict(source=frame, conf=conf, classes=classes, verbose=False)
                centers: List[Tuple[int, int]] = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    for x1, y1, x2, y2 in boxes:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        centers.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

                vehicle_centers_per_frame.append(centers)

                occupied_flags: List[bool] = []
                for i, poly in enumerate(slot_polys):
                    occupied = any(cv2.pointPolygonTest(poly, c, False) >= 0 for c in centers)
                    occupied_flags.append(occupied)
                    color = (0, 0, 255) if occupied else (0, 200, 0)
                    cv2.polylines(frame, [poly], True, color, line_width)
                    x, y, w, h = cv2.boundingRect(poly)
                    cv2.putText(frame, f"P{i+1}:{'OCC' if occupied else 'FREE'}", (x, max(18, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                occupied_count = int(sum(occupied_flags))
                occupied_per_frame.append(occupied_count)
                total_slots = len(slot_polys)
                free_count = total_slots - occupied_count
                cv2.putText(frame, f"Occupied: {occupied_count}/{total_slots}  Free: {free_count}",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 3)
                cv2.putText(frame, f"Occupied: {occupied_count}/{total_slots}  Free: {free_count}",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return frame

            ext = Path(source).suffix.lower()
            is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts"}

            if is_video:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    return {"success": False, "message": "无法打开视频文件"}

                writer = None
                if output_path:
                    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = _open_solution_mp4_writer(output_path, fps, width, height)

                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = process_frame(frame)
                    if writer is not None and writer.isOpened():
                        writer.write(frame)

                cap.release()
                if writer is not None:
                    writer.release()
                    _remux_mp4_for_html5_video(output_path)
            else:
                frame = cv2.imread(source)
                if frame is None:
                    return {"success": False, "message": "无法读取图片文件"}
                frame = process_frame(frame)
                if output_path:
                    cv2.imwrite(output_path, frame)

            avg_occupied = (sum(occupied_per_frame) / len(occupied_per_frame)) if occupied_per_frame else 0.0
            max_occupied = max(occupied_per_frame) if occupied_per_frame else 0
            current_occupied = occupied_per_frame[-1] if occupied_per_frame else 0
            total_slots = len(slot_polys)

            return {
                "success": True,
                "message": "停车管理分析完成",
                "results": {
                    "total_slots": total_slots,
                    "current_occupied": current_occupied,
                    "current_free": total_slots - current_occupied,
                    "avg_occupied": avg_occupied,
                    "max_occupied": max_occupied,
                    "total_frames": total_frames,
                },
                "output_path": output_path
            }
        except Exception as e:
            return {"success": False, "message": f"停车管理失败: {str(e)}"}


# 全局实例
solutions_service = SolutionsService() if ULTRALYTICS_AVAILABLE else None
