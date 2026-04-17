"""
数据集统计服务 - Dataset Statistics Service
提供类别分布、位置热图、维度分析、拆分明细等可视化统计
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import numpy as np

try:
    import PIL.Image
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class StatisticsService:
    """
    数据集统计服务

    提供:
    1. 类别分布 - 按类别划分的标签计数条形图
    2. 位置热图 - 标注的空间分布
    3. 维度分析 - 图像宽度与高度分布
    4. 拆分明细 - 训练/验证/测试样本计数
    """

    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 3600  # 1小时缓存

    def get_dataset_statistics(
        self,
        dataset_path: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        获取数据集完整统计信息

        Returns:
            {
                "success": bool,
                "dataset": str,
                "generated_at": str,
                "summary": {...},
                "class_distribution": {...},
                "spatial_distribution": {...},
                "dimension_analysis": {...},
                "split_details": {...}
            }
        """
        dataset_path = Path(dataset_path)
        cache_key = str(dataset_path.resolve())

        # 检查缓存
        if not force_refresh and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if datetime.now().timestamp() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["statistics"]

        try:
            # 收集所有统计信息
            stats = {
                "success": True,
                "dataset": str(dataset_path),
                "generated_at": datetime.now().isoformat(),
                "summary": self._get_summary(dataset_path),
                "class_distribution": self._get_class_distribution(dataset_path),
                "spatial_distribution": self._get_spatial_distribution(dataset_path),
                "dimension_analysis": self._get_dimension_analysis(dataset_path),
                "split_details": self._get_split_details(dataset_path)
            }

            # 缓存结果
            self.cache[cache_key] = {
                "statistics": stats,
                "timestamp": datetime.now().timestamp()
            }

            return stats

        except Exception as e:
            return {
                "success": False,
                "message": f"统计失败: {str(e)}",
                "error": str(e)
            }

    def _resolve_dirs(self, dataset_path: Path):
        """
        解析数据集目录结构，兼容三种格式：
        1. 平铺: images/ labels/
        2. images 内拆分: images/train/ images/val/ labels/train/
        3. Roboflow: train/images/ train/labels/ valid/images/ ...
        返回 (all_image_dirs, all_label_dirs) 两个列表，每项为 Path
        """
        SPLITS = ["train", "val", "valid", "test"]
        image_dirs: List[Path] = []
        label_dirs: List[Path] = []

        root_img = dataset_path / "images"
        root_lbl = dataset_path / "labels"

        if root_img.exists():
            image_dirs.append(root_img)
        if root_lbl.exists():
            label_dirs.append(root_lbl)

        # Roboflow 结构: train/images/, valid/images/ ...
        for sp in SPLITS:
            sp_img = dataset_path / sp / "images"
            sp_lbl = dataset_path / sp / "labels"
            if sp_img.exists():
                image_dirs.append(sp_img)
            if sp_lbl.exists():
                label_dirs.append(sp_lbl)

        # 兜底：没有任何 images 目录则用数据集根目录
        if not image_dirs:
            image_dirs = [dataset_path]

        return image_dirs, label_dirs

    def _get_summary(self, dataset_path: Path) -> Dict[str, Any]:
        """获取数据集摘要（与 /datasets/{name}/images 使用同一套枚举规则，避免概览与浏览张数不一致）"""
        from backend.modules.data_preparation.dataset_service import dataset_service

        entries = dataset_service.collect_dataset_image_entries(dataset_path)
        image_count = len(entries)
        annotation_count = sum(
            dataset_service.count_label_lines_for_image(p, lbl) for p, _, lbl in entries
        )
        total_size = sum(p.stat().st_size for p, _, _ in entries) if entries else 0

        classes = self._get_classes(dataset_path)

        return {
            "total_images": image_count,
            "total_annotations": annotation_count,
            "total_classes": len(classes),
            "classes": classes,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "avg_annotations_per_image": round(annotation_count / max(image_count, 1), 2)
        }

    def _get_class_distribution(self, dataset_path: Path) -> Dict[str, Any]:
        """获取类别分布"""
        _, label_dirs = self._resolve_dirs(dataset_path)

        if not label_dirs:
            return {
                "success": True,
                "labels": [],
                "counts": [],
                "percentages": [],
                "chart_data": []
            }

        class_counts = defaultdict(int)

        # 解析所有标签文件（跨所有 labels 目录）
        for labels_dir in label_dirs:
            for label_file in labels_dir.rglob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1

        # 获取类别名称
        classes = self._get_classes(dataset_path)

        # 构建结果
        labels = []
        counts = []
        percentages = []
        total = sum(class_counts.values())

        for class_id in sorted(class_counts.keys()):
            class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
            count = class_counts[class_id]

            labels.append(class_name)
            counts.append(count)
            percentages.append(round(count / max(total, 1) * 100, 2))

        # Chart.js 格式数据
        chart_data = {
            "labels": labels,
            "datasets": [{
                "label": "标注数量",
                "data": counts,
                "backgroundColor": self._generate_colors(len(labels)),
                "borderColor": [self._adjust_color(c, -30) for c in self._generate_colors(len(labels))],
                "borderWidth": 2
            }]
        }

        return {
            "success": True,
            "labels": labels,
            "counts": counts,
            "percentages": percentages,
            "total_annotations": total,
            "chart_data": chart_data
        }

    def _get_spatial_distribution(self, dataset_path: Path) -> Dict[str, Any]:
        """获取位置热图数据"""
        image_dirs, label_dirs = self._resolve_dirs(dataset_path)

        if not label_dirs:
            return {
                "success": True,
                "heatmap_data": [],
                "density_map": [],
                "center_points": []
            }

        # 收集所有边界框中心点
        center_points = []
        image_sizes = {}

        # 获取图片尺寸（跨所有 image 目录）
        for images_dir in image_dirs:
            for img_path in images_dir.rglob("*.[jp][pn]g"):
                try:
                    if PIL_AVAILABLE:
                        img = Image.open(img_path)
                        image_sizes[img_path.stem] = (img.width, img.height)
                except:
                    pass

        # 解析标签（跨所有 label 目录）
        for labels_dir in label_dirs:
            for label_file in labels_dir.rglob("*.txt"):
                img_path = label_file.with_suffix(".jpg")
                img_stem = label_file.stem

            img_size = image_sizes.get(img_stem, (640, 480))
            img_w, img_h = img_size

            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO 格式: class_id x_center y_center width height (归一化)
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_w
                        y_center = float(parts[2]) * img_h

                        center_points.append({
                            "x": round(x_center, 1),
                            "y": round(y_center, 1),
                            "class_id": class_id
                        })

        if not center_points:
            return {
                "success": True,
                "heatmap_data": [],
                "density_map": [],
                "center_points": []
            }

        # 生成热力图数据 (10x10 网格)
        heatmap_grid = np.zeros((10, 10))
        for point in center_points:
            grid_x = min(9, int(point["x"] / 64))  # point["x"] 范围 0-640
            grid_y = min(9, int(point["y"] / 48))  # point["y"] 范围 0-480
            heatmap_grid[grid_y][grid_x] += 1

        # 转换为列表
        heatmap_data = heatmap_grid.flatten().tolist()

        # 生成密度等级
        max_val = heatmap_grid.max()
        density_map = []
        for val in heatmap_grid.flatten():
            if val == 0:
                density = 0
            elif val < max_val * 0.2:
                density = 1
            elif val < max_val * 0.4:
                density = 2
            elif val < max_val * 0.6:
                density = 3
            elif val < max_val * 0.8:
                density = 4
            else:
                density = 5
            density_map.append(density)

        return {
            "success": True,
            "total_points": len(center_points),
            "heatmap_data": heatmap_data,
            "density_map": density_map,
            "center_points": center_points[:100],  # 限制返回数量
            "grid_size": 10,
            "chart_data": {
                "type": "heatmap",
                "data": {
                    "labels": [f"x{i}" for i in range(10)],
                    "datasets": [{
                        "label": "标注密度",
                        "data": heatmap_data,
                        "backgroundColor": self._get_heatmap_colors()
                    }]
                }
            }
        }

    def _get_dimension_analysis(self, dataset_path: Path) -> Dict[str, Any]:
        """获取维度分析"""
        image_dirs, _ = self._resolve_dirs(dataset_path)

        widths = []
        heights = []
        aspect_ratios = []

        # 分析图片尺寸（跨所有 image 目录）
        for images_dir in image_dirs:
            for img_path in images_dir.rglob("*.[jp][pn]g"):
                try:
                    if PIL_AVAILABLE:
                        img = Image.open(img_path)
                        widths.append(img.width)
                        heights.append(img.height)
                        aspect_ratios.append(round(img.width / max(img.height, 1), 2))
                except:
                    pass

        if not widths:
            return {
                "success": True,
                "widths": [],
                "heights": [],
                "aspect_ratios": [],
                "statistics": {}
            }

        # 计算统计信息
        stats = {
            "width": {
                "min": min(widths),
                "max": max(widths),
                "mean": round(sum(widths) / len(widths), 1),
                "median": self._median(widths)
            },
            "height": {
                "min": min(heights),
                "max": max(heights),
                "mean": round(sum(heights) / len(heights), 1),
                "median": self._median(heights)
            },
            "aspect_ratio": {
                "min": min(aspect_ratios),
                "max": max(aspect_ratios),
                "mean": round(sum(aspect_ratios) / len(aspect_ratios), 2),
                "median": self._median(aspect_ratios)
            }
        }

        # 分布数据 (分桶)
        width_buckets = self._create_buckets(widths, 10)
        height_buckets = self._create_buckets(heights, 10)
        ratio_buckets = self._create_buckets(aspect_ratios, 5)

        return {
            "success": True,
            "total_images": len(widths),
            "widths": widths,
            "heights": heights,
            "aspect_ratios": aspect_ratios,
            "statistics": stats,
            "width_distribution": {
                "labels": width_buckets["labels"],
                "data": width_buckets["counts"]
            },
            "height_distribution": {
                "labels": height_buckets["labels"],
                "data": height_buckets["counts"]
            },
            "aspect_ratio_distribution": {
                "labels": ratio_buckets["labels"],
                "data": ratio_buckets["counts"]
            },
            "scatter_data": list(zip(widths[:500], heights[:500])),  # 限制数量
            "chart_data": {
                "scatter": {
                    "label": "宽度 x 高度",
                    "data": [{"x": w, "y": h} for w, h in zip(widths[:500], heights[:500])]
                }
            }
        }

    def _get_split_details(self, dataset_path: Path) -> Dict[str, Any]:
        """获取数据拆分详情（与图片浏览同一枚举，避免重复计数或漏计）"""
        from backend.modules.data_preparation.dataset_service import dataset_service

        entries = dataset_service.collect_dataset_image_entries(dataset_path)
        splits = {
            "train": {"images": 0, "annotations": 0},
            "val": {"images": 0, "annotations": 0},
            "test": {"images": 0, "annotations": 0},
            "unlabeled": {"images": 0, "annotations": 0},
        }

        for img_path, sp, lbl_root in entries:
            ann = dataset_service.count_label_lines_for_image(img_path, lbl_root)
            if sp == "train":
                bucket = "train"
            elif sp == "val":
                bucket = "val"
            elif sp == "test":
                bucket = "test"
            else:
                bucket = "unlabeled"
            splits[bucket]["images"] += 1
            splits[bucket]["annotations"] += ann

        total = sum(s["images"] for s in splits.values())
        for split in splits:
            splits[split]["percentage"] = round(
                splits[split]["images"] / max(total, 1) * 100, 1
            )

        # Chart.js 格式数据
        chart_data = {
            "labels": ["训练集", "验证集", "测试集", "未标注"],
            "datasets": [{
                "label": "图片数量",
                "data": [splits[s]["images"] for s in ["train", "val", "test", "unlabeled"]],
                "backgroundColor": [
                    "rgba(54, 162, 235, 0.8)",
                    "rgba(75, 192, 192, 0.8)",
                    "rgba(255, 206, 86, 0.8)",
                    "rgba(201, 203, 207, 0.8)"
                ],
                "borderColor": [
                    "rgba(54, 162, 235, 1)",
                    "rgba(75, 192, 192, 1)",
                    "rgba(255, 206, 86, 1)",
                    "rgba(201, 203, 207, 1)"
                ],
                "borderWidth": 1
            }]
        }

        return {
            "success": True,
            "splits": splits,
            "total_images": total,
            "chart_data": chart_data
        }

    # ==================== 辅助方法 ====================

    def _count_images(self, dir_path: Path) -> int:
        """统计图片数量"""
        if not dir_path.exists():
            return 0

        count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            count += len(list(dir_path.rglob(ext)))
        return count

    def _count_annotations(self, dir_path: Path) -> int:
        """统计标注数量"""
        if not dir_path.exists():
            return 0

        count = 0
        for label_file in dir_path.rglob("*.txt"):
            with open(label_file, 'r') as f:
                count += sum(1 for line in f if line.strip())
        return count

    def _get_classes(self, dataset_path: Path) -> List[str]:
        """获取类别列表"""
        yaml_path = dataset_path / "data.yaml"
        classes = []

        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                content = f.read()
                if 'names:' in content:
                    names_start = content.find('names:') + 7
                    names_content = content[names_start:]
                    for line in names_content.split('\n'):
                        if ':' in line and not line.strip().startswith('#'):
                            try:
                                idx = int(line.split(':')[0].strip())
                                name = line.split(':')[1].strip()
                                while len(classes) <= idx:
                                    classes.append(f"class_{len(classes)}")
                                classes[idx] = name
                            except:
                                pass

        return classes

    def _get_total_size(self, dir_path: Path) -> int:
        """获取目录总大小"""
        if not dir_path.exists():
            return 0

        total = 0
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total

    def _median(self, values: List[float]) -> float:
        """计算中位数"""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return round((sorted_vals[mid - 1] + sorted_vals[mid]) / 2, 1)
        return round(sorted_vals[mid], 1)

    def _create_buckets(self, values: List[float], num_buckets: int) -> Dict:
        """创建数值分桶"""
        if not values:
            return {"labels": [], "counts": []}

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val + 1
        bucket_size = max(1, range_val / num_buckets)

        buckets = [0] * num_buckets
        labels = []

        for v in values:
            bucket_idx = int((v - min_val) / bucket_size)
            bucket_idx = max(0, min(num_buckets - 1, bucket_idx))
            buckets[bucket_idx] += 1

        # 创建标签
        for i in range(num_buckets):
            start = min_val + i * bucket_size
            end = start + bucket_size - 1
            labels.append(f"{round(start, 2)}-{round(end, 2)}")

        return {"labels": labels, "counts": buckets}

    def _generate_colors(self, count: int) -> List[str]:
        """生成颜色列表"""
        base_colors = [
            "rgba(54, 162, 235, 0.8)",
            "rgba(255, 99, 132, 0.8)",
            "rgba(75, 192, 192, 0.8)",
            "rgba(255, 206, 86, 0.8)",
            "rgba(153, 102, 255, 0.8)",
            "rgba(255, 159, 64, 0.8)",
            "rgba(199, 199, 199, 0.8)",
            "rgba(83, 102, 255, 0.8)",
            "rgba(40, 159, 64, 0.8)",
            "rgba(210, 99, 132, 0.8)"
        ]

        colors = []
        for i in range(count):
            colors.append(base_colors[i % len(base_colors)])

        return colors

    def _adjust_color(self, color: str, amount: int) -> str:
        """调整颜色亮度"""
        import re
        match = re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)', color)
        if match:
            r = max(0, min(255, int(match.group(1)) + amount))
            g = max(0, min(255, int(match.group(2)) + amount))
            b = max(0, min(255, int(match.group(3)) + amount))
            if 'a' in color:
                return f"rgba({r}, {g}, {b}, 1)"
            return f"rgb({r}, {g}, {b})"
        return color

    def _get_heatmap_colors(self) -> List[str]:
        """获取热力图颜色"""
        return [
            "rgba(0, 0, 255, 0.1)",
            "rgba(0, 0, 255, 0.2)",
            "rgba(0, 128, 255, 0.4)",
            "rgba(0, 255, 128, 0.6)",
            "rgba(128, 255, 0, 0.8)",
            "rgba(255, 255, 0, 0.9)",
            "rgba(255, 128, 0, 1.0)",
            "rgba(255, 0, 0, 1.0)"
        ]

    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()


# 全局实例
statistics_service = StatisticsService()
