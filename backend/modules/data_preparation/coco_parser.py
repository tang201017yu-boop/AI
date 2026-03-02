"""
COCO 格式解析器
支持 COCO JSON 解析和 COCO→YOLO 格式转换
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math

from backend.core.config import settings


class COCOParser:
    """COCO 格式解析器"""

    # COCO 任务类型
    TASK_DETECT = "detect"
    TASK_SEGMENT = "segment"
    TASK_KEYPOINT = "keypoint"

    def __init__(self):
        self.coco_data: Dict = {}
        self.categories: List[Dict] = []
        self.images: Dict[int, Dict] = {}  # image_id -> image_info
        self.annotations: List[Dict] = []
        self.image_annotations: Dict[int, List[Dict]] = defaultdict(list)  # image_id -> annotations

    def load_coco_json(self, json_path: Path) -> bool:
        """加载 COCO JSON 文件"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.coco_data = json.load(f)

            # 解析基本信息
            self.categories = self.coco_data.get('categories', [])
            self.images = {img['id']: img for img in self.coco_data.get('images', [])}
            self.annotations = self.coco_data.get('annotations', [])

            # 构建 image_id -> annotations 映射
            for ann in self.annotations:
                self.image_annotations[ann['image_id']].append(ann)

            return True
        except Exception as e:
            print(f"加载 COCO JSON 失败: {e}")
            return False

    def get_task_type(self) -> str:
        """检测 COCO 数据集任务类型"""
        if not self.annotations:
            return self.TASK_DETECT

        # 检查是否有分割标注
        for ann in self.annotations:
            if 'segmentation' in ann and ann['segmentation']:
                return self.TASK_SEGMENT
            if 'keypoints' in ann and ann['keypoints']:
                return self.TASK_KEYPOINT

        return self.TASK_DETECT

    def get_categories_map(self) -> Dict[int, str]:
        """获取类别 ID 到名称的映射"""
        return {cat['id']: cat['name'] for cat in self.categories}

    def get_categories_list(self) -> List[Dict]:
        """获取类别列表（按 ID 排序）"""
        return sorted(self.categories, key=lambda x: x.get('id', 0))

    def coco_to_yolo_bbox(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        COCO bbox 转 YOLO 格式
        COCO: [x, y, w, h] (左上角坐标，像素)
        YOLO: [x_center, y_center, w, h] (归一化)
        """
        x, y, w, h = bbox

        # 计算中心点
        x_center = x + w / 2
        y_center = y + h / 2

        # 归一化
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        return x_center_norm, y_center_norm, w_norm, h_norm

    def coco_polygon_to_yolo_bbox(self, segmentation: List, img_width: int, img_height: int) -> Optional[Tuple[float, float, float, float]]:
        """
        COCO 多边形分割转 YOLO bbox
        segmentation: [[x1,y1,x2,y2,...], ...] 或 [x1,y1,x2,y2,...]
        """
        if not segmentation:
            return None

        # 展平所有点
        all_points = []
        for poly in segmentation:
            if isinstance(poly, list):
                all_points.extend(poly)

        if len(all_points) < 4:  # 至少需要2个点
            return None

        # 计算边界框
        x_coords = all_points[0::2]
        y_coords = all_points[1::2]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        return self.coco_to_yolo_bbox(bbox, img_width, img_height)

    def generate_yolo_labels(
        self,
        images_dir: Path,
        output_labels_dir: Path,
        image_id_to_filename: Dict[int, str] = None
    ) -> Dict[str, Any]:
        """
        生成 YOLO 格式标签文件

        Args:
            images_dir: 图像目录
            output_labels_dir: 输出标签目录
            image_id_to_filename: image_id 到文件名的映射（可选）

        Returns:
            生成结果统计
        """
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        categories_map = self.get_categories_map()
        total_annotations = 0
        total_images = 0
        class_counts = defaultdict(int)

        # 构建 image_id -> filename 映射
        if image_id_to_filename is None:
            image_id_to_filename = {}
            for img_id, img_info in self.images.items():
                file_name = img_info.get('file_name', '')
                # 处理可能的路径前缀
                image_id_to_filename[img_id] = Path(file_name).name

        # 处理每个图像
        for image_id, img_info in self.images.items():
            img_width = img_info.get('width', 1)
            img_height = img_info.get('height', 1)

            # 获取图像文件名（不带扩展名）
            original_filename = image_id_to_filename.get(image_id, '')
            label_filename = Path(original_filename).stem + '.txt'
            label_path = output_labels_dir / label_filename

            # 获取该图像的所有标注
            img_annotations = self.image_annotations.get(image_id, [])
            if not img_annotations:
                continue

            # 写入 YOLO 格式标签
            with open(label_path, 'w') as f:
                for ann in img_annotations:
                    category_id = ann.get('category_id', 0)
                    class_counts[category_id] += 1

                    # 获取边界框
                    bbox = ann.get('bbox', [])
                    if not bbox:
                        # 尝试从 segmentation 转换
                        segmentation = ann.get('segmentation', [])
                        if segmentation:
                            bbox_result = self.coco_polygon_to_yolo_bbox(segmentation, img_width, img_height)
                            if bbox_result:
                                x_c, y_c, w, h = bbox_result
                            else:
                                continue
                        else:
                            continue
                    else:
                        x_c, y_c, w, h = self.coco_to_yolo_bbox(bbox, img_width, img_height)

                    # 写入 YOLO 格式行
                    f.write(f"{category_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    total_annotations += 1

            total_images += 1

        # 构建类别列表
        classes_list = []
        for cat in self.get_categories_list():
            cat_id = cat.get('id', 0)
            classes_list.append({
                'id': cat_id,
                'name': cat.get('name', f'class_{cat_id}'),
                'count': class_counts.get(cat_id, 0)
            })

        return {
            'success': True,
            'total_images': total_images,
            'total_annotations': total_annotations,
            'classes': classes_list,
            'task_type': self.get_task_type()
        }

    def generate_data_yaml(self, dataset_name: str, output_dir: Path) -> Path:
        """生成 data.yaml 配置文件"""
        categories_map = self.get_categories_map()

        # 按 ID 排序类别
        sorted_cats = sorted(categories_map.items(), key=lambda x: x[0])

        yaml_content = f"""# {dataset_name} dataset configuration
# Generated from COCO format
# Task: {self.get_task_type()}

path: .
train: images/train
val: images/val

names:
"""
        for cat_id, cat_name in sorted_cats:
            yaml_content += f"    {cat_id}: {cat_name}\n"

        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        return yaml_path


class COCOExporter:
    """COCO 格式导出器"""

    def __init__(self):
        self.images: List[Dict] = []
        self.annotations: List[Dict] = []
        self.categories: List[Dict] = []
        self.next_image_id = 1
        self.next_ann_id = 1
        self._category_map: Dict[int, int] = {}  # original_id -> new_id

    def set_categories(self, categories: List[Dict]):
        """设置类别列表"""
        self.categories = categories
        # 建立类别 ID 映射（从 1 开始连续 ID）
        self._category_map = {cat['id']: i + 1 for i, cat in enumerate(categories)}

    def add_image(self, file_name: str, width: int, height: int, split: str = 'train') -> int:
        """添加图像信息"""
        image_id = self.next_image_id
        self.images.append({
            'id': image_id,
            'file_name': file_name,
            'width': width,
            'height': height,
            'split': split
        })
        self.next_image_id += 1
        return image_id

    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        bbox: List[float],  # [x, y, w, h] COCO 格式
        segmentation: List = None,
        keypoints: List = None,
        area: float = None
    ) -> int:
        """添加标注"""
        ann_id = self.next_ann_id

        # 转换类别 ID
        new_category_id = self._category_map.get(category_id, category_id)

        ann = {
            'id': ann_id,
            'image_id': image_id,
            'category_id': new_category_id,
            'bbox': bbox
        }

        if area:
            ann['area'] = area
        if segmentation:
            ann['segmentation'] = segmentation
        if keypoints:
            ann['keypoints'] = keypoints

        self.annotations.append(ann)
        self.next_ann_id += 1
        return ann_id

    def export_to_json(self) -> Dict:
        """导出为 COCO JSON 格式"""
        # 重新映射类别 ID
        exported_categories = []
        for i, cat in enumerate(self.categories):
            exported_categories.append({
                'id': i + 1,
                'name': cat.get('name', cat.get('name', f'class_{i}')),
                'supercategory': cat.get('supercategory', '')
            })

        return {
            'images': self.images,
            'annotations': self.annotations,
            'categories': exported_categories
        }

    def save_json(self, output_path: Path) -> bool:
        """保存为 JSON 文件"""
        try:
            data = self.export_to_json()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存 COCO JSON 失败: {e}")
            return False


# 全局实例
coco_parser = COCOParser()
coco_exporter = COCOExporter()
