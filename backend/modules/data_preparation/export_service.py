"""
数据集导出服务
支持 YOLO、COCO、NDJSON 格式导出
"""
import json
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from backend.core.config import settings
from backend.modules.data_preparation.coco_parser import COCOExporter


class DatasetExporter:
    """数据集导出器"""

    def __init__(self):
        pass

    def export_dataset(
        self,
        dataset_name: str,
        format: str = "yolo",
        include_splits: List[str] = None,
        output_path: Path = None
    ) -> Dict[str, Any]:
        """
        导出数据集

        Args:
            dataset_name: 数据集名称
            format: 导出格式 (yolo, coco, ndjson)
            include_splits: 包含的拆分 (train, val, test)
            output_path: 输出路径

        Returns:
            导出结果
        """
        dataset_dir = settings.DATASETS_DIR / dataset_name
        if not dataset_dir.exists():
            return {"success": False, "message": "数据集不存在"}

        if include_splits is None:
            include_splits = ["train", "val", "test"]

        # 收集图像和标签
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        if not images_dir.exists():
            return {"success": False, "message": "数据集图像目录不存在"}

        # 获取所有图像
        all_images = []
        for split in include_splits:
            split_dir = images_dir / split
            if split_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
                    all_images.extend(list(split_dir.glob(ext)))
                    all_images.extend(list(split_dir.glob(ext.upper())))

        # 如果根目录也有图像
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            root_images = list(images_dir.glob(ext))
            root_images.extend(list(images_dir.glob(ext.upper())))
            for img in root_images:
                if img not in all_images:
                    all_images.append(img)

        if not all_images:
            return {"success": False, "message": "数据集没有图像"}

        # 根据格式导出
        if format == "yolo":
            return self._export_yolo(dataset_name, dataset_dir, all_images, include_splits, output_path)
        elif format == "coco":
            return self._export_coco(dataset_name, dataset_dir, all_images, include_splits, output_path)
        elif format == "ndjson":
            return self._export_ndjson(dataset_name, dataset_dir, all_images, include_splits, output_path)
        else:
            return {"success": False, "message": f"不支持的格式: {format}"}

    def _export_yolo(
        self,
        dataset_name: str,
        dataset_dir: Path,
        images: List[Path],
        splits: List[str],
        output_path: Path = None
    ) -> Dict[str, Any]:
        """导出为 YOLO 格式"""
        # 创建 ZIP
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 收集标签
            labels_dir = dataset_dir / "labels"
            image_labels = {}

            for img_path in images:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    # 按拆分分类
                    split = 'train'
                    for s in splits:
                        if s in img_path.parts:
                            split = s
                            break

                    if split not in image_labels:
                        image_labels[split] = {}
                    image_labels[split][img_path.name] = label_path.read_text()

            # 收集类别信息
            data_yaml = dataset_dir / "data.yaml"
            if data_yaml.exists():
                zf.writestr("data.yaml", data_yaml.read_text())
            else:
                # 生成默认 data.yaml
                classes = self._get_dataset_classes(dataset_dir)
                yaml_content = f"""# {dataset_name} dataset configuration
path: .
train: images/train
val: images/val

names:
"""
                for i, cls in enumerate(classes):
                    yaml_content += f"    {i}: {cls}\n"
                zf.writestr("data.yaml", yaml_content)

            # 写入图像和标签
            for img_path in images:
                # 确定拆分
                split = 'train'
                for s in splits:
                    if s in img_path.parts:
                        split = s
                        break

                # 写入图像
                img_rel = Path("images") / split / img_path.name
                zf.writestr(str(img_rel), img_path.read_bytes())

                # 写入标签
                if split in image_labels and img_path.name in image_labels[split]:
                    label_rel = Path("labels") / split / f"{img_path.stem}.txt"
                    zf.writestr(str(label_rel), image_labels[split][img_path.name])

        # 保存或返回
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            return {"success": True, "format": "yolo", "path": str(output_path)}

        return {
            "success": True,
            "format": "yolo",
            "data": buffer.getvalue(),
            "filename": f"{dataset_name}_yolo.zip"
        }

    def _export_coco(
        self,
        dataset_name: str,
        dataset_dir: Path,
        images: List[Path],
        splits: List[str],
        output_path: Path = None
    ) -> Dict[str, Any]:
        """导出为 COCO 格式"""
        labels_dir = dataset_dir / "labels"
        data_yaml = dataset_dir / "data.yaml"

        # 获取类别
        classes = self._get_dataset_classes(dataset_dir)
        categories = [{"id": i, "name": name, "supercategory": ""} for i, name in enumerate(classes)]

        # 构建 COCO 数据
        exporter = COCOExporter()
        exporter.set_categories(categories)

        # 处理每个图像
        image_id = 1
        ann_id = 1
        coco_images = []
        coco_annotations = []

        for img_path in sorted(images):
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    width, height = img.size
            except:
                width, height = 640, 480

            # 确定拆分
            split = 'train'
            for s in splits:
                if s in img_path.parts:
                    split = s
                    break

            # 添加图像
            exporter.add_image(img_path.name, width, height, split)

            # 添加标注
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                label_content = label_path.read_text()
                for line in label_content.strip().split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # 转换为 COCO 格式
                        x = (x_center - width / 2) * 640  # 假设原始尺寸
                        y = (y_center - height / 2) * 480
                        w = width * 640
                        h = height * 480

                        exporter.add_annotation(
                            image_id=image_id,
                            category_id=class_id,
                            bbox=[x, y, w, h],
                            area=w * h
                        )
                        ann_id += 1

            image_id += 1

        # 导出 JSON
        coco_data = exporter.export_to_json()

        # 创建 ZIP
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 写入图像
            for img_path in images:
                split = 'train'
                for s in splits:
                    if s in img_path.parts:
                        split = s
                        break

                img_rel = Path("images") / split / img_path.name
                zf.writestr(str(img_rel), img_path.read_bytes())

            # 写入 annotations
            for split in splits:
                ann_file = f"annotations/instances_{split}.json"
                # 筛选当前拆分的图像
                split_images = [img for img in coco_data['images'] if img.get('split') == split]
                split_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] in [img['id'] for img in split_images]]

                split_data = {
                    'images': split_images,
                    'annotations': split_anns,
                    'categories': categories
                }
                zf.writestr(ann_file, json.dumps(split_data, indent=2))

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            return {"success": True, "format": "coco", "path": str(output_path)}

        return {
            "success": True,
            "format": "coco",
            "data": buffer.getvalue(),
            "filename": f"{dataset_name}_coco.zip"
        }

    def _export_ndjson(
        self,
        dataset_name: str,
        dataset_dir: Path,
        images: List[Path],
        splits: List[str],
        output_path: Path = None
    ) -> Dict[str, Any]:
        """导出为 NDJSON 格式（参考 Ultralytics 格式）"""
        labels_dir = dataset_dir / "labels"
        data_yaml = dataset_dir / "data.yaml"

        # 获取类别
        classes = self._get_dataset_classes(dataset_dir)

        # 生成数据集元数据
        metadata = {
            "type": "dataset",
            "task": "detect",
            "name": dataset_name,
            "description": f"Dataset exported from {dataset_name}",
            "class_names": {str(i): name for i, name in enumerate(classes)},
            "version": 1,
            "created_at": datetime.now().isoformat() + "Z",
            "updated_at": datetime.now().isoformat() + "Z"
        }

        # 构建 NDJSON 行
        lines = [json.dumps(metadata)]

        # 处理每个图像
        for img_path in sorted(images):
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    width, height = img.size
            except:
                width, height = 640, 480

            # 确定拆分
            split = 'train'
            for s in splits:
                if s in img_path.parts:
                    split = s
                    break

            # 获取标注
            annotations = {}
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                label_content = label_path.read_text()
                boxes = []
                for line in label_content.strip().split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append([
                            int(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4])
                        ])
                if boxes:
                    annotations["boxes"] = boxes

            # 构建图像行
            image_entry = {
                "type": "image",
                "file": img_path.name,
                "width": width,
                "height": height,
                "split": split
            }

            if annotations:
                image_entry["annotations"] = annotations

            lines.append(json.dumps(image_entry))

        ndjson_content = '\n'.join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ndjson_content)
            return {"success": True, "format": "ndjson", "path": str(output_path)}

        return {
            "success": True,
            "format": "ndjson",
            "data": ndjson_content,
            "filename": f"{dataset_name}.ndjson"
        }

    def _get_dataset_classes(self, dataset_dir: Path) -> List[str]:
        """获取数据集类别"""
        # 从 data.yaml 读取
        data_yaml = dataset_dir / "data.yaml"
        if data_yaml.exists():
            content = data_yaml.read_text()
            if 'names:' in content:
                classes = []
                lines = content.split('\n')
                in_names = False
                for line in lines:
                    if 'names:' in line:
                        in_names = True
                        continue
                    if in_names:
                        if ':' in line:
                            try:
                                idx = int(line.split(':')[0].strip())
                                name = line.split(':')[1].strip()
                                classes.append((idx, name))
                            except:
                                continue
                        elif line.strip() and not line.startswith(' '):
                            break
                if classes:
                    return [name for _, name in sorted(classes)]

        # 从标签文件推断
        labels_dir = dataset_dir / "labels"
        if labels_dir.exists():
            class_ids = set()
            for label_file in labels_dir.rglob("*.txt"):
                try:
                    for line in label_file.read_text().split('\n'):
                        parts = line.strip().split()
                        if parts:
                            class_ids.add(int(parts[0]))
                except:
                    continue

            if class_ids:
                return [f"class_{i}" for i in sorted(class_ids)]

        return ["object"]

    def generate_export_response(
        self,
        dataset_name: str,
        format: str = "yolo",
        include_splits: List[str] = None
    ) -> Tuple[bytes, str, str]:
        """
        生成导出响应

        Returns:
            (data, content_type, filename)
        """
        result = self.export_dataset(dataset_name, format, include_splits)

        if not result.get('success'):
            raise ValueError(result.get('message', '导出失败'))

        if format == "ndjson":
            return (
                result['data'].encode('utf-8'),
                'application/x-ndjson',
                result['filename']
            )
        else:
            return (
                result['data'],
                'application/zip',
                result['filename']
            )


# 全局实例
dataset_exporter = DatasetExporter()
