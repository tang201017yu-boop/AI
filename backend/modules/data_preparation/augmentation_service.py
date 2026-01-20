"""
数据集增强服务 - Data Augmentation Service
基于 Albumentations 库提供图像增强功能

支持:
- 图像分类增强
- 目标检测增强 (边界框)
- 分割增强 (掩码)
- 关键点增强
"""
import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

try:
    import albumentations as A
    import cv2
    import numpy as np
    from PIL import Image
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


@dataclass
class AugmentationConfig:
    """增强配置"""
    # 基础变换开关
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotate: bool = True
    scale: bool = True
    translate: bool = False
    brightness_contrast: bool = True
    hue_saturation: bool = True
    blur: bool = False
    noise: bool = False
    cutout: bool = False

    # 变换参数
    rotate_limit: int = 45
    scale_range: float = 0.1
    translate_range: float = 0.1
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    blur_limit: int = 7
    noise_var: int = 25
    cutout_num: int = 8
    cutout_size: int = 16

    # 输出设置
    num_augmented: int = 5  # 每张图片生成的数量
    output_format: str = "jpg"  # jpg, png


@dataclass
class AugmentationResult:
    """增强结果"""
    success: bool
    total_images: int = 0
    total_augmented: int = 0
    output_dir: str = ""
    message: str = ""
    samples: List[Dict] = field(default_factory=list)


class AugmentationService:
    """数据集增强服务"""

    def __init__(self):
        self.transforms = None
        self.config = None
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def _check_availability(self) -> bool:
        """检查依赖是否可用"""
        if not ALBUMENTATIONS_AVAILABLE:
            raise RuntimeError("Albumentations 库未安装，请运行: pip install albumentations")
        return True

    def create_transform(self, config: AugmentationConfig) -> A.Compose:
        """创建增强变换管道"""
        transforms_list = []

        # 几何变换
        if config.horizontal_flip:
            transforms_list.append(A.HorizontalFlip(p=0.5))

        if config.vertical_flip:
            transforms_list.append(A.VerticalFlip(p=0.5))

        if config.rotate:
            transforms_list.append(A.RandomRotate90())
            transforms_list.append(A.Rotate(limit=config.rotate_limit, p=0.5))

        if config.scale:
            transforms_list.append(A.RandomScale(
                scale_limit=(0.8, 1.2),
                p=0.5
            ))

        if config.translate:
            transforms_list.append(A.ShiftScaleRotate(
                shift_limit=config.translate_range,
                scale_limit=0,
                rotate_limit=0,
                p=0.5
            ))

        # 光照变换
        if config.brightness_contrast:
            transforms_list.append(A.RandomBrightnessContrast(
                brightness_limit=config.brightness_range,
                contrast_limit=config.contrast_range,
                p=0.5
            ))

        if config.hue_saturation:
            transforms_list.append(A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ))

        # 模糊和噪声
        if config.blur:
            transforms_list.append(A.Blur(
                blur_limit=config.blur_limit,
                p=0.2
            ))

        if config.noise:
            transforms_list.append(A.GaussNoise(
                var_limit=(10, config.noise_var),
                p=0.3
            ))

        # Cutout
        if config.cutout:
            transforms_list.append(A.CoarseDropout(
                num_holes=config.cutout_num,
                max_h_size=config.cutout_size,
                max_w_size=config.cutout_size,
                p=0.3
            ))

        return A.Compose(transforms_list)

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """加载图片"""
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            return None

    def _save_image(self, image: np.ndarray, output_path: Path) -> bool:
        """保存图片"""
        try:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if self.config and self.config.output_format == 'png':
                quality = None
            else:
                quality = 95
            cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return True
        except Exception as e:
            print(f"保存图片失败 {output_path}: {e}")
            return False

    def _parse_yolo_bbox(self, label_path: Path, img_width: int, img_height: int) -> Optional[List[Dict]]:
        """解析 YOLO 格式标签"""
        bboxes = []
        if not label_path.exists():
            return bboxes

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO 格式: cx, cy, w, h (归一化)
                        cx, cy, w, h = map(float, parts[1:5])
                        # 转换为 albumentations 格式: x_min, y_min, x_max, y_max
                        x_min = (cx - w/2) * img_width
                        y_min = (cy - h/2) * img_height
                        x_max = (cx + w/2) * img_width
                        y_max = (cy + h/2) * img_height
                        bboxes.append({
                            'class_id': class_id,
                            'x_min': max(0, x_min),
                            'y_min': max(0, y_min),
                            'x_max': min(img_width, x_max),
                            'y_max': min(img_height, y_max)
                        })
        except Exception as e:
            print(f"解析标签失败 {label_path}: {e}")
        return bboxes

    def _save_yolo_bbox(self, bboxes: List[Dict], output_path: Path, img_width: int, img_height: int):
        """保存 YOLO 格式标签"""
        try:
            with open(output_path, 'w') as f:
                for box in bboxes:
                    # 转换为归一化坐标
                    cx = (box['x_min'] + box['x_max']) / 2 / img_width
                    cy = (box['y_min'] + box['y_max']) / 2 / img_height
                    w = (box['x_max'] - box['x_min']) / img_width
                    h = (box['y_max'] - box['y_min']) / img_height
                    f.write(f"{box['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            print(f"保存标签失败 {output_path}: {e}")

    def augment_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        config: Optional[AugmentationConfig] = None
    ) -> AugmentationResult:
        """
        增强数据集

        Args:
            input_dir: 输入数据集目录 (YOLO 格式)
            output_dir: 输出目录
            config: 增强配置

        Returns:
            AugmentationResult: 增强结果
        """
        self._check_availability()
        self.config = config or AugmentationConfig()
        self.transforms = self.create_transform(self.config)

        result = AugmentationResult(success=False)

        # 验证输入目录结构
        images_dir = input_dir / 'images'
        labels_dir = input_dir / 'labels'

        if not images_dir.exists():
            result.message = f"图像目录不存在: {images_dir}"
            return result

        images_dir = images_dir.resolve()
        output_dir = output_dir.resolve()

        # 创建输出目录
        output_images_dir = output_dir / 'images'
        output_labels_dir = output_dir / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图片
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))

        if not image_files:
            result.message = "未找到图片文件"
            return result

        result.total_images = len(image_files)
        total_augmented = 0
        samples = []

        for img_path in image_files:
            img = self._load_image(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # 解析标签
            label_path = labels_dir / f"{img_path.stem}.txt"
            bboxes = self._parse_yolo_bbox(label_path, w, h)

            # 生成增强图片
            for i in range(self.config.num_augmented):
                # 应用变换
                transformed = self.transforms(
                    image=img,
                    bboxes=[b['x_min', 'y_min', 'x_max', 'y_max'] for b in bboxes] if bboxes else [],
                    class_labels=[b['class_id'] for b in bboxes] if bboxes else []
                )

                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_classes = transformed['class_labels']

                # 保存增强图片
                output_name = f"{img_path.stem}_aug{i+1}"
                output_img_path = output_images_dir / f"{output_name}.{self.config.output_format}"
                self._save_image(aug_img, output_img_path)

                # 保存增强标签
                if aug_bboxes and len(aug_bboxes) == len(aug_classes):
                    output_label_path = output_labels_dir / f"{output_name}.txt"
                    formatted_bboxes = []
                    for box, cls in zip(aug_bboxes, aug_classes):
                        formatted_bboxes.append({
                            'class_id': cls,
                            'x_min': box[0],
                            'y_min': box[1],
                            'x_max': box[2],
                            'y_max': box[3]
                        })
                    self._save_yolo_bbox(formatted_bboxes, output_label_path, w, h)

                total_augmented += 1

                # 保存样本信息（限制数量）
                if len(samples) < 10:
                    samples.append({
                        'original': img_path.name,
                        'augmented': f"{output_name}.{self.config.output_format}",
                        'transforms_applied': self._count_transforms()
                    })

        result.success = True
        result.total_augmented = total_augmented
        result.output_dir = str(output_dir)
        result.message = f"成功增强 {result.total_images} 张图片，生成 {total_augmented} 张增强图片"
        result.samples = samples

        # 生成数据集统计信息
        self._save_dataset_info(output_dir, result)

        return result

    def _count_transforms(self) -> int:
        """统计启用的变换数量"""
        if not self.config:
            return 0
        count = 0
        if self.config.horizontal_flip: count += 1
        if self.config.vertical_flip: count += 1
        if self.config.rotate: count += 2
        if self.config.scale: count += 1
        if self.config.translate: count += 1
        if self.config.brightness_contrast: count += 1
        if self.config.hue_saturation: count += 1
        if self.config.blur: count += 1
        if self.config.noise: count += 1
        if self.config.cutout: count += 1
        return count

    def _save_dataset_info(self, output_dir: Path, result: AugmentationResult):
        """保存数据集信息"""
        info = {
            'created_at': datetime.now().isoformat(),
            'original_images': result.total_images,
            'augmented_images': result.total_augmented,
            'augmentation_config': {
                'num_augmented_per_image': self.config.num_augmented if self.config else 5,
                'horizontal_flip': self.config.horizontal_flip if self.config else True,
                'vertical_flip': self.config.vertical_flip if self.config else False,
                'rotate': self.config.rotate if self.config else True,
                'brightness_contrast': self.config.brightness_contrast if self.config else True,
                'hue_saturation': self.config.hue_saturation if self.config else True,
            },
            'samples': result.samples
        }

        info_path = output_dir / 'augmentation_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    def preview_augmentation(
        self,
        image_path: Path,
        config: Optional[AugmentationConfig] = None,
        num_previews: int = 4
    ) -> List[Dict[str, Any]]:
        """
        预览增强效果

        Args:
            image_path: 原图路径
            config: 增强配置
            num_previews: 预览数量

        Returns:
            预览结果列表
        """
        self._check_availability()
        self.config = config or AugmentationConfig()
        self.config.num_augmented = num_previews
        self.transforms = self.create_transform(self.config)

        img = self._load_image(image_path)
        if img is None:
            return []

        results = []
        results.append({
            'type': 'original',
            'data': self._image_to_base64(img)
        })

        for i in range(num_previews):
            transformed = self.transforms(image=img)
            aug_img = transformed['image']
            results.append({
                'type': f'augmented_{i+1}',
                'data': self._image_to_base64(aug_img)
            })

        return results

    def _image_to_base64(self, image: np.ndarray) -> str:
        """图片转 base64"""
        import base64
        from io import BytesIO
        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

    def get_available_transforms(self) -> Dict[str, Any]:
        """获取可用的增强变换列表"""
        self._check_availability()
        return {
            'geometric': {
                'HorizontalFlip': {'description': '水平翻转', 'default': True},
                'VerticalFlip': {'description': '垂直翻转', 'default': False},
                'RandomRotate90': {'description': '随机旋转90度', 'default': True},
                'Rotate': {'description': '随机旋转', 'params': {'limit': '旋转角度范围'}},
                'RandomScale': {'description': '随机缩放', 'params': {'scale_limit': '缩放范围'}},
                'ShiftScaleRotate': {'description': '平移-缩放-旋转组合', 'params': {'shift_limit': '平移范围'}},
            },
            'photometric': {
                'RandomBrightnessContrast': {'description': '随机亮度和对比度', 'params': {'brightness_limit': '亮度范围', 'contrast_limit': '对比度范围'}},
                'HueSaturationValue': {'description': '随机色调、饱和度和明度', 'params': {'hue_shift_limit': '色调范围', 'sat_shift_limit': '饱和度范围'}},
            },
            'noise': {
                'Blur': {'description': '模糊', 'params': {'blur_limit': '模糊核大小'}},
                'GaussNoise': {'description': '高斯噪声', 'params': {'var_limit': '噪声方差范围'}},
                'CoarseDropout': {'description': '随机遮挡 (Cutout)', 'params': {'num_holes': '遮挡数量', 'max_h_size': '遮挡高度', 'max_w_size': '遮挡宽度'}},
            }
        }


# 创建单例实例
augmentation_service = AugmentationService()
