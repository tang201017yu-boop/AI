# -*- coding: utf-8 -*-
"""
工具函数模块 - Utility Functions
提供文件操作、目录管理、压缩解压等通用工具函数
"""
import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# 创建日志记录器
logger = logging.getLogger(__name__)


def allowed_file(filename: str, allowed_extensions: List[str]) -> bool:
    """
    检查文件扩展名是否允许上传
    Args:
        filename: 文件名
        allowed_extensions: 允许的扩展名列表
    Returns:
        bool: 是否允许
    """
    # 检查文件名是否包含扩展名
    result = '.' in filename and \
             filename.rsplit('.', 1)[1].lower() in [ext.lower() for ext in allowed_extensions]
    logger.debug(f"[工具] 检查文件 {filename}: {'允许' if result else '拒绝'}")
    return result


def get_file_size_mb(file_path: str) -> float:
    """
    获取文件大小（MB）
    Args:
        file_path: 文件路径
    Returns:
        float: 文件大小（MB）
    """
    try:
        size = os.path.getsize(file_path) / (1024 * 1024)
        logger.debug(f"[工具] 文件 {file_path} 大小: {size:.2f} MB")
        return size
    except Exception as e:
        logger.error(f"[工具] 获取文件大小失败: {file_path}, 错误: {e}")
        return 0.0


def create_directory(directory: str) -> bool:
    """
    创建目录（如果不存在）
    Args:
        directory: 目录路径
    Returns:
        bool: 是否创建成功
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"[工具] 创建目录成功: {directory}")
        return True
    except Exception as e:
        logger.error(f"[工具] 创建目录失败: {directory}, 错误: {e}")
        return False


def delete_directory(directory: str) -> bool:
    """
    删除目录及其所有内容
    Args:
        directory: 目录路径
    Returns:
        bool: 是否删除成功
    """
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info(f"[工具] 删除目录成功: {directory}")
        else:
            logger.warning(f"[工具] 目录不存在: {directory}")
        return True
    except Exception as e:
        logger.error(f"[工具] 删除目录失败: {directory}, 错误: {e}")
        return False


def list_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    列出目录中的所有文件
    Args:
        directory: 目录路径
        extensions: 文件扩展名过滤（可选）
    Returns:
        List[str]: 文件路径列表
    """
    files = []
    directory_path = Path(directory)

    # 检查目录是否存在
    if not directory_path.exists():
        logger.warning(f"[工具] 目录不存在: {directory}")
        return files

    # 递归遍历所有文件
    for file_path in directory_path.rglob("*"):
        # 只处理文件，不处理目录
        if file_path.is_file():
            # 如果没有指定扩展名过滤，则所有文件都包含
            if extensions is None or file_path.suffix.lower().lstrip('.') in extensions:
                files.append(str(file_path))

    logger.info(f"[工具] 列出文件: {directory}, 找到 {len(files)} 个文件")
    return files


def get_unique_filename(directory: str, filename: str) -> str:
    """
    生成唯一的文件名（避免覆盖已存在的文件）
    Args:
        directory: 目录路径
        filename: 原文件名
    Returns:
        str: 唯一的文件名
    """
    file_path = Path(directory) / filename

    # 如果文件不存在，直接返回原文件名
    if not file_path.exists():
        logger.debug(f"[工具] 文件名无需修改: {filename}")
        return filename

    # 分离文件名和扩展名
    base_name = file_path.stem
    extension = file_path.suffix
    counter = 1

    # 循环查找不存在的文件名
    while file_path.exists():
        new_name = f"{base_name}_{counter}{extension}"
        file_path = Path(directory) / new_name
        counter += 1

    logger.debug(f"[工具] 生成唯一文件名: {filename} -> {file_path.name}")
    return file_path.name


def save_uploaded_file(upload_file, destination: str) -> str:
    """
    保存上传的文件到目标路径
    Args:
        upload_file: FastAPI 上传文件对象
        destination: 目标文件路径
    Returns:
        str: 保存的文件路径
    """
    try:
        # 以二进制写入模式打开目标文件
        with open(destination, "wb") as buffer:
            # 从上传文件复制内容到目标文件
            shutil.copyfileobj(upload_file.file, buffer)
        file_size = get_file_size_mb(destination)
        logger.info(f"[工具] 保存文件成功: {destination}, 大小: {file_size:.2f} MB")
        return destination
    except Exception as e:
        logger.error(f"[工具] 保存文件失败: {destination}, 错误: {e}")
        raise Exception(f"Error saving file: {e}")


def extract_zip(zip_path: str, extract_to: str) -> Dict[str, Any]:
    """
    解压 ZIP 文件，支持大文件（50GB+），保留完整目录结构
    Args:
        zip_path: ZIP 文件路径
        extract_to: 解压目标目录
    Returns:
        Dict: 解压统计信息
    """
    import zipfile
    import os

    # 初始化结果统计
    result = {
        "success": False,
        "total_files": 0,
        "extracted_files": 0,
        "image_files": 0,
        "label_files": 0,
        "skipped_files": 0,
        "errors": []
    }

    logger.info(f"[工具] 开始解压 ZIP: {zip_path} -> {extract_to}")

    try:
        # 打开 ZIP 文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取 ZIP 内所有文件列表
            all_files = zip_ref.namelist()
            result["total_files"] = len(all_files)
            logger.info(f"[工具] ZIP 内共有 {len(all_files)} 个文件/目录")

            # 遍历并解压每个文件
            for file_name in all_files:
                try:
                    # 跳过目录（以 / 结尾）
                    if file_name.endswith('/'):
                        result["skipped_files"] += 1
                        continue

                    # 跳过 macOS 系统文件
                    if file_name.startswith('__MACOSX'):
                        result["skipped_files"] += 1
                        continue

                    # 构建目标文件路径（保留相对目录结构）
                    target_path = os.path.join(extract_to, file_name)

                    # 创建目标目录
                    target_dir = os.path.dirname(target_path)
                    if target_dir:
                        os.makedirs(target_dir, exist_ok=True)

                    # 解压文件
                    with zip_ref.open(file_name) as source:
                        with open(target_path, 'wb') as target:
                            target.write(source.read())

                    result["extracted_files"] += 1

                    # 统计图片文件
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        result["image_files"] += 1
                    # 统计标签文件（YOLO 格式）
                    elif ext == '.txt':
                        result["label_files"] += 1

                except Exception as e:
                    error_msg = f"解压文件失败 {file_name}: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(f"[工具] {error_msg}")

        result["success"] = True
        logger.info(f"[工具] ZIP 解压完成: 提取 {result['extracted_files']} 个文件, "
                   f"图片 {result['image_files']} 个, 标签 {result['label_files']} 个")
        return result

    except Exception as e:
        error_msg = f"打开 ZIP 文件失败: {str(e)}"
        result["errors"].append(error_msg)
        logger.error(f"[工具] {error_msg}")
        return result
