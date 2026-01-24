"""
工具函数 - Utility Functions
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


def allowed_file(filename: str, allowed_extensions: List[str]) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [ext.lower() for ext in allowed_extensions]


def get_file_size_mb(file_path: str) -> float:
    """获取文件大小（MB）"""
    return os.path.getsize(file_path) / (1024 * 1024)


def create_directory(directory: str) -> bool:
    """创建目录"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        return False


def delete_directory(directory: str) -> bool:
    """删除目录"""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        return True
    except Exception as e:
        print(f"Error deleting directory {directory}: {e}")
        return False


def list_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """列出目录中的文件"""
    files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        return files

    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower().lstrip('.') in extensions:
                files.append(str(file_path))

    return files


def get_unique_filename(directory: str, filename: str) -> str:
    """获取唯一文件名"""
    file_path = Path(directory) / filename

    if not file_path.exists():
        return filename

    base_name = file_path.stem
    extension = file_path.suffix
    counter = 1

    while file_path.exists():
        new_name = f"{base_name}_{counter}{extension}"
        file_path = Path(directory) / new_name
        counter += 1

    return file_path.name


def save_uploaded_file(upload_file, destination: str) -> str:
    """保存上传的文件"""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    except Exception as e:
        raise Exception(f"Error saving file: {e}")


def extract_zip(zip_path: str, extract_to: str) -> Dict[str, Any]:
    """
    解压 ZIP 文件
    支持大文件（50GB+），保留完整目录结构
    返回解压统计信息
    """
    import zipfile
    import os

    result = {
        "success": False,
        "total_files": 0,
        "extracted_files": 0,
        "image_files": 0,
        "label_files": 0,
        "skipped_files": 0,
        "errors": []
    }

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取文件列表
            all_files = zip_ref.namelist()
            result["total_files"] = len(all_files)

            # 解压每个文件，保留目录结构
            for file_name in all_files:
                try:
                    # 跳过目录和特殊文件
                    if file_name.endswith('/') or file_name.startswith('__MACOSX'):
                        result["skipped_files"] += 1
                        continue

                    # 解压文件，保留相对路径
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

                    # 统计图片和标签文件
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        result["image_files"] += 1
                    elif ext == '.txt':
                        result["label_files"] += 1

                except Exception as e:
                    result["errors"].append(f"Error extracting {file_name}: {str(e)}")

        result["success"] = True
        return result

    except Exception as e:
        result["errors"].append(f"Error opening zip: {str(e)}")
        return result
