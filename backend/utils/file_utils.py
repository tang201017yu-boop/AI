"""
文件工具函数
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional, Union
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
    """获取唯一文件名（如果文件存在则添加序号）"""
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


def uploads_public_url(file_path: Union[str, Path], uploads_dir: Union[str, Path]) -> str:
    """将上传目录内的绝对路径转为对外 URL（保留子目录，如 cropped-objects/…）。"""
    p = Path(file_path)
    root = Path(uploads_dir)
    try:
        rel = p.relative_to(root)
    except ValueError:
        return f"/uploads/{p.name}"
    return f"/uploads/{rel.as_posix()}"


def save_uploaded_file(upload_file, destination: str) -> str:
    """保存上传的文件"""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    except Exception as e:
        raise Exception(f"Error saving file: {e}")
