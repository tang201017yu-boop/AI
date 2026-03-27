#!/usr/bin/env python3
"""在后端机器上执行此脚本以修复缩略图404问题"""
import re

file_path = 'backend/modules/data_preparation/routes.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old = '''@router.get("/datasets/{name}/thumbnails/{image_name}")
async def get_thumbnail(name: str, image_name: str):
    """获取图片缩略图"""
    thumb_path = dataset_service.get_thumbnail_path(name, image_name)
    if thumb_path and Path(thumb_path).exists():
        from fastapi.responses import FileResponse
        return FileResponse(thumb_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="缩略图不存在")'''

new = '''@router.get("/datasets/{name}/thumbnails/{image_name}")
async def get_thumbnail(name: str, image_name: str):
    """获取图片缩略图，不存在时实时生成"""
    from fastapi.responses import FileResponse, StreamingResponse
    import io

    # 先尝试已生成的缩略图
    thumb_path = dataset_service.get_thumbnail_path(name, image_name)
    if thumb_path and Path(thumb_path).exists():
        return FileResponse(thumb_path, media_type="image/jpeg")

    # 回退：找到原图，实时生成缩略图
    from pathlib import Path as PathLib
    dataset_dir = dataset_service._get_dataset_path(name)
    images_dir = dataset_dir / "images" if (dataset_dir / "images").exists() else dataset_dir / "image"
    image_path = images_dir / image_name
    if not image_path.exists():
        found = list(images_dir.rglob(image_name)) if images_dir.exists() else []
        if not found:
            raise HTTPException(status_code=404, detail="图片不存在")
        image_path = found[0]

    try:
        from PIL import Image
        img = Image.open(image_path)
        img.thumbnail((256, 256), Image.LANCZOS)
        if img.mode in (\'RGBA\', \'P\'):
            img = img.convert(\'RGB\')
        buf = io.BytesIO()
        img.save(buf, format=\'JPEG\', quality=85)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缩略图生成失败: {str(e)}")'''

if old in content:
    content = content.replace(old, new)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('修改成功')
else:
    print('未找到目标代码，可能已经修改过了或代码不匹配')
    # 检查是否已包含新代码
    if 'StreamingResponse' in content and 'rglob(image_name)' in content:
        print('-> 检测到新代码已存在，无需修改')
