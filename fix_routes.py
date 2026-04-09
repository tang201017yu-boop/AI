import re

path = 'backend/api/routes.py'
with open(path, encoding='utf-8') as f:
    content = f.read()

ALL_EXTS = '["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "mp4", "avi", "mov", "mkv", "flv", "wmv"]'
ERR_MSG  = '"不支持的文件类型，请上传图片或视频"'
new_check = f'if not allowed_file(file.filename, {ALL_EXTS}):\n        raise HTTPException(status_code=400, detail={ERR_MSG})'

old_video = 'if not allowed_file(file.filename, ["mp4", "avi", "mov"]):\n        raise HTTPException(status_code=400, detail="Only video files are allowed")'
old_img_vid = 'if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"]):\n        raise HTTPException(status_code=400, detail="Invalid file type")'
old_img = 'if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp"]):\n        raise HTTPException(status_code=400, detail="Invalid file type")'

n1 = content.count(old_video);   content = content.replace(old_video,   new_check)
n2 = content.count(old_img_vid); content = content.replace(old_img_vid, new_check)
n3 = content.count(old_img);     content = content.replace(old_img,     new_check)
print(f'File type checks fixed: video={n1}, img+vid={n2}, img={n3}')

# Add filename sanitization using a helper inline
old_fn = 'filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)'
new_fn  = 'filename = get_unique_filename(str(settings.UPLOADS_DIR), re.sub(r"[^A-Za-z0-9_.-]", "_", (file.filename or "upload"))[:80])'
n4 = content.count(old_fn); content = content.replace(old_fn, new_fn)
print(f'Filename sanitization added: {n4} places')

# Make sure `import re` is at top of file
if '\nimport re\n' not in content:
    content = content.replace('import sys\n', 'import sys\nimport re\n', 1)
    print('Added import re at top')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('All fixes applied successfully!')
