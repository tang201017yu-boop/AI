"""修复 backend/api/routes.py 中所有解决方案端点：
1. 扩展文件类型限制（允许图片+视频）
2. 给所有端点的文件名做 URL 安全清理
"""
import re

ROUTES_PATH = '/root/wuyu/Vision_Platform/backend/api/routes.py'

with open(ROUTES_PATH, encoding='utf-8') as f:
    content = f.read()

original = content

# ── 1. 在文件顶部加 import re（如果没有）──────────────────────────
if 'import re\n' not in content and 'import re\r' not in content:
    content = content.replace(
        'import sys\n',
        'import sys\nimport re\n',
        1
    )
    print('[1] Added import re at top')
else:
    print('[1] import re already present')

# ── 2. 通用文件名清理函数（只加一次）────────────────────────────────
SANITIZE_FUNC = '''
def _sanitize_filename(name: str) -> str:
    """把文件名中的 URL 特殊字符替换为下划线，避免静态文件路径解析错误"""
    if not name:
        return 'upload'
    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else 'bin'
    base = re.sub(r'[^A-Za-z0-9_\\-]', '_', name.rsplit('.', 1)[0])
    return (base[:60] + '.' + ext)

'''

# 插入到 router = APIRouter() 后面
ANCHOR = 'router = APIRouter()\n'
if '_sanitize_filename' not in content and ANCHOR in content:
    content = content.replace(ANCHOR, ANCHOR + SANITIZE_FUNC, 1)
    print('[2] Added _sanitize_filename helper')
else:
    print('[2] _sanitize_filename already present or anchor not found')

# ── 3. 扩展各端点的允许文件类型 ─────────────────────────────────────
IMAGE_EXTS = '["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"]'
ALL_EXTS   = '["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "mp4", "avi", "mov", "mkv", "flv", "wmv"]'
VIDEO_EXTS = '["mp4", "avi", "mov", "mkv", "flv", "wmv"]'

# speed-estimation: 只允许视频 -> 允许图片+视频
n = content.count('if not allowed_file(file.filename, ["mp4", "avi", "mov"]):\n        raise HTTPException(status_code=400, detail="Only video files are allowed")')
content = content.replace(
    'if not allowed_file(file.filename, ["mp4", "avi", "mov"]):\n        raise HTTPException(status_code=400, detail="Only video files are allowed")',
    f'if not allowed_file(file.filename, {ALL_EXTS}):\n        raise HTTPException(status_code=400, detail="不支持的文件类型，请上传图片或视频文件")'
)
print(f'[3] Expanded video-only type checks: {n} places')

# object-counting / heatmap (old) / blur / crop: 只允许常见图片 -> 允许图片+视频
n2 = content.count('if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"]):\n        raise HTTPException(status_code=400, detail="Invalid file type")')
content = content.replace(
    'if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"]):\n        raise HTTPException(status_code=400, detail="Invalid file type")',
    f'if not allowed_file(file.filename, {ALL_EXTS}):\n        raise HTTPException(status_code=400, detail="不支持的文件类型，请上传图片或视频文件")'
)
print(f'[3b] Expanded image+video type checks: {n2} places')

n3 = content.count('if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp"]):\n        raise HTTPException(status_code=400, detail="Invalid file type")')
content = content.replace(
    'if not allowed_file(file.filename, ["jpg", "jpeg", "png", "bmp"]):\n        raise HTTPException(status_code=400, detail="Invalid file type")',
    f'if not allowed_file(file.filename, {ALL_EXTS}):\n        raise HTTPException(status_code=400, detail="不支持的文件类型，请上传图片或视频文件")'
)
print(f'[3c] Expanded image-only type checks: {n3} places')

# ── 4. 给所有端点加文件名清理（替换 file.filename 为清理后的版本）──
# 对每个 get_unique_filename 调用，把 file.filename 替换为 _sanitize_filename(file.filename or "upload")
# 但要排除已经处理过的 heatmap（它用 clean_fname）
old_pattern = 'filename = get_unique_filename(str(settings.UPLOADS_DIR), file.filename)'
new_pattern = 'filename = get_unique_filename(str(settings.UPLOADS_DIR), _sanitize_filename(file.filename or "upload"))'
n4 = content.count(old_pattern)
content = content.replace(old_pattern, new_pattern)
print(f'[4] Added filename sanitization: {n4} places')

if content != original:
    with open(ROUTES_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print('\nAll patches applied successfully!')
else:
    print('\nNo changes made (already patched?)')
