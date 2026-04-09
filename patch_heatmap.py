import re

with open('/root/wuyu/Vision_Platform/backend/api/routes.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '''        filename = get_unique_filename(str(settings.UPLOADS_DIR), fname or "upload.jpg")'''

new = '''        # 清理文件名，移除 = & , 等 URL 特殊字符，防止静态文件 URL 解析出错
        def _sanitize(name):
            ext = name.rsplit('.', 1)[-1].lower() if '.' in name else 'jpg'
            safe = re.sub(r'[^A-Za-z0-9_\\-]', '_', name.rsplit('.', 1)[0]) + '.' + ext
            return safe[:80]
        clean_fname = _sanitize(fname) if fname else 'upload.jpg'
        logger.info(f"[heatmap] 文件名清理: {fname!r} -> {clean_fname!r}")
        filename = get_unique_filename(str(settings.UPLOADS_DIR), clean_fname)'''

if old in content:
    content = content.replace(old, new, 1)
    # 确保 re 已导入（routes.py 顶部已有 import re？不一定，直接在函数内用 import）
    # 在函数内 import re
    content = content.replace(
        'import json\n        import re as _re\n',
        'import json\n'
    )
    # 加 import re 到函数 try 块内
    content = content.replace(
        '        import json\n\n        # 清理文件名',
        '        import json\n        import re\n\n        # 清理文件名'
    )
    with open('/root/wuyu/Vision_Platform/backend/api/routes.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('PATCHED OK')
else:
    print('NOT FOUND')
    idx = content.find('get_unique_filename(str(settings.UPLOADS_DIR), fname')
    print(f'index: {idx}')
    if idx >= 0:
        print(repr(content[idx-20:idx+120]))
