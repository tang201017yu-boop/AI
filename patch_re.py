content = open('/root/wuyu/Vision_Platform/backend/api/routes.py').read()
old = '        import json\n\n        # 保存上传文件\n        # 清理文件名'
new = '        import json\n        import re\n\n        # 保存上传文件\n        # 清理文件名'
if old in content:
    content = content.replace(old, new, 1)
    open('/root/wuyu/Vision_Platform/backend/api/routes.py', 'w').write(content)
    print('OK - re imported')
else:
    # already has import re or different whitespace
    i = content.find('_sanitize')
    print(repr(content[max(0,i-80):i+20]))
