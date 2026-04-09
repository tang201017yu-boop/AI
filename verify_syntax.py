import ast
with open('/root/wuyu/Vision_Platform/backend/api/routes.py') as f:
    src = f.read()
try:
    ast.parse(src)
    print('Syntax OK')
except SyntaxError as e:
    print(f'Syntax Error: {e}')
