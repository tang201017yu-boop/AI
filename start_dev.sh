#!/bin/bash
# ========================================
# 快速测试启动脚本
# ========================================

cd "$(dirname "$0")"

# 使用现有环境或创建新的
if [ -d "yolo_env" ]; then
    source yolo_env/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 检查依赖
python -c "import fastapi, ultralytics, torch" 2>/dev/null || {
    echo "缺少依赖，请运行: uv sync 或 pip install -r requirements.txt"
    exit 1
}

# 启动服务
echo "启动服务在 http://localhost:8000"
echo "API 文档: http://localhost:8000/api/docs"

python app.py
