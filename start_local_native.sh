#!/bin/bash

# OpenCV Platform 本地启动脚本 (不依赖 Docker)
# 支持两种模式：独立模式 / 开发模式（连接远程服务器）

set -e

echo "======================================"
echo "  OpenCV Platform - 本地启动"
echo "======================================"
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 检查参数
MODE="standalone"
if [ "$1" = "--dev" ] || [ "$2" = "--dev" ]; then
    MODE="dev"
fi

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python3 已安装${NC}"

# 检查虚拟环境
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
else
    echo -e "${YELLOW}⚠️  正在创建虚拟环境...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ 虚拟环境创建完成${NC}"
fi

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source venv/bin/activate

# 检查依赖
echo ""
echo "检查依赖..."
pip install --quiet --upgrade pip 2>/dev/null

if [ "$MODE" = "dev" ]; then
    # 开发模式：只需轻量级依赖
    pip install --quiet fastapi uvicorn python-multipart jinja2 httpx 2>/dev/null
    echo -e "${GREEN}✓ 开发模式依赖已安装${NC}"
else
    # 独立模式：需要完整依赖
    if ! pip show ultralytics &> /dev/null; then
        echo -e "${YELLOW}⚠️  安装完整依赖中...${NC}"
        pip install -r requirements.txt
        echo -e "${GREEN}✓ 依赖安装完成${NC}"
    else
        echo -e "${GREEN}✓ 依赖已安装${NC}"
    fi
fi

# 确保目录存在
echo ""
echo "创建数据目录..."
mkdir -p data/datasets data/models data/exports data/uploads data/logs data/annotation_projects data/storage
echo -e "${GREEN}✓ 目录创建完成${NC}"

# 检查端口占用
echo ""
echo "检查端口占用..."
check_port() {
    if lsof -i:$1 &> /dev/null 2>/dev/null; then
        echo -e "${YELLOW}⚠️  端口 $1 已被占用${NC}"
        return 1
    fi
    return 0
}
check_port 8000 || echo "  (如需释放: lsof -ti:8000 | xargs kill -9)"

# 设置环境变量
if [ "$MODE" = "dev" ]; then
    # 开发模式
    export DEV_MODE=true

    # 获取服务器地址
    if [ -n "$2" ] && [ "$1" = "--dev" ]; then
        SERVER_URL="$2"
    elif [ -n "$REMOTE_API_URL" ]; then
        SERVER_URL="$REMOTE_API_URL"
    else
        SERVER_URL="http://localhost:8000"
    fi
    export REMOTE_API_URL="$SERVER_URL"

    echo ""
    echo -e "${BLUE}[开发模式]${NC}"
    echo -e "${CYAN}  🔗 远程服务器: $SERVER_URL${NC}"
else
    # 独立模式
    export DEV_MODE=false
    export STORAGE_BACKEND=local
    export DATABASE_URL=sqlite:///./data/opencv.db
    export UPLOADS_DIR=./data/uploads

    echo ""
    echo -e "${BLUE}[独立模式]${NC}"
    echo -e "${GREEN}  🖥️  GPU 加速: $(python3 -c "import torch; print('可用' if torch.cuda.is_available() else '不可用')" 2>/dev/null || echo "未安装")${NC}"
fi

echo ""
echo "======================================"
if [ "$MODE" = "dev" ]; then
    echo -e "${GREEN}🚀 启动开发模式...${NC}"
else
    echo -e "${GREEN}🚀 启动 OpenCV Platform...${NC}"
fi
echo "======================================"
echo ""
echo "  📁 数据目录: ./data"
if [ "$MODE" = "dev" ]; then
    echo "  🔗 API 代理: -> $SERVER_URL"
    echo "  ⚠️  训练和推理将在远程服务器执行"
else
    echo "  📁 模型保存: ./data/models"
    echo "  📁 上传目录: ./data/uploads"
fi
echo ""
echo "  📱 访问地址: http://localhost:8000"
echo "  📚 API 文档: http://localhost:8000/api/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动
python app.py
