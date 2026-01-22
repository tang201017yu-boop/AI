#!/bin/bash

# OpenCV Platform - 本地开发环境 (MacBook Air M3)
# 只用于前端开发，连接远程服务器后端

set -e

echo "======================================"
echo "  OpenCV Platform - 本地开发环境"
echo "======================================"
echo ""
echo "  MacBook Air M3 (开发机) <---> RTX 5080 (服务器)"
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查 Python
echo -e "${BLUE}[1/4]${NC} 检查环境..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 已安装${NC}"

# 检查 Node.js (可选，用于前端构建)
if command -v node &> /dev/null; then
    echo -e "${GREEN}✓ Node.js 已安装${NC}"
else
    echo -e "${YELLOW}⚠️  Node.js 未安装（可选）${NC}"
fi

# 检查虚拟环境
VENV_DIR="$PROJECT_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
else
    echo -e "${YELLOW}  创建虚拟环境...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 安装轻量级依赖（不需要 GPU）
echo -e "${BLUE}[2/4]${NC} 安装依赖..."
pip install --quiet --upgrade pip

# 只安装前端相关的依赖，不需要 torch/ultralytics
pip install --quiet fastapi uvicorn python-multipart jinja2 aiofiles 2>/dev/null || true

echo -e "${GREEN}✓ 依赖安装完成${NC}"

# 确保目录存在
echo -e "${BLUE}[3/4]${NC} 初始化目录..."
mkdir -p "$PROJECT_DIR"/data/{datasets,models,exports,uploads,logs,annotation_projects}
echo -e "${GREEN}✓ 目录创建完成${NC}"

# 读取服务器配置
echo -e "${BLUE}[4/4]${NC} 检查服务器连接..."
SERVER_URL="${API_SERVER_URL:-http://192.168.1.100:8000}"

if curl -s --connect-timeout 2 "$SERVER_URL/api/v1/system/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 连接到服务器: $SERVER_URL${NC}"
else
    echo -e "${YELLOW}⚠️  无法连接到服务器: $SERVER_URL${NC}"
    echo "   设置环境变量: export API_SERVER_URL=http://<你的服务器IP>:8000"
fi

echo ""
echo "======================================"
echo -e "${GREEN}🚀 启动本地开发环境${NC}"
echo "======================================"
echo ""
echo "  📡 服务器: $SERVER_URL"
echo ""
echo "  工作模式:"
echo "    1. 前端页面 -> 连接到远程服务器"
echo "    2. API 请求 -> 转发到远程服务器"
echo "    3. 代码修改 -> 自动热重载"
echo ""
echo "  访问地址: http://localhost:8000"
echo "  API 代理: -> $SERVER_URL"
echo ""
echo "  按 Ctrl+C 停止"
echo ""

# 启动 uvicorn，代理 API 请求到远程服务器
cd "$PROJECT_DIR"
exec uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir "$PROJECT_DIR/frontend" \
    --reload-dir "$PROJECT_DIR/backend" \
    --log-level info
