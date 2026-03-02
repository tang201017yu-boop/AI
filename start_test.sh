#!/bin/bash
# ========================================
# YOLO Platform 启动脚本
# 使用 uv 管理环境
# ========================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  YOLO Platform${NC}"
echo -e "${BLUE}  AI原生应用: 让AI帮你写代码${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}错误: uv 未安装${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} uv: $(uv --version)"

# 虚拟环境
ENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$ENV_DIR" ]; then
    echo -e "${YELLOW}创建虚拟环境...${NC}"
    uv venv "$ENV_DIR" --python=python3.12
fi

source "$ENV_DIR/bin/activate"

# 安装依赖
echo -e "${YELLOW}检查依赖...${NC}"
uv pip install -r "$PROJECT_ROOT/requirements.min.txt" --python "$ENV_DIR/bin/python" 2>/dev/null || true
uv pip install psycopg2-binary --python "$ENV_DIR/bin/python" 2>/dev/null || true

export PYTHONPATH="$PROJECT_ROOT"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  启动后端服务 (端口 8000)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Web界面:   http://localhost:8000"
echo -e "API文档:   http://localhost:8000/api/docs"
echo -e "前端:      已集成在后端 (静态文件)"
echo ""
echo -e "${CYAN}AI原生应用示例:${NC}"
echo -e "  • 检测隧洞裂缝: '检测隧洞裂缝，超过5mm且有渗水要报警'"
echo -e "  • 安全施工:    '工人没戴安全帽要报警'"
echo -e "  • 边坡滑坡:    '检测边坡滑坡风险'"
echo ""

exec "$ENV_DIR/bin/python" app.py
