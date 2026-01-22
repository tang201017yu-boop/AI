#!/bin/bash

# OpenCV Platform - RTX 5080 工作站部署脚本
# 优化目标：充分发挥 GPU 性能，支持混合精度训练和推理

set -e

echo "======================================"
echo "  OpenCV Platform - RTX 5080 部署"
echo "======================================"
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_DIR/data"
VENV_DIR="$PROJECT_DIR/venv"

# 检测 CUDA
echo -e "${BLUE}[1/6]${NC} 检测 CUDA 环境..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA 驱动已安装${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  GPU: RTX 5080 (推测)"
else
    echo -e "${YELLOW}⚠️  未检测到 NVIDIA 驱动，将使用 CPU 模式${NC}"
fi

# 检查 Python
echo -e "${BLUE}[2/6]${NC} 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 已安装${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "  版本: $PYTHON_VERSION"

# 检查 CUDA
echo -e "${BLUE}[3/6]${NC} 检查 CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}✓ CUDA $CUDA_VERSION 已安装${NC}"
else
    echo -e "${YELLOW}⚠️  CUDA 未安装，将使用 CPU 推理（较慢）${NC}"
fi

# 创建虚拟环境
echo -e "${BLUE}[4/6]${NC} 配置虚拟环境..."
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
else
    echo -e "${YELLOW}  创建虚拟环境...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ 虚拟环境创建完成${NC}"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 安装/更新依赖
echo -e "${BLUE}[5/6]${NC} 安装依赖..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# 安装 PyTorch GPU 版本（如果 CUDA 可用）
if command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}  安装 PyTorch GPU 版本...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 > /dev/null 2>&1 || true
else
    echo -e "${YELLOW}  安装 PyTorch CPU 版本...${NC}"
    pip install torch torchvision torchaudio > /dev/null 2>&1 || true
fi

# 安装项目依赖
pip install -r "$PROJECT_DIR/requirements.txt" > /dev/null 2>&1
echo -e "${GREEN}✓ 依赖安装完成${NC}"

# 确保目录存在
echo -e "${BLUE}[6/6]${NC} 初始化数据目录..."
mkdir -p "$DATA_DIR"/{datasets,models,exports,uploads,logs,annotation_projects}
echo -e "${GREEN}✓ 目录创建完成${NC}"

# 停止 Docker 服务（避免端口冲突）
echo ""
echo -e "${YELLOW}停止 Docker 服务（避免端口冲突）...${NC}"
docker compose -f docker-compose.dev.yml down 2>/dev/null || true
docker compose -f docker-compose.gpu.yml down 2>/dev/null || true
docker rm -f opencv-postgres opencv-minio opencv-platform-gpu 2>/dev/null || true

# 设置环境变量（GPU 优化）
export STORAGE_BACKEND=local
export DATABASE_URL=sqlite:///./data/opencv.db
export UPLOADS_DIR=./data/uploads
export DEFAULT_EPOCHS=100
export DEFAULT_BATCH_SIZE=32
export DEFAULT_AMP=True
export DEFAULT_WORKERS=8
export CUDA_LAUNCH_BLOCKING=0

echo ""
echo "======================================"
echo -e "${GREEN}🚀 启动 OpenCV Platform (RTX 5080)${NC}"
echo "======================================"
echo ""
echo "  🔥 GPU 模式: $(command -v nvidia-smi &> /dev/null && echo '启用' || echo '未检测到')"
echo "  📁 数据目录: $DATA_DIR"
echo "  📁 模型目录: $DATA_DIR/models"
echo "  📁 上传目录: $DATA_DIR/uploads"
echo ""
echo "  📍 访问地址: http://localhost:8000"
echo "  📚 API 文档: http://localhost:8000/api/docs"
echo ""
echo "  训练优化配置:"
echo "    - 默认批次大小: 32"
echo "    - 混合精度训练: 启用"
echo "    - 数据加载 workers: 8"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动
cd "$PROJECT_DIR"
exec python app.py
