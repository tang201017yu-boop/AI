#!/bin/bash

# OpenCV Platform 本地快速启动脚本 (macOS M3)
# 使用 Apple Silicon 优化配置

set -e

echo "======================================"
echo "  OpenCV Platform - 本地启动 (M3)"
echo "======================================"
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker 未安装，请先安装 Docker Desktop for Mac${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker 未运行，正在启动...${NC}"
    open -a Docker
    echo "等待 Docker 启动..."
    for i in {1..30}; do
        if docker info &> /dev/null; then
            echo -e "${GREEN}✓ Docker 已启动${NC}"
            break
        fi
        sleep 2
    done
fi

echo -e "${GREEN}✓ Docker 运行正常${NC}"

# 检查端口占用
check_port() {
    if lsof -i:$1 &> /dev/null; then
        echo -e "${YELLOW}⚠️  端口 $1 已被占用${NC}"
        return 1
    fi
    return 0
}

echo ""
echo "检查端口占用..."
check_port 8000 || echo "  (如需强制释放: lsof -ti:8000 | xargs kill -9)"
check_port 5432 || echo "  (如需强制释放: lsof -ti:5432 | xargs kill -9)"
check_port 9000 || echo "  (如需强制释放: lsof -ti:9000 | xargs kill -9)"

echo ""
echo "停止旧容器（如果有）..."
docker compose $COMPOSE_FILE down 2>/dev/null || true
docker rm -f opencv-postgres opencv-minio opencv-platform-dev opencv-platform-m3 2>/dev/null || true

echo ""
echo "构建镜像（Apple Silicon 优化）..."
echo "⏱️  首次构建需要 5-15 分钟，请耐心等待..."
echo ""

# 检测架构
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "🍎 检测到 Apple Silicon (M1/M2/M3)"
    # 检查是否有 GPU 支持的镜像，没有则用 CPU 版本
    IMAGE_TYPE="CPU (M3 优化)"
else
    echo "🖥️  检测到 Intel Mac"
    IMAGE_TYPE="CPU"
fi

# 检测架构并选择合适的配置
if [ "$ARCH" = "arm64" ]; then
    echo "🍎 使用 Apple Silicon M3 优化配置"
    COMPOSE_FILE="-f docker-compose.m3.yml"
else
    echo "🖥️  使用标准配置"
    COMPOSE_FILE="-f docker-compose.dev.yml"
fi

echo ""
echo "构建镜像..."
docker compose $COMPOSE_FILE build --no-cache

echo ""
echo "启动服务..."
docker compose $COMPOSE_FILE up -d

echo ""
echo "等待服务启动..."
sleep 5

# 检查服务状态
echo ""
echo "检查服务状态..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/system/health &> /dev/null; then
        echo -e "${GREEN}✓ 服务启动成功！${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ 服务启动失败${NC}"
        echo "查看日志: docker compose $COMPOSE_FILE logs"
    fi
    sleep 2
done

echo ""
echo "======================================"
echo -e "${GREEN}🎉  启动完成！${NC}"
echo "======================================"
echo ""
echo "  📱 访问地址: http://localhost:8000"
echo "  📚 API 文档: http://localhost:8000/api/docs"
echo ""
echo "常用命令:"
echo "  查看日志: docker compose $COMPOSE_FILE logs -f"
echo "  停止服务: docker compose $COMPOSE_FILE down"
echo "  重启服务: docker compose $COMPOSE_FILE restart"
echo ""

# 自动打开浏览器
if command -v open &> /dev/null; then
    echo "正在打开浏览器..."
    sleep 2
    open http://localhost:8000
fi
