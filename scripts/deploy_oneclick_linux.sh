#!/usr/bin/env bash
# Linux 一键部署：配置 Docker DNS/镜像加速、准备目录、启动生产栈（CPU）。
# 可选后台构建 GPU 镜像（耗时长，不阻塞上线）。
#
# 用法（在仓库根目录或任意路径）：
#   sudo bash scripts/deploy_oneclick_linux.sh
#   sudo bash scripts/deploy_oneclick_linux.sh --gpu-bg
#
# 环境变量（可选）：
#   APP_DIR=/root/AI          项目目录（默认：本脚本所在仓库根目录）
#   REPO_URL=...              无本地代码时从此克隆
#   BRANCH=dev                分支

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APP="$(cd "$SCRIPT_DIR/.." && pwd)"

REPO_URL="${REPO_URL:-https://github.com/tang201017yu-boop/AI.git}"
BRANCH="${BRANCH:-dev}"
APP_DIR="${APP_DIR:-$DEFAULT_APP}"
GPU_BG=0
[[ "${1:-}" == "--gpu-bg" ]] && GPU_BG=1

if [[ "${EUID:-0}" -ne 0 ]]; then
  echo "请用 root 执行（需要写 /etc/docker）：sudo bash $0 $*"
  exit 1
fi

echo "==> [1/6] 配置 Docker DNS 与镜像加速（已备份旧 daemon.json）"
mkdir -p /etc/docker
if [[ -f /etc/docker/daemon.json ]]; then
  cp /etc/docker/daemon.json "/etc/docker/daemon.json.bak.$(date +%s)"
fi
cat >/etc/docker/daemon.json <<'JSON'
{
  "dns": ["223.5.5.5", "119.29.29.29", "8.8.8.8"],
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://docker.m.daocloud.io"
  ]
}
JSON
systemctl daemon-reload
systemctl restart docker
sleep 2

echo "==> [2/6] 拉取基础镜像（验证网络）"
docker pull python:3.12-slim

echo "==> [3/6] 准备项目目录: $APP_DIR"
if [[ ! -f "$APP_DIR/docker-compose.prod.yml" ]]; then
  if [[ -d "$APP_DIR/.git" ]]; then
    echo "目录存在但缺少 compose，请检查 APP_DIR"
    exit 1
  fi
  mkdir -p "$(dirname "$APP_DIR")"
  git clone -b "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"
git fetch origin 2>/dev/null || true
git checkout "$BRANCH" 2>/dev/null || true
git pull origin "$BRANCH" 2>/dev/null || true

if [[ ! -f ".env" ]] && [[ -f ".env.example" ]]; then
  cp .env.example .env
  echo "已创建 .env"
fi

# Dockerfile.gpu 需要 frontend/；若仅有 frontend-react 则建软链
if [[ ! -e frontend ]] && [[ -d frontend-react ]]; then
  ln -sfn frontend-react frontend
  echo "已创建 frontend -> frontend-react"
fi
mkdir -p config logs

echo "==> [4/6] 清理可能失败的残留网络（忽略错误）"
docker network rm ai_opencv-network 2>/dev/null || true

echo "==> [5/6] 构建并启动生产栈（CPU，端口 8000）"
export DOCKER_BUILDKIT=1
docker compose -f docker-compose.prod.yml up -d --build

echo "==> [6/6] 状态"
docker compose -f docker-compose.prod.yml ps

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo ""
echo "部署完成。浏览器访问: http://${IP}:8000"
echo "健康检查: curl -s http://127.0.0.1:8000/api/v1/system/health"
echo "日志: docker compose -f $APP_DIR/docker-compose.prod.yml logs -f opencv-platform"

if [[ "$GPU_BG" -eq 1 ]]; then
  echo ""
  echo "后台构建 GPU 镜像（日志: /root/gpu-build.log）..."
  nohup bash -lc "cd '$APP_DIR' && export DOCKER_BUILDKIT=1 && docker compose -f docker-compose.gpu.yml build opencv-platform-gpu" \
    >/root/gpu-build.log 2>&1 &
  echo "查看进度: tail -f /root/gpu-build.log"
  echo "构建完成后可: cd $APP_DIR && docker compose -f docker-compose.prod.yml down && docker compose -f docker-compose.gpu.yml up -d"
fi
