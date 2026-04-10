#!/usr/bin/env bash
# Linux 一键部署（含常见网络修复）：IPv4 优先、可选关 IPv6、系统 DNS、Docker 直连 Hub、启动生产栈。
# 可选后台构建 GPU 镜像。
#
# 用法：
#   sudo bash scripts/deploy_oneclick_linux.sh
#   sudo bash scripts/deploy_oneclick_linux.sh --gpu-bg
#
# 环境变量（可选）：
#   APP_DIR=/root/AI
#   REPO_URL=https://github.com/tang201017yu-boop/AI.git
#   BRANCH=dev
#   USE_DOCKER_MIRROR=1          启用 DaoCloud（内网 DNS 差时勿开）
#   SKIP_IPV6_FIX=1               不关闭 IPv6（默认会关，避免 Docker Hub 走 V6 失败）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APP="$(cd "$SCRIPT_DIR/.." && pwd)"

REPO_URL="${REPO_URL:-https://github.com/tang201017yu-boop/AI.git}"
BRANCH="${BRANCH:-dev}"
APP_DIR="${APP_DIR:-$DEFAULT_APP}"
GPU_BG=0
[[ "${1:-}" == "--gpu-bg" ]] && GPU_BG=1

if [[ "${EUID:-0}" -ne 0 ]]; then
  echo "请用 root 执行: sudo bash $0 $*"
  exit 1
fi

echo "==> [1/8] 系统网络：IPv4 优先 + 修复 systemd-resolved DNS"
if ! grep -q 'precedence ::ffff:0:0/96' /etc/gai.conf 2>/dev/null; then
  echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf
fi
mkdir -p /etc/systemd/resolved.conf.d
cat >/etc/systemd/resolved.conf.d/vision-platform-dns.conf <<'EOF'
[Resolve]
DNS=223.5.5.5 8.8.8.8
FallbackDNS=119.29.29.29
EOF
if systemctl is-active --quiet systemd-resolved 2>/dev/null; then
  systemctl restart systemd-resolved
fi

if [[ "${SKIP_IPV6_FIX:-0}" != "1" ]]; then
  echo "==> [2/8] 关闭 IPv6（避免 registry-1.docker.io 走 IPv6 报 network unreachable）"
  sysctl -w net.ipv6.conf.all.disable_ipv6=1 >/dev/null
  sysctl -w net.ipv6.conf.default.disable_ipv6=1 >/dev/null
  cat >/etc/sysctl.d/99-vision-platform-ipv4-docker.conf <<'EOF'
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
EOF
  sysctl --system >/dev/null 2>&1 || true
else
  echo "==> [2/8] 已跳过 IPv6 修复（SKIP_IPV6_FIX=1）"
fi

echo "==> [3/8] Docker：去掉镜像加速碎片、仅保留 DNS、直连 Docker Hub"
mkdir -p /etc/docker /root/docker.bak.vision
if [[ -f /etc/docker/daemon.json ]]; then
  cp /etc/docker/daemon.json "/root/docker.bak.vision/daemon.json.$(date +%s)"
fi
if [[ -d /etc/docker/daemon.json.d ]]; then
  mv /etc/docker/daemon.json.d/*.json /root/docker.bak.vision/ 2>/dev/null || true
fi
if [[ "${USE_DOCKER_MIRROR:-0}" == "1" ]]; then
  cat >/etc/docker/daemon.json <<'JSON'
{
  "dns": ["223.5.5.5", "119.29.29.29", "8.8.8.8"],
  "registry-mirrors": ["https://docker.m.daocloud.io"]
}
JSON
else
  cat >/etc/docker/daemon.json <<'JSON'
{
  "dns": ["223.5.5.5", "119.29.29.29", "8.8.8.8"]
}
JSON
fi
systemctl daemon-reload
systemctl restart docker
sleep 3

echo "==> [4/8] 验证：拉取 python:3.12-slim"
if ! docker pull python:3.12-slim; then
  echo "拉取失败。可尝试：检查出口防火墙是否放行 443；或设置内网 HTTP 代理后重试。"
  exit 1
fi

echo "==> [5/8] 准备项目目录: $APP_DIR"
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
git fetch github 2>/dev/null || true
git checkout "$BRANCH" 2>/dev/null || true
git pull origin "$BRANCH" 2>/dev/null || git pull github "$BRANCH" 2>/dev/null || true

if [[ ! -f ".env" ]] && [[ -f ".env.example" ]]; then
  cp .env.example .env
  echo "已创建 .env"
fi

if [[ ! -e frontend ]] && [[ -d frontend-react ]]; then
  ln -sfn frontend-react frontend
  echo "已创建 frontend -> frontend-react"
fi
mkdir -p config logs

echo "==> [6/8] 清理可能失败的残留网络"
docker network rm ai_opencv-network 2>/dev/null || true

echo "==> [7/8] 构建并启动生产栈（CPU，端口 8000）"
export DOCKER_BUILDKIT=1
docker compose -f docker-compose.prod.yml up -d --build

echo "==> [8/8] 状态"
docker compose -f docker-compose.prod.yml ps

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo ""
echo "部署完成。浏览器: http://${IP}:8000"
echo "健康检查: curl -s http://127.0.0.1:8000/api/v1/system/health"
echo "日志: docker compose -f $APP_DIR/docker-compose.prod.yml logs -f opencv-platform"

if [[ "$GPU_BG" -eq 1 ]]; then
  echo ""
  echo "后台构建 GPU（日志 /root/gpu-build.log）..."
  nohup bash -lc "cd '$APP_DIR' && export DOCKER_BUILDKIT=1 && docker compose -f docker-compose.gpu.yml build opencv-platform-gpu" \
    >/root/gpu-build.log 2>&1 &
  echo "进度: tail -f /root/gpu-build.log"
  echo "完成后: cd $APP_DIR && docker compose -f docker-compose.prod.yml down && docker compose -f docker-compose.gpu.yml up -d"
fi
