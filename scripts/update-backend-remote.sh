#!/usr/bin/env bash
# 在已克隆本仓库的服务器上执行：拉取最新代码 → 更新依赖/镜像 → 重启后端。
#
# 使用（SSH 登录到 192.168.x.x 后，进入项目根目录）:
#   chmod +x scripts/update-backend-remote.sh
#   bash scripts/update-backend-remote.sh
#
# 环境变量:
#   GIT_BRANCH=main              git pull 使用的远程分支（默认：当前分支）
#   GIT_REMOTE=origin            默认 origin
#   USE_VENV=1                   不用 Docker，改用本目录 venv + uvicorn（需已安装 python3-venv）
#   UVICORN_PORT=8000            USE_VENV=1 时监听端口
#   COMPOSE_FILE=docker-compose.prod.yml   Docker 模式下的 compose 文件
#   COMPOSE_SERVICE=opencv-platform          仅重建并启动该服务（加快速度）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-}"
if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
fi

echo "==> 仓库: $ROOT"
echo "==> 拉取代码: $REMOTE / $BRANCH"
git fetch "$REMOTE"
git pull "$REMOTE" "$BRANCH"

if [[ "${USE_VENV:-0}" == "1" ]]; then
  echo "==> 模式: 本机 venv + uvicorn"
  if ! command -v python3 &>/dev/null; then
    echo "未找到 python3"; exit 1
  fi
  [[ -d venv ]] || python3 -m venv venv
  ./venv/bin/pip install -U pip
  ./venv/bin/pip install -r requirements.txt
  mkdir -p logs
  PORT="${UVICORN_PORT:-8000}"
  if command -v fuser &>/dev/null; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  else
    pkill -f "uvicorn app:app.*--port ${PORT}" 2>/dev/null || true
    pkill -f "uvicorn app:app" 2>/dev/null || true
  fi
  sleep 1
  nohup ./venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port "$PORT" >>logs/uvicorn.log 2>&1 &
  echo "==> uvicorn 已在后台启动，端口 $PORT，日志: $ROOT/logs/uvicorn.log"
  echo "    健康检查: curl -s http://127.0.0.1:${PORT}/api/v1/system/health"
  exit 0
fi

if ! command -v docker &>/dev/null; then
  echo "未找到 docker。若要用本机 Python 启动，请设置 USE_VENV=1 后重试。"
  exit 1
fi
if ! docker compose version &>/dev/null; then
  echo "需要 Docker Compose v2（docker compose）"; exit 1
fi

COMPOSE="${COMPOSE_FILE:-docker-compose.prod.yml}"
if [[ ! -f "$COMPOSE" ]]; then
  echo "未找到 $COMPOSE，请设置 COMPOSE_FILE 或把 compose 文件放到项目根目录。"
  exit 1
fi

SVC="${COMPOSE_SERVICE:-opencv-platform}"
echo "==> 模式: Docker Compose 文件=$COMPOSE 服务=$SVC"
export DOCKER_BUILDKIT=1
# 一条命令完成构建并用新镜像重建容器（避免仅 up 时仍跑旧层）
docker compose -f "$COMPOSE" up -d --build "$SVC"

echo "==> 容器状态"
docker compose -f "$COMPOSE" ps "$SVC" || true

IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
echo ""
echo "完成。健康检查: curl -s http://127.0.0.1:8000/api/v1/system/health"
[[ -n "$IP" ]] && echo "局域网可访问: http://${IP}:8000"
echo "查看日志: docker compose -f $ROOT/$COMPOSE logs -f $SVC"
