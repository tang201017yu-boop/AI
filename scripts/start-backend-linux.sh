#!/usr/bin/env bash
# 在 Linux 服务器项目根目录启动后端（venv + uvicorn，不依赖 Docker）
#
# 用法:
#   bash scripts/start-backend-linux.sh [/root/Vision_Platform]
#
# 环境变量:
#   SKIP_PIP=1     跳过 pip install（venv 已装好依赖时快速重启）
#   API_PORT=8000  监听端口
#   TORCH_FORCE_CPU=1  强制 PyTorch 用 CPU（仍报 CUDA no kernel 时可设，或升级 cu128 见 scripts/install-pytorch-gpu-cu128.sh）

set -euo pipefail
ROOT="${1:-/root/Vision_Platform}"
ROOT="${ROOT//$'\r'/}"
cd "$ROOT"

if [[ ! -f "app.py" ]]; then
  echo "未找到 app.py，请确认 ROOT=$ROOT 为项目根目录"
  exit 1
fi

if ! command -v python3 &>/dev/null; then
  echo "请先安装: apt install -y python3 python3-venv python3-pip"
  exit 1
fi

echo "==> 目录: $ROOT"
[[ -d venv ]] || python3 -m venv venv
if [[ "${SKIP_PIP:-0}" != "1" ]]; then
  echo "==> 安装依赖（首次较慢，可另开 screen/tmux；仅重启可设 SKIP_PIP=1）"
  ./venv/bin/pip install -q -U pip
  ./venv/bin/pip install -q -r requirements.txt
else
  echo "==> 已跳过 pip（SKIP_PIP=1）"
fi

mkdir -p logs
PORT="${API_PORT:-8000}"
if command -v fuser &>/dev/null; then
  # fuser -k 默认会把 PID 打到 stdout，与后面 echo 连在一起；全部重定向
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
else
  pkill -f "uvicorn app:app" 2>/dev/null || true
fi
sleep 1

nohup ./venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port "$PORT" >>logs/uvicorn.log 2>&1 &
echo $! > /tmp/vision-platform-uvicorn.pid
echo "==> 已后台启动 uvicorn PID=$(cat /tmp/vision-platform-uvicorn.pid) 端口=$PORT"
echo "==> 日志: $ROOT/logs/uvicorn.log"
HEALTH_URL="http://127.0.0.1:${PORT}/api/v1/system/health"
code="000"
for _ in $(seq 1 45); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$HEALTH_URL" 2>/dev/null || true)
  if [[ "$code" == "200" ]]; then
    break
  fi
  sleep 1
done
if [[ "$code" == "200" ]]; then
  echo "==> 健康检查: HTTP $code"
else
  echo "==> 健康检查: HTTP ${code:-000}（若服务仍在加载，请过几秒再: curl -s -o /dev/null -w '%{http_code}' $HEALTH_URL ）"
  echo "==> 或查看: tail -50 $ROOT/logs/uvicorn.log"
fi
