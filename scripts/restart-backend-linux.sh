#!/usr/bin/env bash
# 在 Linux 服务器上「一键重启」后端：不装依赖，只结束占用的 API 端口并 nohup 启动 uvicorn。
# 与 start-backend-linux.sh 相同，但固定 SKIP_PIP=1 以加快操作。
#
# 用法（SSH 登录 192.168.2.102 后，任选其一）:
#   在项目根目录:
#     chmod +x scripts/restart-backend-linux.sh
#     bash scripts/restart-backend-linux.sh
#   若项目不在当前目录，传入绝对路径，例如:
#     bash /root/wuyu/Vision_Platform/scripts/restart-backend-linux.sh /root/wuyu/Vision_Platform
#   从本机一条 SSH 远程执行（把路径换成服务器上真实根目录）:
#     ssh root@192.168.2.102 'bash /root/Vision_Platform/scripts/restart-backend-linux.sh /root/Vision_Platform'

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"
export SKIP_PIP=1
exec "$SCRIPT_DIR/start-backend-linux.sh" "${1:-$ROOT_DEFAULT}"
