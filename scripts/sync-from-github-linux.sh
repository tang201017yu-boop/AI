#!/usr/bin/env bash
# 在 Linux 服务器（如 192.168.2.102）上执行：从 GitHub 克隆或拉取 dev 分支。
#
# 用法：
#   chmod +x scripts/sync-from-github-linux.sh
#   ./scripts/sync-from-github-linux.sh
#   # 或指定目录：
#   ./scripts/sync-from-github-linux.sh /opt/Vision_Platform
#
# 本机（Windows）无法代你 SSH；请 SSH 登录到 Linux 后在本仓库目录执行，或只复制下面「一行版」在任意目录执行。

set -euo pipefail

TARGET="${1:-$HOME/Vision_Platform}"
REPO="https://github.com/tang201017yu-boop/AI.git"
BRANCH="dev"

if [[ -d "$TARGET/.git" ]]; then
  echo "==> 已存在仓库，拉取更新: $TARGET"
  cd "$TARGET"
  git fetch origin
  git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
  git pull origin "$BRANCH"
  echo "==> 当前提交: $(git rev-parse --short HEAD) ($(git log -1 --oneline))"
else
  if [[ -d "$TARGET" ]]; then
    # 目录已存在但不是 git 仓库时，git clone 会报错「已存在且非空」
    BAK="${TARGET}.bak.$(date +%Y%m%d%H%M%S)"
    echo "==> 目录已存在且无 .git，先备份为: $BAK"
    mv "$TARGET" "$BAK"
  fi
  echo "==> 克隆 $BRANCH 到: $TARGET"
  git clone -b "$BRANCH" "$REPO" "$TARGET"
  echo "==> 当前提交: $(git -C "$TARGET" rev-parse --short HEAD) ($(git -C "$TARGET" log -1 --oneline))"
fi

echo "==> 完成。"
