#Requires -Version 5.1
# 从 Windows 本机 SSH 到 Linux 服务器，后台启动/重启 API（与服务器上 start-backend-linux.sh 一致）。
# 前提：能无交互登录目标机（例如已配置 root 的 SSH 公钥）；不要用错误的 Linux 账号。
#
# 用法（在项目根或任意目录）:
#   powershell -ExecutionPolicy Bypass -File scripts/restart-backend-remote.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/restart-backend-remote.ps1 -RemoteRoot /root/wuyu/Vision_Platform
param(
  [string]$LinuxHost = "192.168.2.102",
  [string]$LinuxUser = "root",
  [string]$RemoteRoot = "/root/Vision_Platform"
)

$ErrorActionPreference = "Stop"
$sshTarget = "${LinuxUser}@${LinuxHost}"
$remoteCmd = "SKIP_PIP=1 bash $RemoteRoot/scripts/start-backend-linux.sh $RemoteRoot"
Write-Host "==> $sshTarget : $remoteCmd" -ForegroundColor Cyan
ssh $sshTarget $remoteCmd
