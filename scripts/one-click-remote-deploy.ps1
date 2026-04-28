#Requires -Version 5.1
<#
.SYNOPSIS
  一键：同步代码到 Linux -> 放行 8000 -> 启动后端（venv + uvicorn）

.DESCRIPTION
  1) 调用 one-shot-windows-to-linux.ps1（git bundle + scp + 远程 clone）
  2) SSH：ufw/firewalld 放行 8000/tcp
  3) SSH：bash scripts/start-backend-linux.sh

  同步时 data/ 默认会从本次备份目录合并回 {RemoteDir}/data/（与 one-shot 内说明一致）；
  见 one-shot-windows-to-linux.ps1 顶部注释。若需全新空 data/，传 -SkipDataRestore。

  依赖：OpenSSH（ssh/scp）、本机当前分支 dev 有提交、能登录 root@192.168.2.102

.PARAMETER SkipDataRestore
  传入 one-shot，不从 .bak 时间戳目录合并 data/

.PARAMETER SkipSync
  仅重启/装依赖，不再打 bundle（代码已在服务器上时加快）

.PARAMETER SkipPip
  等价于服务器上 SKIP_PIP=1，不跑 pip install

.EXAMPLE
  cd c:\Users\Lenovo\work\Vision_Platform
  .\scripts\one-click-remote-deploy.ps1
#>
param(
  [string]$LinuxHost = "192.168.2.102",
  [string]$LinuxUser = "root",
  [string]$RemoteDir = "/root/Vision_Platform",
  [string]$Branch = "dev",
  [switch]$SkipSync,
  [switch]$SkipPip,
  [switch]$SkipDataRestore
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$sshTarget = "${LinuxUser}@${LinuxHost}"

function Send-BashScript([string]$Text, [string]$SshTarget) {
  $unix = ($Text -replace "`r`n", "`n") -replace "`r", "`n"
  $unix | ssh -o StrictHostKeyChecking=accept-new $SshTarget bash -s
}

$oneShot = Join-Path $ScriptDir "one-shot-windows-to-linux.ps1"
if (-not (Test-Path $oneShot)) { Write-Error "Missing: $oneShot" }

if (-not $SkipSync) {
  Write-Host "========== [1/3] Sync code (bundle -> server) ==========" -ForegroundColor Cyan
  & $oneShot -LinuxHost $LinuxHost -LinuxUser $LinuxUser -RemoteDir $RemoteDir -Branch $Branch -SkipDataRestore:$SkipDataRestore
} else {
  Write-Host "========== [1/3] Skip sync (SkipSync) ==========" -ForegroundColor Yellow
}

Write-Host "========== [2/3] Firewall port 8000 ==========" -ForegroundColor Cyan
$fw = @'
set -e
if command -v ufw >/dev/null 2>&1; then
  ufw allow 8000/tcp comment vision-platform 2>/dev/null || ufw allow 8000/tcp || true
  echo "[ufw] 8000 allowed"
fi
if command -v firewall-cmd >/dev/null 2>&1; then
  firewall-cmd --permanent --add-port=8000/tcp 2>/dev/null || true
  firewall-cmd --reload 2>/dev/null || true
  echo "[firewalld] 8000 allowed"
fi
echo "[firewall] done"
'@
Send-BashScript $fw $sshTarget

Write-Host "========== [3/3] Start backend ==========" -ForegroundColor Cyan
$sp = if ($SkipPip) { '1' } else { '0' }
$remoteStart = @(
  'set -e'
  ('export SKIP_PIP=' + $sp)
  ('cd "' + $RemoteDir + '" || exit 1')
  ('test -f "' + $RemoteDir + '/scripts/start-backend-linux.sh" || exit 1')
  ('bash "' + $RemoteDir + '/scripts/start-backend-linux.sh" "' + $RemoteDir + '"')
) -join "`n"
Send-BashScript $remoteStart $sshTarget

Write-Host ""
Write-Host "========== Done ==========" -ForegroundColor Green
Write-Host "API:    http://${LinuxHost}:8000/api/docs" -ForegroundColor Green
Write-Host "Health: http://${LinuxHost}:8000/api/v1/system/health" -ForegroundColor Green
