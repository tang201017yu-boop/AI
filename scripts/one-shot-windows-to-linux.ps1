#Requires -Version 5.1
<#
.SYNOPSIS
  从本机 Windows 仓库一键同步到 Linux（走 SSH/SCP，服务器无需访问 GitHub）。

.DESCRIPTION
  使用 git bundle 打包当前 dev 分支，上传到目标机后 git clone，含完整提交历史。
  依赖：已安装 OpenSSH 客户端（ssh/scp），且能 ssh 登录目标机。

.PARAMETER LinuxHost
  默认 192.168.2.102

.PARAMETER LinuxUser
  默认 root

.PARAMETER RemoteDir
  远程目录，默认 /root/Vision_Platform

.PARAMETER Branch
  默认 dev

.EXAMPLE
  cd c:\Users\Lenovo\work\Vision_Platform
  .\scripts\one-shot-windows-to-linux.ps1
#>
param(
  [string]$LinuxHost = "192.168.2.102",
  [string]$LinuxUser = "root",
  [string]$RemoteDir = "/root/Vision_Platform",
  [string]$Branch = "dev"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$sshTarget = "${LinuxUser}@${LinuxHost}"
$bundleName = "vision-platform-${Branch}.bundle"
$bundlePath = Join-Path $env:TEMP $bundleName

Write-Host ('==> Repo: ' + $RepoRoot)
Write-Host ('==> Bundle branch: ' + $Branch + ' -> ' + $bundlePath)

# 确保分支存在（show-ref --quiet 成功时无输出，不能靠返回值判断）
git show-ref --verify --quiet "refs/heads/$Branch" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Error "Local branch missing: $Branch"
}

git bundle create $bundlePath $Branch
if (-not (Test-Path $bundlePath)) { Write-Error "git bundle create failed" }

$remoteBundle = "/tmp/$bundleName"
Write-Host ('==> Upload: ' + $sshTarget + ':' + $remoteBundle)
scp -o StrictHostKeyChecking=accept-new $bundlePath "${sshTarget}:$remoteBundle"

$bakSuffix = Get-Date -Format "yyyyMMddHHmmss"
# 含 || 的行须用单引号，避免 PowerShell 解析
$remoteLines = @(
  "set -e"
  'echo "==> backup old dir if exists"'
  "if [ -d `"$RemoteDir`" ]; then"
  ('  mv "' + $RemoteDir + '" "' + $RemoteDir + '.bak.' + $bakSuffix + '" || true')
  "fi"
  'echo "==> git clone from bundle (no internet needed)"'
  "git clone -b $Branch `"$remoteBundle`" `"$RemoteDir`""
  "rm -f `"$remoteBundle`""
  ('echo "==> done: ' + $RemoteDir + '"')
  ('git -C "' + $RemoteDir + '" log -1 --pretty=oneline')
)
# 只用 LF，避免 bash 收到 CRLF 把参数拆坏
$remoteScript = ($remoteLines -join "`n") -replace "`r`n", "`n"

Write-Host '==> Remote: bash clone...'
$remoteScript | ssh -o StrictHostKeyChecking=accept-new $sshTarget bash -s

Remove-Item -Force $bundlePath -ErrorAction SilentlyContinue
Write-Host ('==> Done. SSH=' + $sshTarget + ' Path=' + $RemoteDir)
