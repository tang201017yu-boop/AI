#Requires -Version 5.1
<#
.SYNOPSIS
  从本机 Windows 仓库一键同步到 Linux（走 SSH/SCP，服务器无需访问 GitHub）。

.DESCRIPTION
  使用 git bundle 打包当前 dev 分支，上传到目标机后 git clone，含完整提交历史。
  依赖：已安装 OpenSSH 客户端（ssh/scp），且能 ssh 登录目标机。

  重要 - data/ 与 venv/：
  - 仓库不跟踪 data/，bundle clone 得到的是“空/默认”的 data/，原先服务器上的
    上传、数据集、标注项目等会留在本次脚本自动改名的备份目录
    {RemoteDir}.bak.{时间戳} 中。
  - 默认在 clone 之后，若该备份下存在 data/，会用 rsync -a 合并到新的 {RemoteDir}/data/
    （与手工 cp -a 等效，不删新目录中已有文件，同名则覆盖为备份里版本）。
  - venv/ 也不进 Git；合并 data 后请在服务器上从旧备份恢复 venv，或重新 pip
    install。另一套代码目录中的数据（例如 /root/wuyu/Vision_Platform/data）不会
    自动合并，需你自行 rsync 到 {RemoteDir}/data/。

.PARAMETER SkipDataRestore
  加此开关则不从 .bak.时间戳 合并 data/（全新空 data）。

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

.EXAMPLE
  # 不合并 data/，保持 clone 后默认空 data
  .\scripts\one-shot-windows-to-linux.ps1 -SkipDataRestore
#>
param(
  [string]$LinuxHost = "192.168.2.102",
  [string]$LinuxUser = "root",
  [string]$RemoteDir = "/root/Vision_Platform",
  [string]$Branch = "dev",
  [switch]$SkipDataRestore
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
$dataBak = "${RemoteDir}.bak.$bakSuffix/data"
# 含 || 的行须用单引号，避免 PowerShell 解析
$remoteLines = [System.Collections.Generic.List[string]]::new()
$remoteLines.Add("set -e")
$remoteLines.Add('echo "==> backup old dir if exists"')
$remoteLines.Add("if [ -d `"$RemoteDir`" ]; then")
$remoteLines.Add('  mv "' + $RemoteDir + '" "' + $RemoteDir + '.bak.' + $bakSuffix + '" || true')
$remoteLines.Add("fi")
$remoteLines.Add('echo "==> git clone from bundle (no internet needed)"')
$remoteLines.Add("git clone -b $Branch `"$remoteBundle`" `"$RemoteDir`"")
$remoteLines.Add("rm -f `"$remoteBundle`"")
if (-not $SkipDataRestore) {
  $remoteLines.Add('echo "==> ensure data/ exists"')
  $remoteLines.Add("mkdir -p `"$RemoteDir/data`"")
  $remoteLines.Add("if [ -d `"$dataBak`" ]; then")
  $remoteLines.Add('  echo "==> merge data/ from backup (bundle has no user data in git)"')
  $remoteLines.Add("  rsync -a `"$dataBak/`" `"$RemoteDir/data/`"")
  $remoteLines.Add("else")
  $remoteLines.Add("  echo '(no backup data dir, skip data merge)'")
  $remoteLines.Add("fi")
} else {
  $remoteLines.Add('echo "==> SkipDataRestore: 不合并 data/"')
}
$remoteLines.Add('echo "==> done: ' + $RemoteDir + '"')
$remoteLines.Add('git -C "' + $RemoteDir + '" log -1 -s --pretty=%h')
# 含 || 的行须用单引号，避免 PowerShell 解析
$remoteLines = $remoteLines.ToArray()
# 只用 LF，避免 bash 收到 CRLF 把参数拆坏
$remoteScript = ($remoteLines -join "`n") -replace "`r`n", "`n"

Write-Host '==> Remote: bash clone...'
$remoteScript | ssh -o StrictHostKeyChecking=accept-new $sshTarget bash -s

Remove-Item -Force $bundlePath -ErrorAction SilentlyContinue
Write-Host ('==> Done. SSH=' + $sshTarget + ' Path=' + $RemoteDir)
Write-Host '==> 提示: 服务器上请从备份恢复 venv 或 pip install；其他目录(如 wuyu)的 data 需自行 rsync 到 Path/data/' -ForegroundColor DarkYellow
