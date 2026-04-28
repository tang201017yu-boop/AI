# 结束占用 8000 的进程并启动本机 API（与 package.json 中 dev:backend 一致）
# 用法（项目根目录）:  powershell -ExecutionPolicy Bypass -File scripts/restart-backend.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$port = 8000
Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction Stop } catch {}
}
Start-Sleep -Seconds 1
$py = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "未找到 venv\Scripts\python.exe" -ForegroundColor Red
    exit 1
}
Write-Host "启动: uvicorn app:app --host 127.0.0.1 --port $port --reload" -ForegroundColor Cyan
& $py -m uvicorn app:app --host 127.0.0.1 --port $port --reload
