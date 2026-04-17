# 在「已克隆本仓库」的 Windows 服务器上执行（在仓库根目录打开 PowerShell）:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\scripts\update-backend-remote.ps1
#
# 可选环境变量:
#   $env:GIT_BRANCH = "main"
#   $env:USE_VENV = "1"        # 强制不用 Docker，改用 venv
#   $env:UVICORN_PORT = "8000"

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
Write-Host "==> 仓库: $Root"

$branch = if ($env:GIT_BRANCH) { $env:GIT_BRANCH } else { (git rev-parse --abbrev-ref HEAD).Trim() }
Write-Host "==> git pull ($branch)"
git fetch origin
git pull origin $branch

$dockerOk = (Get-Command docker -ErrorAction SilentlyContinue) -and ($env:USE_VENV -ne "1")

if (-not $dockerOk) {
    Write-Host "==> 模式: venv + uvicorn"
    if (-not (Test-Path ".\venv\Scripts\python.exe")) { python -m venv venv }
    & .\venv\Scripts\pip.exe install -U pip
    & .\venv\Scripts\pip.exe install -r requirements.txt
    New-Item -ItemType Directory -Force -Path ".\logs" | Out-Null
    $port = if ($env:UVICORN_PORT) { $env:UVICORN_PORT } else { "8000" }
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
    $logOut = Join-Path $Root "logs\uvicorn.log"
    $logErr = Join-Path $Root "logs\uvicorn.err.log"
    $py = Join-Path $Root "venv\Scripts\python.exe"
    Start-Process -FilePath $py -ArgumentList @(
        "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", $port
    ) -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr
    Write-Host "==> 已启动 uvicorn 端口 $port，日志: $logOut / $logErr"
    Write-Host "    健康检查: Invoke-WebRequest http://127.0.0.1:$port/api/v1/system/health -UseBasicParsing"
    exit 0
}

$compose = if ($env:COMPOSE_FILE) { $env:COMPOSE_FILE } else { "docker-compose.prod.yml" }
$svc = if ($env:COMPOSE_SERVICE) { $env:COMPOSE_SERVICE } else { "opencv-platform" }
if (-not (Test-Path (Join-Path $Root $compose))) {
    Write-Error "未找到 $compose，可设置 USE_VENV=1 使用本机 Python。"
}
Write-Host "==> 模式: docker compose -f $compose 服务 $svc"
$env:DOCKER_BUILDKIT = "1"
docker compose -f $compose up -d --build $svc
docker compose -f $compose ps $svc
Write-Host "完成。健康检查: http://127.0.0.1:8000/api/v1/system/health"
