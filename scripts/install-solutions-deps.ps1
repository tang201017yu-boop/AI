# 智能方案（对象计数等）依赖：在「本机 venv」一键安装 shapely、lap
# 用法（项目根目录）:  powershell -ExecutionPolicy Bypass -File scripts/install-solutions-deps.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$py = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "未找到 venv，请先: python -m venv venv" -ForegroundColor Red
    exit 1
}
& $py -m pip install -U "shapely>=2.0.0" "lap>=0.5.0"
& $py -c "import shapely; import lap; print('shapely OK:', shapely.__version__, '| lap OK')"
Write-Host "完成。若 API 跑在另一台机，请在那台 venv 同样: pip install shapely lap" -ForegroundColor Cyan
