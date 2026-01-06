"""
OpenCV Platform - 主应用入口
基于 Ultralytics YOLO 的开源计算机视觉平台
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from config.config import settings
from backend.api.routes import router
from backend.services.dataset_service import dataset_service
from backend.services.yolo_service import yolo_service

# 版本戳 - 用于缓存破坏
APP_VERSION_TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 Ultralytics YOLO 的开源计算机视觉平台，提供数据标注、模型训练、API 部署的完整工作流",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 缓存控制中间件
class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # 对静态资源设置缓存策略
        if request.url.path.startswith("/static/"):
            # 静态资源缓存1小时，但必须重新验证
            response.headers["Cache-Control"] = "public, max-age=3600, must-revalidate"
            response.headers["ETag"] = APP_VERSION_TIMESTAMP
        elif request.url.path.endswith(".html") or request.url.path in ["/", "/inference", "/training", "/models", "/datasets", "/annotation", "/solutions"]:
            # HTML页面不缓存
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response

# 添加缓存控制中间件
app.add_middleware(CacheControlMiddleware)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = project_root / "frontend" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 模板引擎
templates_dir = project_root / "frontend"
templates = Jinja2Templates(directory=str(templates_dir))

# 注册 API 路由
app.include_router(router, prefix="/api/v1", tags=["API"])


@app.on_event("startup")
async def warmup_services():
    """预加载模型与数据集索引，减少首次访问延迟"""
    loop = asyncio.get_running_loop()
    tasks = []
    if yolo_service:
        tasks.append(loop.run_in_executor(None, yolo_service.list_models))
    tasks.append(loop.run_in_executor(None, dataset_service.list_datasets))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ==================== 前端路由 ====================
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """首页"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/inference", response_class=HTMLResponse)
async def inference_page(request: Request):
    """推理页面"""
    return templates.TemplateResponse("inference.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    """训练页面"""
    return templates.TemplateResponse("training.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """模型管理页面"""
    return templates.TemplateResponse("models.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    """数据集管理页面"""
    return templates.TemplateResponse("datasets.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/annotation", response_class=HTMLResponse)
async def annotation_page(request: Request):
    """本地数据标注页面"""
    return templates.TemplateResponse("annotation.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/solutions", response_class=HTMLResponse)
async def solutions_page(request: Request):
    """Ultralytics Solutions 页面"""
    return templates.TemplateResponse("solutions.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


if __name__ == "__main__":
    import uvicorn
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║         OpenCV Platform - YOLO Edition                   ║
    ║         开源计算机视觉平台                                ║
    ║                                                          ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  🚀 Server starting...                                   ║
    ║  📍 API: http://localhost:{settings.API_PORT}                       ║
    ║  📖 Docs: http://localhost:{settings.API_PORT}/api/docs            ║
    ║  🏷️  Label Studio: {settings.LABEL_STUDIO_URL}       ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
