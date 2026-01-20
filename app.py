"""
OpenCV Platform - 主应用入口
基于 Ultralytics YOLO 的开源计算机视觉平台

项目结构:
├── backend/
│   ├── core/              # 核心模块
│   │   ├── config.py      # 配置管理
│   │   ├── yolo_engine.py # YOLO 引擎
│   │   └── utils.py       # 工具函数
│   ├── modules/           # 功能模块
│   │   ├── data_preparation/   # 数据准备模块
│   │   │   ├── dataset_service.py
│   │   │   ├── annotation_service.py
│   │   │   └── sam_service.py
│   │   ├── training/           # 训练模块
│   │   │   └── training_service.py
│   │   ├── inference/          # 推理模块
│   │   │   └── inference_service.py
│   │   └── solutions/          # 解决方案模块（独立）
│   │       └── solutions_service.py
│   └── api/
│       └── routes.py     # API 路由（从各模块导入）
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
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

# 使用新的核心配置
from backend.core.config import settings
from backend.core.yolo_engine import yolo_engine
from backend.core.database import postgres_service
from backend.core.s3_storage import s3_service

# 导入各模块路由
from backend.modules.data_preparation.routes import router as data_prep_router
from backend.modules.training.routes import router as training_router
from backend.modules.inference.routes import router as inference_router
from backend.modules.solutions.routes import router as solutions_router

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
            response.headers["Cache-Control"] = "public, max-age=3600, must-revalidate"
            response.headers["ETag"] = APP_VERSION_TIMESTAMP
        elif request.url.path.endswith(".html") or request.url.path in ["/", "/inference", "/training", "/models", "/datasets", "/annotation", "/solutions", "/image_browser", "/augmentation"]:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response

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

# 挂载上传文件目录
uploads_dir = settings.UPLOADS_DIR
if uploads_dir.exists():
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# 模板引擎
templates_dir = project_root / "frontend"
templates = Jinja2Templates(directory=str(templates_dir))

# ==================== 注册模块路由 ====================

# 数据准备模块路由
app.include_router(data_prep_router, prefix="/api/v1", tags=["数据准备"])

# 训练模块路由
app.include_router(training_router, prefix="/api/v1", tags=["模型训练"])

# 推理模块路由
app.include_router(inference_router, prefix="/api/v1", tags=["推理测试"])

# 解决方案模块路由
app.include_router(solutions_router, prefix="/api/v1", tags=["智能解决方案"])


@app.on_event("startup")
async def warmup_services():
    """预加载模型与数据集索引，减少首次访问延迟"""
    loop = asyncio.get_running_loop()
    tasks = []

    if yolo_engine:
        tasks.append(loop.run_in_executor(None, lambda: yolo_engine._iter_model_paths()))

    # 预加载数据集列表
    from backend.modules.data_preparation import dataset_service
    tasks.append(loop.run_in_executor(None, dataset_service.list_datasets))

    # 初始化 PostgreSQL 数据库
    print("正在初始化 PostgreSQL 数据库...")
    try:
        postgres_service.init_database()
        print("PostgreSQL 数据库初始化完成")
    except Exception as e:
        print(f"PostgreSQL 初始化失败: {e}")

    # 初始化 S3 存储
    print("正在初始化 S3 存储...")
    try:
        s3_service.init_storage()
        print("S3 存储初始化完成")
    except Exception as e:
        print(f"S3 存储初始化失败: {e}")

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


@app.get("/image_browser", response_class=HTMLResponse)
async def image_browser_page(request: Request):
    """图像浏览器页面"""
    return templates.TemplateResponse("image_browser.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/augmentation", response_class=HTMLResponse)
async def augmentation_page(request: Request):
    """数据增强页面"""
    return templates.TemplateResponse("augmentation.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/training_monitor", response_class=HTMLResponse)
async def training_monitor_page(request: Request):
    """训练监控页面"""
    return templates.TemplateResponse("training_monitor.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request):
    """项目管理页面"""
    return templates.TemplateResponse("projects.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """模型库页面"""
    return templates.TemplateResponse("models.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/model", response_class=HTMLResponse)
async def model_detail_page(request: Request):
    """模型详情页面"""
    return templates.TemplateResponse("model_detail.html", {
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
