"""
================================================================================
 OpenCV Platform - 主应用入口
================================================================================
 基于 Ultralytics YOLO 的开源计算机视觉平台

 功能模块:
   - 数据准备: 数据集管理、图像标注、数据增强
   - 模型训练: YOLO 系列模型训练、超参数调优
   - 推理测试: 图片/视频推理、实时监控
   - 智能解决方案: 对象计数、热图生成、速度估算等

 项目结构:
   backend/
   ├── core/              # 核心模块
   │   ├── config.py      # 配置管理（环境变量、默认参数）
   │   ├── yolo_engine.py # YOLO 引擎（模型加载、推理）
   │   ├── database.py    # PostgreSQL 数据库服务
   │   └── s3_storage.py  # S3/MinIO 存储服务
   ├── modules/           # 功能模块
   │   ├── data_preparation/   # 数据准备模块
   │   ├── training/           # 训练模块
   │   ├── inference/          # 推理模块
   │   └── solutions/          # 解决方案模块
   └── api/               # API 路由

 部署模式:
   - 独立模式: 本地运行所有服务（需要 GPU）
   - 开发模式: 本地前端 + 远程后端 API（MacBook 开发用）
================================================================================
"""
import sys
import asyncio
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# ============================================================================
# 日志配置
# ============================================================================
# 创建日志目录
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler(LOG_DIR / "app.log", encoding='utf-8')  # 文件输出
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 路径初始化
# ============================================================================
# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 打印启动信息
logger.info("=" * 60)
logger.info("OpenCV Platform 启动中...")
logger.info(f"项目路径: {project_root}")

# ============================================================================
# FastAPI 应用初始化
# ============================================================================
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from backend.core.config import settings

# ============================================================================
# 模式配置
# ============================================================================
# DEV_MODE: 开发模式（本地前端 + 远程 API）
#   - true: API 请求转发到远程服务器
#   - false: 本地执行所有 API
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

# REMOTE_API_URL: 远程服务器地址（开发模式使用）
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "http://localhost:8000")

# 版本戳 - 用于缓存破坏（每次启动生成新时间戳）
APP_VERSION_TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

logger.info(f"运行模式: {'开发模式' if DEV_MODE else '独立模式'}")
if DEV_MODE:
    logger.info(f"远程API地址: {REMOTE_API_URL}")

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME if not DEV_MODE else f"{settings.APP_NAME} (开发模式)",
    version=settings.APP_VERSION,
    description="基于 Ultralytics YOLO 的开源计算机视觉平台，提供数据标注、模型训练、API 部署的完整工作流",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# ============================================================================
# 中间件
# ============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    记录所有 incoming 请求的详细信息
    """
    async def dispatch(self, request: Request, call_next):
        # 记录请求信息
        logger.debug(f"请求: {request.method} {request.url.path}")

        # 记录请求头（排除敏感信息）
        headers = dict(request.headers)
        headers.pop("authorization", None)
        headers.pop("cookie", None)
        if headers:
            logger.debug(f"请求头: {headers}")

        # 继续处理请求
        response = await call_next(request)

        # 记录响应信息
        logger.debug(f"响应状态: {response.status_code}")

        return response


class ProxyAPIMiddleware(BaseHTTPMiddleware):
    """
    API 代理中间件（开发模式使用）
    将 /api/v1/* 请求转发到远程服务器
    """
    async def dispatch(self, request: Request, call_next):
        # 只代理 /api/v1/ 开头的请求
        if DEV_MODE and str(request.url.path).startswith("/api/v1/"):
            import httpx

            # 构建远程请求 URL
            remote_url = f"{REMOTE_API_URL.rstrip('/')}{request.url.path}"

            logger.info(f"[代理] {request.method} {request.url.path} -> {remote_url}")

            # 获取请求体
            body = await request.body()

            # 转发请求到远程服务器
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    headers = dict(request.headers)
                    headers.pop("host", None)

                    request_method = request.method
                    remote_response = await client.request(
                        method=request_method,
                        url=remote_url,
                        headers=headers,
                        content=body,
                        params=request.query_params
                    )

                    logger.info(f"[代理] 响应: {remote_response.status_code}")

                    return JSONResponse(
                        content=remote_response.json(),
                        status_code=remote_response.status_code
                    )
            except httpx.TimeoutException:
                logger.error(f"[代理] 请求超时: {remote_url}")
                return JSONResponse(
                    status_code=504,
                    content={"success": False, "message": "网关超时：远程服务器响应超时"}
                )
            except Exception as e:
                logger.error(f"[代理] 请求失败: {e}")
                return JSONResponse(
                    status_code=502,
                    content={"success": False, "message": f"代理请求失败: {str(e)}"}
                )

        return await call_next(request)


class CacheControlMiddleware(BaseHTTPMiddleware):
    """
    缓存控制中间件
    - 静态资源: 缓存 1 小时
    - HTML 页面: 不缓存（开发模式）
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # 对静态资源设置缓存策略
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=3600, must-revalidate"
            response.headers["ETag"] = APP_VERSION_TIMESTAMP
            logger.debug(f"[缓存] 静态资源: {request.url.path}")
        # HTML 页面不缓存
        elif request.url.path.endswith(".html") or request.url.path in [
            "/", "/inference", "/training", "/models", "/datasets",
            "/annotation", "/solutions", "/image_browser", "/augmentation"
        ]:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            logger.debug(f"[缓存] 页面(无缓存): {request.url.path}")

        return response


# 注册中间件（顺序很重要：先代理，再日志，最后缓存）
app.add_middleware(ProxyAPIMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(CacheControlMiddleware)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("中间件注册完成: CORS, 缓存控制, 日志, 代理(开发模式)")

# ============================================================================
# 静态文件服务
# ============================================================================
# 挂载静态文件
static_dir = project_root / "frontend" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"静态文件服务: /static -> {static_dir}")

# 挂载上传文件目录
uploads_dir = settings.UPLOADS_DIR
if uploads_dir.exists():
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")
    logger.info(f"上传文件服务: /uploads -> {uploads_dir}")

# 挂载标注项目图片目录
annotation_images_dir = settings.ANNOTATION_PROJECTS_DIR
if annotation_images_dir.exists():
    app.mount("/annotation-images", StaticFiles(directory=str(annotation_images_dir)), name="annotation_images")
    logger.info(f"标注图片服务: /annotation-images -> {annotation_images_dir}")

# 模板引擎
templates_dir = project_root / "frontend"
templates = Jinja2Templates(directory=str(templates_dir))
logger.info(f"模板引擎: {templates_dir}")

# ============================================================================
# 路由注册
# ============================================================================
from backend.api.routes import router as api_router
from backend.modules.data_preparation.routes import router as data_prep_router
from backend.modules.training.routes import router as training_router
from backend.modules.inference.routes import router as inference_router
from backend.modules.solutions.routes import router as solutions_router

# 基础 API 路由
app.include_router(api_router, prefix="/api/v1", tags=["基础功能"])

# 数据准备模块路由
app.include_router(data_prep_router, prefix="/api/v1", tags=["数据准备"])
logger.info("路由注册: /api/v1/data-preparation -> 数据准备模块")

# 训练模块路由
app.include_router(training_router, prefix="/api/v1", tags=["模型训练"])
logger.info("路由注册: /api/v1/training -> 模型训练模块")

# 推理模块路由
app.include_router(inference_router, prefix="/api/v1", tags=["推理测试"])
logger.info("路由注册: /api/v1/inference -> 推理测试模块")

# 解决方案模块路由
app.include_router(solutions_router, prefix="/api/v1", tags=["智能解决方案"])
logger.info("路由注册: /api/v1/solutions -> 智能解决方案模块")

# ============================================================================
# 启动事件
# ============================================================================
@app.on_event("startup")
async def warmup_services():
    """
    启动预热
    - 开发模式: 跳过 GPU 初始化
    - 独立模式: 预加载模型、初始化数据库和存储
    """
    logger.info("=" * 50)
    logger.info("服务启动中...")

    if DEV_MODE:
        # 开发模式：只打印信息，不初始化后端服务
        logger.info("=" * 50)
        logger.info("  [开发模式] 🏠")
        logger.info(f"  远程API: {REMOTE_API_URL}")
        logger.info("  API 请求将转发到远程服务器执行")
        logger.info("=" * 50)
        return

    # 独立模式：初始化所有服务
    loop = asyncio.get_running_loop()
    tasks = []

    # 1. 预加载模型路径
    try:
        from backend.core.yolo_engine import yolo_engine
        if yolo_engine:
            tasks.append(loop.run_in_executor(None, lambda: yolo_engine._iter_model_paths()))
            logger.info("模型引擎已加载")
    except Exception as e:
        logger.error(f"模型引擎初始化失败: {e}")

    # 2. 预加载数据集列表
    try:
        from backend.modules.data_preparation import dataset_service
        tasks.append(loop.run_in_executor(None, dataset_service.list_datasets))
        logger.info("数据集服务已加载")
    except Exception as e:
        logger.error(f"数据集服务初始化失败: {e}")

    # 3. 初始化 PostgreSQL 数据库
    logger.info("正在初始化 PostgreSQL 数据库...")
    try:
        from backend.core.database import postgres_service
        postgres_service.init_database()
        logger.info("PostgreSQL 数据库初始化完成")
    except Exception as e:
        logger.warning(f"PostgreSQL 初始化失败（使用 SQLite）: {e}")

    # 4. 初始化 S3 存储
    logger.info("正在初始化 S3 存储...")
    try:
        from backend.core.s3_storage import s3_service
        s3_service.init_storage()
        logger.info("S3 存储初始化完成")
    except Exception as e:
        logger.warning(f"S3 存储初始化失败（使用本地存储）: {e}")

    # 等待所有任务完成
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("=" * 50)
    logger.info("服务启动完成！")
    logger.info(f"访问地址: http://localhost:{settings.API_PORT}")
    logger.info(f"API 文档: http://localhost:{settings.API_PORT}/api/docs")
    logger.info("=" * 50)


# ============================================================================
# 前端路由
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """首页"""
    logger.debug("访问首页")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/inference", response_class=HTMLResponse)
async def inference_page(request: Request):
    """推理页面"""
    logger.debug("访问推理页面")
    return templates.TemplateResponse("inference.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    """训练页面"""
    logger.debug("访问训练页面")
    return templates.TemplateResponse("training.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """模型管理页面"""
    logger.debug("访问模型管理页面")
    return templates.TemplateResponse("models.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    """数据集管理页面"""
    logger.debug("访问数据集管理页面")
    return templates.TemplateResponse("datasets.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/annotation", response_class=HTMLResponse)
async def annotation_page(request: Request):
    """本地数据标注页面"""
    logger.debug("访问数据标注页面")
    return templates.TemplateResponse("annotation.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/solutions", response_class=HTMLResponse)
async def solutions_page(request: Request):
    """Ultralytics Solutions 页面"""
    logger.debug("访问智能解决方案页面")
    return templates.TemplateResponse("solutions.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/image_browser", response_class=HTMLResponse)
async def image_browser_page(request: Request):
    """图像浏览器页面"""
    logger.debug("访问图像浏览器页面")
    return templates.TemplateResponse("image_browser.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/augmentation", response_class=HTMLResponse)
async def augmentation_page(request: Request):
    """数据增强页面"""
    logger.debug("访问数据增强页面")
    return templates.TemplateResponse("augmentation.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/training_monitor", response_class=HTMLResponse)
async def training_monitor_page(request: Request):
    """训练监控页面"""
    logger.debug("访问训练监控页面")
    return templates.TemplateResponse("training_monitor.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/projects", response_class=HTMLResponse)
async def projects_page(request: Request):
    """项目管理页面"""
    logger.debug("访问项目管理页面")
    return templates.TemplateResponse("projects.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


@app.get("/model", response_class=HTMLResponse)
async def model_detail_page(request: Request):
    """模型详情页面"""
    logger.debug("访问模型详情页面")
    return templates.TemplateResponse("model_detail.html", {
        "request": request,
        "version": APP_VERSION_TIMESTAMP
    })


# ============================================================================
# 健康检查端点
# ============================================================================

@app.get("/api/v1/system/health")
async def health_check():
    """
    系统健康检查
    用于负载均衡器和服务监控
    """
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "mode": "dev" if DEV_MODE else "standalone",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    # 打印启动信息
    print(f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║         OpenCV Platform - YOLO Edition                         ║
    ║         开源计算机视觉平台                                      ║
    ║                                                                ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  运行模式: {'开发模式 (代理远程API)' if DEV_MODE else '独立模式 (本地GPU)'}                      ║
    ║  服务端口: {settings.API_PORT}                                             ║
    ║  访问地址: http://localhost:{settings.API_PORT}                              ║
    ║  API 文档: http://localhost:{settings.API_PORT}/api/docs                   ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    # 启动服务
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
