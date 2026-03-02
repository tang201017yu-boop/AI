"""
================================================================================
 OpenCV Platform - 后端 API 服务
================================================================================
 基于 Ultralytics YOLO 的开源计算机视觉平台 API

 功能模块:
   - 数据准备: 数据集管理、图像标注、数据增强
   - 模型训练: YOLO 系列模型训练、超参数调优
   - 推理测试: 图片/视频推理、实时监控
   - 智能解决方案: 对象计数、热图生成、速度估算等

 项目结构:
   backend/
   ├── core/              # 核心模块
   │   ├── config.py      # 配置管理
   │   ├── yolo_engine.py # YOLO 引擎
   │   ├── database.py    # 数据库服务
   │   └── s3_storage.py  # S3/MinIO 存储服务
   ├── modules/           # 功能模块
   │   ├── data_preparation/   # 数据准备模块
   │   ├── training/           # 训练模块
   │   ├── inference/          # 推理模块
   │   ├── solutions/          # 解决方案模块
   │   └── ai_native/          # AI原生应用模块
   └── api/               # API 路由

 前端: 独立运行在 http://localhost:3000 (React + Vite)
================================================================================
"""
import sys
import asyncio
import os
import logging
from pathlib import Path
from datetime import datetime

# 检测GPU显存，如果不足则禁用CUDA（必须在导入torch之前设置）
try:
    import torch
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 2:
            print(f"[启动] GPU显存不足 ({gpu_memory:.1f}GB < 2GB)，强制使用CPU模式")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
except:
    pass

# ============================================================================
# 日志配置
# ============================================================================
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "app.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 路径初始化
# ============================================================================
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

logger.info("=" * 60)
logger.info("OpenCV Platform API 服务启动中...")
logger.info(f"项目路径: {project_root}")

# ============================================================================
# FastAPI 应用初始化
# ============================================================================
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from backend.core.config import settings

# ============================================================================
# 模式配置
# ============================================================================
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "http://localhost:8000")

logger.info(f"运行模式: {'开发模式' if DEV_MODE else '独立模式'}")
if DEV_MODE:
    logger.info(f"远程API地址: {REMOTE_API_URL}")

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 Ultralytics YOLO 的开源计算机视觉平台 API",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# ============================================================================
# 中间件
# ============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.debug(f"请求: {request.method} {request.url.path}")
        response = await call_next(request)
        logger.debug(f"响应状态: {response.status_code}")
        return response


class ProxyAPIMiddleware(BaseHTTPMiddleware):
    """API 代理中间件（开发模式使用）"""
    async def dispatch(self, request: Request, call_next):
        if DEV_MODE and str(request.url.path).startswith("/api/v1/"):
            import httpx
            remote_url = f"{REMOTE_API_URL.rstrip('/')}{request.url.path}"
            logger.info(f"[代理] {request.method} {request.url.path} -> {remote_url}")

            body = await request.body()
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    headers = dict(request.headers)
                    headers.pop("host", None)
                    remote_response = await client.request(
                        method=request.method,
                        url=remote_url,
                        headers=headers,
                        content=body,
                        params=request.query_params
                    )
                    return JSONResponse(
                        content=remote_response.json(),
                        status_code=remote_response.status_code
                    )
            except httpx.TimeoutException:
                return JSONResponse(
                    status_code=504,
                    content={"success": False, "message": "网关超时"}
                )
            except Exception as e:
                return JSONResponse(
                    status_code=502,
                    content={"success": False, "message": f"代理请求失败: {str(e)}"}
                )
        return await call_next(request)


# 注册中间件
app.add_middleware(ProxyAPIMiddleware)
app.add_middleware(LoggingMiddleware)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("中间件注册完成: CORS, 日志, 代理(开发模式)")

# ============================================================================
# 静态文件服务
# ============================================================================
uploads_dir = settings.UPLOADS_DIR
if uploads_dir.exists():
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")
    logger.info(f"上传文件服务: /uploads -> {uploads_dir}")

annotation_images_dir = settings.ANNOTATION_PROJECTS_DIR
if annotation_images_dir.exists():
    app.mount("/annotation-images", StaticFiles(directory=str(annotation_images_dir)), name="annotation_images")
    logger.info(f"标注图片服务: /annotation-images -> {annotation_images_dir}")

# ============================================================================
# 路由注册
# ============================================================================
from backend.api.routes import router as api_router
from backend.modules.data_preparation.routes import router as data_prep_router
from backend.modules.training.routes import router as training_router
from backend.modules.inference.routes import router as inference_router
from backend.modules.solutions.routes import router as solutions_router
from backend.modules.ai_native.routes import router as ai_native_router

# 基础 API 路由
app.include_router(api_router, prefix="/api/v1", tags=["基础功能"])

# 数据准备模块路由
app.include_router(data_prep_router, prefix="/api/v1", tags=["数据准备"])

# 训练模块路由
app.include_router(training_router, prefix="/api/v1", tags=["模型训练"])

# 推理模块路由
app.include_router(inference_router, prefix="/api/v1", tags=["推理测试"])

# 解决方案模块路由
app.include_router(solutions_router, prefix="/api/v1", tags=["智能解决方案"])

# AI原生应用路由
app.include_router(ai_native_router, prefix="/api/v1", tags=["AI原生应用"])

logger.info("所有 API 路由注册完成")

# ============================================================================
# 启动事件
# ============================================================================
@app.on_event("startup")
async def warmup_services():
    logger.info("=" * 50)
    logger.info("服务启动中...")

    if DEV_MODE:
        logger.info("[开发模式] API 请求将转发到远程服务器执行")
        logger.info("=" * 50)
        return

    # 独立模式：初始化所有服务
    loop = asyncio.get_running_loop()
    tasks = []

    try:
        from backend.core.yolo_engine import yolo_engine
        if yolo_engine:
            tasks.append(loop.run_in_executor(None, lambda: yolo_engine._iter_model_paths()))
            logger.info("模型引擎已加载")
    except Exception as e:
        logger.error(f"模型引擎初始化失败: {e}")

    try:
        from backend.modules.data_preparation import dataset_service
        tasks.append(loop.run_in_executor(None, dataset_service.list_datasets))
        logger.info("数据集服务已加载")
    except Exception as e:
        logger.error(f"数据集服务初始化失败: {e}")

    try:
        from backend.core.database import postgres_service
        postgres_service.init_database()
        logger.info("PostgreSQL 数据库初始化完成")
    except Exception as e:
        logger.warning(f"PostgreSQL 初始化失败（使用 SQLite）: {e}")

    try:
        from backend.core.s3_storage import s3_service
        s3_service.init_storage()
        logger.info("S3 存储初始化完成")
    except Exception as e:
        logger.warning(f"S3 存储初始化失败（使用本地存储）: {e}")

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("=" * 50)
    logger.info("服务启动完成！")
    logger.info(f"API 地址: http://localhost:{settings.API_PORT}")
    logger.info(f"API 文档: http://localhost:{settings.API_PORT}/api/docs")
    logger.info("=" * 50)


# ============================================================================
# 健康检查端点
# ============================================================================
@app.get("/api/v1/system/health")
async def health_check():
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

    print(f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║         OpenCV Platform - API 服务                             ║
    ║                                                                ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  运行模式: {'开发模式 (代理远程API)' if DEV_MODE else '独立模式 (本地GPU)'}                      ║
    ║  服务端口: {settings.API_PORT}                                             ║
    ║  API 地址: http://localhost:{settings.API_PORT}                              ║
    ║  API 文档: http://localhost:{settings.API_PORT}/api/docs                   ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
