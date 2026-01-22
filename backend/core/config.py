"""
配置管理模块 - Configuration Management
"""
import os
from pathlib import Path
from typing import List

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
EXPORTS_DIR = DATA_DIR / "exports"
UPLOADS_DIR = DATA_DIR / "uploads"
ANNOTATION_PROJECTS_DIR = DATA_DIR / "annotation_projects"
FALLBACK_DATASETS_DIR = BASE_DIR / "datasets"
FALLBACK_MODELS_DIR = BASE_DIR / "models"

# 确保目录存在
for directory in [DATA_DIR, DATASETS_DIR, MODELS_DIR, EXPORTS_DIR, UPLOADS_DIR, ANNOTATION_PROJECTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Settings:
    """应用配置"""

    # 应用信息
    APP_NAME: str = os.getenv("APP_NAME", "OpenCV Platform")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # 服务端口
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # 路径配置
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    DATASETS_DIR: Path = DATASETS_DIR
    MODELS_DIR: Path = MODELS_DIR
    EXPORTS_DIR: Path = EXPORTS_DIR
    UPLOADS_DIR: Path = UPLOADS_DIR
    ANNOTATION_PROJECTS_DIR: Path = ANNOTATION_PROJECTS_DIR

    # 模型配置
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "yolo26n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

    # 训练配置 (RTX 5080 优化 - 16GB显存)
    DEFAULT_EPOCHS: int = int(os.getenv("DEFAULT_EPOCHS", "100"))
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))  # RTX 5080 可用较大批次
    DEFAULT_IMG_SIZE: int = int(os.getenv("DEFAULT_IMG_SIZE", "640"))
    MAX_TRAINING_WORKERS: int = max(1, int(os.getenv("MAX_TRAINING_WORKERS", "8")))

    # GPU 优化配置
    DEFAULT_AMP: bool = os.getenv("DEFAULT_AMP", "True").lower() == "true"  # 混合精度
    DEFAULT_WORKERS: int = int(os.getenv("DEFAULT_WORKERS", "8"))
    DEFAULT_CACHE: str = os.getenv("DEFAULT_CACHE", "ram")

    # CUDA 优化
    CUDA_LAUNCH_BLOCKING: int = int(os.getenv("CUDA_LAUNCH_BLOCKING", "0"))

    # RTX 5080 特定优化
    CUDA_TF32: bool = os.getenv("CUDA_TF32", "True").lower() == "true"  # 启用 TF32 加速
    CUDNN_BENCHMARK: bool = os.getenv("CUDNN_BENCHMARK", "True").lower() == "true"  # cuDNN 自动调优

    # 缓存与扫描配置
    MODEL_METADATA_CACHE_TTL: int = int(os.getenv("MODEL_METADATA_CACHE_TTL", "30"))
    DATASET_CACHE_TTL: int = int(os.getenv("DATASET_CACHE_TTL", "10"))

    # API 配置
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "500")) * 1024 * 1024
    ALLOWED_EXTENSIONS: List[str] = os.getenv(
        "ALLOWED_EXTENSIONS",
        "jpg,jpeg,png,bmp,mp4,avi,mov,zip"
    ).split(",")

    # CORS 配置
    CORS_ORIGINS: List[str] = ["*"]

    # ==================== PostgreSQL 配置 ====================
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "opencv_platform")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_POOL_SIZE: int = int(os.getenv("POSTGRES_POOL_SIZE", "10"))

    @property
    def postgres_url(self) -> str:
        """获取 PostgreSQL 连接 URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # ==================== S3/MinIO 配置 ====================
    S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "opencv-platform")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    S3_USE_SSL: bool = os.getenv("S3_USE_SSL", "false").lower() == "true"

    @property
    def s3_public_url(self) -> str:
        """获取 S3 公开访问 URL"""
        return f"{self.S3_ENDPOINT_URL}/{self.S3_BUCKET}"

    # ==================== 存储后端选择 ====================
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")  # s3 或 local
    USE_VECTOR_SEARCH: bool = os.getenv("USE_VECTOR_SEARCH", "false").lower() == "true"  # 默认关闭

    @staticmethod
    def _unique_paths(paths: List[Path]) -> List[Path]:
        seen = set()
        unique: List[Path] = []
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(resolved)
        return unique

    @staticmethod
    def _parse_extra_paths(env_key: str) -> List[Path]:
        raw_value = os.getenv(env_key, "")
        results: List[Path] = []
        for item in raw_value.split(","):
            candidate = item.strip()
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists():
                results.append(path)
        return results

    @property
    def model_search_paths(self) -> List[Path]:
        paths: List[Path] = [self.MODELS_DIR]
        paths.extend(self._parse_extra_paths("EXTRA_MODEL_DIRS"))
        if FALLBACK_MODELS_DIR.exists():
            paths.append(FALLBACK_MODELS_DIR)
        return self._unique_paths([p for p in paths if p.exists()])

    @property
    def dataset_search_paths(self) -> List[Path]:
        paths: List[Path] = [self.DATASETS_DIR]
        paths.extend(self._parse_extra_paths("EXTRA_DATASET_DIRS"))
        if FALLBACK_DATASETS_DIR.exists():
            paths.append(FALLBACK_DATASETS_DIR)
        return self._unique_paths([p for p in paths if p.exists()])


settings = Settings()
