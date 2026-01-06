"""
配置管理模块
"""
import os
from pathlib import Path
from typing import List

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
EXPORTS_DIR = DATA_DIR / "exports"
UPLOADS_DIR = DATA_DIR / "uploads"
FALLBACK_DATASETS_DIR = BASE_DIR / "datasets"
FALLBACK_MODELS_DIR = BASE_DIR / "models"

# 确保目录存在
for directory in [DATA_DIR, DATASETS_DIR, MODELS_DIR, EXPORTS_DIR, UPLOADS_DIR]:
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
    DATA_DIR: Path = DATA_DIR
    DATASETS_DIR: Path = DATASETS_DIR
    MODELS_DIR: Path = MODELS_DIR
    EXPORTS_DIR: Path = EXPORTS_DIR
    UPLOADS_DIR: Path = UPLOADS_DIR
    
    # 模型配置
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "yolo11n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    
    # 训练配置
    DEFAULT_EPOCHS: int = int(os.getenv("DEFAULT_EPOCHS", "100"))
    DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "16"))
    DEFAULT_IMG_SIZE: int = int(os.getenv("DEFAULT_IMG_SIZE", "640"))
    MAX_TRAINING_WORKERS: int = max(1, int(os.getenv("MAX_TRAINING_WORKERS", "1")))
    
    # 缓存与扫描配置
    MODEL_METADATA_CACHE_TTL: int = int(os.getenv("MODEL_METADATA_CACHE_TTL", "30"))
    DATASET_CACHE_TTL: int = int(os.getenv("DATASET_CACHE_TTL", "10"))
    
    # API 配置
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "50")) * 1024 * 1024  # 转换为字节
    ALLOWED_EXTENSIONS: List[str] = os.getenv(
        "ALLOWED_EXTENSIONS", 
        "jpg,jpeg,png,bmp,mp4,avi,mov"
    ).split(",")
    
    # CORS 配置
    CORS_ORIGINS: List[str] = ["*"]  # 生产环境应该限制具体域名

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
