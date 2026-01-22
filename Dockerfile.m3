# OpenCV Platform Dockerfile - Apple Silicon M3 优化版本
# 使用 CPU + MPS (Metal Performance Shaders) 加速

# ============== 第一阶段：Python 基础 ==============
FROM python:3.12-slim-bookworm AS base

LABEL maintainer="OpenCV Platform"
LABEL description="基于 Ultralytics YOLO 的开源计算机视觉平台 (Apple Silicon M3)"

# 配置 pip 源
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_TIMEOUT=120
ENV PIP_RETRIES=5

# 使用阿里云镜像源（解决国内网络问题）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============== 第二阶段：构建依赖 ==============
FROM base AS builder

WORKDIR /build

# 安装依赖
COPY requirements.txt .

# 分步安装（避免一次性下载超时）
RUN pip install --no-cache-dir ultralytics

RUN pip install --no-cache-dir \
    opencv-python supervision Pillow pyyaml albumentations \
    psycopg2-binary psycopg[binary] pgvector boto3 requests \
    fastapi uvicorn[standard] python-multipart jinja2

# ============== 第三阶段：运行时 ==============
FROM base AS runtime

LABEL maintainer="OpenCV Platform"
LABEL description="基于 Ultralytics YOLO 的开源计算机视觉平台 (Apple Silicon M3)"

# pip 配置
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 跨阶段复制依赖
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# 复制应用代码
COPY backend /app/backend
COPY config /app/config
COPY frontend /app/frontend
COPY scripts /app/scripts
COPY app.py requirements.txt /app/

# 创建目录
WORKDIR /app
RUN mkdir -p data/datasets data/models data/exports data/uploads logs frontend/static

# 环境变量（MPS 加速）
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV MPS_DETERMINISTIC_REDUCTION=0

# 非 root 用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod -R 777 /app/data /app/logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/system/health').read()" || exit 1

CMD ["python", "app.py"]
