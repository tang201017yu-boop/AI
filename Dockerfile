# OpenCV Platform Dockerfile - 开发/测试版本（CPU 优化）
# 优化目标：体积小、拉取快、适合国内服务器

# ============== 第一阶段：PyTorch 基础 ==============
FROM python:3.12-slim-bookworm AS pytorch-base

LABEL maintainer="OpenCV Platform"
LABEL description="基于 Ultralytics YOLO 的开源计算机视觉平台"

# 配置国内 pip 源（仅清华，稳定）
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_TIMEOUT=120
ENV PIP_RETRIES=5

# 安装 PyTorch（增加超时和重试）
RUN pip install --no-cache-dir \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        torchaudio==2.5.1+cpu \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple/torch \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    || pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# ============== 第二阶段：构建依赖 ==============
FROM pytorch-base AS builder

WORKDIR /build

# 安装编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 复制依赖文件
COPY requirements.txt .

# 分步安装依赖，避免一次性下载大文件超时
# 先安装 ultralytics（会同时安装大部分依赖）
RUN pip install --no-cache-dir ultralytics

# 再安装其他依赖（过滤掉已安装的）
RUN pip install --no-cache-dir \
    opencv-python supervision Pillow pyyaml albumentations \
    psycopg2-binary psycopg[binary] pgvector boto3 requests \
    fastapi uvicorn python-multipart jinja2

# ============== 第三阶段：运行时 ==============
FROM python:3.12-slim-bookworm AS runtime

LABEL maintainer="OpenCV Platform"
LABEL description="基于 Ultralytics YOLO 的开源计算机视觉平台 (Python 3.12)"

# 配置国内 pip 源
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# 使用阿里云镜像
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libxcb1 \
    libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 跨阶段复制（分层复制优化）
COPY --from=pytorch-base /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=pytorch-base /usr/local/bin/ /usr/local/bin/

# 复制应用代码
COPY backend /app/backend
COPY config /app/config
COPY frontend /app/frontend
COPY scripts /app/scripts
COPY app.py requirements.txt /app/

# 创建目录
WORKDIR /app
RUN mkdir -p data/datasets data/models data/exports data/uploads logs frontend/static

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV NVIDIA_DISABLE_REQUIRE=1

# 非 root 用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod -R 777 /app/data /app/logs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/system/health').read()" || exit 1

CMD ["python", "app.py"]
