# OpenCV Platform Dockerfile - 开发/测试版本（CPU 优化）
# 优化目标：体积小、拉取快、适合国内服务器

# ============== 第一阶段：构建依赖 ==============
FROM python:3.12-slim AS builder

LABEL maintainer="OpenCV Platform"
LABEL description="基于 Ultralytics YOLO 的开源计算机视觉平台"

WORKDIR /build

# 配置国内 pip 源（提速 5 倍+）
# 使用环境变量而非 pip config 命令，更快生效
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
# 加速下载
ENV PIP_CONNECTION_TIMEOUT=60
ENV PIP_RETRIES=3
ENV PIP_FIND_LINKS=https://mirror.sjtu.edu.cn/pytorch-wheels

# 预编译 wheel 无需编译工具，跳过 apt-get 加速构建

# 复制依赖文件并安装
COPY requirements.txt .

# 使用预编译 wheel（CPU 版本，无需编译）
# 优先使用清华镜像，fallback 到官方
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision torchaudio \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple/torch \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --trusted-host pypi.tuna.tsinghua.edu.cn || \
    pip install --no-cache-dir --prefix=/install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# 再安装其他依赖（使用 --prefer-binary 优先使用二进制包）
RUN pip download --no-cache-dir -r requirements.txt -d /tmp/wheels && \
    pip install --no-cache-dir --prefix=/install \
        --no-index --find-links=/tmp/wheels \
        --prefer-binary \
        -r requirements.txt

# ============== 第二阶段：运行时 ==============
FROM python:3.12-slim AS runtime

LABEL maintainer="OpenCV Platform"
LABEL description="基于 Ultralytics YOLO 的开源计算机视觉平台 (Python 3.12)"

# 配置国内 pip 源
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# 使用阿里云镜像站作为 apt 源（国内服务器提速）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装运行时系统依赖 (Debian trixie 适配)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libxcb1 \
    libxkbcommon0 \
    libdrm2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 从第一阶段复制已安装的 Python 包
COPY --from=builder /install /usr/local

# 创建工作目录和必要目录
WORKDIR /app
RUN mkdir -p data/datasets data/models data/exports data/uploads logs frontend/static

# 复制应用代码（分层复制利用缓存）
COPY backend /app/backend
COPY config /app/config
COPY frontend /app/frontend
COPY scripts /app/scripts
COPY app.py requirements.txt /app/

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV NVIDIA_DISABLE_REQUIRE=1

# 非 root 用户运行（安全）
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod -R 777 /app/data /app/logs

USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/system/health').read()" || exit 1

# 启动命令
CMD ["python", "app.py"]
