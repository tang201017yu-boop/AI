# OpenCV Platform - 分离式开发架构指南

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                      本地开发环境 (MacBook Air M3)               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  VS Code / PyCharm                                       │   │
│  │  ├── 前端代码 (HTML/CSS/JS)                              │   │
│  │  └── 后端代码 (Python/FastAPI)                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ HTTP API 代理                    │
│                              ▼                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ 网络 (局域网/公网)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      服务器端 (RTX 5080 工作站)                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Docker Compose                                          │   │
│  │  ├── PostgreSQL (数据库)                                 │   │
│  │  ├── MinIO (S3 存储)                                     │   │
│  │  └── OpenCV Platform (GPU 加速)                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│              GPU 训练 │ 模型存储 │ 数据集管理                    │
└─────────────────────────────────────────────────────────────────┘
```

## 优势

| 方面 | 传统单机型 | 分离式架构 |
|------|-----------|-----------|
| **开发体验** | 受限的 Mac 本地环境 | 流畅的本地开发 |
| **GPU 资源** | M3 不支持 CUDA | 充分利用 RTX 5080 |
| **依赖安装** | 复杂的 ARM64 兼容 | 只需轻量级依赖 |
| **协作** | 环境不一致 | 统一服务器环境 |

## 快速开始

### 1. 服务器端部署 (RTX 5080)

在服务器上执行：

```bash
# 克隆项目
git clone <your-repo>
cd YOLO-

# 设置环境变量
cp .env.example .env
# 编辑 .env，修改:
# STORAGE_BACKEND=s3
# POSTGRES_HOST=postgres

# 启动完整服务
docker compose -f docker-compose.gpu.yml up -d

# 验证服务
curl http://localhost:8000/api/v1/system/health
```

### 2. 本地开发环境 (MacBook Air M3)

```bash
# 克隆项目（如果还没克隆）
git clone <your-repo>
cd YOLO-

# 设置服务器地址
export REMOTE_API_URL="http://<你的服务器IP>:8000"
export DEV_MODE=true

# 启动本地开发环境
./start_dev_local.sh
```

或使用配置文件：

```bash
# 创建 .env.local
echo "DEV_MODE=true" > .env.local
echo "REMOTE_API_URL=http://192.168.1.100:8000" >> .env.local

# 启动
./start_dev_local.sh
```

## 环境变量配置

### 本地开发环境 (.env.local)

```bash
# 启用开发模式
DEV_MODE=true

# 服务器 API 地址（修改为你的服务器IP）
REMOTE_API_URL=http://192.168.1.100:8000

# 可选：自定义本地端口
API_PORT=8000
```

### 服务器端 (.env)

```bash
# 存储后端
STORAGE_BACKEND=s3

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=opencv_platform
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# S3/MinIO
S3_ENDPOINT_URL=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=opencv-platform
```

## 工作流程

### 开发流程

```
1. 本地修改代码
   ├── 前端: frontend/annotation.html
   └── 后端: backend/modules/xxx/

2. 本地服务自动重载
   └── 浏览器刷新查看效果

3. API 请求自动转发到服务器
   └── 训练/推理任务在 GPU 上执行

4. 提交代码
   └── git push 到仓库
```

### 数据管理

| 数据类型 | 存储位置 | 访问方式 |
|---------|---------|---------|
| 数据集 | 服务器 MinIO | Web 界面上传 |
| 模型 | 服务器 MinIO | Web 界面下载 |
| 标注项目 | 服务器 MinIO | Web 界面管理 |
| 上传文件 | 服务器 MinIO | Web 界面上传 |

## 常用命令

### 本地开发 (MacBook)

```bash
# 启动开发环境
./start_dev_local.sh

# 后台运行
nohup ./start_dev_local.sh > logs/dev.log 2>&1 &

# 查看日志
tail -f logs/dev.log

# 停止
pkill -f "uvicorn app:app"
```

### 服务器部署 (RTX 5080)

```bash
# 启动所有服务
docker compose -f docker-compose.gpu.yml up -d

# 只重启应用
docker compose -f docker-compose.gpu.yml restart opencv-platform-gpu

# 查看日志
docker logs -f opencv-platform-gpu

# 查看 GPU 使用
docker stats opencv-platform-gpu

# 停止服务
docker compose -f docker-compose.gpu.yml down
```

### SSH 隧道（远程访问）

如果服务器在局域网内，需要通过 SSH 访问：

```bash
# 从 MacBook 连接到服务器
ssh -L 8000:localhost:8000 user@server-ip

# 然后本地访问 http://localhost:8000
```

## 故障排查

### 本地开发问题

**问题：无法连接到服务器**

```bash
# 检查服务器是否运行
curl http://<服务器IP>:8000/api/v1/system/health

# 检查防火墙设置
ssh user@server "docker ps"
```

**问题：代理请求失败**

```bash
# 确认环境变量
echo $REMOTE_API_URL

# 测试直接连接
curl $REMOTE_API_URL/api/v1/models/list
```

### 服务器端问题

**问题：GPU 不可用**

```bash
# 检查 nvidia-smi
ssh user@server "nvidia-smi"

# 检查 Docker GPU
docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi
```

**问题：端口冲突**

```bash
# 停止占用端口的进程
ssh user@server "lsof -ti:8000 | xargs kill -9"
```

## 开发建议

### 1. 代码同步

使用 Rsync 同步代码到服务器：

```bash
# 同步代码到服务器
rsync -avz --delete \
    --exclude='venv' \
    --exclude='data' \
    --exclude='logs' \
    ./ user@server:/path/to/YOLO-/
```

或使用 Git：

```bash
# 本地提交
git add .
git commit -m "fix: xxx"
git push

# 服务器拉取
ssh user@server "cd /path/to/YOLO- && git pull"
```

### 2. 前端调试

在 MacBook 上开发前端时，可以使用浏览器开发者工具：

```bash
# Network 标签查看 API 请求
# Console 标签查看错误日志
# 代理会自动转发请求到服务器
```

### 3. 后端调试

在服务器上查看实时日志：

```bash
# Docker 日志
docker logs -f opencv-platform-gpu

# 或进入容器调试
docker exec -it opencv-platform-gpu bash
```

## 目录结构

```
YOLO-/
├── backend/              # 后端代码
│   ├── core/            # 核心模块
│   ├── modules/         # 功能模块
│   └── api/             # API 路由
├── frontend/            # 前端代码
│   ├── static/          # 静态资源
│   ├── index.html       # 主页面
│   └── *.html           # 各功能页面
├── data/                # 数据目录（服务器端）
│   ├── datasets/        # 数据集
│   ├── models/          # 模型
│   ├── exports/         # 导出
│   └── uploads/         # 上传
├── scripts/             # 工具脚本
├── start_dev_local.sh   # 本地开发启动脚本
├── start_rtx5080.sh     # 服务器启动脚本
├── docker-compose.gpu.yml  # GPU 编排配置
└── DEVELOPMENT.md       # 本文档
```

## 访问地址

| 环境 | 地址 |
|------|------|
| **本地开发** | http://localhost:8000 |
| **服务器直接** | http://<服务器IP>:8000 |
| **API 文档** | http://localhost:8000/api/docs |
| **MinIO 控制台** | http://<服务器IP>:9001 |

## 下一步

1. 在服务器上运行 `./start_rtx5080.sh`
2. 在 MacBook 上运行 `./start_dev_local.sh`
3. 打开浏览器访问 http://localhost:8000
4. 开始开发！
