# OpenCV Platform - RTX 5080 工作站部署指南

## 系统要求

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA RTX 5080 (16GB显存) |
| CPU | Intel Core i7/i9 或 AMD Ryzen 7/9 |
| 内存 | 32GB DDR5 |
| 存储 | 1TB NVMe SSD |
| CUDA | 12.4+ |
| Python | 3.10+ |

## 快速部署

### 方式一：本地 Python 部署（推荐用于开发）

```bash
# 克隆项目
git clone <your-repo>
cd YOLO-

# 运行部署脚本
./start_rtx5080.sh
```

**特点：**
- 无 Docker 开销，性能更好
- 方便调试和开发
- 自动检测并安装 GPU 驱动

### 方式二：Docker GPU 部署（推荐用于生产）

```bash
# 构建并启动
docker compose -f docker-compose.gpu.yml up -d

# 查看日志
docker logs -f opencv-platform-gpu

# 停止服务
docker compose -f docker-compose.gpu.yml down
```

## GPU 优化配置

### PyTorch 混合精度训练

```python
from backend.core.config import settings

# 启用 AMP (自动混合精度)
with torch.cuda.amp.autocast():
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### CUDA 优化环境变量

```bash
# 在 .env 文件中设置
CUDA_TF32=1           # 启用 TF32 加速（RTX 30/40 系列）
CUDNN_BENCHMARK=1     # cuDNN 自动调优
CUDA_LAUNCH_BLOCKING=0 # 异步 CUDA 调用
```

### 训练优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 32 | RTX 5080 16GB 可用较大批次 |
| img_size | 640 | 输入图像大小 |
| epochs | 100 | 训练轮数 |
| workers | 8 | 数据加载线程数 |
| amp | True | 混合精度训练 |

## 目录结构

```
YOLO-/
├── data/                  # 数据目录
│   ├── datasets/          # 数据集
│   ├── models/            # 训练好的模型
│   ├── exports/           # 导出的模型
│   ├── uploads/           # 上传文件
│   └── annotation_projects/ # 标注项目
├── backend/               # 后端代码
├── frontend/              # 前端代码
├── scripts/               # 脚本工具
├── logs/                  # 日志文件
├── venv/                  # Python 虚拟环境
├── start_rtx5080.sh       # 启动脚本
├── Dockerfile.gpu         # GPU Docker 镜像
├── docker-compose.gpu.yml # GPU 编排配置
└── DEPLOY_RTX5080.md      # 本文档
```

## 常用命令

### 本地部署

```bash
# 启动服务
./start_rtx5080.sh

# 后台运行
nohup ./start_rtx5080.sh > logs/server.log 2>&1 &

# 查看日志
tail -f logs/server.log
```

### Docker 部署

```bash
# 启动所有服务
docker compose -f docker-compose.gpu.yml up -d

# 只启动应用
docker compose -f docker-compose.gpu.yml up -d opencv-platform-gpu

# 重启应用（代码变更后）
docker compose -f docker-compose.gpu.yml restart opencv-platform-gpu

# 查看 GPU 使用
docker stats opencv-platform-gpu

# 进入容器调试
docker exec -it opencv-platform-gpu bash
```

## 性能调优

### 1. 训练时

```python
# 使用较大的 batch size
model.train(data="dataset.yaml", batch=32, epochs=100, imgsz=640)

# 启用缓存
model.train(data="dataset.yaml", cache="ram")

# 多 workers
model.train(data="dataset.yaml", workers=8)
```

### 2. 推理时

```python
# 批量推理
results = model.predict(source, batch=16, imgsz=640)

# 半精度推理
model.predict(source, half=True)
```

### 3. GPU 监控

```bash
# 实时监控 GPU
nvidia-smi -l 1

# 查看 GPU 内存
nvidia-smi -q -d MEMORY

# 查看 GPU 利用率
nvidia-smi -q -d UTILIZATION
```

## 故障排查

### GPU 无法使用

```bash
# 检查 CUDA
nvcc --version

# 检查 PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"

# 检查 GPU 驱动
nvidia-smi
```

### Docker GPU 问题

```bash
# 重新安装 nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 显存不足

```bash
# 减小 batch size
export DEFAULT_BATCH_SIZE=16

# 减小图像大小
export DEFAULT_IMG_SIZE=512
```

## 访问地址

| 服务 | 地址 |
|------|------|
| Web 应用 | http://localhost:8000 |
| API 文档 | http://localhost:8000/api/docs |
| MinIO 控制台 | http://localhost:9001 |

## 下一步

1. 访问 http://localhost:8000
2. 在"模型训练"页面创建新项目
3. 上传数据集并开始训练
4. 在"推理测试"页面验证模型效果
