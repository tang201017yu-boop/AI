**仓库：** [tang201017yu-boop/AI](https://github.com/tang201017yu-boop/AI)（私有，分支 `dev`）

# OpenCV Platform - YOLO Edition

<div align="center">

![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

**基于 Ultralytics YOLO 的开源计算机视觉平台**

提供数据标注、模型训练、推理部署的完整工作流

[功能特性](#功能特性) • [快速开始](#快速开始) • [使用文档](#使用文档) • [API 文档](#api-文档)

</div>

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用文档](#使用文档)
- [API 文档](#api-文档)
- [架构设计](#架构设计)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

**OpenCV Platform** 是一个基于 Ultralytics YOLO 的开源计算机视觉平台，提供从数据标注到模型部署的完整工作流。

### 核心特性

```
数据标注 → 模型训练 → 模型管理 → 推理部署
```

**v2.2 更新：**
- **训练功能增强**：支持手动选择优化器（AdamW/SGD/Adam等）和自动匹配
- **默认模型更新**：YOLO26 系列成为默认首选模型
- **微调参数优化**：可手动配置学习率、马赛克增强、预热轮数等
- **国内部署优化**：Docker 镜像构建针对国内网络环境深度优化
- 模型评估功能：验证模型性能并获取 mAP、Precision、Recall 等指标
- 训练恢复支持：从检查点继续中断的训练

**v2.1 更新：**
- 新增项目管理功能（项目创建、活动日志、模型迁移）
- 新增模型库管理（上传、验证图表、17种格式导出）
- 新增训练监控（实时损失曲线、GPU监控、检查点管理）
- 新增图像浏览器（多种视图、筛选、全屏查看）
- 增强推理测试（拖拽上传、参数调节、示例图片）

---

## 功能特性

### 数据标注
- 基于 Supervision 的智能标注系统
- 自动标注：使用预训练模型快速标注
- 手动标注：支持检测框、分割多边形、关键点、旋转框
- SAM 智能分割辅助标注
- 导出为 YOLO 格式数据集

### 数据增强
- 基于 Albumentations 库的图像增强
- 支持 YOLO 格式边界框保持
- 几何变换：翻转、旋转、缩放、平移
- 光照变换：亮度、对比度、色调、饱和度
- 噪声与特效：模糊、高斯噪声、随机遮挡 (Cutout)
- 实时预览增强效果
- 生成增强数据集用于模型训练

### 模型训练
- 基于 Ultralytics YOLO (YOLO26/ YOLO11/ YOLO8)
- **YOLO26 默认首选**：最新一代 YOLO 模型作为默认训练选项
- **优化器选择**：支持手动选择（AdamW/SGD/Adam/NAdam/RAdam/RMSProp）或自动匹配
- **微调参数**：可配置初始学习率(lr0)、最终学习率因子(lrf)、预热轮数、马赛克增强
- **训练恢复**：支持从检查点继续中断的训练
- 实时训练进度监控
- 损失曲线和性能指标可视化
- 自动保存最佳模型权重
- 支持 CPU/GPU 训练
- 检查点管理

### 模型管理
- 模型上传与元数据解析
- 验证图表（混淆矩阵、PR曲线、F1曲线）
- 17种格式导出（ONNX、TensorRT、CoreML、TFLite等）
- 模型对比与迁移
- 活动日志追踪

### 推理部署
- RESTful API 接口
- 单张/批量图片推理
- 实时检测结果可视化
- Web UI 在线推理测试
- 置信度/IoU 参数调节

### Ultralytics Solutions
- 对象计数
- 热图生成
- 速度估算
- 距离计算
- 对象模糊
- 对象裁剪
- 队列管理

---

## 技术栈

### 后端
- **FastAPI** - 高性能 Web 框架
- **Ultralytics YOLO** - 计算机视觉模型库
- **PyTorch** - 深度学习框架
- **OpenCV** - 图像处理库

### 前端
- **原生 JavaScript** - 轻量级前端
- **HTML5 + CSS3** - 现代化 UI
- **Chart.js** - 图表可视化
- **响应式设计** - 适配各种设备

### 部署
- **Docker** - 容器化部署
- **Docker Compose** - 服务编排
- **Uvicorn** - ASGI 服务器

---

## 项目结构

```
YOLO-/
├── app.py                      # FastAPI 主应用入口
├── requirements.txt            # Python 依赖
├── Dockerfile                  # Docker 镜像配置
├── docker-compose.yml          # Docker 编排配置
├── .env.example                # 环境配置示例
├── .gitignore                  # Git 忽略文件
│
├── backend/                    # 后端代码
│   ├── core/                   # 核心模块
│   │   ├── config.py           # 配置管理
│   │   ├── yolo_engine.py      # YOLO 引擎
│   │   └── utils.py            # 工具函数
│   │
│   ├── modules/                # 功能模块
│   │   ├── data_preparation/   # 数据准备模块
│   │   │   ├── routes.py       # API 路由
│   │   │   ├── dataset_service.py    # 数据集服务
│   │   │   ├── annotation_service.py # 标注服务
│   │   │   ├── sam_service.py        # SAM 分割服务
│   │   │   ├── augmentation_service.py # 数据增强服务
│   │   │   ├── storage_service.py    # 智能存储服务
│   │   │   └── statistics_service.py # 统计服务
│   │   │
│   │   ├── training/           # 训练模块
│   │   │   ├── routes.py       # API 路由
│   │   │   ├── training_service.py   # 训练服务
│   │   │   ├── export_service.py     # 导出服务
│   │   │   ├── project_service.py    # 项目管理服务
│   │   │   └── model_service.py      # 模型管理服务
│   │   │
│   │   ├── inference/          # 推理模块
│   │   │   ├── routes.py       # API 路由
│   │   │   └── inference_service.py  # 推理服务
│   │   │
│   │   └── solutions/          # 解决方案模块
│   │       ├── routes.py       # API 路由
│   │       └── solutions_service.py  # 解决方案服务
│   │
│   └── api/
│       └── routes.py           # API 路由（从各模块导入）
│
├── frontend/                   # 前端页面
│   ├── index.html              # 首页
│   ├── inference.html          # 在线推理页面
│   ├── training.html           # 模型训练页面
│   ├── training_monitor.html   # 训练监控页面
│   ├── models.html             # 模型库页面
│   ├── model_detail.html       # 模型详情页面
│   ├── projects.html           # 项目管理页面
│   ├── datasets.html           # 数据集管理页面
│   ├── annotation.html         # 数据标注页面
│   ├── image_browser.html      # 图像浏览器页面
│   ├── solutions.html          # 解决方案页面
│   ├── augmentation.html       # 数据增强页面
│   │
│   └── static/                 # 静态资源
│       ├── css/
│       │   ├── style.css       # 主样式
│       │   ├── models.css      # 模型页面样式
│       │   └── ...
│       ├── js/
│       │   ├── api.js          # API 封装
│       │   └── ...
│       └── samples/            # 示例图片
│
├── data/                       # 数据目录
│   ├── datasets/               # 数据集存储
│   ├── models/                 # 模型文件
│   ├── exports/                # 导出文件
│   ├── uploads/                # 上传文件
│   └── annotation_projects/    # 标注项目
│
└── scripts/                    # 脚本工具
    ├── setup.sh                # 环境设置
    └── ...
```

---

## 快速开始

### 前置要求

- **Docker & Docker Compose** (Docker 部署)
- **Python 3.10+** (本地开发)
- **CUDA** (GPU 支持，可选)

### Docker 部署

```bash
# 1. 克隆项目
git clone <repository-url>
cd YOLO-

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件配置 GPU 支持等选项

# 3. 开发环境启动（热重载，CPU 模式）
docker compose -f docker-compose.dev.yml up -d --build

# 生产环境启动
docker compose -f docker-compose.prod.yml up -d --build

# 4. 查看日志
docker compose -f docker-compose.dev.yml logs -f

# 5. 停止服务
docker compose -f docker-compose.dev.yml down
```

**访问应用：**
- OpenCV Platform: http://localhost:8000
- API 文档: http://localhost:8000/api/docs

#### 部署优化（国内服务器）

Dockerfile 已针对国内服务器深度优化，支持阿里云、腾讯云等主流国内服务器：

| 优化项 | 说明 | 预期收益 |
|--------|------|----------|
| 清华 pip 源 | `pypi.tuna.tsinghua.edu.cn` | 下载提速 5 倍+ |
| PyTorch 镜像 | 优先清华镜像，fallback 官方 | PyTorch 下载提速 3 倍 |
| 阿里云 apt 源 | `mirrors.aliyun.com` | 系统包下载提速 2-4 倍 |
| 上海交大镜像 | PyTorch wheels 镜像站 | 额外加速保障 |
| slim 基础镜像 | `python:3.12-slim` | 体积减小 40% |
| 多阶段构建 | 分离编译依赖和运行时 | 最终镜像再省 30% |
| 离线 wheel 安装 | `pip download` + `pip install --no-index` | 网络抖动时稳定 |
| 构建上下文优化 | `.dockerignore` 排除大文件 | 构建更快 |

**推荐构建方式：**
```bash
# 标准构建
docker build -t opencv-platform .

# 或启用 BuildKit 加速
DOCKER_BUILDKIT=1 docker build -t opencv-platform .

# 或使用 docker buildx（支持多平台）
docker buildx build --platform linux/amd64 -t opencv-platform .
```

**GPU 支持（需要 NVIDIA Docker）：**
```yaml
# docker-compose.yml 中添加
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**离线构建（内网服务器）：**
```bash
# 1. 在有外网的机器下载 wheel 包
pip download -r requirements.txt -d ./wheels

# 2. 构建时使用本地 wheel
docker build --build-arg PIP_WHEELS_DIR=./wheels -t opencv-platform:local .
```

### 本地开发

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python app.py

# 或使用 uvicorn（支持热重载）
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 使用文档

### 1. 数据标注

1. 访问标注页面
2. 创建新项目，上传图片
3. 使用自动标注或手动绘制检测框
4. 导出为 YOLO 格式

### 2. 模型训练

1. 准备 YOLO 格式数据集
2. 访问训练页面
3. 配置参数：
   - **模型类型**：默认 YOLO26n，支持 YOLO26/11/8 全系列
   - **优化器**：选择自动匹配（推荐）或手动指定（AdamW/SGD/Adam/NAdam/RAdam/RMSProp）
   - **训练参数**：轮数、批次大小、图像尺寸
   - **微调参数**（可选）：初始学习率、最终学习率因子、预热轮数、马赛克增强
4. 开始训练，实时监控进度
5. 训练中断后可从检查点恢复

### 3. 模型管理

1. 访问模型库页面
2. 上传 .pt 模型文件
3. 查看模型元数据和性能指标
4. 导出为目标部署格式

### 4. 推理测试

1. 访问模型详情页的测试标签
2. 上传或选择示例图片
3. 调整置信度、IoU 参数
4. 查看检测结果

---

## API 文档

### 系统信息

#### GET `/api/v1/system/info`
获取系统信息

### 数据集接口

#### GET `/api/v1/datasets/list`
获取数据集列表

#### POST `/api/v1/datasets/upload`
上传数据集

### 数据增强接口

#### GET `/api/v1/augmentation/transforms`
获取可用的增强变换列表

#### POST `/api/v1/augmentation/preview`
预览增强效果（上传图片，返回原图和增强图）

#### POST `/api/v1/datasets/{name}/augment`
增强数据集（简单参数配置）

#### POST `/api/v1/datasets/{name}/augment/custom`
增强数据集（自定义 JSON 配置）

#### GET `/api/v1/datasets/augmented/list`
获取增强数据集列表

### 训练接口

#### POST `/api/v1/training/start`
开始训练

#### GET `/api/v1/training/status/{task_id}`
获取训练状态

#### GET `/api/v1/training/{task_id}/metrics`
获取训练指标

#### POST `/api/v1/training/validate`
验证模型性能，返回 mAP@0.5、mAP@0.5:0.95、Precision、Recall、F1 等指标

#### POST `/api/v1/training/resume`
从检查点恢复训练

#### GET `/api/v1/training/{task_id}/checkpoints`
获取训练检查点列表

#### GET `/api/v1/training/tasks`
列出所有训练任务

#### POST `/api/v1/training/cancel/{task_id}`
取消训练任务

### 模型接口

#### POST `/api/v1/models/upload`
上传模型

#### GET `/api/v1/models`
获取模型列表

#### GET `/api/v1/models/{model_id}`
获取模型详情

#### POST `/api/v1/models/{model_id}/export`
导出模型

#### POST `/api/v1/models/{model_id}/infer`
模型推理

### 项目接口

#### POST `/api/v1/projects`
创建项目

#### GET `/api/v1/projects`
获取项目列表

#### GET `/api/v1/projects/{project_id}/activity`
获取项目活动日志

完整 API 文档请访问：http://localhost:8000/api/docs

---

## 架构设计

### 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Web Browser                        │
│                 (Frontend UI)                        │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/REST API
┌────────────────────▼────────────────────────────────┐
│              FastAPI Application                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ Routes                                        │   │
│  │ ├── data_preparation/                         │   │
│  │ ├── training/                                 │   │
│  │ ├── inference/                                │   │
│  │ └── solutions/                                │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──────┐ ┌──▼─────────┐ ┌▼──────────────┐
│ Ultralytics  │ │  Services  │ │  File System  │
│    YOLO      │ │  (Core)    │ │   Storage     │
└──────────────┘ └────────────┘ └───────────────┘
```

### 数据流

```
原始图片 → 数据标注 → 导出数据集 → 模型训练 → 模型管理 → 推理部署
```

---

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

⭐ 如果这个项目对你有帮助，请给它一个 Star！⭐

</div>
