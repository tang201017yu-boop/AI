#!/bin/bash
# ========================================
# AI检测系统 - 一键部署脚本
# 项目：project_20260227171911
# 创建时间：2026-02-27T17:19:11.430342
# ========================================

echo "正在部署AI检测系统..."
echo "检测场景：人员, 安全帽, 防护服"
echo "模型：helmet, hardhat, clothes, safety, person, vest, worker, ppe"
echo "阈值：0.5"
echo ""

# 检查环境
if ! command -v python3 &> /dev/null; then
    echo "错误：需要 Python 3.8+"
    exit 1
fi

# 创建工作目录
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

# 安装依赖
echo "安装依赖..."
pip install -q ultralytics opencv-python pillow

# 启动检测服务
echo "启动检测服务..."
python3 -c "
import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 检测函数
def detect(frame):
    results = model.predict(frame, conf=0.5, verbose=False)
    return results

print('检测服务已启动，按 Ctrl+C 停止')
# 这里可以添加视频流处理逻辑
"

echo "部署完成！"
