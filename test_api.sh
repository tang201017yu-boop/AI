#!/bin/bash
# ========================================
# API 测试脚本
# ========================================

BASE_URL="http://localhost:8000/api/v1"

echo "=========================================="
echo "  YOLO Platform API 测试"
echo "=========================================="
echo ""

# 测试健康检查
echo "1. 测试健康检查..."
curl -s "$BASE_URL/system/health" | python -m json.tool 2>/dev/null || echo "服务未启动"
echo ""

# 测试系统信息
echo "2. 测试系统信息..."
curl -s "$BASE_URL/system/info" | python -m json.tool 2>/dev/null || echo "服务未启动"
echo ""

# 测试数据集列表
echo "3. 测试数据集列表..."
curl -s "$BASE_URL/datasets/list" | python -m json.tool 2>/dev/null || echo "服务未启动"
echo ""

# 测试模型列表
echo "4. 测试模型列表..."
curl -s "$BASE_URL/models/list" | python -m json.tool 2>/dev/null || echo "服务未启动"
echo ""

# 测试解决方案列表
echo "5. 测试解决方案列表..."
curl -s "$BASE_URL/solutions/list" | python -m json.tool 2>/dev/null || echo "服务未启动"
echo ""

echo "=========================================="
echo "  测试完成"
echo "=========================================="
