# 训练完成详情查看功能设计

## 需求概述

用户希望能够查看已完成训练的详细信息，包括训练指标、训练曲线、验证结果图片，并支持导出模型、推理测试、继续训练等操作。

## 问题分析

1. **训练状态不同步** - `experiments.json` 中所有任务都标记为 `"status": "running"`，但实际训练早已完成
2. **两套系统并存** - `training_service.experiments` 和 `yolo_engine.training_tasks` 是两套独立的记录，系统重启后 `training_tasks` 为空
3. **前端获取不到已完成训练** - `/training/tasks` API 返回空列表
4. **详情数据缺失** - 没有专门获取已完成训练详情的 API

## 设计方案

### 1. 后端 API 设计

#### 1.1 获取实验列表
- **现有 API**: `GET /api/v1/experiments`
- **问题**: status 字段不准确
- **修复**: 训练完成时同步更新 experiments.json 中的 status

#### 1.2 获取单个实验详情
- **新 API**: `GET /api/v1/experiments/{task_id}`
- **返回**:
  - 基础信息（task_id, project_name, model_type, dataset_path, epochs, batch_size）
  - 训练状态（status, created_at, completed_at）
  - 最终指标（best_metrics: mAP50, mAP50-95, precision, recall）
  - 训练曲线数据（loss_history, metrics_history）
  - 模型路径（checkpoint_path）

#### 1.3 获取验证结果图片
- **新 API**: `GET /api/v1/experiments/{task_id}/results`
- **返回**: 验证结果图片列表（confusion_matrix, PR_curve, F1_curve等）

#### 1.4 导出模型
- **新 API**: `POST /api/v1/experiments/{task_id}/export`
- **参数**: format (onnx/tflite/torchscript/pytorch)
- **返回**: 导出结果

#### 1.5 推理测试
- **新 API**: `POST /api/v1/experiments/{task_id}/infer`
- **参数**: image_url, conf_threshold, iou_threshold
- **返回**: 推理结果

#### 1.6 继续训练
- **新 API**: `POST /api/v1/experiments/{task_id}/resume`
- **参数**: epochs, batch_size, resume_from_best
- **返回**: 新任务ID

### 2. 前端页面设计

#### 2.1 训练历史列表页
- 使用 `/experiments` API 获取列表
- 显示任务名称、模型类型、状态、创建时间
- 点击某一行跳转到 `/training/{taskId}/details`

#### 2.2 详情页 `/training/:taskId/details`

**顶部信息区**
- 任务名称、项目名称、模型类型
- 训练状态标签（已完成/失败）
- 创建时间、完成时间

**指标卡片区**
- mAP@0.5（最佳）
- mAP@0.5:0.95（最佳）
- Precision（最佳）
- Recall（最佳）
- 总Epochs / 最佳Epoch

**训练曲线区**
- Loss曲线（box_loss, cls_loss, dfl_loss）
- 性能指标曲线（mAP50, mAP50-95, precision, recall）
- 支持切换查看

**验证结果图片区**
- 混淆矩阵
- PR曲线
- F1曲线
- 召回率曲线

**操作按钮区**
- 导出模型（跳转到导出页面或弹窗选择格式）
- 推理测试（跳转到推理页面并预选该模型）
- 继续训练（跳转到训练页面并预填配置）

### 3. 状态同步修复

训练完成时同时更新两处状态：
1. `yolo_engine.training_tasks[task_id].status = "completed"`
2. `training_service.experiments[task_id]["status"] = "completed"`
3. `training_service._save_experiments()`

## 数据流

```
用户点击训练历史
    ↓
GET /experiments → 返回实验列表（含错误status）
    ↓
用户点击某条记录
    ↓
GET /experiments/{task_id} → 返回完整详情
    ↓
GET /experiments/{task_id}/results → 返回验证图片
    ↓
渲染详情页
```

## 验收标准

1. 训练历史列表正确显示已完成的任务（status = "completed"）
2. 详情页正确显示所有训练指标
3. 训练曲线图表正确渲染
4. 验证结果图片正确显示
5. 导出模型功能正常工作
6. 推理测试功能正常工作
7. 继续训练功能正常工作
