# 智能标注功能优化设计

**日期**: 2026-03-03
**参考**: Ultralytics 智能标注文档

## 1. 概述

优化智能标注功能，提供完整的交互式标注体验，支持 SAM 点点击分割、批量同类标注、YOLO+SAM 联动等功能。

## 2. 功能需求

### 2.1 SAM 点点击分割
- 正样本点 (label=1): 标识要分割的目标
- 负样本点 (label=0): 标识要排除的区域
- 迭代优化: 多次点击细化分割结果

### 2.2 批量同类标注
- 输入类别名称，自动标注图片中所有该类别对象
- YOLO 检测 + SAM 分割联动

### 2.3 YOLO+SAM 联动
- YOLO 检测所有对象
- 对每个检测框应用 SAM 精细分割

### 2.4 标注管理
- 标注列表展示
- 类别管理
- 标注选择、删除、修改
- 导出 YOLO 格式

## 3. 技术架构

### 3.1 前端 (React)
- AnnotationCanvas: 交互式画布组件
- AnnotationToolbar: 工具栏
- AnnotationPanel: 结果面板

### 3.2 后端 (Python)
- SAMService: SAM 分割服务
- SmartAnnotationService: 智能标注服务
- 新增批量同类标注 API

## 4. API 设计

### 4.1 点点击分割
```
POST /sam/predict
Body: { points: [[x, y], ...], labels: [1, 0, ...] }
```

### 4.2 批量同类标注
```
POST /sam/batch-sam-label
Body: { image_path, class_name, model_name, confidence }
```

### 4.3 迭代优化
```
POST /sam/iterative-predict
Body: { points, labels, use_mask_input: true }
```

## 5. 验收标准

- [ ] 支持正/负样本点交互
- [ ] 支持批量同类标注
- [ ] 支持 YOLO+SAM 联动
- [ ] 标注管理功能完整
- [ ] 导出 YOLO 格式正确
