# AI Vision Platform 前端重构设计方案

**日期**: 2026-02-28
**项目**: YOLO- 计算机视觉平台前后端分离重构
**状态**: 已批准

## 1. 项目概述

将现有的 Flask/Jinja2 模板前端重构为 React + TypeScript SPA，实现前后端完全分离。

### 目标
- 现代化前端架构体验和可维护，提升开发性
- 明亮企业风设计风格，专业且易于使用
- 复用现有后端 API，无需修改后端代码

## 2. 技术架构

| 层级 | 技术选型 | 理由 |
|------|----------|------|
| 构建工具 | Vite | 快速开发体验，热更新 |
| 框架 | React 18 | 成熟生态，组件化 |
| 语言 | TypeScript | 类型安全，减少运行时错误 |
| 路由 | React Router v6 | 官方推荐路由方案 |
| 状态 | Zustand | 轻量级，比 Redux 简洁 |
| HTTP | Axios | 请求拦截，统一错误处理 |
| 样式 | CSS Modules | 作用域隔离，防止样式冲突 |

## 3. 前端项目结构

```
frontend/
├── public/
│   └── favicon.svg
├── src/
│   ├── assets/              # 静态资源
│   │   └── logo.svg
│   ├── components/          # 通用组件
│   │   ├── common/         # 基础组件 (Button, Card, Input...)
│   │   ├── layout/         # 布局组件 (Navbar, Sidebar, Footer)
│   │   └── features/      # 功能组件 (Upload, Preview...)
│   ├── pages/              # 页面组件
│   │   ├── Home/          # 首页仪表盘
│   │   ├── Inference/      # 模型推理
│   │   ├── Training/       # 模型训练
│   │   ├── Models/         # 模型管理
│   │   ├── Datasets/       # 数据集管理
│   │   ├── Annotation/     # 智能标注
│   │   ├── Solutions/      # 智能方案
│   │   ├── ImageBrowser/   # 图像浏览器
│   │   ├── Augmentation/   # 数据增强
│   │   ├── TrainingMonitor/# 训练监控
│   │   └── Projects/        # 项目管理
│   ├── services/           # API 服务层
│   │   └── api.ts          # Axios 实例 + 接口封装
│   ├── stores/             # Zustand 状态管理
│   │   └── useAppStore.ts
│   ├── hooks/              # 自定义 Hooks
│   ├── types/              # TypeScript 类型定义
│   │   └── index.ts
│   ├── styles/             # 全局样式
│   │   ├── variables.css   # CSS 变量
│   │   └── global.css      # 全局重置
│   ├── utils/              # 工具函数
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── .env.example
```

## 4. 设计规范 - 明亮企业风

### 颜色系统

```css
:root {
  /* 主色 - 专业蓝 */
  --primary-50: #EBF5FF;
  --primary-100: #D6EBFF;
  --primary-500: #0066CC;
  --primary-600: #0052A3;
  --primary-700: #003D7A;

  /*  accent - 活力橙 */
  --accent-500: #FF6B35;
  --accent-600: #E55A28;

  /* 中性色 */
  --gray-50: #F8FAFC;
  --gray-100: #F1F5F9;
  --gray-200: #E2E8F0;
  --gray-300: #CBD5E1;
  --gray-500: #64748B;
  --gray-700: #334155;
  --gray-900: #0F172A;

  /* 功能色 */
  --success: #10B981;
  --warning: #F59E0B;
  --error: #EF4444;
  --info: #3B82F6;
}
```

### 字体

- **标题**: "DM Sans", sans-serif (Google Fonts)
- **正文**: "Source Sans 3", sans-serif (Google Fonts)
- **代码**: "JetBrains Mono", monospace

### 布局

- 容器最大宽度: 1400px
- 侧边栏宽度: 260px
- 卡片圆角: 12px
- 间距基准: 4px (0.25rem)

## 5. 页面清单

| 页面 | 路由 | 功能 |
|------|------|------|
| 首页仪表盘 | `/` | 系统概览、统计卡片、工作流引导 |
| 模型推理 | `/inference` | 图片/视频上传、推理配置、结果展示 |
| 模型训练 | `/training` | 训练参数配置、开始训练 |
| 模型管理 | `/models` | 模型列表、详情、删除 |
| 数据集管理 | `/datasets` | 数据集上传、查看、导出 |
| 智能标注 | `/annotation` | SAM分割、YOLO预标注 |
| 智能方案 | `/solutions` | Ultralytics Solutions 演示 |
| 图像浏览器 | `/image-browser` | 图片查看、缩放 |
| 数据增强 | `/augmentation` | 增强配置、预览 |
| 训练监控 | `/training-monitor` | 实时训练指标 |
| 项目管理 | `/projects` | 项目列表、创建 |

## 6. API 对接

### 对接策略

1. **创建 Axios 实例** - 基础配置、超时、错误处理
2. **接口封装** - 每个模块对应一个服务文件
3. **类型定义** - 根据后端 Schema 定义 TypeScript 类型

### 基础接口 (来自现有 API)

- `GET /api/v1/system/info` - 系统信息
- `GET /api/v1/datasets` - 数据集列表
- `GET /api/v1/models` - 模型列表
- `POST /api/v1/inference` - 推理接口
- `POST /api/v1/training/start` - 开始训练

## 7. 实现计划

详见 `docs/plans/2026-02-28-frontend-impl-plan.md`

## 8. 验收标准

- [ ] 所有 11 个页面可正常访问
- [ ] API 请求正确发送并接收响应
- [ ] 响应式布局适配移动端
- [ ] 页面切换无白屏
- [ ] 样式与设计规范一致
