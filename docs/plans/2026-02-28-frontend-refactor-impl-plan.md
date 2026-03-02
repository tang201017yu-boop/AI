# AI Vision Platform 前端重构实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 YOLO- 计算机视觉平台从 Jinja2 模板重构为 React + TypeScript SPA，实现前后端完全分离

**Architecture:** 使用 Vite + React 18 + TypeScript 构建单页应用，通过 Axios 对接现有后端 API，采用 Zustand 进行状态管理，CSS Modules 实现样式隔离

**Tech Stack:** React 18, TypeScript, Vite, React Router v6, Zustand, Axios, CSS Modules

---

## 阶段 1: 项目初始化

### Task 1: 创建前端项目目录结构

**Files:**
- Create: `frontend-react/`
- Create: `frontend-react/public/`
- Create: `frontend-react/src/`
- Create: `frontend-react/src/assets/`
- Create: `frontend-react/src/components/`
- Create: `frontend-react/src/pages/`
- Create: `frontend-react/src/services/`
- Create: `frontend-react/src/stores/`
- Create: `frontend-react/src/hooks/`
- Create: `frontend-react/src/types/`
- Create: `frontend-react/src/styles/`
- Create: `frontend-react/src/utils/`

**Step 1: 创建目录**

```bash
mkdir -p frontend-react/{public,src/{assets,components/{common,layout,features},pages/{Home,Inference,Training,Models,Datasets,Annotation,Solutions,ImageBrowser,Augmentation,TrainingMonitor,Projects},services,stores,hooks,types,styles,utils}}
```

**Step 2: 验证**

```bash
ls -la frontend-react/src/
```

Expected: 显示所有子目录

---

### Task 2: 创建 package.json

**Files:**
- Create: `frontend-react/package.json`

**Step 1: 写入配置文件**

```json
{
  "name": "ai-vision-platform",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "zustand": "^4.4.7",
    "axios": "^1.6.2"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.3.3",
    "vite": "^5.0.8"
  }
}
```

---

### Task 3: 创建 Vite 配置

**Files:**
- Create: `frontend-react/vite.config.ts`

**Step 1: 写入配置**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
```

---

### Task 4: 创建 TypeScript 配置

**Files:**
- Create: `frontend-react/tsconfig.json`

**Step 1: 写入配置**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

**Step 2: 创建 tsconfig.node.json**

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

---

### Task 5: 创建 index.html 入口

**Files:**
- Create: `frontend-react/index.html`

**Step 1: 写入 HTML**

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI Vision Platform - 智能视觉平台</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Sans+3:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

---

## 阶段 2: 样式系统

### Task 6: 创建 CSS 变量定义

**Files:**
- Create: `frontend-react/src/styles/variables.css`

**Step 1: 写入 CSS 变量**

```css
:root {
  /* 主色 - 专业蓝 */
  --primary-50: #EBF5FF;
  --primary-100: #D6EBFF;
  --primary-200: #ADD6FF;
  --primary-300: #85C2FF;
  --primary-400: #5CADFF;
  --primary-500: #0066CC;
  --primary-600: #0052A3;
  --primary-700: #003D7A;
  --primary-800: #002952;
  --primary-900: #001429;

  /* accent - 活力橙 */
  --accent-50: #FFF0EB;
  --accent-100: #FFD9CC;
  --accent-200: #FFB399;
  --accent-300: #FF8D66;
  --accent-400: #FF6B35;
  --accent-500: #E55A28;
  --accent-600: #CC4A1C;
  --accent-700: #993A15;
  --accent-800: #66290D;
  --accent-900: #331907;

  /* 中性色 */
  --gray-50: #F8FAFC;
  --gray-100: #F1F5F9;
  --gray-200: #E2E8F0;
  --gray-300: #CBD5E1;
  --gray-400: #94A3B8;
  --gray-500: #64748B;
  --gray-600: #475569;
  --gray-700: #334155;
  --gray-800: #1E293B;
  --gray-900: #0F172A;

  /* 功能色 */
  --success: #10B981;
  --success-light: #D1FAE5;
  --warning: #F59E0B;
  --warning-light: #FEF3C7;
  --error: #EF4444;
  --error-light: #FEE2E2;
  --info: #3B82F6;
  --info-light: #DBEAFE;

  /* 语义色 */
  --text-primary: var(--gray-900);
  --text-secondary: var(--gray-500);
  --text-muted: var(--gray-400);
  --bg-primary: #FFFFFF;
  --bg-secondary: var(--gray-50);
  --bg-tertiary: var(--gray-100);
  --border-color: var(--gray-200);
  --border-light: var(--gray-100);

  /* 间距 */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.25rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-10: 2.5rem;
  --space-12: 3rem;

  /* 圆角 */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
  --radius-full: 9999px;

  /* 阴影 */
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
  --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);

  /* 过渡 */
  --transition-fast: 150ms ease;
  --transition-base: 200ms ease;
  --transition-slow: 300ms ease;

  /* 布局 */
  --max-width: 1400px;
  --sidebar-width: 260px;
  --navbar-height: 64px;
}
```

---

### Task 7: 创建全局样式

**Files:**
- Create: `frontend-react/src/styles/global.css`

**Step 1: 写入全局样式**

```css
@import './variables.css';

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  font-size: 16px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

body {
  font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--text-primary);
  background: var(--bg-secondary);
  line-height: 1.6;
}

h1, h2, h3, h4, h5, h6 {
  font-family: 'DM Sans', sans-serif;
  font-weight: 600;
  line-height: 1.3;
}

h1 { font-size: 2.5rem; }
h2 { font-size: 2rem; }
h3 { font-size: 1.5rem; }
h4 { font-size: 1.25rem; }
h5 { font-size: 1rem; }
h6 { font-size: 0.875rem; }

a {
  color: var(--primary-500);
  text-decoration: none;
  transition: color var(--transition-fast);
}

a:hover {
  color: var(--primary-600);
}

button {
  font-family: inherit;
  cursor: pointer;
}

img {
  max-width: 100%;
  height: auto;
}

/* 滚动条 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--gray-100);
}

::-webkit-scrollbar-thumb {
  background: var(--gray-300);
  border-radius: var(--radius-full);
}

::-webkit-scrollbar-thumb:hover {
  background: var(--gray-400);
}

/* 工具类 */
.container {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 0 var(--space-6);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
```

---

## 阶段 3: 类型定义

### Task 8: 创建 TypeScript 类型

**Files:**
- Create: `frontend-react/src/types/index.ts`

**Step 1: 写入类型定义**

```typescript
// 系统相关
export interface SystemInfo {
  total_models: number;
  total_datasets: number;
  gpu_available: boolean;
  gpu_info?: string;
  ultralytics_version: string;
}

// 数据集
export interface Dataset {
  id: string;
  name: string;
  path: string;
  image_count: number;
  created_at: string;
  updated_at: string;
}

export interface DatasetUploadResponse {
  success: boolean;
  dataset_id: string;
  message: string;
}

// 模型
export interface Model {
  id: string;
  name: string;
  type: string;
  path: string;
  created_at: string;
  metrics?: ModelMetrics;
}

export interface ModelMetrics {
  precision?: number;
  recall?: number;
  mAP50?: number;
  mAP50_95?: number;
}

// 训练
export interface TrainingConfig {
  model_type: string;
  dataset_id: string;
  epochs: number;
  batch_size: number;
  image_size: number;
  optimizer: string;
  learning_rate: number;
}

export interface TrainingStatus {
  status: 'idle' | 'running' | 'completed' | 'failed';
  current_epoch: number;
  total_epochs: number;
  loss?: number;
  metrics?: Record<string, number>;
}

// 推理
export interface InferenceRequest {
  model_id: string;
  image_path?: string;
  image_data?: string;
  conf_threshold?: number;
  iou_threshold?: number;
}

export interface InferenceResult {
  success: boolean;
  image_path: string;
  detections: Detection[];
  inference_time: number;
}

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // x1, y1, x2, y2
}

// 标注
export interface AnnotationProject {
  id: string;
  name: string;
  dataset_id: string;
  classes: string[];
  status: 'pending' | 'in_progress' | 'completed';
  progress: number;
}

// 通用
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}
```

---

## 阶段 4: API 服务层

### Task 9: 创建 Axios 实例和 API 服务

**Files:**
- Create: `frontend-react/src/services/api.ts`

**Step 1: 创建 Axios 实例**

```typescript
import axios, { AxiosInstance, AxiosError } from 'axios';
import type { ApiResponse } from '../types';

const api: AxiosInstance = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证 token
    return config;
  },
  (error) => Promise.reject(error)
);

// 响应拦截器
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiResponse<unknown>>) => {
    const message = error.response?.data?.message || error.message || '请求失败';
    console.error('API Error:', message);
    return Promise.reject(new Error(message));
  }
);

export default api;

// ============ 系统 API ============
export const systemApi = {
  getInfo: () => api.get<ApiResponse<SystemInfo>>('/system/info'),
  getHealth: () => api.get('/system/health'),
};

// ============ 数据集 API ============
export const datasetApi = {
  api.get<Api list: () =>Response<Dataset[]>>('/datasets'),
  get: (id: string) => api.get<ApiResponse<Dataset>>(`/datasets/${id}`),
  upload: (formData: FormData) =>
    api.post<ApiResponse<DatasetUploadResponse>>('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  delete: (id: string) => api.delete(`/datasets/${id}`),
};

// ============ 模型 API ============
export const modelApi = {
  list: () => api.get<ApiResponse<Model[]>>('/models'),
  get: (id: string) => api.get<ApiResponse<Model>>(`/models/${id}`),
  upload: (formData: FormData) =>
    api.post<ApiResponse<Model>>('/models/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  delete: (id: string) => api.delete(`/models/${id}`),
};

// ============ 训练 API ============
export const trainingApi = {
  start: (config: TrainingConfig) =>
    api.post<ApiResponse<{ task_id: string }>>('/training/start', config),
  status: (taskId: string) =>
    api.get<ApiResponse<TrainingStatus>>(`/training/status/${taskId}`),
  stop: (taskId: string) => api.post(`/training/stop/${taskId}`),
  getHistory: () => api.get('/training/history'),
};

// ============ 推理 API ============
export const inferenceApi = {
  image: (request: InferenceRequest) =>
    api.post<ApiResponse<InferenceResult>>('/inference/image', request),
  video: (request: InferenceRequest) =>
    api.post<ApiResponse<InferenceResult>>('/inference/video', request),
  batch: (requests: InferenceRequest[]) =>
    api.post<ApiResponse<InferenceResult[]>>('/inference/batch', { requests }),
};

// ============ 标注 API ============
export const annotationApi = {
  listProjects: () => api.get<ApiResponse<AnnotationProject[]>>('/annotation/projects'),
  getProject: (id: string) => api.get<ApiResponse<AnnotationProject>>(`/annotation/projects/${id}`),
  createProject: (data: Partial<AnnotationProject>) =>
    api.post<ApiResponse<AnnotationProject>>('/annotation/projects', data),
  getImages: (projectId: string) =>
    api.get(`/annotation/projects/${projectId}/images`),
  saveAnnotations: (projectId: string, imageId: string, annotations: unknown) =>
    api.post(`/annotation/projects/${projectId}/images/${imageId}/annotations`, { annotations }),
};
```

---

## 阶段 5: 状态管理

### Task 10: 创建 Zustand Store

**Files:**
- Create: `frontend-react/src/stores/useAppStore.ts`

**Step 1: 创建 Store**

```typescript
import { create } from 'zustand';
import type { SystemInfo, Dataset, Model, TrainingStatus } from '../types';

interface AppState {
  // 系统信息
  systemInfo: SystemInfo | null;
  setSystemInfo: (info: SystemInfo) => void;

  // 数据集
  datasets: Dataset[];
  setDatasets: (datasets: Dataset[]) => void;
  addDataset: (dataset: Dataset) => void;
  removeDataset: (id: string) => void;

  // 模型
  models: Model[];
  setModels: (models: Model[]) => void;
  addModel: (model: Model) => void;
  removeModel: (id: string) => void;

  // 训练状态
  trainingStatus: TrainingStatus | null;
  setTrainingStatus: (status: TrainingStatus | null) => void;

  // UI 状态
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  // 加载状态
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // 系统信息
  systemInfo: null,
  setSystemInfo: (info) => set({ systemInfo: info }),

  // 数据集
  datasets: [],
  setDatasets: (datasets) => set({ datasets }),
  addDataset: (dataset) => set((state) => ({ datasets: [...state.datasets, dataset] })),
  removeDataset: (id) => set((state) => ({
    datasets: state.datasets.filter((d) => d.id !== id)
  })),

  // 模型
  models: [],
  setModels: (models) => set({ models }),
  addModel: (model) => set((state) => ({ models: [...state.models, model] })),
  removeModel: (id) => set((state) => ({
    models: state.models.filter((m) => m.id !== id)
  })),

  // 训练状态
  trainingStatus: null,
  setTrainingStatus: (status) => set({ trainingStatus: status }),

  // UI 状态
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  // 加载状态
  loading: false,
  setLoading: (loading) => set({ loading }),
}));
```

---

## 阶段 6: 通用组件

### Task 11: 创建 Button 组件

**Files:**
- Create: `frontend-react/src/components/common/Button/Button.tsx`
- Create: `frontend-react/src/components/common/Button/Button.module.css`

**Step 1: 写入 Button CSS**

```css
.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-4);
  font-family: 'DM Sans', sans-serif;
  font-size: 0.9375rem;
  font-weight: 500;
  line-height: 1.5;
  border: none;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-fast);
  white-space: nowrap;
}

.button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* 变体 */
.primary {
  background: var(--primary-500);
  color: white;
}

.primary:hover:not(:disabled) {
  background: var(--primary-600);
}

.secondary {
  background: var(--gray-100);
  color: var(--gray-700);
  border: 1px solid var(--border-color);
}

.secondary:hover:not(:disabled) {
  background: var(--gray-200);
}

.accent {
  background: var(--accent-500);
  color: white;
}

.accent:hover:not(:disabled) {
  background: var(--accent-600);
}

.ghost {
  background: transparent;
  color: var(--primary-500);
}

.ghost:hover:not(:disabled) {
  background: var(--primary-50);
}

/* 尺寸 */
.sm {
  padding: var(--space-1) var(--space-3);
  font-size: 0.8125rem;
}

.md {
  padding: var(--space-2) var(--space-4);
}

.lg {
  padding: var(--space-3) var(--space-6);
  font-size: 1rem;
}

/* 图标 */
.iconOnly {
  padding: var(--space-2);
}

.iconOnly.sm { padding: var(--space-1); }
.iconOnly.lg { padding: var(--space-3); }
```

**Step 2: 写入 Button 组件**

```tsx
import React from 'react';
import styles from './Button.module.css';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'accent' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
  iconOnly?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  iconOnly = false,
  className = '',
  disabled,
  ...props
}) => {
  const classes = [
    styles.button,
    styles[variant],
    styles[size],
    iconOnly && styles.iconOnly,
    className,
  ].filter(Boolean).join(' ');

  return (
    <button className={classes} disabled={disabled || loading} {...props}>
      {loading ? (
        <span className={styles.spinner} />
      ) : (
        <>
          {icon}
          {!iconOnly && children}
        </>
      )}
    </button>
  );
};
```

**Step 3: 创建 index 导出**

```typescript
export { Button } from './Button';
```

---

### Task 12: 创建 Card 组件

**Files:**
- Create: `frontend-react/src/components/common/Card/Card.tsx`
- Create: `frontend-react/src/components/common/Card/Card.module.css`

**Step 1: 写入 Card CSS**

```css
.card {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  transition: all var(--transition-base);
}

.card:hover {
  border-color: var(--primary-200);
  box-shadow: var(--shadow-md);
}

.elevated {
  box-shadow: var(--shadow-md);
}

.elevated:hover {
  box-shadow: var(--shadow-lg);
}

.highlight {
  background: linear-gradient(135deg, var(--primary-50), var(--gray-50));
  border-color: var(--primary-200);
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--space-4);
  padding-bottom: var(--space-4);
  border-bottom: 1px solid var(--border-light);
}

.title {
  font-family: 'DM Sans', sans-serif;
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--text-primary);
}

.icon {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--primary-50);
  border-radius: var(--radius-md);
  font-size: 1.25rem;
}
```

**Step 2: 写入 Card 组件**

```tsx
import React from 'react';
import styles from './Card.module.css';

interface CardProps {
  children: React.ReactNode;
  variant?: 'default' | 'elevated' | 'highlight';
  className?: string;
}

interface CardHeaderProps {
  title: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({ children, variant = 'default', className = '' }) => {
  const classes = [styles.card, variant !== 'default' && styles[variant], className]
    .filter(Boolean)
    .join(' ');
  return <div className={classes}>{children}</div>;
};

export const CardHeader: React.FC<CardHeaderProps> = ({ title, icon, action }) => (
  <div className={styles.header}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
      {icon && <div className={styles.icon}>{icon}</div>}
      <h3 className={styles.title}>{title}</h3>
    </div>
    {action && <div>{action}</div>}
  </div>
);
```

---

### Task 13: 创建 Input 组件

**Files:**
- Create: `frontend-react/src/components/common/Input/Input.tsx`
- Create: `frontend-react/src/components/common/Input/Input.module.css`

**Step 1: 写入 Input CSS**

```css
.wrapper {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-primary);
}

.required::after {
  content: '*';
  color: var(--error);
  margin-left: var(--space-1);
}

.input {
  width: 100%;
  padding: var(--space-2) var(--space-3);
  font-size: 0.9375rem;
  color: var(--text-primary);
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  transition: all var(--transition-fast);
}

.input:focus {
  outline: none;
  border-color: var(--primary-500);
  box-shadow: 0 0 0 3px var(--primary-50);
}

.input::placeholder {
  color: var(--text-muted);
}

.input:disabled {
  background: var(--gray-50);
  cursor: not-allowed;
}

.error {
  border-color: var(--error);
}

.error:focus {
  box-shadow: 0 0 0 3px var(--error-light);
}

.errorText {
  font-size: 0.8125rem;
  color: var(--error);
}

.helper {
  font-size: 0.8125rem;
  color: var(--text-muted);
}
```

**Step 2: 写入 Input 组件**

```tsx
import React from 'react';
import styles from './Input.module.css';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helper?: string;
}

export const Input: React.FC<InputProps> = ({
  label,
  error,
  helper,
  required,
  className = '',
  ...props
}) => {
  return (
    <div className={styles.wrapper}>
      {label && (
        <label className={`${styles.label} ${required ? styles.required : ''}`}>
          {label}
        </label>
      )}
      <input
        className={`${styles.input} ${error ? styles.error : ''} ${className}`}
        required={required}
        {...props}
      />
      {error && <span className={styles.errorText}>{error}</span>}
      {!error && helper && <span className={styles.helper}>{helper}</span>}
    </div>
  );
};
```

---

### Task 14: 创建 StatCard 组件

**Files:**
- Create: `frontend-react/src/components/common/StatCard/StatCard.tsx`
- Create: `frontend-react/src/components/common/StatCard/StatCard.module.css`

**Step 1: 写入 StatCard CSS**

```css
.statCard {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all var(--transition-base);
}

.statCard::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--primary-500);
}

.statCard:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.value {
  font-family: 'DM Sans', sans-serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--primary-600);
  margin-bottom: var(--space-1);
}

.label {
  font-size: 0.875rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.accent::before {
  background: var(--accent-500);
}

.accent .value {
  color: var(--accent-600);
}

.success::before {
  background: var(--success);
}

.success .value {
  color: var(--success);
}
```

**Step 2: 写入 StatCard 组件**

```tsx
import React from 'react';
import styles from './StatCard.module.css';

interface StatCardProps {
  value: string | number;
  label: string;
  variant?: 'default' | 'accent' | 'success';
}

export const StatCard: React.FC<StatCardProps> = ({ value, label, variant = 'default' }) => {
  return (
    <div className={`${styles.statCard} ${variant !== 'default' ? styles[variant] : ''}`}>
      <div className={styles.value}>{value}</div>
      <div className={styles.label}>{label}</div>
    </div>
  );
};
```

---

### Task 15: 创建 Upload 组件

**Files:**
- Create: `frontend-react/src/components/features/Upload/Upload.tsx`
- Create: `frontend-react/src/components/features/Upload/Upload.module.css`

**Step 1: 写入 Upload CSS**

```css
.uploadArea {
  border: 2px dashed var(--border-color);
  border-radius: var(--radius-lg);
  padding: var(--space-10);
  text-align: center;
  cursor: pointer;
  transition: all var(--transition-base);
  background: var(--gray-50);
}

.uploadArea:hover,
.dragOver {
  border-color: var(--primary-500);
  background: var(--primary-50);
}

.icon {
  font-size: 3rem;
  margin-bottom: var(--space-4);
  color: var(--text-muted);
}

.text {
  font-size: 1rem;
  color: var(--text-primary);
  margin-bottom: var(--space-2);
}

.hint {
  font-size: 0.875rem;
  color: var(--text-muted);
}

.input {
  display: none;
}

.fileList {
  margin-top: var(--space-4);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.fileItem {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-2) var(--space-3);
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
}

.fileName {
  font-size: 0.875rem;
  color: var(--text-primary);
}

.fileSize {
  font-size: 0.75rem;
  color: var(--text-muted);
}
```

**Step 2: 写入 Upload 组件**

```tsx
import React, { useCallback } from 'react';
import styles from './Upload.module.css';

interface UploadProps {
  accept?: string;
  multiple?: boolean;
  onFilesSelected: (files: File[]) => void;
  hint?: string;
}

export const Upload: React.FC<UploadProps> = ({
  accept,
  multiple = false,
  onFilesSelected,
  hint = '点击或拖拽文件到此处上传',
}) => {
  const [dragOver, setDragOver] = React.useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      onFilesSelected(Array.from(e.target.files));
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files) {
      onFilesSelected(Array.from(e.dataTransfer.files));
    }
  }, [onFilesSelected]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  return (
    <div
      className={`${styles.uploadArea} ${dragOver ? styles.dragOver : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={() => document.getElementById('file-input')?.click()}
    >
      <input
        type="file"
        id="file-input"
        className={styles.input}
        accept={accept}
        multiple={multiple}
        onChange={handleChange}
      />
      <div className={styles.icon}>📁</div>
      <div className={styles.text}>{hint}</div>
      {accept && <div className={styles.hint}>支持格式: {accept}</div>}
    </div>
  );
};
```

---

### Task 16: 导出所有通用组件

**Files:**
- Create: `frontend-react/src/components/common/index.ts`

```typescript
export { Button } from './Button';
export { Card, CardHeader } from './Card';
export { Input } from './Input';
export { StatCard } from './StatCard';
```

**Files:**
- Create: `frontend-react/src/components/features/index.ts`

```typescript
export { Upload } from './Upload';
```

---

## 阶段 7: 布局组件

### Task 17: 创建 Navbar 组件

**Files:**
- Create: `frontend-react/src/components/layout/Navbar/Navbar.tsx`
- Create: `frontend-react/src/components/layout/Navbar/Navbar.module.css`

**Step 1: 写入 Navbar CSS**

```css
.navbar {
  position: sticky;
  top: 0;
  z-index: 100;
  height: var(--navbar-height);
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-color);
  box-shadow: var(--shadow-sm);
}

.container {
  max-width: var(--max-width);
  height: 100%;
  margin: 0 auto;
  padding: 0 var(--space-6);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logoSection {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.logo {
  height: 36px;
}

.logoText {
  font-family: 'DM Sans', sans-serif;
  font-size: 1.375rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--primary-500), var(--primary-700));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.nav {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}

.navLink {
  padding: var(--space-2) var(--space-3);
  font-size: 0.9375rem;
  font-weight: 500;
  color: var(--text-secondary);
  border-radius: var(--radius-md);
  transition: all var(--transition-fast);
}

.navLink:hover {
  color: var(--primary-500);
  background: var(--primary-50);
}

.navLinkActive {
  color: var(--primary-600);
  background: var(--primary-50);
}
```

**Step 2: 写入 Navbar 组件**

```tsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import styles from './Navbar.module.css';

const navItems = [
  { path: '/', label: '首页' },
  { path: '/inference', label: '模型推理' },
  { path: '/training', label: '模型训练' },
  { path: '/solutions', label: '智能方案' },
  { path: '/models', label: '模型管理' },
  { path: '/datasets', label: '数据集' },
  { path: '/annotation', label: '智能标注' },
  { path: '/augmentation', label: '数据增强' },
];

export const Navbar: React.FC = () => {
  return (
    <nav className={styles.navbar}>
      <div className={styles.container}>
        <div className={styles.logoSection}>
          <img src="/logo.svg" alt="Logo" className={styles.logo} />
          <span className={styles.logoText}>AI Vision</span>
        </div>
        <div className={styles.nav}>
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
};
```

---

### Task 18: 创建 Layout 组件

**Files:**
- Create: `frontend-react/src/components/layout/Layout/Layout.tsx`
- Create: `frontend-react/src/components/layout/Layout/Layout.module.css`

**Step 1: 写入 Layout CSS**

```css
.layout {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.main {
  flex: 1;
  padding: var(--space-6) 0;
}

.footer {
  background: var(--bg-primary);
  border-top: 1px solid var(--border-color);
  padding: var(--space-6) 0;
  text-align: center;
}

.footerContent {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 0 var(--space-6);
}

.footerText {
  font-size: 0.875rem;
  color: var(--text-muted);
}
```

**Step 2: 写入 Layout 组件**

```tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import { Navbar } from '../Navbar/Navbar';
import styles from './Layout.module.css';

export const Layout: React.FC = () => {
  return (
    <div className={styles.layout}>
      <Navbar />
      <main className={styles.main}>
        <div className="container">
          <Outlet />
        </div>
      </main>
      <footer className={styles.footer}>
        <div className={styles.footerContent}>
          <p className={styles.footerText}>
            © 2024 AI Vision Platform. 基于 Ultralytics YOLO 与 Supervision.
          </p>
        </div>
      </footer>
    </div>
  );
};
```

**Step 3: 创建导出**

```typescript
export { Layout } from './Layout';
export { Navbar } from './Navbar';
```

---

## 阶段 8: 页面组件

### Task 19: 创建首页 Home

**Files:**
- Create: `frontend-react/src/pages/Home/Home.tsx`
- Create: `frontend-react/src/pages/Home/Home.module.css`

**Step 1: 写入 Home CSS**

```css
.hero {
  text-align: center;
  padding: var(--space-12) 0;
  position: relative;
}

.heroTitle {
  font-family: 'DM Sans', sans-serif;
  font-size: 3rem;
  font-weight: 800;
  color: var(--text-primary);
  margin-bottom: var(--space-4);
}

.heroTitleAccent {
  background: linear-gradient(135deg, var(--primary-500), var(--accent-500));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.heroSubtitle {
  font-size: 1.25rem;
  color: var(--text-secondary);
  max-width: 600px;
  margin: 0 auto var(--space-8);
  line-height: 1.7;
}

.heroButtons {
  display: flex;
  gap: var(--space-4);
  justify-content: center;
}

.section {
  margin-bottom: var(--space-8);
}

.grid4 {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--space-4);
}

.grid3 {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-6);
}

.grid2 {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: var(--space-6);
}

@media (max-width: 1024px) {
  .grid4 { grid-template-columns: repeat(2, 1fr); }
  .grid3 { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 768px) {
  .grid4, .grid3, .grid2 { grid-template-columns: 1fr; }
  .heroTitle { font-size: 2rem; }
}

.workflowStep {
  position: relative;
  padding-left: var(--space-10);
  padding-bottom: var(--space-6);
}

.workflowStep::before {
  content: attr(data-step);
  position: absolute;
  left: 0;
  top: 0;
  width: 2.5rem;
  height: 2.5rem;
  background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  color: white;
  font-size: 1rem;
}

.workflowStep::after {
  content: '';
  position: absolute;
  left: 1.125rem;
  top: 3rem;
  width: 2px;
  height: calc(100% - 1.5rem);
  background: var(--border-color);
}

.workflowStep:last-child::after {
  display: none;
}

.workflowTitle {
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: var(--space-2);
  color: var(--text-primary);
}

.workflowDesc {
  font-size: 0.9375rem;
  color: var(--text-secondary);
  line-height: 1.7;
}

.solutionCard {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  text-align: center;
  transition: all var(--transition-base);
}

.solutionCard:hover {
  border-color: var(--primary-200);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.solutionIcon {
  font-size: 2rem;
  margin-bottom: var(--space-3);
}

.solutionTitle {
  font-weight: 600;
  margin-bottom: var(--space-1);
  color: var(--text-primary);
}

.solutionDesc {
  font-size: 0.875rem;
  color: var(--text-secondary);
}
```

**Step 2: 写入 Home 组件**

```tsx
import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Card, CardHeader, Button, StatCard } from '../../components/common';
import { systemApi } from '../../services/api';
import type { SystemInfo } from '../../types';
import styles from './Home.module.css';

const solutions = [
  { icon: '📊', title: '对象计数', desc: '统计区域对象数量' },
  { icon: '🔥', title: '热图分析', desc: '可视化检测密度' },
  { icon: '🚗', title: '速度估算', desc: '计算移动对象速度' },
  { icon: '📏', title: '距离计算', desc: '测量对象间距离' },
  { icon: '🔒', title: '隐私保护', desc: '对象模糊处理' },
  { icon: '✂️', title: '对象裁剪', desc: '自动提取检测对象' },
  { icon: '👥', title: '队列管理', desc: '监控队列长度' },
  { icon: '🏮', title: '虚拟围栏', desc: '区域入侵检测' },
];

export const Home: React.FC = () => {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  useEffect(() => {
    systemApi.getInfo()
      .then((res) => setSystemInfo(res.data.data || null))
      .catch(console.error);
  }, []);

  return (
    <div>
      {/* Hero Section */}
      <div className={styles.hero}>
        <h1 className={styles.heroTitle}>
          AI <span className={styles.heroTitleAccent}>Vision</span> Platform
        </h1>
        <p className={styles.heroSubtitle}>
          基于 Ultralytics YOLO 与 Supervision 的新一代智能视觉平台
          <br />
          融合 AI 驱动的数据标注、模型训练与部署解决方案
        </p>
        <div className={styles.heroButtons}>
          <Link to="/annotation">
            <Button variant="primary" size="lg">⭐ 智能标注</Button>
          </Link>
          <Link to="/training">
            <Button variant="secondary" size="lg">🚀 开始训练</Button>
          </Link>
        </div>
      </div>

      {/* System Stats */}
      <div className={styles.section}>
        <Card>
          <CardHeader title="系统状态" icon="📈" />
          <div className={styles.grid4}>
            <StatCard value={systemInfo?.total_models || '-'} label="模型数量" />
            <StatCard value={systemInfo?.total_datasets || '-'} label="数据集数量" variant="accent" />
            <StatCard value={systemInfo?.gpu_available ? 'GPU' : 'CPU'} label="计算资源" variant="success" />
            <StatCard value={systemInfo?.ultralytics_version || 'YOLO'} label="YOLO 版本" />
          </div>
        </Card>
      </div>

      {/* AI Workflow */}
      <div className={styles.section}>
        <Card>
          <CardHeader title="AI 工作流" icon="⚙️" />
          <div className={styles.grid3}>
            <div className={styles.workflowStep} data-step="1">
              <h3 className={styles.workflowTitle}>
                <span style={{ color: 'var(--primary-500)' }}>01.</span> Supervision 智能标注
              </h3>
              <p className={styles.workflowDesc}>
                集成 Supervision 智能标注系统，支持 SAM 自动分割、YOLO 预标注，一键生成高质量训练数据
              </p>
              <Link to="/annotation">
                <Button variant="primary" size="sm" style={{ marginTop: 'var(--space-4)' }}>
                  开始标注 →
                </Button>
              </Link>
            </div>
            <div className={styles.workflowStep} data-step="2">
              <h3 className={styles.workflowTitle}>
                <span style={{ color: 'var(--primary-500)' }}>02.</span> 模型训练
              </h3>
              <p className={styles.workflowDesc}>
                基于 Ultralytics YOLO 框架，支持多模型训练、超参数调优、实时监控训练进度
              </p>
              <Link to="/training">
                <Button variant="secondary" size="sm" style={{ marginTop: 'var(--space-4)' }}>
                  开始训练 →
                </Button>
              </Link>
            </div>
            <div className={styles.workflowStep} data-step="3">
              <h3 className={styles.workflowTitle}>
                <span style={{ color: 'var(--primary-500)' }}>03.</span> 智能部署
              </h3>
              <p className={styles.workflowDesc}>
                一键部署训练模型，提供 RESTful API，支持 TensorRT 加速、批量推理
              </p>
              <Link to="/inference">
                <Button variant="secondary" size="sm" style={{ marginTop: 'var(--space-4)' }}>
                  开始推理 →
                </Button>
              </Link>
            </div>
          </div>
        </Card>
      </div>

      {/* Solutions */}
      <div className={styles.section}>
        <Card>
          <CardHeader title="Ultralytics 智能解决方案" icon="💡" />
          <div className={styles.grid4}>
            {solutions.map((item, index) => (
              <div key={index} className={styles.solutionCard}>
                <div className={styles.solutionIcon}>{item.icon}</div>
                <div className={styles.solutionTitle}>{item.title}</div>
                <div className={styles.solutionDesc}>{item.desc}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
};
```

---

### Task 20: 创建推理页面

**Files:**
- Create: `frontend-react/src/pages/Inference/Inference.tsx`
- Create: `frontend-react/src/pages/Inference/Inference.module.css`

**Step 1: 写入 Inference CSS**

```css
.container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-6);
}

@media (max-width: 1024px) {
  .container { grid-template-columns: 1fr; }
}

.panel {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
}

.panelTitle {
  font-family: 'DM Sans', sans-serif;
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: var(--space-4);
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.form {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.select {
  width: 100%;
  padding: var(--space-2) var(--space-3);
  font-size: 0.9375rem;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  background: var(--bg-primary);
  cursor: pointer;
}

.previewArea {
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--gray-50);
  border: 1px dashed var(--border-color);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.previewImage {
  max-width: 100%;
  max-height: 400px;
  object-fit: contain;
}

.resultOverlay {
  position: absolute;
  top: var(--space-2);
  right: var(--space-2);
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
}

.detectionList {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  margin-top: var(--space-4);
}

.detectionItem {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-2) var(--space-3);
  background: var(--gray-50);
  border-radius: var(--radius-md);
  font-size: 0.875rem;
}

.detectionClass {
  font-weight: 500;
}

.detectionConf {
  color: var(--primary-600);
  font-weight: 600;
}
```

**Step 2: 写入 Inference 组件**

```tsx
import React, { useState } from 'react';
import { Card, Button, Input } from '../../components/common';
import { Upload } from '../../components/features';
import { modelApi, inferenceApi } from '../../services/api';
import styles from './Inference.module.css';

export const Inference: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [confThreshold, setConfThreshold] = useState(0.25);

  const handleFileSelect = (files: File[]) => {
    if (files[0]) {
      setImageFile(files[0]);
      setImagePreview(URL.createObjectURL(files[0]));
      setResult(null);
    }
  };

  const handleInference = async () => {
    if (!selectedModel || !imageFile) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('model_id', selectedModel);
      formData.append('conf_threshold', confThreshold.toString());
      const res = await inferenceApi.image({
        model_id: selectedModel,
        image_path: imageFile.name,
        conf_threshold: confThreshold,
      });
      setResult(res.data.data);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.panel}>
        <h2 className={styles.panelTitle}>📤 上传图片</h2>
        <Upload
          accept="image/*"
          onFilesSelected={handleFileSelect}
          hint="点击或拖拽图片到此处"
        />
        {imagePreview && (
          <div className={styles.previewArea} style={{ marginTop: 'var(--space-4)' }}>
            <img src={imagePreview} alt="Preview" className={styles.previewImage} />
          </div>
        )}
      </div>

      <div className={styles.panel}>
        <h2 className={styles.panelTitle}>⚙️ 推理配置</h2>
        <div className={styles.form}>
          <div>
            <label className={styles.label}>选择模型</label>
            <select
              className={styles.select}
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="">请选择模型</option>
              <option value="yolo11n">YOLO11n</option>
              <option value="yolo11s">YOLO11s</option>
              <option value="yolo11m">YOLO11m</option>
            </select>
          </div>
          <Input
            type="number"
            label="置信度阈值"
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
            min={0}
            max={1}
            step={0.05}
          />
          <Button
            variant="primary"
            onClick={handleInference}
            loading={loading}
            disabled={!selectedModel || !imageFile}
          >
            开始推理
          </Button>
        </div>

        {result && (
          <div className={styles.detectionList}>
            <h4>检测结果 ({result.detections?.length || 0} 个对象)</h4>
            {result.detections?.map((det: any, idx: number) => (
              <div key={idx} className={styles.detectionItem}>
                <span className={styles.detectionClass}>{det.class_name}</span>
                <span className={styles.detectionConf}>
                  {(det.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
```

---

### Task 21: 创建数据集页面

**Files:**
- Create: `frontend-react/src/pages/Datasets/Datasets.tsx`
- Create: `frontend-react/src/pages/Datasets/Datasets.module.css`

**Step 1: 写入 Datasets CSS**

```css
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-6);
}

.title {
  font-family: 'DM Sans', sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
}

.toolbar {
  display: flex;
  gap: var(--space-3);
}

.search {
  width: 300px;
}

.table {
  width: 100%;
  border-collapse: collapse;
}

.table th,
.table td {
  padding: var(--space-3) var(--space-4);
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

.table th {
  font-weight: 600;
  color: var(--text-secondary);
  font-size: 0.8125rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  background: var(--gray-50);
}

.table tr:hover {
  background: var(--gray-50);
}

.tableName {
  font-weight: 500;
  color: var(--text-primary);
}

.tableMeta {
  font-size: 0.8125rem;
  color: var(--text-muted);
}

.badge {
  display: inline-block;
  padding: var(--space-1) var(--space-2);
  font-size: 0.75rem;
  font-weight: 500;
  border-radius: var(--radius-full);
  background: var(--primary-50);
  color: var(--primary-600);
}

.empty {
  text-align: center;
  padding: var(--space-12);
  color: var(--text-muted);
}

.emptyIcon {
  font-size: 4rem;
  margin-bottom: var(--space-4);
}
```

**Step 2: 写入 Datasets 组件**

```tsx
import React, { useEffect, useState } from 'react';
import { Card, Button, Input } from '../../components/common';
import { Upload } from '../../components/features';
import { datasetApi } from '../../services/api';
import type { Dataset } from '../../types';
import styles from './Datasets.module.css';

export const Datasets: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const res = await datasetApi.list();
      setDatasets(res.data.data || []);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (files: File[]) => {
    if (!files[0]) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', files[0]);
      await datasetApi.upload(formData);
      await loadDatasets();
    } catch (error) {
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <div className={styles.header}>
        <h1 className={styles.title}>数据集管理</h1>
        <div className={styles.toolbar}>
          <Input placeholder="搜索数据集..." className={styles.search} />
          <Upload
            accept=".zip,.tar.gz"
            onFilesSelected={handleUpload}
            hint=""
          />
        </div>
      </div>

      <Card>
        {loading ? (
          <div className={styles.empty}>加载中...</div>
        ) : datasets.length === 0 ? (
          <div className={styles.empty}>
            <div className={styles.emptyIcon}>📁</div>
            <p>暂无数据集</p>
            <p>上传数据集开始使用</p>
          </div>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>名称</th>
                <th>图片数量</th>
                <th>创建时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((dataset) => (
                <tr key={dataset.id}>
                  <td>
                    <div className={styles.tableName}>{dataset.name}</div>
                  </td>
                  <td>{dataset.image_count}</td>
                  <td className={styles.tableMeta}>
                    {new Date(dataset.created_at).toLocaleDateString()}
                  </td>
                  <td>
                    <Button variant="ghost" size="sm">查看</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
};
```

---

### Task 22: 创建其他页面

**Files:**
- Create: `frontend-react/src/pages/Training/Training.tsx`
- Create: `frontend-react/src/pages/Models/Models.tsx`
- Create: `frontend-react/src/pages/Annotation/Annotation.tsx`
- Create: `frontend-react/src/pages/Solutions/Solutions.tsx`
- Create: `frontend-react/src/pages/Augmentation/Augmentation.tsx`
- Create: `frontend-react/src/pages/ImageBrowser/ImageBrowser.tsx`
- Create: `frontend-react/src/pages/TrainingMonitor/TrainingMonitor.tsx`
- Create: `frontend-react/src/pages/Projects/Projects.tsx`

**每个页面模板:**

```tsx
import React from 'react';
import { Card, CardHeader, Button } from '../../components/common';

export const [PageName]: React.FC = () => {
  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans' }}>
        [页面标题]
      </h1>
      <Card>
        <CardHeader icon="📌" title="功能开发中" />
        <p style={{ color: 'var(--text-secondary)' }}>
          该页面正在积极开发中，敬请期待...
        </p>
      </Card>
    </div>
  );
};
```

---

## 阶段 9: 应用入口

### Task 23: 创建主应用文件

**Files:**
- Create: `frontend-react/src/App.tsx`
- Create: `frontend-react/src/main.tsx`

**Step 1: 写入 main.tsx**

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { App } from './App';
import './styles/global.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
```

**Step 2: 写入 App.tsx**

```tsx
import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout';
import { Home } from './pages/Home/Home';
import { Inference } from './pages/Inference/Inference';
import { Training } from './pages/Training/Training';
import { Models } from './pages/Models/Models';
import { Datasets } from './pages/Datasets/Datasets';
import { Annotation } from './pages/Annotation/Annotation';
import { Solutions } from './pages/Solutions/Solutions';
import { Augmentation } from './pages/Augmentation/Augmentation';
import { ImageBrowser } from './pages/ImageBrowser/ImageBrowser';
import { TrainingMonitor } from './pages/TrainingMonitor/TrainingMonitor';
import { Projects } from './pages/Projects/Projects';

export const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="inference" element={<Inference />} />
        <Route path="training" element={<Training />} />
        <Route path="models" element={<Models />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="annotation" element={<Annotation />} />
        <Route path="solutions" element={<Solutions />} />
        <Route path="augmentation" element={<Augmentation />} />
        <Route path="image-browser" element={<ImageBrowser />} />
        <Route path="training-monitor" element={<TrainingMonitor />} />
        <Route path="projects" element={<Projects />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
};
```

---

## 阶段 10: 静态资源

### Task 24: 创建 Logo

**Files:**
- Create: `frontend-react/public/favicon.svg`
- Create: `frontend-react/public/logo.svg`

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0066CC"/>
      <stop offset="100%" style="stop-color:#FF6B35"/>
    </linearGradient>
  </defs>
  <rect width="100" height="100" rx="20" fill="url(#grad)"/>
  <text x="50" y="65" font-family="Arial" font-size="40" font-weight="bold" fill="white" text-anchor="middle">Y</text>
</svg>
```

---

## 阶段 11: 开发脚本

### Task 25: 创建启动脚本

**Files:**
- Create: `frontend-react/start.sh`

```bash
#!/bin/bash
cd "$(dirname "$0")"
npm install
npm run dev
```

---

## 执行顺序

1. **阶段 1-5**: 项目基础架构 (初始化、样式、类型、API、Store)
2. **阶段 6**: 通用组件 (Button, Card, Input, StatCard, Upload)
3. **阶段 7**: 布局组件 (Navbar, Layout)
4. **阶段 8**: 页面组件 (Home, Inference, Datasets + 8个占位页面)
5. **阶段 9**: 应用入口 (App.tsx, main.tsx)
6. **阶段 10-11**: 静态资源和启动脚本

---

## 验证步骤

每个页面完成后:
1. 运行 `npm run dev` 启动开发服务器
2. 访问 http://localhost:3000
3. 验证页面渲染和导航
4. 验证 API 请求正常
5. 检查控制台无错误
