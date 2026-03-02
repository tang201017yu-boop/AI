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
