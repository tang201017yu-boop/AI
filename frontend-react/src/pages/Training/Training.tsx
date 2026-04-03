import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Card, CardHeader, Button, Input, StatCard } from '../../components/common';
import { datasetApi, trainingApi, modelApi, projectApi } from '../../services/api';
import type { Dataset, TrainingConfig, Model, Project } from '../../types';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

// 工作流程步骤
type Step = 'project' | 'configure' | 'train' | 'monitor' | 'export';

interface StepInfo {
  key: Step;
  title: string;
  icon: string;
}

const STEPS: StepInfo[] = [
  { key: 'project', title: '项目', icon: '📁' },
  { key: 'configure', title: '配置', icon: '⚙️' },
  { key: 'train', title: '训练', icon: '🚀' },
  { key: 'monitor', title: '监控', icon: '📈' },
  { key: 'export', title: '导出', icon: '📦' },
];

interface PretrainedModel {
  name: string;
  display: string;
  installed: boolean;
  path?: string;
}

interface ChartData {
  epochs: number[];
  losses: {
    box_loss: number[];
    cls_loss: number[];
    dfl_loss: number[];
  };
  metrics_history: Array<{
    epoch: number;
    'metrics/mAP50(B)': number;
    'metrics/mAP50-95(B)': number;
    'metrics/precision(B)': number;
    'metrics/recall(B)': number;
  }>;
  best_metrics: {
    mAP50: number;
    mAP50_95: number;
    precision: number;
    recall: number;
  };
}

interface SystemInfo {
  gpu_available: boolean;
  gpu_name?: string;
  gpu_memory_total?: number;
  gpu_memory_used?: number;
  gpu_utilization?: number;
  cpu_percent?: number;
  memory_total?: number;
  memory_used?: number;
  memory_percent?: number;
}

type MonitorTab = 'overview' | 'charts' | 'console' | 'system';
type ChartTabType = 'loss' | 'metrics';

export const Training: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [currentStep, setCurrentStep] = useState<Step>('project');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [pretrainedModels, setPretrainedModels] = useState<PretrainedModel[]>([]);
  const [trainedModels, setTrainedModels] = useState<Model[]>([]);

  // Project 阶段
  const [projectName, setProjectName] = useState('');
  const [projectDescription, setProjectDescription] = useState('');
  const [taskType, setTaskType] = useState<'detect' | 'segment' | 'classify' | 'pose' | 'obb'>('detect');
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);

  // Configure 阶段
  const [selectedDataset, setSelectedDataset] = useState('');
  const [modelSource, setModelSource] = useState<'pretrained' | 'trained'>('pretrained');
  const [selectedPretrainedModel, setSelectedPretrainedModel] = useState('');
  const [selectedTrainedModel, setSelectedTrainedModel] = useState('');
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [imageSize, setImageSize] = useState(640);
  const [learningRate, setLearningRate] = useState(0.01);

  // Train 阶段
  const [training, setTraining] = useState(false);
  const [taskId, setTaskId] = useState('');
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

  // 训练历史
  const [history, setHistory] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Monitor 阶段
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [monitorTab, setMonitorTab] = useState<MonitorTab>('overview');
  const [chartTab, setChartTab] = useState<ChartTabType>('loss');
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [monitorMetrics, setMonitorMetrics] = useState<{loss?: number; mAP50?: number; precision?: number; recall?: number; status?: string}>({});
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Export 阶段
  const [exportLoading, setExportLoading] = useState(false);
  const [exportMessage, setExportMessage] = useState('');

  const handleExport = async (format: string) => {
    if (!taskId && !currentTaskId) {
      setExportMessage('请先完成训练');
      return;
    }
    const id = taskId || currentTaskId || '';
    setExportLoading(true);
    setExportMessage('');
    try {
      const res = await trainingApi.exportExperimentModel(id, format);
      const result = res.data;
      if (result?.success && result?.data?.export_path) {
        const filePath = result.data.export_path;
        const filename = filePath.split('/').pop() || `${id}.${format}`;
        // 根据路径选择静态服务前缀
        const prefix = filePath.includes('YOLO-') ? '/yolo-models' : '/models';
        // 提取相对于挂载目录的路径
        const relativePath = filePath.includes('YOLO-')
          ? filePath.replace('/root/wuyu/YOLO-/data/models/', '')
          : filename;
        const link = document.createElement('a');
        link.href = `${prefix}/${relativePath}`;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        setExportMessage(`${format.toUpperCase()} 导出成功！`);
      } else {
        setExportMessage(result?.data?.message || result?.message || '导出失败');
      }
    } catch (e: any) {
      setExportMessage(e?.response?.data?.detail || e?.message || '导出请求失败');
    } finally {
      setExportLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
    loadPretrainedModels();
    loadTrainedModels();
    loadProjects();
    loadHistory();
  }, []);

  const loadProjects = async () => {
    try {
      const res = await projectApi.list();
      // 处理多种响应格式
      const responseData = res.data as any;
      const projects = responseData?.projects || responseData?.data?.projects || [];
      setProjects(projects);
    } catch (error) {
      console.error('Failed to load projects:', error);
    }
  };

  // 从 URL 参数恢复任务状态
  useEffect(() => {
    const taskIdFromUrl = searchParams.get('taskId');
    if (taskIdFromUrl && !currentTaskId) {
      setCurrentTaskId(taskIdFromUrl);
      setTaskId(taskIdFromUrl);
      setCurrentStep('monitor');
      setTraining(true);
    }
  }, [searchParams]);

  // 系统信息轮询（仅在 system tab 激活时）
  useEffect(() => {
    if (monitorTab !== 'system') return;
    const loadSys = async () => {
      try {
        const res = await trainingApi.getSystemInfo();
        if (res.data?.data) setSystemInfo(res.data.data);
      } catch (e) { console.error(e); }
    };
    loadSys();
    const interval = setInterval(loadSys, 3000);
    return () => clearInterval(interval);
  }, [monitorTab]);

  // 日志自动滚动
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // 定期检查训练状态
  useEffect(() => {
    if (!currentTaskId) return;

    let lastEpoch = -1;
    let errorCount = 0;
    let stopped = false;

    const stop = (clearId: ReturnType<typeof setInterval>) => {
      if (!stopped) {
        stopped = true;
        clearInterval(clearId);
      }
    };

    const interval = setInterval(async () => {
      if (stopped) return;
      try {
        const [statusRes, chartRes] = await Promise.all([
          trainingApi.status(currentTaskId),
          trainingApi.getChartData(currentTaskId).catch(() => null),
        ]);
        errorCount = 0; // 成功后重置错误计数
        const data = (statusRes.data as any)?.data || statusRes.data as any;
        if (data) {
          setTrainingProgress(data.progress || 0);
          setCurrentEpoch(data.current_epoch || 0);
          setTotalEpochs(data.total_epochs || 0);
          setMonitorMetrics({
            loss: data.metrics?.losses?.box_loss,
            mAP50: data.metrics?.latest?.['metrics/mAP50(B)'],
            precision: data.metrics?.latest?.['metrics/precision(B)'],
            recall: data.metrics?.latest?.['metrics/recall(B)'],
            status: data.status,
          });

          if (chartRes?.data?.data) {
            setChartData(chartRes.data.data);
          }

          const epoch = data.current_epoch || 0;
          if (epoch !== lastEpoch && data.status === 'running') {
            lastEpoch = epoch;
            const mAP = data.metrics?.latest?.['metrics/mAP50(B)'];
            const loss = data.metrics?.losses?.box_loss;
            setLogs(prev => [...prev.slice(-100), `[${new Date().toLocaleTimeString()}] Epoch ${epoch}/${data.total_epochs || epochs} - Loss: ${loss?.toFixed(4) ?? '-'} - mAP50: ${mAP ? (mAP * 100).toFixed(1) + '%' : '-'}`]);
          }

          const status = data.status;
          if (status === 'completed' || status === 'failed' || status === 'cancelled') {
            stop(interval);
            setTraining(false);
            setCurrentTaskId(null);
            if (status === 'completed') {
              setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] 训练完成！`]);
            }
          }
        }
      } catch (e: any) {
        // 404 表示任务已不存在（服务器重启等情况），直接停止轮询
        if (e?.response?.status === 404 || e?.status === 404) {
          stop(interval);
          setTraining(false);
          setCurrentTaskId(null);
          return;
        }
        errorCount++;
        if (errorCount >= 3) {
          stop(interval);
          setTraining(false);
          setCurrentTaskId(null);
          setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] 连接失败，训练任务可能已中断`]);
        }
      }
    }, 3000);

    return () => stop(interval);
  }, [currentTaskId]);

  const loadDatasets = async () => {
    try {
      const res = await datasetApi.list();
      const datasets = (res.data as any)?.datasets || (res.data as any)?.data?.datasets || [];
      setDatasets(datasets);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadPretrainedModels = async () => {
    try {
      const res = await modelApi.getPretrainedModels();
      const models = (res.data as any)?.models || (res.data as any)?.data?.models || [];
      setPretrainedModels(models);
      const installed = models.find((m: PretrainedModel) => m.installed);
      if (installed) {
        setSelectedPretrainedModel(installed.name);
      }
    } catch (error) {
      console.error('Failed to load pretrained models:', error);
    }
  };

  const loadTrainedModels = async () => {
    try {
      const res = await modelApi.getUserModels();
      const responseData = res.data as any;
      const allModels = responseData?.models || [];
      const models = allModels.filter((m: any) => m.source === 'uploaded' || m.source === 'training' || m.source === 'project_model');
      setTrainedModels(models);
    } catch (error) {
      console.error('Failed to load trained models:', error);
    }
  };

  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await trainingApi.getHistory();
      const list = (res.data as any)?.experiments || (res.data as any)?.data?.experiments || (res.data as any) || [];
      setHistory(Array.isArray(list) ? list : []);
    } catch (error) {
      console.error('Failed to load history:', error);
    } finally {
      setHistoryLoading(false);
    }
  };

  // 检查当前步骤是否完成
  const canProceed = (): boolean => {
    switch (currentStep) {
      case 'project':
        return projectName.trim().length > 0 || currentProject !== null;
      case 'configure':
        return selectedDataset.length > 0 &&
          ((modelSource === 'pretrained' && !!selectedPretrainedModel) ||
           (modelSource === 'trained' && !!selectedTrainedModel));
      case 'train':
        return !training;
      case 'monitor':
        return true;
      case 'export':
        return true;
      default:
        return false;
    }
  };

  // 进入下一步
  // 创建项目
  const handleCreateProject = async () => {
    if (!projectName.trim()) return;
    try {
      const res = await projectApi.create({
        name: projectName,
        description: projectDescription,
        task_type: taskType,
        settings: {
          default_model_type: modelSource === 'pretrained' ? selectedPretrainedModel : 'yolo11n',
          default_epochs: epochs,
          default_batch_size: batchSize,
          default_img_size: imageSize,
        },
      });
      const project = (res.data as any)?.project || (res.data as any)?.data;
      if (project) {
        setCurrentProject(project);
        setProjects(prev => [project, ...prev]);
        return project;
      }
    } catch (error) {
      console.error('Failed to create project:', error);
    }
    return null;
  };

  // 进入下一步
  const handleNext = async () => {
    const currentIndex = STEPS.findIndex(s => s.key === currentStep);

    // 如果是项目阶段，先创建项目
    if (currentStep === 'project' && !currentProject) {
      const project = await handleCreateProject();
      if (!project) {
        alert('创建项目失败，请重试');
        return;
      }
    }

    if (currentIndex < STEPS.length - 1) {
      setCurrentStep(STEPS[currentIndex + 1].key);
    }
  };

  // 返回上一步
  const handlePrev = () => {
    const currentIndex = STEPS.findIndex(s => s.key === currentStep);
    if (currentIndex > 0) {
      setCurrentStep(STEPS[currentIndex - 1].key);
    }
  };

  // 开始训练
  const handleStartTraining = async () => {
    if (!selectedDataset) return;
    if (modelSource === 'pretrained' && !selectedPretrainedModel) return;
    if (modelSource === 'trained' && !selectedTrainedModel) return;
    if (training) return;

    setTraining(true);
    try {
      const config: TrainingConfig = {
        project_name: projectName || `train_${Date.now()}`,
        project_id: currentProject?.id, // 关联项目ID
        dataset_path: selectedDataset,
        model_type: modelSource === 'pretrained' ? selectedPretrainedModel : selectedTrainedModel,
        epochs,
        batch_size: batchSize,
        img_size: imageSize,
        optimizer: 'auto',
        lr0: learningRate,
      };
      const res = await trainingApi.start(config);
      const newTaskId = (res.data as any)?.data?.task_id || (res.data as any)?.task_id || '';
      setTaskId(newTaskId);
      setCurrentTaskId(newTaskId);
      setTotalEpochs(epochs);
      setLogs([`[${new Date().toLocaleTimeString()}] 训练任务已启动，Task ID: ${newTaskId}`]);
      setCurrentStep('monitor');
      setMonitorTab('overview');
    } catch (error) {
      console.error(error);
      setTraining(false);
    }
  };

  const handleStopTraining = async () => {
    if (!currentTaskId) return;
    try {
      await trainingApi.stop(currentTaskId);
      setTraining(false);
      setCurrentTaskId(null);
    } catch (error) {
      console.error(error);
    }
  };

  // 渲染步骤指示器
  const renderStepIndicator = () => (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      marginBottom: 'var(--space-6)',
      gap: 'var(--space-2)'
    }}>
      {STEPS.map((step, index) => {
        const isActive = step.key === currentStep;
        const currentIndex = STEPS.findIndex(s => s.key === currentStep);
        const isCompleted = index < currentIndex;

        return (
          <div
            key={step.key}
            onClick={() => !training && setCurrentStep(step.key)}
            style={{
              display: 'flex',
              alignItems: 'center',
              cursor: training ? 'not-allowed' : 'pointer',
              opacity: training && !isActive ? 0.5 : 1,
            }}
          >
            <div style={{
              width: 40,
              height: 40,
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: isActive ? 'var(--color-primary)' : isCompleted ? 'var(--color-success)' : 'var(--color-bg-secondary)',
              color: isActive || isCompleted ? '#fff' : 'var(--color-text-secondary)',
              fontWeight: 600,
              fontSize: 14,
              transition: 'all 0.2s',
            }}>
              {isCompleted ? '✓' : step.icon}
            </div>
            <span style={{
              marginLeft: 8,
              fontWeight: isActive ? 600 : 400,
              color: isActive ? 'var(--color-text)' : 'var(--color-text-secondary)',
            }}>
              {step.title}
            </span>
            {index < STEPS.length - 1 && (
              <div style={{
                width: 40,
                height: 2,
                background: isCompleted ? 'var(--color-success)' : 'var(--color-border)',
                margin: '0 8px',
              }} />
            )}
          </div>
        );
      })}
    </div>
  );

  // 渲染当前步骤内容
  const renderStepContent = () => {
    switch (currentStep) {
      case 'project':
        return (
          <Card>
            <CardHeader icon="📁" title="选择或创建项目" />
            <div style={{ maxWidth: 600, margin: '0 auto' }}>
              {/* 项目选择下拉框 */}
              {projects.length > 0 && (
                <div style={{ marginBottom: 'var(--space-4)' }}>
                  <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500 }}>选择已有项目</label>
                  <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                  <select
                    className="input"
                    value={currentProject?.id || ''}
                    onChange={(e) => {
                      const projectId = e.target.value;
                      if (projectId) {
                        const project = projects.find(p => p.id === projectId);
                        if (project) {
                          setCurrentProject(project);
                          setProjectName(project.name);
                          setProjectDescription(project.description || '');
                        }
                      } else {
                        setCurrentProject(null);
                      }
                    }}
                    style={{ flex: 1, padding: '10px', borderRadius: '8px', border: '1px solid #e5e7eb', background: '#fff' }}
                  >
                    <option value="">-- 选择已有项目 --</option>
                    {projects.map((project) => (
                      <option key={project.id} value={project.id}>
                        {project.name} {project.models_count ? `(${project.models_count} 个模型)` : ''}
                      </option>
                    ))}
                  </select>
                  {currentProject && (
                    <Button
                      variant="danger"
                      size="sm"
                      onClick={async () => {
                        if (!confirm(`确定删除项目 "${currentProject.name}" 吗？此操作不可恢复。`)) return;
                        try {
                          await projectApi.delete(currentProject.id);
                          setProjects(prev => prev.filter(p => p.id !== currentProject.id));
                          setCurrentProject(null);
                          setProjectName('');
                        } catch (e) {
                          console.error(e);
                          alert('删除项目失败');
                        }
                      }}
                    >
                      删除
                    </Button>
                  )}
                  </div>
                </div>
              )}

              <div style={{ marginTop: 'var(--space-4)', borderTop: '1px solid #e5e7eb', paddingTop: 'var(--space-4)' }}>
                <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500 }}>
                  {projects.length > 0 ? '或创建新项目' : '项目名称'}
                </label>
                <Input
                  label=""
                  value={projectName}
                  onChange={(e) => {
                    setProjectName(e.target.value);
                    setCurrentProject(null); // 清除已选项目
                  }}
                  placeholder="输入新项目名称，用于组织训练任务"
                />
              </div>

              {!currentProject && (
                <>
                  <div style={{ marginTop: 'var(--space-4)' }}>
                    <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500 }}>项目描述</label>
                    <textarea
                      value={projectDescription}
                      onChange={(e) => setProjectDescription(e.target.value)}
                      placeholder="描述这个项目的用途（可选）"
                      style={{
                        width: '100%',
                        padding: '10px',
                        borderRadius: '8px',
                        border: '1px solid #e5e7eb',
                        fontFamily: 'inherit',
                        fontSize: '14px',
                        minHeight: '80px',
                        resize: 'vertical'
                      }}
                    />
                  </div>
                  <div style={{ marginTop: 'var(--space-4)' }}>
                    <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500 }}>任务类型</label>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
                      {[
                        { key: 'detect', label: '检测', icon: '🔍', desc: '目标检测' },
                        { key: 'segment', label: '分割', icon: '✂️', desc: '实例分割' },
                        { key: 'classify', label: '分类', icon: '🏷️', desc: '图像分类' },
                        { key: 'pose', label: '姿态', icon: '🧘', desc: '姿态估计' },
                        { key: 'obb', label: '定向', icon: '📐', desc: '定向检测' },
                      ].map((type) => (
                        <button
                          key={type.key}
                          type="button"
                          onClick={() => setTaskType(type.key as any)}
                          style={{
                            padding: '12px 8px',
                            border: taskType === type.key ? '2px solid #3b82f6' : '1px solid #e5e7eb',
                            borderRadius: '8px',
                            background: taskType === type.key ? '#eff6ff' : '#ffffff',
                            cursor: 'pointer',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            gap: '4px',
                            transition: 'all 0.2s'
                          }}
                        >
                          <span style={{ fontSize: '20px' }}>{type.icon}</span>
                          <span style={{ fontWeight: 500, fontSize: '13px' }}>{type.label}</span>
                          <span style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>{type.desc}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}
              <p style={{ color: 'var(--color-text-secondary)', marginTop: 16, fontSize: 14 }}>
                {currentProject ? `当前项目: ${currentProject.name}` : '项目将保存到 /data/models/projects/ 目录'}
              </p>
            </div>
          </Card>
        );

      case 'configure':
        return (
          <Card>
            <CardHeader icon="⚙️" title="配置训练参数" />
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 'var(--space-4)' }}>
              <div>
                <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500 }}>选择数据集</label>
                <select
                  className="input"
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  disabled={training}
                  style={{ width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid #e5e7eb', background: '#fff' }}
                >
                  <option value="">请选择数据集</option>
                  {datasets.map((ds) => (
                    <option key={ds.name} value={ds.name}>{ds.name} ({ds.num_images || ds.image_count || 0} 张图片)</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500 }}>模型来源</label>
                <select
                  className="input"
                  value={modelSource}
                  onChange={(e) => setModelSource(e.target.value as 'pretrained' | 'trained')}
                  disabled={training}
                  style={{ width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid #e5e7eb', background: '#fff' }}
                >
                  <option value="pretrained">预训练基础模型</option>
                  <option value="trained">用户训练模型</option>
                </select>
              </div>

              {modelSource === 'pretrained' && pretrainedModels.length > 0 && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500 }}>选择预训练模型</label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 8 }}>
                    {pretrainedModels.map((model) => (
                      <button
                        key={model.name}
                        type="button"
                        onClick={() => setSelectedPretrainedModel(model.name)}
                        disabled={training}
                        style={{
                          padding: '12px 8px',
                          border: selectedPretrainedModel === model.name ? '2px solid #3b82f6' : '1px solid #e5e7eb',
                          borderRadius: 8,
                          background: selectedPretrainedModel === model.name ? '#eff6ff' : '#ffffff',
                          cursor: training ? 'not-allowed' : 'pointer',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          width: '100%'
                        }}
                      >
                        <div style={{ fontWeight: 500, fontSize: 14 }}>{model.name}</div>
                        <div style={{ fontSize: 12, color: model.installed ? 'var(--color-success)' : 'var(--color-text-secondary)' }}>
                          {model.installed ? '✓ 已安装' : '○ 将下载'}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {modelSource === 'trained' && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <label style={{ display: 'block', marginBottom: 'var(--space-2)', fontWeight: 500 }}>选择训练好的模型</label>
                  <select
                    className="input"
                    value={selectedTrainedModel}
                    onChange={(e) => setSelectedTrainedModel(e.target.value)}
                    disabled={training}
                    style={{ width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid #e5e7eb', background: '#fff' }}
                  >
                    <option value="">请选择模型</option>
                    {trainedModels.map((model) => (
                      <option key={(model as any).path || model.name} value={(model as any).path || model.name}>
                        {(model as any).project ? `[${(model as any).project}] ${model.name.includes('best') ? '最优权重' : model.name.includes('last') ? '最终权重' : model.name}` : model.name}{(model as any).mAP50 ? ` (mAP: ${(((model as any).mAP50) * 100).toFixed(1)}%)` : ''}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <Input type="number" label="训练轮数 (Epochs)" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value))} min={1} max={1000} disabled={training} />
              <Input type="number" label="批量大小 (Batch Size)" value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value))} min={1} max={128} disabled={training} />
              <Input type="number" label="图像尺寸" value={imageSize} onChange={(e) => setImageSize(parseInt(e.target.value))} min={320} max={1280} step={32} disabled={training} />
              <Input type="number" label="学习率" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))} min={0.0001} max={0.1} step={0.001} disabled={training} />
            </div>
          </Card>
        );

      case 'train':
        return (
          <Card>
            <CardHeader icon="🚀" title="开始训练" />
            <div style={{ textAlign: 'center', padding: 'var(--space-6)' }}>
              <div style={{ fontSize: 48, marginBottom: 'var(--space-4)' }}>🔥</div>
              <h2 style={{ marginBottom: 'var(--space-4)' }}>{projectName || '训练任务'}</h2>
              <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-4)' }}>
                数据集: {selectedDataset} | 模型: {modelSource === 'pretrained' ? selectedPretrainedModel : selectedTrainedModel}
              </p>
              <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-4)' }}>
                Epochs: {epochs} | Batch: {batchSize} | Image Size: {imageSize}
              </p>

              {!training ? (
                <Button variant="primary" size="lg" onClick={handleStartTraining}>
                  🚀 开始训练
                </Button>
              ) : (
                <div>
                  <div style={{ marginBottom: 'var(--space-4)' }}>
                    <div style={{
                      width: '100%',
                      height: 24,
                      background: 'var(--color-bg-secondary)',
                      borderRadius: 12,
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${trainingProgress}%`,
                        height: '100%',
                        background: 'linear-gradient(90deg, #3b82f6, #10b981)',
                        transition: 'width 0.3s ease'
                      }} />
                    </div>
                    <p style={{ marginTop: 8, color: 'var(--color-text-secondary)' }}>
                      Epoch {currentEpoch}/{totalEpochs} - {trainingProgress.toFixed(1)}%
                    </p>
                  </div>
                  <Button variant="secondary" size="lg" onClick={handleStopTraining}>
                    停止训练
                  </Button>
                </div>
              )}
            </div>
          </Card>
        );

      case 'monitor':
        return (
          <div>
            {/* 进度条 */}
            {training && (
              <Card style={{ marginBottom: 'var(--space-4)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontWeight: 600 }}>{projectName || taskId}</span>
                  <span style={{ color: '#f59e0b', fontWeight: 500 }}>训练中</span>
                </div>
                <div style={{ height: 20, background: '#f1f5f9', borderRadius: 10, overflow: 'hidden', marginBottom: 8 }}>
                  <div style={{
                    width: `${trainingProgress}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                    transition: 'width 0.5s ease',
                    display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 6
                  }}>
                    {trainingProgress > 10 && <span style={{ color: '#fff', fontSize: 11, fontWeight: 600 }}>{trainingProgress.toFixed(1)}%</span>}
                  </div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 13, color: '#6b7280' }}>Epoch {currentEpoch} / {totalEpochs}</span>
                  <Button variant="secondary" size="sm" onClick={handleStopTraining}>⏹ 停止训练</Button>
                </div>
              </Card>
            )}

            {/* 统计卡片 */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-3)', marginBottom: 'var(--space-4)' }}>
              <StatCard value={training || monitorMetrics.status === 'running' ? '训练中' : (monitorMetrics.status === 'completed' ? '已完成' : monitorMetrics.status === 'failed' ? '失败' : monitorMetrics.status ? '已停止' : '加载中')} label="状态" variant={monitorMetrics.status === 'completed' ? 'success' : training || monitorMetrics.status === 'running' ? 'accent' : 'default'} />
              <StatCard value={`${currentEpoch}/${totalEpochs || epochs}`} label="Epoch" />
              <StatCard value={monitorMetrics.mAP50 ? `${(monitorMetrics.mAP50 * 100).toFixed(1)}%` : '-'} label="mAP@0.5" variant="accent" />
              <StatCard value={monitorMetrics.loss?.toFixed(4) || '-'} label="Box Loss" />
            </div>

            {/* Tab 栏 */}
            <div style={{ display: 'flex', gap: 4, marginBottom: 'var(--space-4)', borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
              {(['overview', 'charts', 'console', 'system'] as MonitorTab[]).map(tab => (
                <button
                  key={tab}
                  onClick={() => setMonitorTab(tab)}
                  style={{
                    padding: '8px 20px',
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer',
                    fontWeight: monitorTab === tab ? 600 : 400,
                    color: monitorTab === tab ? 'var(--color-primary)' : '#6b7280',
                    borderBottom: monitorTab === tab ? '2px solid var(--color-primary)' : '2px solid transparent',
                    marginBottom: -1,
                    fontSize: 14,
                  }}
                >
                  {{ overview: '概览', charts: '图表', console: '日志', system: '系统' }[tab]}
                </button>
              ))}
            </div>

            {/* Tab 内容 */}
            {monitorTab === 'overview' && (
              <Card>
                <CardHeader icon="📋" title="训练信息" />
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-3)' }}>
                  {[
                    ['项目', projectName || '-'],
                    ['任务 ID', taskId || '-'],
                    ['数据集', selectedDataset || '-'],
                    ['基础模型', modelSource === 'pretrained' ? selectedPretrainedModel : selectedTrainedModel],
                    ['Epochs', epochs],
                    ['Batch Size', batchSize],
                    ['图像尺寸', imageSize],
                    ['学习率', learningRate],
                  ].map(([k, v]) => (
                    <div key={String(k)} style={{ padding: '10px 12px', background: '#f8fafc', borderRadius: 8 }}>
                      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 2 }}>{k}</div>
                      <div style={{ fontWeight: 600, fontSize: 14 }}>{v}</div>
                    </div>
                  ))}
                </div>
                {!training && monitorMetrics.status === 'completed' && (
                  <div style={{ marginTop: 'var(--space-4)', textAlign: 'center' }}>
                    <Button variant="primary" onClick={() => setCurrentStep('export')}>📦 导出模型</Button>
                  </div>
                )}
              </Card>
            )}

            {monitorTab === 'charts' && (
              <Card>
                <div style={{ display: 'flex', gap: 8, marginBottom: 'var(--space-4)' }}>
                  {(['loss', 'metrics'] as ChartTabType[]).map(t => (
                    <Button key={t} variant={chartTab === t ? 'primary' : 'secondary'} size="sm" onClick={() => setChartTab(t)}>
                      {t === 'loss' ? 'Loss 曲线' : '指标曲线'}
                    </Button>
                  ))}
                </div>
                {!chartData ? (
                  <div style={{ textAlign: 'center', padding: 'var(--space-8)', color: '#6b7280' }}>暂无图表数据，训练进行中...</div>
                ) : chartTab === 'loss' ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData.epochs.map((ep, i) => ({
                      epoch: ep,
                      box_loss: chartData.losses.box_loss[i] || 0,
                      cls_loss: chartData.losses.cls_loss[i] || 0,
                      dfl_loss: chartData.losses.dfl_loss[i] || 0,
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="box_loss" stroke="#3b82f6" dot={false} name="Box Loss" />
                      <Line type="monotone" dataKey="cls_loss" stroke="#f59e0b" dot={false} name="Cls Loss" />
                      <Line type="monotone" dataKey="dfl_loss" stroke="#10b981" dot={false} name="DFL Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData.metrics_history.map(m => ({
                      epoch: m.epoch,
                      mAP50: (m['metrics/mAP50(B)'] || 0) * 100,
                      mAP50_95: (m['metrics/mAP50-95(B)'] || 0) * 100,
                      precision: (m['metrics/precision(B)'] || 0) * 100,
                      recall: (m['metrics/recall(B)'] || 0) * 100,
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis unit="%" />
                      <Tooltip formatter={(v: any) => (typeof v === 'number' ? v.toFixed(1) : v) + '%'} />
                      <Legend />
                      <Line type="monotone" dataKey="mAP50" stroke="#3b82f6" dot={false} name="mAP@0.5" />
                      <Line type="monotone" dataKey="mAP50_95" stroke="#8b5cf6" dot={false} name="mAP@0.5:0.95" />
                      <Line type="monotone" dataKey="precision" stroke="#10b981" dot={false} name="Precision" />
                      <Line type="monotone" dataKey="recall" stroke="#f59e0b" dot={false} name="Recall" />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </Card>
            )}

            {monitorTab === 'console' && (
              <Card>
                <CardHeader icon="📟" title="训练日志" />
                <div style={{
                  background: '#0f172a', borderRadius: 8, padding: 16,
                  height: 320, overflowY: 'auto', fontFamily: 'monospace', fontSize: 12
                }}>
                  {logs.length === 0 ? (
                    <span style={{ color: '#64748b' }}>等待训练开始...</span>
                  ) : (
                    logs.map((log, i) => (
                      <div key={i} style={{ color: '#94a3b8', marginBottom: 2 }}>{log}</div>
                    ))
                  )}
                  <div ref={logsEndRef} />
                </div>
              </Card>
            )}

            {monitorTab === 'system' && (
              <Card>
                <CardHeader icon="🖥️" title="系统资源" />
                {!systemInfo ? (
                  <div style={{ textAlign: 'center', padding: 'var(--space-6)', color: '#6b7280' }}>加载中...</div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-3)' }}>
                    <div style={{ padding: 16, background: '#f8fafc', borderRadius: 8 }}>
                      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>GPU</div>
                      <div style={{ fontWeight: 700, fontSize: 16 }}>{systemInfo.gpu_available ? (systemInfo.gpu_name || 'Available') : 'N/A'}</div>
                      {systemInfo.gpu_available && <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>利用率: {systemInfo.gpu_utilization?.toFixed(1) ?? '-'}%</div>}
                    </div>
                    <div style={{ padding: 16, background: '#f8fafc', borderRadius: 8 }}>
                      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>GPU 显存</div>
                      <div style={{ fontWeight: 700, fontSize: 16 }}>
                        {systemInfo.gpu_available ? `${((systemInfo.gpu_memory_used || 0) / 1024).toFixed(1)} / ${((systemInfo.gpu_memory_total || 0) / 1024).toFixed(1)} GB` : 'N/A'}
                      </div>
                    </div>
                    <div style={{ padding: 16, background: '#f8fafc', borderRadius: 8 }}>
                      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>CPU</div>
                      <div style={{ fontWeight: 700, fontSize: 16 }}>{systemInfo.cpu_percent?.toFixed(1) ?? '-'}%</div>
                    </div>
                    <div style={{ padding: 16, background: '#f8fafc', borderRadius: 8 }}>
                      <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>内存</div>
                      <div style={{ fontWeight: 700, fontSize: 16 }}>
                        {`${((systemInfo.memory_used || 0) / 1024).toFixed(1)} / ${((systemInfo.memory_total || 0) / 1024).toFixed(1)} GB`}
                      </div>
                      <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>使用率: {systemInfo.memory_percent?.toFixed(1) ?? '-'}%</div>
                    </div>
                  </div>
                )}
              </Card>
            )}
          </div>
        );

      case 'export':
        return (
          <Card>
            <CardHeader icon="📦" title="导出模型" />
            <div style={{ padding: 'var(--space-4)' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
                <Button variant="secondary" disabled={exportLoading} onClick={() => handleExport('onnx')}>
                  📦 ONNX
                </Button>
                <Button variant="secondary" disabled={exportLoading} onClick={() => handleExport('tensorrt')}>
                  ⚡ TensorRT
                </Button>
                <Button variant="secondary" disabled={exportLoading} onClick={() => handleExport('tflite')}>
                  📱 TFLite
                </Button>
              </div>
              {exportMessage && (
                <div style={{ textAlign: 'center', padding: 'var(--space-3)', color: exportMessage.includes('成功') ? '#10b981' : '#ef4444', fontWeight: 500 }}>
                  {exportMessage}
                </div>
              )}
              <div style={{ textAlign: 'center' }}>
                <Button variant="primary" onClick={() => {
                  setCurrentStep('project');
                  setProjectName('');
                  setSelectedDataset('');
                }}>
                  开始新训练
                </Button>
              </div>
            </div>
          </Card>
        );

      default:
        return null;
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)' }}>
        <h1 style={{ fontFamily: 'DM Sans', fontWeight: 700, margin: 0 }}>模型训练</h1>
      </div>

      {/* 统计卡片 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        <StatCard value={datasets.length} label="可用数据集" />
        <StatCard value={pretrainedModels.filter(m => m.installed).length} label="已安装模型" variant="accent" />
        <StatCard value={trainedModels.length} label="训练完成" variant="success" />
        <StatCard value={training ? '训练中' : '空闲'} label="训练状态" variant={training ? 'accent' : 'default'} />
      </div>

      {/* 步骤指示器 */}
      {renderStepContent()}

      {/* 导航按钮 */}
      <div style={{ marginTop: 'var(--space-6)', display: 'flex', justifyContent: 'space-between' }}>
        <Button
          variant="secondary"
          onClick={handlePrev}
          disabled={currentStep === 'project' || training}
        >
          上一步
        </Button>
        {currentStep !== 'train' && currentStep !== 'export' && (
          <Button
            variant="primary"
            onClick={handleNext}
            disabled={!canProceed()}
          >
            下一步
          </Button>
        )}
      </div>

      {/* 训练历史 */}
      <Card style={{ marginTop: 'var(--space-8)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
          <CardHeader icon="📋" title="训练历史" />
          <Button variant="secondary" size="sm" onClick={loadHistory}>刷新</Button>
        </div>
        {historyLoading ? (
          <p style={{ color: 'var(--text-secondary)' }}>加载中...</p>
        ) : history.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>暂无训练记录</p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['项目名', '状态', 'Epochs', 'mAP@0.5', '数据集', '操作'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 12px', color: 'var(--text-secondary)', fontWeight: 500 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((item: any) => {
                  const statusColor = item.status === 'completed' ? '#10b981' : item.status === 'running' ? '#f59e0b' : '#ef4444';
                  const statusLabel = item.status === 'completed' ? '已完成' : item.status === 'running' ? '训练中' : item.status === 'failed' ? '失败' : item.status;
                  const mAP = item.best_metrics?.mAP50 ?? item.metrics?.mAP50;
                  return (
                    <tr key={item.task_id || item.id} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 500 }}>{item.project_name || item.name || '-'}</td>
                      <td style={{ padding: '10px 12px' }}>
                        <span style={{ color: statusColor, fontWeight: 600 }}>{statusLabel}</span>
                      </td>
                      <td style={{ padding: '10px 12px' }}>{item.config?.epochs ?? item.epochs ?? '-'}</td>
                      <td style={{ padding: '10px 12px' }}>{mAP != null ? `${(mAP * 100).toFixed(1)}%` : '-'}</td>
                      <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>{item.config?.dataset_path || item.dataset || '-'}</td>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => {
                            setCurrentTaskId(item.task_id || item.id);
                            setTaskId(item.task_id || item.id);
                            setProjectName(item.project_name || item.name || '');
                            setCurrentStep('monitor');
                          }}
                        >
                          查看
                        </Button>
                        <Button
                          variant="danger"
                          size="sm"
                          onClick={async () => {
                            if (!confirm(`确定删除训练记录 "${item.project_name || item.name}" 吗？`)) return;
                            try {
                              await trainingApi.deleteExperiment(item.task_id || item.id);
                              await loadHistory();
                            } catch (e) {
                              console.error(e);
                            }
                          }}
                        >
                          删除
                        </Button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
};
