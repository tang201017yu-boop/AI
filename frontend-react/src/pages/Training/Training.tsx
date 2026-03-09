import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Card, CardHeader, Button, Input, StatCard } from '../../components/common';
import { datasetApi, trainingApi, modelApi, projectApi } from '../../services/api';
import type { Dataset, TrainingConfig, Model, Project } from '../../types';

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

  // Monitor 阶段
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);

  useEffect(() => {
    loadDatasets();
    loadPretrainedModels();
    loadTrainedModels();
    loadProjects();
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

  // 定期检查训练状态
  useEffect(() => {
    if (!currentTaskId) return;

    const interval = setInterval(async () => {
      try {
        const res = await trainingApi.status(currentTaskId);
        const data = res.data?.data || res.data;
        if (data) {
          setTrainingProgress(data.progress || 0);
          setCurrentEpoch(data.current_epoch || 0);
          setTotalEpochs(data.total_epochs || 0);

          const status = data.status;
          if (status === 'completed' || status === 'failed' || status === 'cancelled') {
            setTraining(false);
            if (status === 'completed') {
              setCurrentStep('monitor');
            }
          }
        }
      } catch (e) {
        console.error(e);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [currentTaskId]);

  const loadDatasets = async () => {
    try {
      const res = await datasetApi.list();
      const datasets = res.data?.datasets || res.data?.data?.datasets || [];
      setDatasets(datasets);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadPretrainedModels = async () => {
    try {
      const res = await modelApi.getPretrainedModels();
      const models = res.data?.models || res.data?.data?.models || [];
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
      const res = await modelApi.list();
      const models = res.data || res.data?.models || res.data?.data?.models || [];
      setTrainedModels(models);
    } catch (error) {
      console.error('Failed to load trained models:', error);
    }
  };

  // 检查当前步骤是否完成
  const canProceed = (): boolean => {
    switch (currentStep) {
      case 'project':
        return projectName.trim().length > 0 || currentProject !== null;
      case 'configure':
        return selectedDataset.length > 0 &&
          ((modelSource === 'pretrained' && selectedPretrainedModel) ||
           (modelSource === 'trained' && selectedTrainedModel));
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
      const project = res.data?.project || res.data?.data;
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
      const newTaskId = res.data?.data?.task_id || res.data?.task_id || '';
      setTaskId(newTaskId);
      setCurrentTaskId(newTaskId);
      setTotalEpochs(epochs);

      // 跳转到监控页面，同时传递当前任务ID
      navigate(`/training-monitor?taskId=${newTaskId}`);
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
                      }
                    }}
                    style={{ width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid #e5e7eb', background: '#fff' }}
                  >
                    <option value="">-- 选择已有项目 --</option>
                    {projects.map((project) => (
                      <option key={project.id} value={project.id}>
                        {project.name} {project.models_count ? `(${project.models_count} 个模型)` : ''}
                      </option>
                    ))}
                  </select>
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
                      <option key={model.name} value={model.name}>
                        {model.name} {model.mAP50 ? `(mAP: ${(model.mAP50 * 100).toFixed(1)}%)` : ''}
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
                  <Button variant="danger" size="lg" onClick={handleStopTraining}>
                    停止训练
                  </Button>
                </div>
              )}
            </div>
          </Card>
        );

      case 'monitor':
        return (
          <Card>
            <CardHeader icon="📈" title="训练监控" />
            <div style={{ padding: 'var(--space-4)' }}>
              {training ? (
                <div>
                  <div style={{
                    width: '100%',
                    height: 24,
                    background: 'var(--color-bg-secondary)',
                    borderRadius: 12,
                    overflow: 'hidden',
                    marginBottom: 'var(--space-4)'
                  }}>
                    <div style={{
                      width: `${trainingProgress}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, #3b82f6, #10b981)',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-4)' }}>
                    <StatCard value={currentEpoch} label="当前 Epoch" />
                    <StatCard value={totalEpochs} label="总 Epochs" />
                    <StatCard value={`${trainingProgress.toFixed(1)}%`} label="进度" variant="accent" />
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <Button variant="secondary" onClick={() => navigate(`/training-monitor?taskId=${currentTaskId || taskId}`)}>
                      打开详细监控页面
                    </Button>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: 'var(--space-6)' }}>
                  <div style={{ fontSize: 48, marginBottom: 'var(--space-4)' }}>✅</div>
                  <h3>训练已完成</h3>
                  <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-4)' }}>
                    模型已保存到 /data/models/{projectName}
                  </p>
                  <div style={{ display: 'flex', gap: 'var(--space-3)', justifyContent: 'center' }}>
                    <Button variant="secondary" onClick={() => navigate(`/training-monitor?taskId=${taskId}`)}>
                      查看训练详情
                    </Button>
                    <Button variant="primary" onClick={() => setCurrentStep('export')}>
                      导出模型
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </Card>
        );

      case 'export':
        return (
          <Card>
            <CardHeader icon="📦" title="导出模型" />
            <div style={{ padding: 'var(--space-4)' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
                <Button variant="secondary" onClick={() => alert('导出为 ONNX 格式')}>
                  📦 ONNX
                </Button>
                <Button variant="secondary" onClick={() => alert('导出为 TensorRT 格式')}>
                  ⚡ TensorRT
                </Button>
                <Button variant="secondary" onClick={() => alert('导出为 TFLite 格式')}>
                  📱 TFLite
                </Button>
              </div>
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
        <Button variant="secondary" onClick={() => navigate('/training-monitor')}>
          📈 训练监控
        </Button>
      </div>

      {/* 统计卡片 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        <StatCard value={datasets.length} label="可用数据集" />
        <StatCard value={pretrainedModels.filter(m => m.installed).length} label="已安装模型" variant="accent" />
        <StatCard value={trainedModels.length} label="训练完成" variant="success" />
        <StatCard value={training ? '训练中' : '空闲'} label="训练状态" variant={training ? 'warning' : 'default'} />
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
    </div>
  );
};
