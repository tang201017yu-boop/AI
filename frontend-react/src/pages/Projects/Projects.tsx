import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardHeader, Button, StatCard, Input, Modal } from '../../components/common';
import { annotationApi, datasetApi, trainingApi, modelApi, projectApi } from '../../services/api';

interface ProjectCard {
  id: string;
  name: string;
  description?: string;
  task_type?: string;
  status?: string;
  created_at: string;
  updated_at?: string;
  model?: any;
  metrics?: any;
  image_count?: number;
  type: 'training' | 'annotation' | 'dataset';
}

export const Projects: React.FC = () => {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectCard[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [annotationProjects, setAnnotationProjects] = useState<any[]>([]);
  const [trainingProjects, setTrainingProjects] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeFilter, setActiveFilter] = useState<'all' | 'training' | 'annotation' | 'dataset'>('all');
  // 创建/编辑项目Modal
  const [showModal, setShowModal] = useState(false);
  const [editingProject, setEditingProject] = useState<any>(null);
  const [projectForm, setProjectForm] = useState({ name: '', description: '', default_epochs: 100, default_batch_size: 32 });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [annoRes, datasetRes, trainingRes, modelsRes, trainingProjRes] = await Promise.all([
        annotationApi.listProjects(),
        datasetApi.list(),
        trainingApi.getHistory(),
        modelApi.list(),
        projectApi.list(),
      ]);

      const annoData = annoRes.data as any;
      setAnnotationProjects(annoData?.data || annoData || []);
      const datasetRd = datasetRes.data as any;
      const datasetsData = datasetRd?.datasets || datasetRd?.data?.datasets || [];
      setDatasets(datasetsData);

      // 加载训练项目
      const tProjRd = trainingProjRes.data as any;
      const tProjects = tProjRd?.projects || tProjRd?.data?.projects || [];
      setTrainingProjects(tProjects);

      // 从训练任务构建项目列表
      const trainingRd = trainingRes.data as any;
      const tasks = trainingRd?.tasks || trainingRd || [];
      const projectMap = new Map();

      // 处理训练任务
      for (const task of tasks) {
        const name = task.project_name || task.task_id;
        if (!projectMap.has(`train_${name}`)) {
          projectMap.set(`train_${name}`, {
            id: `train_${name}`,
            name: name,
            description: task.description || '',
            task_type: task.task_type || 'detect',
            status: task.status,
            created_at: task.created_at,
            updated_at: task.updated_at,
            metrics: task.metrics,
            type: 'training' as const,
          });
        }
      }

      // 关联模型信息
      const modelsRd = modelsRes.data as any;
      const models: any[] = modelsRd?.models || modelsRd?.data?.models || modelsRd || [];
      for (const model of models) {
        const pathParts = model.path.split('/');
        const modelsIndex = pathParts.indexOf('models');
        const projectName = modelsIndex >= 0 ? pathParts[modelsIndex + 1] : null;
        const key = `train_${projectName}`;
        if (projectName && projectMap.has(key)) {
          projectMap.get(key).model = model;
        }
      }

      // 处理训练项目（来自 projectApi）
      for (const tp of tProjects) {
        const key = `tproject_${tp.id}`;
        projectMap.set(key, {
          id: `tproject_${tp.id}`,
          name: tp.name,
          description: tp.description,
          task_type: tp.settings?.default_model_type || 'detect',
          status: tp.status,
          created_at: tp.created_at,
          updated_at: tp.updated_at,
          models_count: tp.models_count || 0,
          type: 'training' as const,
        });
      }

      // 处理标注项目
      for (const p of (annoData?.data || annoData || [])) {
        projectMap.set(`anno_${p.id}`, {
          id: p.id,
          name: p.name,
          description: p.description,
          task_type: p.task_type,
          created_at: p.created_at,
          image_count: p.images?.length || 0,
          type: 'annotation' as const,
        });
      }

      // 处理数据集
      for (const ds of datasetsData) {
        projectMap.set(`dataset_${ds.name}`, {
          id: ds.name,
          name: ds.name,
          description: ds.description || '',
          task_type: ds.task_type,
          created_at: ds.created_at || ds.createdAt,
          image_count: ds.num_images || ds.image_count || 0,
          type: 'dataset' as const,
        });
      }

      setProjects(Array.from(projectMap.values()));
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // 创建训练项目
  const handleCreateProject = async () => {
    try {
      await projectApi.create({
        name: projectForm.name,
        description: projectForm.description,
        settings: {
          default_epochs: projectForm.default_epochs,
          default_batch_size: projectForm.default_batch_size,
        }
      });
      setShowModal(false);
      setProjectForm({ name: '', description: '', default_epochs: 100, default_batch_size: 32 });
      loadData();
    } catch (error) {
      console.error('创建项目失败:', error);
      alert('创建项目失败');
    }
  };

  // 编辑训练项目
  const handleEditProject = async () => {
    if (!editingProject) return;
    try {
      // 去除 tproject_ 前缀
      const apiId = editingProject.id.replace('tproject_', '');
      await projectApi.update(apiId, {
        name: projectForm.name,
        description: projectForm.description,
      });
      setShowModal(false);
      setEditingProject(null);
      setProjectForm({ name: '', description: '', default_epochs: 100, default_batch_size: 32 });
      loadData();
    } catch (error) {
      console.error('更新项目失败:', error);
      alert('更新项目失败');
    }
  };

  // 删除训练项目
  const handleDeleteProject = async (projectId: string) => {
    if (!confirm('确定要删除这个训练项目吗？')) return;
    try {
      // 去除 tproject_ 前缀
      const apiId = projectId.replace('tproject_', '');
      await projectApi.delete(apiId);
      loadData();
    } catch (error) {
      console.error('删除项目失败:', error);
      alert('删除项目失败');
    }
  };

  const getStatusBadge = (status?: string) => {
    if (!status) return null;
    const colors: Record<string, string> = {
      completed: '#10b981',
      running: '#f59e0b',
      failed: '#ef4444',
      cancelled: '#6b7280',
      active: '#3b82f6',
    };
    const labels: Record<string, string> = {
      completed: '已完成',
      running: '训练中',
      failed: '失败',
      cancelled: '已取消',
      active: '活跃',
    };
    return (
      <span style={{
        display: 'inline-block',
        padding: '2px 8px',
        borderRadius: 12,
        fontSize: 11,
        background: colors[status] || '#6b7280',
        color: '#fff',
        fontWeight: 500,
      }}>
        {labels[status] || status}
      </span>
    );
  };

  const getTypeBadge = (type: string) => {
    const badges: Record<string, { bg: string; color: string; label: string }> = {
      training: { bg: '#dbeafe', color: '#1d4ed8', label: '训练' },
      annotation: { bg: '#fef3c7', color: '#b45309', label: '标注' },
      dataset: { bg: '#d1fae5', color: '#047857', label: '数据集' },
    };
    const badge = badges[type] || badges.dataset;
    return (
      <span style={{
        display: 'inline-block',
        padding: '2px 8px',
        borderRadius: 12,
        fontSize: 11,
        background: badge.bg,
        color: badge.color,
        fontWeight: 500,
      }}>
        {badge.label}
      </span>
    );
  };

  const getTaskTypeLabel = (type?: string) => {
    const labels: Record<string, string> = {
      detect: '目标检测',
      segment: '实例分割',
      classify: '图像分类',
      pose: '姿态估计',
      obb: '定向检测',
    };
    return labels[type || 'detect'] || type;
  };

  const filteredProjects = activeFilter === 'all'
    ? projects
    : projects.filter(p => p.type === activeFilter);

  const trainingCount = projects.filter(p => p.type === 'training').length;
  const annotationCount = projects.filter(p => p.type === 'annotation').length;
  const datasetCount = projects.filter(p => p.type === 'dataset').length;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)' }}>
        <h1 style={{ fontFamily: 'DM Sans', fontWeight: 700, margin: 0 }}>项目管理</h1>
        <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
          <Button variant="secondary" onClick={() => {
            setEditingProject(null);
            setProjectForm({ name: '', description: '', default_epochs: 100, default_batch_size: 32 });
            setShowModal(true);
          }}>
            + 新建训练项目
          </Button>
          <Button variant="primary" onClick={() => navigate('/training')}>
            前往训练
          </Button>
        </div>
      </div>

      {/* 统计卡片 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        <StatCard value={projects.length} label="总项目数" />
        <StatCard value={trainingCount} label="训练项目" variant="accent" />
        <StatCard value={annotationCount} label="标注项目" variant="warning" />
        <StatCard value={datasetCount} label="数据集" variant="success" />
      </div>

      {/* 筛选标签 */}
      <div style={{ display: 'flex', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
        {[
          { key: 'all', label: '全部', count: projects.length },
          { key: 'training', label: '训练', count: trainingCount },
          { key: 'annotation', label: '标注', count: annotationCount },
          { key: 'dataset', label: '数据集', count: datasetCount },
        ].map(filter => (
          <button
            key={filter.key}
            onClick={() => setActiveFilter(filter.key as any)}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: '20px',
              background: activeFilter === filter.key ? 'var(--color-primary)' : 'var(--bg-secondary)',
              color: activeFilter === filter.key ? '#fff' : 'var(--color-text-secondary)',
              cursor: 'pointer',
              fontWeight: 500,
              fontSize: '14px',
              transition: 'all 0.2s',
            }}
          >
            {filter.label} ({filter.count})
          </button>
        ))}
      </div>

      {/* 项目卡片列表 */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
          <p style={{ color: 'var(--color-text-secondary)' }}>加载中...</p>
        </div>
      ) : filteredProjects.length === 0 ? (
        <Card>
          <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
            <div style={{ fontSize: '48px', marginBottom: 'var(--space-4)' }}>📁</div>
            <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-4)' }}>
              {activeFilter === 'all' ? '暂无项目' : `暂无${activeFilter === 'training' ? '训练' : activeFilter === 'annotation' ? '标注' : '数据集'}项目`}
            </p>
            <Button variant="primary" onClick={() => navigate('/training')}>
              创建第一个项目
            </Button>
          </div>
        </Card>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 'var(--space-4)' }}>
          {filteredProjects.map((project) => (
            <Card
              key={project.id}
              variant="elevated"
              style={{
                cursor: 'pointer',
                transition: 'transform 0.2s, box-shadow 0.2s',
              }}
              onMouseEnter={(e: React.MouseEvent<HTMLDivElement>) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.1)';
              }}
              onMouseLeave={(e: React.MouseEvent<HTMLDivElement>) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '';
              }}
              onClick={() => {
                if (project.type === 'training') {
                  navigate(`/training-monitor?taskId=${project.id.replace('train_', '')}`);
                } else if (project.type === 'annotation') {
                  navigate('/annotation');
                } else {
                  navigate('/training');
                }
              }}
            >
              {/* 卡片头部 */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 'var(--space-3)' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                    <h4 style={{ fontWeight: 600, margin: 0, fontSize: '16px' }}>{project.name}</h4>
                    {getTypeBadge(project.type)}
                  </div>
                  {project.status && getStatusBadge(project.status)}
                </div>
                <div style={{ fontSize: '20px' }}>
                  {project.type === 'training' ? '🚀' : project.type === 'annotation' ? '✏️' : '📊'}
                </div>
              </div>

              {/* 描述 */}
              {project.description && (
                <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-2)', lineHeight: 1.5 }}>
                  {project.description.length > 60 ? project.description.substring(0, 60) + '...' : project.description}
                </p>
              )}

              {/* 信息 */}
              <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-3)' }}>
                <div style={{ display: 'flex', gap: 'var(--space-4)' }}>
                  <span>📋 {getTaskTypeLabel(project.task_type)}</span>
                  {project.image_count !== undefined && (
                    <span>🖼️ {project.image_count} 张图片</span>
                  )}
                </div>
              </div>

              {/* 训练指标 */}
              {project.type === 'training' && project.metrics && (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '8px',
                  padding: 'var(--space-3)',
                  background: 'var(--bg-secondary)',
                  borderRadius: '8px',
                  marginBottom: 'var(--space-3)'
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--color-primary)' }}>
                      {project.metrics?.latest?.['metrics/mAP50(B)']
                        ? `${(project.metrics.latest['metrics/mAP50(B)'] * 100).toFixed(1)}%`
                        : '-'}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>mAP@0.5</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '14px', fontWeight: 600, color: '#10b981' }}>
                      {project.metrics?.latest?.['metrics/precision(B)']
                        ? `${(project.metrics.latest['metrics/precision(B)'] * 100).toFixed(1)}%`
                        : '-'}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>Precision</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '14px', fontWeight: 600, color: '#f59e0b' }}>
                      {project.metrics?.latest?.['metrics/recall(B)']
                        ? `${(project.metrics.latest['metrics/recall(B)'] * 100).toFixed(1)}%`
                        : '-'}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)' }}>Recall</div>
                  </div>
                </div>
              )}

              {/* 模型信息 */}
              {project.model && (
                <p style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-2)' }}>
                  🤖 模型: {project.model.name || project.model.path?.split('/').pop()}
                </p>
              )}

              {/* 底部操作 */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'auto', paddingTop: 'var(--space-3)', borderTop: '1px solid var(--color-border)' }}>
                <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)' }}>
                  {project.created_at ? new Date(project.created_at).toLocaleDateString() : ''}
                </span>
                <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                  {project.type === 'training' && (
                    <>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate(`/training-monitor?taskId=${project.id.replace('train_', '')}`);
                        }}
                      >
                        监控
                      </Button>
                      <Button
                        variant="primary"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate('/inference');
                        }}
                      >
                        推理
                      </Button>
                      {project.id.startsWith('tproject_') && (
                        <>
                          <Button
                            variant="secondary"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditingProject(project);
                              setProjectForm({
                                name: project.name,
                                description: project.description || '',
                                default_epochs: 100,
                                default_batch_size: 32
                              });
                              setShowModal(true);
                            }}
                          >
                            编辑
                          </Button>
                          <Button
                            variant="secondary"
                            size="sm"
                            style={{ color: '#ef4444' }}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteProject(project.id);
                            }}
                          >
                            删除
                          </Button>
                        </>
                      )}
                    </>
                  )}
                  {project.type === 'annotation' && (
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate('/annotation');
                      }}
                    >
                      标注
                    </Button>
                  )}
                  {project.type === 'dataset' && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate('/training');
                      }}
                    >
                      训练
                    </Button>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* 创建/编辑项目 Modal */}
    <Modal
      isOpen={showModal}
      onClose={() => {
        setShowModal(false);
        setEditingProject(null);
        setProjectForm({ name: '', description: '', default_epochs: 100, default_batch_size: 32 });
      }}
      title={editingProject ? '编辑训练项目' : '新建训练项目'}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)', minWidth: '400px' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>项目名称</label>
          <Input
            value={projectForm.name}
            onChange={(e) => setProjectForm({ ...projectForm, name: e.target.value })}
            placeholder="输入项目名称"
          />
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>描述</label>
          <textarea
            value={projectForm.description}
            onChange={(e) => setProjectForm({ ...projectForm, description: e.target.value })}
            placeholder="输入项目描述"
            style={{
              width: '100%',
              height: '80px',
              padding: '8px 12px',
              border: '1px solid var(--color-border)',
              borderRadius: '6px',
              fontSize: '14px',
              fontFamily: 'inherit',
              resize: 'vertical',
              backgroundColor: 'var(--bg-primary)',
              color: 'var(--color-text)',
            }}
          />
        </div>
        {!editingProject && (
          <>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>默认训练轮数</label>
              <Input
                type="number"
                value={projectForm.default_epochs}
                onChange={(e) => setProjectForm({ ...projectForm, default_epochs: parseInt(e.target.value) || 100 })}
                placeholder="100"
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>默认批次大小</label>
              <Input
                type="number"
                value={projectForm.default_batch_size}
                onChange={(e) => setProjectForm({ ...projectForm, default_batch_size: parseInt(e.target.value) || 32 })}
                placeholder="32"
              />
            </div>
          </>
        )}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 'var(--space-2)', marginTop: 'var(--space-4)' }}>
          <Button variant="secondary" onClick={() => {
            setShowModal(false);
            setEditingProject(null);
            setProjectForm({ name: '', description: '', default_epochs: 100, default_batch_size: 32 });
          }}>
            取消
          </Button>
          <Button variant="primary" onClick={editingProject ? handleEditProject : handleCreateProject}>
            {editingProject ? '保存' : '创建'}
          </Button>
        </div>
      </div>
    </Modal>
  </div>
);
};
