import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardHeader, Button, StatCard } from '../../components/common';
import { trainingApi } from '../../services/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

interface ExperimentDetail {
  task_id: string;
  project_name: string;
  model_type: string;
  dataset_path: string;
  epochs: number;
  batch_size: number;
  status: string;
  created_at: string;
  best_metrics?: {
    mAP50: number;
    mAP50_95: number;
    precision: number;
    recall: number;
  };
  checkpoint_path?: string;
  chart_data?: {
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
  };
}

interface ResultImage {
  name: string;
  url: string;
}

export const TrainingDetails: React.FC = () => {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null);
  const [resultImages, setResultImages] = useState<ResultImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'charts' | 'results'>('overview');

  useEffect(() => {
    if (taskId) {
      loadExperimentDetail();
    }
  }, [taskId]);

  const loadExperimentDetail = async () => {
    if (!taskId) return;
    try {
      setLoading(true);
      // 获取实验详情
      const [expRes, resultsRes] = await Promise.all([
        trainingApi.getExperiment(taskId),
        trainingApi.getExperimentResults(taskId)
      ]);

      if (expRes.data?.data) {
        setExperiment(expRes.data.data);
      }

      if (resultsRes.data?.data?.images) {
        setResultImages(resultsRes.data.data.images);
      }
    } catch (e) {
      console.error('加载实验详情失败:', e);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'var(--color-warning)';
      case 'completed': return 'var(--color-success)';
      case 'failed': return 'var(--color-danger)';
      case 'cancelled': return 'var(--color-text-secondary)';
      default: return 'var(--color-text-secondary)';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return '训练中';
      case 'pending': return '等待中';
      case 'completed': return '已完成';
      case 'failed': return '失败';
      case 'cancelled': return '已取消';
      default: return status;
    }
  };

  // 准备图表数据
  const getLossChartData = () => {
    if (!experiment?.chart_data?.epochs) return [];
    return experiment.chart_data.epochs.map((epoch, idx) => ({
      epoch,
      box_loss: experiment.chart_data?.losses.box_loss[idx] || 0,
      cls_loss: experiment.chart_data?.losses.cls_loss[idx] || 0,
      dfl_loss: experiment.chart_data?.losses.dfl_loss[idx] || 0,
    }));
  };

  const getMetricsChartData = () => {
    if (!experiment?.chart_data?.metrics_history) return [];
    return experiment.chart_data.metrics_history.map(m => ({
      epoch: m.epoch,
      mAP50: (m['metrics/mAP50(B)'] || 0) * 100,
      mAP50_95: (m['metrics/mAP50-95(B)'] || 0) * 100,
      precision: (m['metrics/precision(B)'] || 0) * 100,
      recall: (m['metrics/recall(B)'] || 0) * 100,
    }));
  };

  const handleExport = async (format: string) => {
    if (!taskId) return;
    try {
      await trainingApi.exportExperimentModel(taskId, format);
      alert('导出成功！');
    } catch (e) {
      console.error('导出失败:', e);
      alert('导出失败');
    }
  };

  const handleInfer = () => {
    if (!experiment?.checkpoint_path) {
      alert('模型文件不存在');
      return;
    }
    navigate(`/inference?model=${taskId}`);
  };

  const handleResume = () => {
    navigate(`/training?resume=${taskId}`);
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
        <p>加载中...</p>
      </div>
    );
  }

  if (!experiment) {
    return (
      <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
        <p>实验不存在</p>
        <Button variant="primary" onClick={() => navigate('/training-monitor')}>
          返回训练监控
        </Button>
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)' }}>
        <h1 style={{ fontFamily: 'DM Sans', fontWeight: 700, margin: 0 }}>
          训练详情
          <Button variant="ghost" size="sm" onClick={() => navigate('/training-monitor')} style={{ marginLeft: '16px' }}>
            ← 返回训练监控
          </Button>
        </h1>
        <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
          <Button variant="primary" size="sm" onClick={handleInfer}>
            🔍 推理测试
          </Button>
          <Button variant="secondary" size="sm" onClick={handleResume}>
            🔄 继续训练
          </Button>
          <Button variant="secondary" size="sm" onClick={() => {
            const format = prompt('请输入导出格式 (onnx/torchscript/tflite/pytorch)', 'onnx');
            if (format) handleExport(format);
          }}>
            📦 导出模型
          </Button>
        </div>
      </div>

      {/* 基本信息 */}
      <Card style={{ marginBottom: 'var(--space-4)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.25rem' }}>{experiment.project_name}</h2>
            <p style={{ margin: 'var(--space-2) 0 0', color: 'var(--text-secondary)' }}>
              模型: {experiment.model_type} | 数据集: {experiment.dataset_path} | Epochs: {experiment.epochs}
            </p>
          </div>
          <div style={{
            padding: 'var(--space-2) var(--space-4)',
            borderRadius: 'var(--radius-md)',
            background: getStatusColor(experiment.status) + '20',
            color: getStatusColor(experiment.status),
            fontWeight: 600
          }}>
            {getStatusText(experiment.status)}
          </div>
        </div>
      </Card>

      {/* 标签页 */}
      <div style={{ display: 'flex', gap: 'var(--space-2)', marginBottom: 'var(--space-4)', borderBottom: '1px solid var(--color-border)', paddingBottom: 'var(--space-2)' }}>
        {[
          { key: 'overview', label: '概览', icon: '📊' },
          { key: 'charts', label: '训练曲线', icon: '📈' },
          { key: 'results', label: '验证结果', icon: '🖼️' },
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            style={{
              padding: '8px 16px',
              border: 'none',
              background: activeTab === tab.key ? 'var(--color-primary)' : 'transparent',
              color: activeTab === tab.key ? '#fff' : 'var(--color-text-secondary)',
              borderRadius: '8px 8px 0 0',
              cursor: 'pointer',
              fontWeight: 500,
              transition: 'all 0.2s'
            }}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* 概览标签页 */}
      {activeTab === 'overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
            <StatCard
              value={experiment.best_metrics?.mAP50 ? `${(experiment.best_metrics.mAP50 * 100).toFixed(1)}%` : '-'}
              label="mAP@0.5 (最佳)"
              variant="accent"
            />
            <StatCard
              value={experiment.best_metrics?.mAP50_95 ? `${(experiment.best_metrics.mAP50_95 * 100).toFixed(1)}%` : '-'}
              label="mAP@0.5:0.95 (最佳)"
            />
            <StatCard
              value={experiment.best_metrics?.precision ? `${(experiment.best_metrics.precision * 100).toFixed(1)}%` : '-'}
              label="Precision (最佳)"
            />
            <StatCard
              value={experiment.best_metrics?.recall ? `${(experiment.best_metrics.recall * 100).toFixed(1)}%` : '-'}
              label="Recall (最佳)"
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
            <Card>
              <CardHeader icon="⚙️" title="训练参数" />
              <div style={{ fontSize: '0.875rem' }}>
                <p><strong>模型类型:</strong> {experiment.model_type}</p>
                <p><strong>数据集:</strong> {experiment.dataset_path}</p>
                <p><strong>训练轮数:</strong> {experiment.epochs}</p>
                <p><strong>批量大小:</strong> {experiment.batch_size}</p>
              </div>
            </Card>
            <Card>
              <CardHeader icon="⏱️" title="时间信息" />
              <div style={{ fontSize: '0.875rem' }}>
                <p><strong>开始时间:</strong> {experiment.created_at ? new Date(experiment.created_at).toLocaleString() : '-'}</p>
                <p><strong>状态:</strong> {getStatusText(experiment.status)}</p>
                {experiment.checkpoint_path && (
                  <p><strong>模型路径:</strong> {experiment.checkpoint_path}</p>
                )}
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* 训练曲线标签页 */}
      {activeTab === 'charts' && (
        <div>
          <Card>
            <CardHeader icon="📉" title="Loss Curves" />
            <div style={{ height: '400px' }}>
              {getLossChartData().length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={getLossChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="epoch" stroke="var(--color-text-secondary)" />
                    <YAxis stroke="var(--color-text-secondary)" />
                    <Tooltip
                      contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--color-border)', borderRadius: '8px' }}
                      labelStyle={{ color: 'var(--color-text)' }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="box_loss" name="Box Loss" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="cls_loss" name="Cls Loss" stroke="#10b981" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="dfl_loss" name="DFL Loss" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)' }}>
                  暂无训练曲线数据
                </div>
              )}
            </div>
          </Card>

          <Card style={{ marginTop: 'var(--space-4)' }}>
            <CardHeader icon="📊" title="Performance Metrics" />
            <div style={{ height: '400px' }}>
              {getMetricsChartData().length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={getMetricsChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="epoch" stroke="var(--color-text-secondary)" />
                    <YAxis stroke="var(--color-text-secondary)" domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--color-border)', borderRadius: '8px' }}
                      labelStyle={{ color: 'var(--color-text)' }}
                      formatter={(value) => value != null ? `${Number(value).toFixed(1)}%` : '-'}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="mAP50" name="mAP@0.5" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="mAP50_95" name="mAP@0.5:0.95" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="precision" name="Precision" stroke="#10b981" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="recall" name="Recall" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)' }}>
                  暂无性能指标数据
                </div>
              )}
            </div>
          </Card>
        </div>
      )}

      {/* 验证结果标签页 */}
      {activeTab === 'results' && (
        <div>
          {resultImages.length > 0 ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 'var(--space-4)' }}>
              {resultImages.map((img) => (
                <Card key={img.name}>
                  <img
                    src={img.url}
                    alt={img.name}
                    style={{ width: '100%', borderRadius: 'var(--radius-md)', maxHeight: '400px', objectFit: 'contain' }}
                  />
                  <p style={{ marginTop: 'var(--space-2)', fontSize: '0.875rem', textAlign: 'center' }}>{img.name}</p>
                </Card>
              ))}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 'var(--space-8)', color: 'var(--text-secondary)' }}>
              暂无验证结果图片
            </div>
          )}
        </div>
      )}
    </div>
  );
};
