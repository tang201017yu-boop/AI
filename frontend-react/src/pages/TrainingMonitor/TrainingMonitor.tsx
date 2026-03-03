import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardHeader, Button, StatCard } from '../../components/common';
import { trainingApi } from '../../services/api';
import type { TrainingStatus } from '../../types';

interface TaskInfo {
  id: string;
  name: string;
  status: string;
  progress: number;
  currentEpoch: number;
  totalEpochs: number;
  metrics: {
    loss?: number;
    mAP50?: number;
    mAP50_95?: number;
    precision?: number;
    recall?: number;
  };
  createdAt: string;
}

export const TrainingMonitor: React.FC = () => {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [currentTask, setCurrentTask] = useState<TaskInfo | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // 加载训练任务列表
  useEffect(() => {
    loadTasks();
  }, []);

  // 轮询获取训练状态
  useEffect(() => {
    if (!selectedTask) return;

    const interval = setInterval(async () => {
      try {
        const res = await trainingApi.status(selectedTask);
        const data = res.data?.data || res.data;

        if (data) {
          const task: TaskInfo = {
            id: data.task_id,
            name: data.task_id,
            status: data.status,
            progress: data.progress || 0,
            currentEpoch: data.current_epoch || 0,
            totalEpochs: data.total_epochs || 1,
            metrics: {
              loss: data.metrics?.losses?.box_loss,
              mAP50: data.metrics?.latest?.['metrics/mAP50(B)'],
              mAP50_95: data.metrics?.latest?.['metrics/mAP50-95(B)'],
              precision: data.metrics?.latest?.['metrics/precision(B)'],
              recall: data.metrics?.latest?.['metrics/recall(B)'],
            },
            createdAt: data.created_at,
          };

          setCurrentTask(task);

          // 添加日志
          const logMsg = `[${new Date().toLocaleTimeString()}] Epoch ${task.currentEpoch}/${task.totalEpochs} - Loss: ${task.metrics.loss?.toFixed(4) || '-'} - mAP50: ${((task.metrics.mAP50 || 0) * 100).toFixed(1)}%`;
          setLogs(prev => [...prev.slice(-50), logMsg]);

          // 如果任务完成或失败，更新任务列表
          if (task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled') {
            loadTasks();
          }
        }
      } catch (e) {
        console.error(e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [selectedTask]);

  // 自动滚动日志
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const loadTasks = async () => {
    try {
      const res = await trainingApi.getHistory();
      // 简化处理，实际应该从后端获取任务列表
      // 这里假设有一个运行中的任务
    } catch (e) {
      console.error(e);
    }
  };

  const handleStop = async () => {
    if (!selectedTask) return;
    try {
      await trainingApi.stop(selectedTask);
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] 训练已停止`]);
    } catch (e) {
      console.error(e);
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

  // 如果没有选中的任务，显示任务选择
  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>
        训练监控
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate('/training')}
          style={{ marginLeft: '16px' }}
        >
          ← 返回训练
        </Button>
      </h1>

      {/* 进度条 */}
      {currentTask && currentTask.status === 'running' && (
        <Card style={{ marginBottom: 'var(--space-4)', background: 'var(--bg-secondary)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-2)' }}>
            <span style={{ fontWeight: 600 }}>
              {currentTask.name}
            </span>
            <span style={{ color: getStatusColor(currentTask.status) }}>
              {getStatusText(currentTask.status)}
            </span>
          </div>

          {/* 进度条 */}
          <div style={{
            height: '24px',
            background: 'var(--bg-primary)',
            borderRadius: '12px',
            overflow: 'hidden',
            position: 'relative',
            marginBottom: 'var(--space-3)'
          }}>
            <div style={{
              width: `${currentTask.progress}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
              transition: 'width 0.5s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              paddingRight: '8px'
            }}>
              <span style={{ color: '#fff', fontSize: '12px', fontWeight: 600 }}>
                {currentTask.progress.toFixed(1)}%
              </span>
            </div>
          </div>

          {/* 控制按钮 */}
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            <Button variant="danger" size="sm" onClick={handleStop}>
              ⏹ 停止训练
            </Button>
          </div>
        </Card>
      )}

      {/* 统计卡片 */}
      {currentTask ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
          <StatCard
            value={getStatusText(currentTask.status)}
            label="状态"
            variant={currentTask.status === 'running' ? 'warning' : currentTask.status === 'completed' ? 'success' : 'default'}
          />
          <StatCard
            value={`${currentTask.currentEpoch}/${currentTask.totalEpochs}`}
            label="当前轮次"
          />
          <StatCard
            value={currentTask.metrics.mAP50 ? `${(currentTask.metrics.mAP50 * 100).toFixed(1)}%` : '-'}
            label="mAP@0.5"
            variant="accent"
          />
          <StatCard
            value={currentTask.metrics.loss?.toFixed(4) || '-'}
            label="Loss"
          />
        </div>
      ) : (
        <div style={{
          textAlign: 'center',
          padding: 'var(--space-8)',
          background: 'var(--bg-secondary)',
          borderRadius: 'var(--radius-lg)',
          marginBottom: 'var(--space-6)'
        }}>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-4)' }}>
            当前没有正在进行的训练任务
          </p>
          <Button variant="primary" onClick={() => navigate('/training')}>
            前往训练页面
          </Button>
        </div>
      )}

      {/* 训练曲线和详情 */}
      {currentTask && (
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 'var(--space-6)' }}>
          <Card>
            <CardHeader icon="📈" title="训练进度" />
            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
              {/* 简化版本：显示进度百分比 */}
              <div style={{ textAlign: 'center' }}>
                <div style={{
                  fontSize: '64px',
                  fontWeight: 700,
                  background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}>
                  {currentTask.progress.toFixed(0)}%
                </div>
                <p style={{ color: 'var(--text-secondary)' }}>
                  Epoch {currentTask.currentEpoch} / {currentTask.totalEpochs}
                </p>
              </div>
            </div>
          </Card>

          <div>
            <Card style={{ marginBottom: 'var(--space-4)' }}>
              <CardHeader icon="⚙️" title="训练参数" />
              <div style={{ fontSize: '0.875rem' }}>
                <p><strong>模型:</strong> {currentTask.name}</p>
                <p><strong>数据集:</strong> -</p>
                <p><strong>批量大小:</strong> 16</p>
                <p><strong>图像尺寸:</strong> 640</p>
                <p><strong>优化器:</strong> AdamW</p>
                <p><strong>学习率:</strong> 0.01</p>
              </div>
            </Card>

            <Card>
              <CardHeader icon="⏱️" title="时间信息" />
              <div style={{ fontSize: '0.875rem' }}>
                <p><strong>开始时间:</strong> {currentTask.createdAt ? new Date(currentTask.createdAt).toLocaleString() : '-'}</p>
                <p><strong>当前状态:</strong> {getStatusText(currentTask.status)}</p>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* 实时日志 */}
      {logs.length > 0 && (
        <Card style={{ marginTop: 'var(--space-6)' }}>
          <CardHeader icon="📊" title="实时日志" />
          <div style={{
            fontFamily: 'monospace',
            fontSize: '0.75rem',
            background: '#1e1e1e',
            color: '#d4d4d4',
            padding: 'var(--space-4)',
            borderRadius: 'var(--radius-md)',
            maxHeight: '250px',
            overflow: 'auto'
          }}>
            {logs.map((log, idx) => (
              <p key={idx} style={{ margin: '2px 0' }}>{log}</p>
            ))}
            <div ref={logsEndRef} />
          </div>
        </Card>
      )}
    </div>
  );
};
