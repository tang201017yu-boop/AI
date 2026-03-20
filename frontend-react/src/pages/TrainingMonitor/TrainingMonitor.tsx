import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Card, CardHeader, Button, StatCard } from '../../components/common';
import { trainingApi } from '../../services/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area
} from 'recharts';
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
  gpu_memory_reserved?: number;
  gpu_utilization?: number;
  cpu_percent?: number;
  memory_total?: number;
  memory_used?: number;
  memory_percent?: number;
}

// 子标签页类型
type TabType = 'overview' | 'charts' | 'console' | 'system';
// Charts 子标签页
type ChartTabType = 'loss' | 'metrics';

export const TrainingMonitor: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [currentTask, setCurrentTask] = useState<TaskInfo | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // 新增状态
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [chartTab, setChartTab] = useState<ChartTabType>('loss');
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loadingChart, setLoadingChart] = useState(false);

  // 加载训练任务列表
  useEffect(() => {
    loadTasks().then(() => {
      const taskIdFromUrl = searchParams.get('taskId');
      if (taskIdFromUrl) {
        setSelectedTask(taskIdFromUrl);
      }
    });
  }, [searchParams]);

  // 轮询获取训练状态和图表数据
  useEffect(() => {
    if (!selectedTask) return;

    let lastEpoch = -1;

    const interval = setInterval(async () => {
      try {
        const [statusRes, chartRes] = await Promise.all([
          trainingApi.status(selectedTask),
          activeTab === 'charts' ? trainingApi.getChartData(selectedTask) : Promise.resolve(null)
        ]);

        const data = (statusRes.data as any)?.data || statusRes.data;

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

          // 更新图表数据
          if (chartRes?.data?.data) {
            setChartData(chartRes.data.data);
          }

          // 添加日志
          if (task.currentEpoch !== lastEpoch && task.status === 'running') {
            lastEpoch = task.currentEpoch;
            const logMsg = `[${new Date().toLocaleTimeString()}] Epoch ${task.currentEpoch}/${task.totalEpochs} - Loss: ${task.metrics.loss?.toFixed(4) || '-'} - mAP50: ${((task.metrics.mAP50 || 0) * 100).toFixed(1)}%`;
            setLogs(prev => [...prev.slice(-50), logMsg]);
          }

          if (task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled') {
            loadTasks();
          }
        }
      } catch (e) {
        console.error(e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [selectedTask, chartTab]);

  // 加载系统信息
  useEffect(() => {
    if (activeTab !== 'system') return;

    const loadSystemInfo = async () => {
      try {
        const res = await trainingApi.getSystemInfo();
        if (res.data?.data) {
          setSystemInfo(res.data.data);
        }
      } catch (e) {
        console.error(e);
      }
    };

    loadSystemInfo();
    const interval = setInterval(loadSystemInfo, 3000);
    return () => clearInterval(interval);
  }, [activeTab]);

  // 自动滚动日志
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const loadTasks = async () => {
    try {
      const res = await trainingApi.getHistory();
      // 使用 experiments API 返回的数据格式
      const tasksData = res.data?.experiments || res.data || [];
      const taskList: TaskInfo[] = tasksData.map((t: any) => ({
        id: t.task_id,
        name: t.project_name || t.task_id,
        status: t.status,
        progress: t.progress || 0,
        currentEpoch: t.current_epoch || 0,
        totalEpochs: t.epochs || 1,
        metrics: {
          loss: t.metrics?.losses?.box_loss,
          mAP50: t.best_metrics?.mAP50 || t.metrics?.latest?.['metrics/mAP50(B)'],
          mAP50_95: t.best_metrics?.['mAP50-95'] || t.metrics?.latest?.['metrics/mAP50-95(B)'],
          precision: t.best_metrics?.precision || t.metrics?.latest?.['metrics/precision(B)'],
          recall: t.best_metrics?.recall || t.metrics?.latest?.['metrics/recall(B)'],
        },
        createdAt: t.created_at,
      }));
      setTasks(taskList);

      const taskIdFromUrl = searchParams.get('taskId');
      if (taskIdFromUrl && taskList.find(t => t.id === taskIdFromUrl)) {
        setSelectedTask(taskIdFromUrl);
      } else if (!selectedTask && taskList.length > 0) {
        const runningTask = taskList.find(t => t.status === 'running');
        if (runningTask) {
          setSelectedTask(runningTask.id);
        } else if (taskList.length > 0) {
          setSelectedTask(taskList[0].id);
        }
      }
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

  // 准备图表数据
  const getLossChartData = () => {
    if (!chartData?.epochs) return [];
    return chartData.epochs.map((epoch, idx) => ({
      epoch,
      box_loss: chartData.losses.box_loss[idx] || 0,
      cls_loss: chartData.losses.cls_loss[idx] || 0,
      dfl_loss: chartData.losses.dfl_loss[idx] || 0,
    }));
  };

  const getMetricsChartData = () => {
    if (!chartData?.metrics_history) return [];
    return chartData.metrics_history.map(m => ({
      epoch: m.epoch,
      mAP50: (m['metrics/mAP50(B)'] || 0) * 100,
      mAP50_95: (m['metrics/mAP50-95(B)'] || 0) * 100,
      precision: (m['metrics/precision(B)'] || 0) * 100,
      recall: (m['metrics/recall(B)'] || 0) * 100,
    }));
  };

  // 渲染子标签页内容
  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'charts':
        return renderCharts();
      case 'console':
        return renderConsole();
      case 'system':
        return renderSystem();
      default:
        return null;
    }
  };

  // 概览标签页
  const renderOverview = () => (
    <>
      {/* 进度条 */}
      {currentTask && currentTask.status === 'running' && (
        <Card style={{ marginBottom: 'var(--space-4)', background: 'var(--bg-secondary)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-2)' }}>
            <span style={{ fontWeight: 600 }}>{currentTask.name}</span>
            <span style={{ color: getStatusColor(currentTask.status) }}>{getStatusText(currentTask.status)}</span>
          </div>
          <div style={{ height: '24px', background: 'var(--bg-primary)', borderRadius: '12px', overflow: 'hidden', position: 'relative', marginBottom: 'var(--space-3)' }}>
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
              <span style={{ color: '#fff', fontSize: '12px', fontWeight: 600 }}>{currentTask.progress.toFixed(1)}%</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            <Button variant="primary" size="sm" onClick={handleStop}>⏹ 停止训练</Button>
          </div>
        </Card>
      )}

      {/* 统计卡片 */}
      {currentTask ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
          <StatCard value={getStatusText(currentTask.status)} label="状态" variant={currentTask.status === 'completed' ? 'success' : 'default'} />
          <StatCard value={`${currentTask.currentEpoch}/${currentTask.totalEpochs}`} label="当前轮次" />
          <StatCard value={currentTask.metrics.mAP50 ? `${(currentTask.metrics.mAP50 * 100).toFixed(1)}%` : '-'} label="mAP@0.5" variant="accent" />
          <StatCard value={currentTask.metrics.loss?.toFixed(4) || '-'} label="Loss" />
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: 'var(--space-8)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', marginBottom: 'var(--space-6)' }}>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-4)' }}>当前没有正在进行的训练任务</p>
          <Button variant="primary" onClick={() => navigate(`/training?taskId=${selectedTask || ''}`)}>前往训练页面</Button>
        </div>
      )}

      {/* 训练历史列表 */}
      <Card style={{ marginBottom: 'var(--space-6)' }}>
        <CardHeader icon="📋" title="训练历史" />
        {tasks.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
                  <th style={{ textAlign: 'left', padding: 'var(--space-2)', color: 'var(--color-text-secondary)' }}>任务名称</th>
                  <th style={{ textAlign: 'left', padding: 'var(--space-2)', color: 'var(--color-text-secondary)' }}>模型</th>
                  <th style={{ textAlign: 'left', padding: 'var(--space-2)', color: 'var(--color-text-secondary)' }}>Epochs</th>
                  <th style={{ textAlign: 'left', padding: 'var(--space-2)', color: 'var(--color-text-secondary)' }}>状态</th>
                  <th style={{ textAlign: 'left', padding: 'var(--space-2)', color: 'var(--color-text-secondary)' }}>mAP@0.5</th>
                  <th style={{ textAlign: 'left', padding: 'var(--space-2)', color: 'var(--color-text-secondary)' }}>创建时间</th>
                  <th style={{ textAlign: 'left', padding: 'var(--space-2)', color: 'var(--color-text-secondary)' }}>操作</th>
                </tr>
              </thead>
              <tbody>
                {tasks.map(task => (
                  <tr
                    key={task.id}
                    onClick={() => navigate(`/training/${task.id}/details`)}
                    style={{
                      cursor: 'pointer',
                      borderBottom: '1px solid var(--color-border)',
                      background: selectedTask === task.id ? 'var(--bg-secondary)' : 'transparent'
                    }}
                  >
                    <td style={{ padding: 'var(--space-2)' }}>{task.name}</td>
                    <td style={{ padding: 'var(--space-2)' }}>{task.id}</td>
                    <td style={{ padding: 'var(--space-2)' }}>{task.totalEpochs}</td>
                    <td style={{ padding: 'var(--space-2)' }}>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: '4px',
                        background: getStatusColor(task.status) + '20',
                        color: getStatusColor(task.status),
                        fontSize: '0.875rem'
                      }}>
                        {getStatusText(task.status)}
                      </span>
                    </td>
                    <td style={{ padding: 'var(--space-2)' }}>
                      {task.metrics.mAP50 ? `${(task.metrics.mAP50 * 100).toFixed(1)}%` : '-'}
                    </td>
                    <td style={{ padding: 'var(--space-2)', fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>
                      {task.createdAt ? new Date(task.createdAt).toLocaleString() : '-'}
                    </td>
                    <td style={{ padding: 'var(--space-2)', display: 'flex', gap: 'var(--space-2)' }}>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate(`/training/${task.id}/details`);
                        }}
                      >
                        查看详情
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={async (e) => {
                          e.stopPropagation();
                          if (confirm(`确定要删除训练记录 "${task.name}" 吗？`)) {
                            try {
                              await trainingApi.deleteExperiment(task.id);
                              loadTasks();
                            } catch (err) {
                              console.error('删除失败:', err);
                              alert('删除失败');
                            }
                          }
                        }}
                        style={{ color: 'var(--color-danger)' }}
                      >
                        删除
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ textAlign: 'center', padding: 'var(--space-4)', color: 'var(--color-text-secondary)' }}>
            暂无训练历史
          </p>
        )}
      </Card>

      {/* 训练曲线和详情 */}
      {currentTask && (
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 'var(--space-6)' }}>
          <Card>
            <CardHeader icon="📈" title="训练进度" />
            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '64px', fontWeight: 700, background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                  {currentTask.progress.toFixed(0)}%
                </div>
                <p style={{ color: 'var(--text-secondary)' }}>Epoch {currentTask.currentEpoch} / {currentTask.totalEpochs}</p>
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
    </>
  );

  // 图表标签页
  const renderCharts = () => (
    <div>
      {/* Charts 子标签页 */}
      <div style={{ display: 'flex', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
        <Button variant={chartTab === 'loss' ? 'primary' : 'secondary'} size="sm" onClick={() => setChartTab('loss')}>Loss Curves</Button>
        <Button variant={chartTab === 'metrics' ? 'primary' : 'secondary'} size="sm" onClick={() => setChartTab('metrics')}>Performance Metrics</Button>
      </div>

      {chartTab === 'loss' && (
        <Card>
          <CardHeader icon="📉" title="Loss Curves" />
          <div style={{ height: '400px' }}>
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
          </div>
        </Card>
      )}

      {chartTab === 'metrics' && (
        <Card>
          <CardHeader icon="📊" title="Performance Metrics" />
          <div style={{ height: '400px' }}>
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
          </div>
        </Card>
      )}
    </div>
  );

  // 控制台标签页
  const renderConsole = () => (
    <Card>
      <CardHeader icon="📝" title="Live Logs" />
      <div style={{
        fontFamily: 'monospace',
        fontSize: '0.8rem',
        background: '#1e1e1e',
        color: '#d4d4d4',
        padding: 'var(--space-4)',
        borderRadius: 'var(--radius-md)',
        height: '500px',
        overflow: 'auto'
      }}>
        {logs.length === 0 ? (
          <p style={{ color: '#666' }}>暂无日志</p>
        ) : (
          logs.map((log, idx) => {
            let color = '#d4d4d4';
            if (log.includes('error') || log.includes('Error') || log.includes('失败')) color = '#ef4444';
            else if (log.includes('warning') || log.includes('Warning') || log.includes('警告')) color = '#f59e0b';
            else if (log.includes('completed') || log.includes('完成')) color = '#10b981';
            return <p key={idx} style={{ margin: '2px 0', color }}>{log}</p>;
          })
        )}
        <div ref={logsEndRef} />
      </div>
    </Card>
  );

  // 系统标签页
  const renderSystem = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 'var(--space-4)' }}>
      {/* GPU 信息 */}
      <Card>
        <CardHeader icon="🖥️" title="GPU" />
        {systemInfo?.gpu_available ? (
          <div>
            <p><strong>GPU:</strong> {systemInfo.gpu_name || 'Unknown'}</p>
            <div style={{ marginTop: 'var(--space-4)' }}>
              <p style={{ marginBottom: '8px' }}>显存使用</p>
              <div style={{ height: '20px', background: 'var(--bg-primary)', borderRadius: '10px', overflow: 'hidden' }}>
                <div style={{
                  width: `${((systemInfo.gpu_memory_used || 0) / (systemInfo.gpu_memory_total || 1)) * 100}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                  transition: 'width 0.3s ease'
                }} />
              </div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                {((systemInfo.gpu_memory_used || 0) / 1024 / 1024 / 1024).toFixed(1)} GB / {((systemInfo.gpu_memory_total || 0) / 1024 / 1024 / 1024).toFixed(1)} GB
              </p>
            </div>
          </div>
        ) : (
          <p style={{ color: 'var(--text-secondary)' }}>GPU 不可用</p>
        )}
      </Card>

      {/* CPU 信息 */}
      <Card>
        <CardHeader icon="💻" title="CPU" />
        <div>
          <p><strong>CPU 使用率</strong></p>
          <div style={{ height: '20px', background: 'var(--bg-primary)', borderRadius: '10px', overflow: 'hidden', marginTop: '8px' }}>
            <div style={{
              width: `${systemInfo?.cpu_percent || 0}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #10b981, #3b82f6)',
              transition: 'width 0.3s ease'
            }} />
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '4px' }}>{systemInfo?.cpu_percent?.toFixed(1) || 0}%</p>
        </div>
      </Card>

      {/* 内存信息 */}
      <Card style={{ gridColumn: '1 / -1' }}>
        <CardHeader icon="🧠" title="Memory" />
        <div>
          <div style={{ height: '24px', background: 'var(--bg-primary)', borderRadius: '12px', overflow: 'hidden' }}>
            <div style={{
              width: `${systemInfo?.memory_percent || 0}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #f59e0b, #ef4444)',
              transition: 'width 0.3s ease'
            }} />
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '8px' }}>
            {((systemInfo?.memory_used || 0) / 1024 / 1024 / 1024).toFixed(1)} GB / {((systemInfo?.memory_total || 0) / 1024 / 1024 / 1024).toFixed(1)} GB ({systemInfo?.memory_percent?.toFixed(1) || 0}%)
          </p>
        </div>
      </Card>
    </div>
  );

  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>
        训练监控
        <Button variant="ghost" size="sm" onClick={() => navigate(`/training?taskId=${selectedTask || ''}`)} style={{ marginLeft: '16px' }}>
          ← 返回训练
        </Button>
      </h1>

      {/* 主标签页 */}
      <div style={{ display: 'flex', gap: 'var(--space-2)', marginBottom: 'var(--space-4)', borderBottom: '1px solid var(--color-border)', paddingBottom: 'var(--space-2)' }}>
        {[
          { key: 'overview', label: 'Overview', icon: '📊' },
          { key: 'charts', label: 'Charts', icon: '📈' },
          { key: 'console', label: 'Console', icon: '💬' },
          { key: 'system', label: 'System', icon: '⚙️' },
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as TabType)}
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

      {/* 标签页内容 */}
      {renderTabContent()}
    </div>
  );
};
