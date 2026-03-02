import React, { useState } from 'react';
import { Card, CardHeader, Button, StatCard } from '../../components/common';

export const TrainingMonitor: React.FC = () => {
  const [trainingTasks] = useState([
    { id: '1', name: 'yolo11m-training', status: 'running', epoch: 45, totalEpochs: 100, loss: 0.234, map: 0.823 },
  ]);

  const isTraining = trainingTasks.some(t => t.status === 'running');

  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>训练监控</h1>

      {isTraining ? (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
            <StatCard value="训练中" label="状态" variant="warning" />
            <StatCard value="45/100" label="当前轮次" />
            <StatCard value="0.234" label="Loss" variant="accent" />
            <StatCard value="82.3%" label="mAP@0.5" variant="success" />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 'var(--space-6)' }}>
            <Card>
              <CardHeader icon="📈" title="训练曲线" />
              <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
                <p style={{ color: 'var(--text-secondary)' }}>训练曲线图表区域</p>
              </div>
            </Card>

            <div>
              <Card style={{ marginBottom: 'var(--space-4)' }}>
                <CardHeader icon="⚙️" title="训练参数" />
                <div style={{ fontSize: '0.875rem' }}>
                  <p><strong>模型:</strong> YOLO11m</p>
                  <p><strong>数据集:</strong> coco128</p>
                  <p><strong>批量大小:</strong> 16</p>
                  <p><strong>图像尺寸:</strong> 640</p>
                  <p><strong>优化器:</strong> SGD</p>
                  <p><strong>学习率:</strong> 0.01</p>
                </div>
              </Card>

              <Card>
                <CardHeader icon="⏱️" title="时间信息" />
                <div style={{ fontSize: '0.875rem' }}>
                  <p><strong>开始时间:</strong> 2024-01-15 10:30:00</p>
                  <p><strong>已运行:</strong> 2小时 15分钟</p>
                  <p><strong>预计剩余:</strong> 3小时 25分钟</p>
                </div>
              </Card>
            </div>
          </div>

          <Card style={{ marginTop: 'var(--space-6)' }}>
            <CardHeader icon="📊" title="实时日志" />
            <div style={{ fontFamily: 'monospace', fontSize: '0.75rem', background: '#1e1e1e', color: '#d4d4d4', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', maxHeight: '200px', overflow: 'auto' }}>
              <p>[10:30:00] Starting training...</p>
              <p>[10:30:05] Loading model: yolo11m.pt</p>
              <p>[10:30:10] Dataset: coco128 (128 images)</p>
              <p>[10:30:15] Training started with batch size 16</p>
              <p>[10:35:00] Epoch 1/100 - loss: 2.345 - mAP: 0.123</p>
              <p>[10:40:00] Epoch 2/100 - loss: 1.876 - mAP: 0.234</p>
              <p>...</p>
              <p>[12:45:00] Epoch 45/100 - loss: 0.234 - mAP: 0.823</p>
            </div>
          </Card>
        </>
      ) : (
        <Card>
          <CardHeader icon="📈" title="实时训练指标" />
          <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-4)' }}>
              当前没有正在进行的训练任务
            </p>
            <Button variant="primary">前往训练页面</Button>
          </div>
        </Card>
      )}
    </div>
  );
};
