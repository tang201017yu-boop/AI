import React, { useState, useEffect } from 'react';
import { Card, CardHeader, Button, Input, StatCard } from '../../components/common';
import { datasetApi, trainingApi } from '../../services/api';
import type { Dataset, TrainingConfig } from '../../types';

export const Training: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [modelType, setModelType] = useState('yolo11m');
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [imageSize, setImageSize] = useState(640);
  const [learningRate, setLearningRate] = useState(0.01);
  const [training, setTraining] = useState(false);
  const [taskId, setTaskId] = useState('');

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const res = await datasetApi.list();
      setDatasets(res.data?.datasets || res.data?.data?.datasets || []);
    } catch (error) {
      console.error(error);
    }
  };

  const handleStartTraining = async () => {
    if (!selectedDataset) return;
    setTraining(true);
    try {
      const config: TrainingConfig = {
        model_type: modelType,
        dataset_id: selectedDataset,
        epochs,
        batch_size: batchSize,
        image_size: imageSize,
        optimizer: 'SGD',
        learning_rate: learningRate,
      };
      const res = await trainingApi.start(config);
      setTaskId(res.data?.data?.task_id || res.data?.task_id || '');
    } catch (error) {
      console.error(error);
    } finally {
      setTraining(false);
    }
  };

  const modelTypes = [
    { value: 'yolo26n', label: 'YOLO26n (nano) ⚡', desc: '最新最快，适合CPU' },
    { value: 'yolo26s', label: 'YOLO26s (small)', desc: '最新快速，适合轻量级' },
    { value: 'yolo26m', label: 'YOLO26m (medium)', desc: '最新平衡，推荐' },
    { value: 'yolo26l', label: 'YOLO26l (large)', desc: '最新高精度' },
    { value: 'yolo26x', label: 'YOLO26x (xlarge)', desc: '最新最高精度' },
    { value: 'yolo11n', label: 'YOLO11n (nano)', desc: '经典最快，适合CPU' },
    { value: 'yolo11s', label: 'YOLO11s (small)', desc: '经典快速，适合轻量级' },
    { value: 'yolo11m', label: 'YOLO11m (medium)', desc: '经典平衡，推荐' },
    { value: 'yolo11l', label: 'YOLO11l (large)', desc: '经典高精度' },
    { value: 'yolo11x', label: 'YOLO11x (xlarge)', desc: '经典最高精度' },
  ];

  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>模型训练</h1>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        <StatCard value={datasets.length} label="可用数据集" />
        <StatCard value={modelTypes.length} label="模型类型" variant="accent" />
        <StatCard value={training ? '训练中' : '空闲'} label="训练状态" variant={training ? 'warning' : 'success'} />
        <StatCard value={taskId || '-'} label="当前任务" />
      </div>

      <Card>
        <CardHeader icon="🚀" title="训练配置" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 'var(--space-4)' }}>
          <div>
            <label style={{ display: 'block', marginBottom: 'var(--space-1)', fontWeight: 500 }}>模型类型</label>
            <select
              className="input"
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              style={{ width: '100%', padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
            >
              {modelTypes.map((m) => (
                <option key={m.value} value={m.value}>{m.label} - {m.desc}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 'var(--space-1)', fontWeight: 500 }}>选择数据集</label>
            <select
              className="input"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              style={{ width: '100%', padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
            >
              <option value="">请选择数据集</option>
              {datasets.map((ds) => (
                <option key={ds.name} value={ds.name}>{ds.name} ({ds.num_images || ds.image_count || 0} 张图片)</option>
              ))}
            </select>
          </div>

          <Input
            type="number"
            label="训练轮数 (Epochs)"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value))}
            min={1}
            max={1000}
          />

          <Input
            type="number"
            label="批量大小 (Batch Size)"
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            min={1}
            max={128}
          />

          <Input
            type="number"
            label="图像尺寸"
            value={imageSize}
            onChange={(e) => setImageSize(parseInt(e.target.value))}
            min={320}
            max={1280}
            step={32}
          />

          <Input
            type="number"
            label="学习率"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            min={0.0001}
            max={0.1}
            step={0.001}
          />
        </div>

        <div style={{ marginTop: 'var(--space-6)', display: 'flex', gap: 'var(--space-3)' }}>
          <Button
            variant="primary"
            size="lg"
            onClick={handleStartTraining}
            disabled={!selectedDataset || training}
            loading={training}
          >
            {training ? '训练中...' : '开始训练'}
          </Button>
          {training && (
            <Button variant="secondary" size="lg">
              停止训练
            </Button>
          )}
        </div>
      </Card>
    </div>
  );
};
