import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardHeader, Button, Input, StatCard } from '../../components/common';
import { datasetApi, trainingApi, modelApi } from '../../services/api';
import type { Dataset, TrainingConfig, Model } from '../../types';

interface PretrainedModel {
  name: string;
  display: string;
  installed: boolean;
  path?: string;
}

export const Training: React.FC = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [pretrainedModels, setPretrainedModels] = useState<PretrainedModel[]>([]);
  const [trainedModels, setTrainedModels] = useState<Model[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [modelSource, setModelSource] = useState<'pretrained' | 'trained'>('pretrained');
  const [selectedPretrainedModel, setSelectedPretrainedModel] = useState('yolo11m');
  const [selectedTrainedModel, setSelectedTrainedModel] = useState('');
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [imageSize, setImageSize] = useState(640);
  const [learningRate, setLearningRate] = useState(0.01);
  const [training, setTraining] = useState(false);
  const [taskId, setTaskId] = useState('');
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

  useEffect(() => {
    loadDatasets();
    loadPretrainedModels();
    loadTrainedModels();
  }, []);

  // 定期检查训练状态
  useEffect(() => {
    if (!currentTaskId) return;

    const interval = setInterval(async () => {
      try {
        const res = await trainingApi.status(currentTaskId);
        const status = res.data?.data?.status || res.data?.status;
        if (status === 'completed' || status === 'failed' || status === 'cancelled') {
          setTraining(false);
          setCurrentTaskId(null);
          clearInterval(interval);
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
      setDatasets(res.data?.datasets || res.data?.data?.datasets || []);
    } catch (error) {
      console.error(error);
    }
  };

  const loadPretrainedModels = async () => {
    try {
      const res = await modelApi.getPretrainedModels();
      setPretrainedModels(res.data?.data?.models || res.data?.models || []);
    } catch (error) {
      console.error(error);
    }
  };

  const loadTrainedModels = async () => {
    try {
      const res = await modelApi.list();
      setTrainedModels(res.data?.models || res.data?.data?.models || []);
    } catch (error) {
      console.error(error);
    }
  };

  const handleStartTraining = async () => {
    if (!selectedDataset) return;
    if (modelSource === 'pretrained' && !selectedPretrainedModel) return;
    if (modelSource === 'trained' && !selectedTrainedModel) return;

    setTraining(true);
    try {
      const config: TrainingConfig = {
        project_name: `train_${Date.now()}`,
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

      // 跳转到监控页面
      setTimeout(() => {
        navigate('/training-monitor');
      }, 1000);
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

  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>模型训练</h1>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        <StatCard value={datasets.length} label="可用数据集" />
        <StatCard value={pretrainedModels.filter(m => m.installed).length} label="已安装基础模型" variant="accent" />
        <StatCard value={trainedModels.length} label="训练完成模型" variant="success" />
        <StatCard value={training ? '训练中' : '空闲'} label="训练状态" variant={training ? 'warning' : 'default'} />
      </div>

      <Card>
        <CardHeader icon="🚀" title="训练配置" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 'var(--space-4)' }}>
          <div>
            <label style={{ display: 'block', marginBottom: 'var(--space-1)', fontWeight: 500 }}>选择数据集</label>
            <select
              className="input"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              disabled={training}
              style={{ width: '100%', padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
            >
              <option value="">请选择数据集</option>
              {datasets.map((ds) => (
                <option key={ds.name} value={ds.name}>{ds.name} ({ds.num_images || ds.image_count || 0} 张图片)</option>
              ))}
            </select>
          </div>

          {/* 模型来源选择 */}
          <div>
            <label style={{ display: 'block', marginBottom: 'var(--space-1)', fontWeight: 500 }}>模型来源</label>
            <select
              className="input"
              value={modelSource}
              onChange={(e) => setModelSource(e.target.value as 'pretrained' | 'trained')}
              disabled={training}
              style={{ width: '100%', padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
            >
              <option value="pretrained">预训练基础模型</option>
              <option value="trained">用户训练模型</option>
            </select>
          </div>

          {/* 预训练模型选择 */}
          {modelSource === 'pretrained' && (
            <div style={{ gridColumn: '1 / -1' }}>
              <label style={{ display: 'block', marginBottom: 'var(--space-1)', fontWeight: 500 }}>选择预训练模型</label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 'var(--space-2)' }}>
                {pretrainedModels.map((model) => (
                  <button
                    key={model.name}
                    onClick={() => setSelectedPretrainedModel(model.name)}
                    disabled={training}
                    style={{
                      padding: 'var(--space-2)',
                      border: selectedPretrainedModel === model.name ? '2px solid var(--color-primary)' : '1px solid var(--border)',
                      borderRadius: 'var(--radius-md)',
                      background: selectedPretrainedModel === model.name ? 'var(--color-primary-light)' : 'var(--bg-primary)',
                      cursor: training ? 'not-allowed' : 'pointer',
                      opacity: training ? 0.6 : 1,
                      textAlign: 'left'
                    }}
                  >
                    <div style={{ fontWeight: 500, fontSize: '14px' }}>{model.name}</div>
                    <div style={{ fontSize: '12px', color: model.installed ? 'var(--color-success)' : 'var(--text-secondary)' }}>
                      {model.installed ? '✓ 已安装' : '○ 将下载'}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 用户训练模型选择 */}
          {modelSource === 'trained' && (
            <div style={{ gridColumn: '1 / -1' }}>
              <label style={{ display: 'block', marginBottom: 'var(--space-1)', fontWeight: 500 }}>选择训练好的模型</label>
              <select
                className="input"
                value={selectedTrainedModel}
                onChange={(e) => setSelectedTrainedModel(e.target.value)}
                disabled={training}
                style={{ width: '100%', padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
              >
                <option value="">请选择模型</option>
                {trainedModels.map((model) => (
                  <option key={model.id || model.name} value={model.name}>
                    {model.name} {model.mAP50 ? `(mAP: ${(model.mAP50 * 100).toFixed(1)}%)` : ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          <Input
            type="number"
            label="训练轮数 (Epochs)"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value))}
            min={1}
            max={1000}
            disabled={training}
          />

          <Input
            type="number"
            label="批量大小 (Batch Size)"
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            min={1}
            max={128}
            disabled={training}
          />

          <Input
            type="number"
            label="图像尺寸"
            value={imageSize}
            onChange={(e) => setImageSize(parseInt(e.target.value))}
            min={320}
            max={1280}
            step={32}
            disabled={training}
          />

          <Input
            type="number"
            label="学习率"
            value={learningRate}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            min={0.0001}
            max={0.1}
            step={0.001}
            disabled={training}
          />
        </div>

        <div style={{ marginTop: 'var(--space-6)', display: 'flex', gap: 'var(--space-3)' }}>
          <Button
            variant="primary"
            size="lg"
            onClick={handleStartTraining}
            disabled={!selectedDataset || training || (modelSource === 'pretrained' && !selectedPretrainedModel) || (modelSource === 'trained' && !selectedTrainedModel)}
            loading={training}
          >
            {training ? '训练中...' : '开始训练'}
          </Button>
          {training && (
            <Button variant="danger" size="lg" onClick={handleStopTraining}>
              停止训练
            </Button>
          )}
          <Button
            variant="secondary"
            size="lg"
            onClick={() => navigate('/training-monitor')}
          >
            查看监控
          </Button>
        </div>
      </Card>
    </div>
  );
};
