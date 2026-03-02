import React, { useEffect, useState } from 'react';
import { Card, CardHeader, Button, StatCard } from '../../components/common';
import { modelApi } from '../../services/api';
import type { Model } from '../../types';
import styles from './Models.module.css';

export const Models: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const res = await modelApi.list();
      setModels(res.data?.data || []);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-6)' }}>
        <h1 style={{ fontFamily: 'DM Sans', fontWeight: 700 }}>模型管理</h1>
        <Button variant="primary">上传模型</Button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        <StatCard value={models.length} label="本地模型" />
        <StatCard value="YOLO11" label="最新版本" variant="accent" />
        <StatCard value="0" label="训练中" variant="warning" />
        <StatCard value="完成" label="训练完成" variant="success" />
      </div>

      {loading ? (
        <Card>
          <p style={{ color: 'var(--text-secondary)' }}>加载中...</p>
        </Card>
      ) : models.length === 0 ? (
        <Card>
          <CardHeader icon="🤖" title="模型列表" />
          <div style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-4)' }}>
              暂无本地模型，请上传 YOLO 模型文件 (.pt)
            </p>
            <Button variant="primary">上传模型</Button>
          </div>
        </Card>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--space-4)' }}>
          {models.map((model, idx) => (
            <Card key={idx} variant="elevated">
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
                <span style={{ fontSize: '1.5rem' }}>🤖</span>
                <div>
                  <h3 style={{ fontWeight: 600, margin: 0 }}>{model.name}</h3>
                  <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>{model.model_type || 'YOLO'}</span>
                </div>
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-3)' }}>
                <p>路径: {model.path}</p>
                <p>任务: {model.task || '检测'}</p>
                {model.classes && <p>类别数: {model.classes.length}</p>}
              </div>
              <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                <Button variant="secondary" size="sm">推理</Button>
                <Button variant="ghost" size="sm">删除</Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};
