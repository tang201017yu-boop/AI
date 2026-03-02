import React, { useState, useEffect } from 'react';
import { Card, CardHeader, Button, StatCard } from '../../components/common';
import { annotationApi, datasetApi } from '../../services/api';

export const Projects: React.FC = () => {
  const [annotationProjects, setAnnotationProjects] = useState<any[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [annoRes, datasetRes] = await Promise.all([
        annotationApi.listProjects(),
        datasetApi.list(),
      ]);
      setAnnotationProjects(annoRes.data?.data || annoRes.data || []);
      setDatasets(datasetRes.data?.datasets || datasetRes.data?.data?.datasets || []);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>项目管理</h1>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
        <StatCard value={annotationProjects.length + datasets.length} label="总项目数" />
        <StatCard value={annotationProjects.length} label="标注项目" variant="accent" />
        <StatCard value={datasets.length} label="数据集" variant="success" />
        <StatCard value="活跃" label="项目状态" />
      </div>

      <Card style={{ marginBottom: 'var(--space-6)' }}>
        <CardHeader icon="✏️" title="标注项目" />
        {loading ? (
          <p style={{ color: 'var(--text-secondary)' }}>加载中...</p>
        ) : annotationProjects.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>暂无标注项目</p>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 'var(--space-4)' }}>
            {annotationProjects.map((project) => (
              <Card key={project.id} variant="elevated">
                <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-1)' }}>{project.name}</h4>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
                  {project.description || '无描述'}
                </p>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  创建于: {new Date(project.created_at).toLocaleDateString()}
                </p>
              </Card>
            ))}
          </div>
        )}
      </Card>

      <Card>
        <CardHeader icon="📊" title="数据集" />
        {loading ? (
          <p style={{ color: 'var(--text-secondary)' }}>加载中...</p>
        ) : datasets.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>暂无数据集</p>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 'var(--space-4)' }}>
            {datasets.map((dataset, idx) => (
              <Card key={idx} variant="elevated">
                <h4 style={{ fontWeight: 600, marginBottom: 'var(--space-1)' }}>{dataset.name}</h4>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
                  任务类型: {dataset.task_type || '检测'}
                </p>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  图片数: {dataset.num_images || dataset.image_count || 0}
                </p>
                {dataset.classes && (
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    类别: {dataset.classes.join(', ')}
                  </p>
                )}
              </Card>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
};
