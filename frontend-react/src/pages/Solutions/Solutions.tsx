import React, { useEffect, useState } from 'react';
import { Card, CardHeader, Button } from '../../components/common';
import { solutionsApi } from '../../services/api';

interface Solution {
  name: string;
  title: string;
  description: string;
  input_types: string[];
  features: string[];
}

export const Solutions: React.FC = () => {
  const [solutions, setSolutions] = useState<Solution[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSolutions();
  }, []);

  const loadSolutions = async () => {
    try {
      const res = await solutionsApi.list();
      setSolutions(res.data?.solutions || res.data?.data?.solutions || []);
    } catch (error) {
      console.error(error);
      // 使用默认解决方案列表
      setSolutions([
        { name: 'object-counting', title: '对象计数', description: '统计区域对象数量', input_types: ['image', 'video'], features: ['区域计数', '进出统计', '分类计数'] },
        { name: 'heatmap', title: '热图生成', description: '可视化检测密度', input_types: ['image', 'video'], features: ['密度可视化', '热点分析'] },
        { name: 'speed-estimation', title: '速度估算', description: '计算移动对象速度', input_types: ['video'], features: ['实时测速', '速度统计'] },
        { name: 'distance-calculation', title: '距离计算', description: '测量对象间距离', input_types: ['image'], features: ['对象间距', '空间分析'] },
        { name: 'object-blur', title: '对象模糊', description: '对象模糊处理', input_types: ['image', 'video'], features: ['隐私保护', '人脸模糊'] },
        { name: 'object-crop', title: '对象裁剪', description: '自动提取检测对象', input_types: ['image'], features: ['自动裁剪', '批量提取'] },
        { name: 'queue-management', title: '队列管理', description: '监控队列长度', input_types: ['video'], features: ['队列计数', '流量分析'] },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const getIcon = (name: string) => {
    const icons: Record<string, string> = {
      'object-counting': '📊',
      'heatmap': '🔥',
      'speed-estimation': '🚗',
      'distance-calculation': '📏',
      'object-blur': '🔒',
      'object-crop': '✂️',
      'queue-management': '👥',
    };
    return icons[name] || '🎯';
  };

  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>智能解决方案</h1>

      {loading ? (
        <p style={{ color: 'var(--text-secondary)' }}>加载中...</p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--space-4)' }}>
          {solutions.map((item, idx) => (
            <Card key={idx} variant="elevated" style={{ cursor: 'pointer' }}>
              <div style={{ fontSize: '2rem', marginBottom: 'var(--space-2)' }}>{getIcon(item.name)}</div>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: 'var(--space-1)' }}>{item.title}</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: 'var(--space-3)' }}>{item.description}</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-1)' }}>
                {item.features?.slice(0, 3).map((feature, fIdx) => (
                  <span key={fIdx} style={{ fontSize: '0.75rem', background: 'var(--primary-100)', color: 'var(--primary-700)', padding: '2px 8px', borderRadius: '4px' }}>
                    {feature}
                  </span>
                ))}
              </div>
              <Button variant="primary" size="sm" style={{ marginTop: 'var(--space-3)', width: '100%' }}>
                使用
              </Button>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};
