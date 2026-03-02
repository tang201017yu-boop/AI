import React, { useState } from 'react';
import { Card, CardHeader, Button, Input } from '../../components/common';

export const Augmentation: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState('');
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  const augmentations = [
    { name: 'flip', label: '水平翻转', icon: '↔️' },
    { name: 'flip_vertical', label: '垂直翻转', icon: '↕️' },
    { name: 'rotate', label: '旋转', icon: '🔄' },
    { name: 'scale', label: '缩放', icon: '🔍' },
    { name: 'brightness', label: '亮度调整', icon: '☀️' },
    { name: 'contrast', label: '对比度调整', icon: '◐' },
    { name: 'saturation', label: '饱和度调整', icon: '🎨' },
    { name: 'blur', label: '模糊', icon: '🌫️' },
    { name: 'noise', label: '噪声', icon: '📺' },
    { name: 'crop', label: '裁剪', icon: '✂️' },
  ];

  return (
    <div>
      <h1 style={{ marginBottom: 'var(--space-6)', fontFamily: 'DM Sans', fontWeight: 700 }}>数据增强</h1>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 'var(--space-6)' }}>
        <Card>
          <CardHeader icon="🖼️" title="增强预览" />
          {previewImage ? (
            <div style={{ textAlign: 'center' }}>
              <img src={previewImage} alt="Preview" style={{ maxWidth: '100%', borderRadius: 'var(--radius-md)' }} />
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 'var(--space-8)', color: 'var(--text-secondary)' }}>
              <p>请选择数据集和增强选项</p>
            </div>
          )}
        </Card>

        <div>
          <Card style={{ marginBottom: 'var(--space-4)' }}>
            <CardHeader icon="📊" title="选择数据集" />
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              style={{ width: '100%', padding: 'var(--space-2)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
            >
              <option value="">请选择数据集</option>
              <option value="workers">安全装备数据集</option>
            </select>
          </Card>

          <Card>
            <CardHeader icon="⚙️" title="增强选项" />
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 'var(--space-2)' }}>
              {augmentations.map((aug) => (
                <label key={aug.name} style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', cursor: 'pointer', padding: 'var(--space-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}>
                  <input type="checkbox" />
                  <span>{aug.icon} {aug.label}</span>
                </label>
              ))}
            </div>

            <div style={{ marginTop: 'var(--space-4)' }}>
              <Input type="number" label="增强倍数" value={5} min={1} max={20} />
            </div>

            <div style={{ marginTop: 'var(--space-4)', display: 'flex', gap: 'var(--space-2)' }}>
              <Button variant="primary" style={{ flex: 1 }}>生成增强数据</Button>
              <Button variant="secondary" style={{ flex: 1 }}>预览</Button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
