import React from 'react';
import type { SAMAnnotation } from '../../types';

interface Props {
  annotations: SAMAnnotation[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onClassChange: (id: string, newClass: string) => void;
  onExport: () => void;
  classes: string[];
}

export const AnnotationPanel: React.FC<Props> = ({
  annotations,
  selectedId,
  onSelect,
  onDelete,
  onClassChange,
  onExport,
  classes,
}) => {
  // 按类别分组统计
  const stats = annotations.reduce((acc, ann, idx) => {
    const key = ann.class;
    if (!acc[key]) acc[key] = [];
    acc[key].push({ ...ann, idx });
    return acc;
  }, {} as Record<string, (SAMAnnotation & { idx: number })[]>);

  return (
    <div style={{
      width: '300px',
      background: '#fff',
      borderRadius: '8px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      display: 'flex',
      flexDirection: 'column',
      maxHeight: '600px',
    }}>
      {/* 头部 */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid #e2e8f0',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <h3 style={{ margin: 0, fontSize: '16px' }}>标注结果</h3>
        <span style={{ fontSize: '12px', color: '#64748b' }}>
          {annotations.length} 个对象
        </span>
      </div>

      {/* 标注列表 */}
      <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
        {annotations.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '32px 16px',
            color: '#94a3b8',
          }}>
            <div style={{ fontSize: '32px', marginBottom: '8px' }}>📦</div>
            <p>暂无标注</p>
            <p style={{ fontSize: '12px' }}>
              使用点标注或框选工具开始
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {annotations.map((ann, idx) => (
              <div
                key={idx}
                onClick={() => onSelect(String(idx))}
                style={{
                  padding: '10px 12px',
                  borderRadius: '6px',
                  border: selectedId === String(idx) ? '2px solid #3b82f6' : '1px solid #e2e8f0',
                  background: selectedId === String(idx) ? '#eff6ff' : '#fff',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 500, fontSize: '14px' }}>{ann.class}</span>
                  <span style={{ fontSize: '12px', color: '#64748b' }}>
                    {ann.confidence?.toFixed(2)}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                  {ann.bbox ? `框: [${ann.bbox.slice(0, 2).join(', ')}]` : '无边界框'}
                </div>
                <div style={{ display: 'flex', gap: '4px', marginTop: '8px' }}>
                  <select
                    value={ann.class}
                    onChange={(e) => {
                      e.stopPropagation();
                      onClassChange(String(idx), e.target.value);
                    }}
                    onClick={(e) => e.stopPropagation()}
                    style={{
                      flex: 1,
                      padding: '4px',
                      fontSize: '12px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '4px',
                    }}
                  >
                    {classes.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm('确定要删除这个标注吗？')) {
                        onDelete(String(idx));
                      }
                    }}
                    style={{
                      padding: '4px 8px',
                      border: 'none',
                      borderRadius: '4px',
                      background: '#fee2e2',
                      color: '#ef4444',
                      cursor: 'pointer',
                      fontSize: '12px',
                    }}
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 底部操作 */}
      {annotations.length > 0 && (
        <div style={{
          padding: '12px 16px',
          borderTop: '1px solid #e2e8f0',
        }}>
          <button
            onClick={onExport}
            style={{
              width: '100%',
              padding: '10px',
              border: 'none',
              borderRadius: '6px',
              background: '#3b82f6',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 500,
            }}
          >
            📥 导出 YOLO 格式
          </button>
        </div>
      )}
    </div>
  );
};

export default AnnotationPanel;
