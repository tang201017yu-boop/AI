import React from 'react';
import type { SAMAnnotation } from '../../types';

interface Props {
  annotations: SAMAnnotation[];
  selectedId: string | null;
  hoveredId?: string | null;
  onSelect: (id: string) => void;
  onHover?: (id: string | null) => void;
  onDelete: (id: string) => void;
  onClassChange: (id: string, newClass: string) => void;
  onBulkClassChange?: (ids: string[], newClass: string) => void;
  onBulkDelete?: (ids: string[]) => void;
  onSelectionIdsChange?: (ids: string[]) => void;
  onExport: () => void;
  classes: string[];
  variant?: 'card' | 'studio';
}

export const AnnotationPanel: React.FC<Props> = ({
  annotations,
  selectedId,
  hoveredId = null,
  onSelect,
  onHover,
  onDelete,
  onClassChange,
  onBulkClassChange,
  onBulkDelete,
  onSelectionIdsChange,
  onExport,
  classes,
  variant = 'card',
}) => {
  const studio = variant === 'studio';
  const line = studio ? '#3c3c3c' : '#e2e8f0';
  const bg = studio ? '#252526' : '#fff';
  const bg2 = studio ? '#2d2d2d' : '#f8fafc';
  const surface = studio ? '#2d2d2d' : '#fff';
  const fgTitle = studio ? '#fafafa' : 'inherit';
  const fgMuted = studio ? '#a1a1aa' : '#64748b';
  const rowIdleBg = studio ? '#2d2d2d' : '#fff';
  const rowSelBg = studio ? '#1e3a5f' : '#eff6ff';
  const rowHoverBg = studio ? '#3f3f1e' : '#fffbeb';

  const [selectedIds, setSelectedIds] = React.useState<string[]>([]);
  const [bulkClass, setBulkClass] = React.useState<string>(classes[0] || '');
  const [lastAnchorIdx, setLastAnchorIdx] = React.useState<number | null>(null);

  React.useEffect(() => {
    setBulkClass((prev) => prev || classes[0] || '');
  }, [classes]);

  React.useEffect(() => {
    // 标注列表变化后，剔除失效索引
    setSelectedIds((prev) => prev.filter((id) => {
      const idx = parseInt(id, 10);
      return !Number.isNaN(idx) && idx >= 0 && idx < annotations.length;
    }));
  }, [annotations.length]);

  React.useEffect(() => {
    onSelectionIdsChange?.(selectedIds);
  }, [selectedIds, onSelectionIdsChange]);

  const toggleMultiSelect = (id: string) => {
    setSelectedIds((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
  };

  const handleRowClick = (idx: number, e: React.MouseEvent<HTMLDivElement>) => {
    const id = String(idx);
    const isCtrlLike = e.ctrlKey || e.metaKey;
    const isShift = e.shiftKey;

    if (isShift && lastAnchorIdx !== null) {
      const start = Math.min(lastAnchorIdx, idx);
      const end = Math.max(lastAnchorIdx, idx);
      const range = Array.from({ length: end - start + 1 }, (_, i) => String(start + i));
      setSelectedIds((prev) => Array.from(new Set([...prev, ...range])));
      onSelect(id);
      return;
    }

    if (isCtrlLike) {
      toggleMultiSelect(id);
      onSelect(id);
      setLastAnchorIdx(idx);
      return;
    }

    setSelectedIds([id]);
    onSelect(id);
    setLastAnchorIdx(idx);
  };

  const selectAll = () => {
    setSelectedIds(annotations.map((_, i) => String(i)));
  };

  const clearSelect = () => setSelectedIds([]);

  const applyBulkClass = () => {
    if (!onBulkClassChange || selectedIds.length === 0 || !bulkClass) return;
    onBulkClassChange(selectedIds, bulkClass);
    clearSelect();
  };

  const applyBulkDelete = () => {
    if (!onBulkDelete || selectedIds.length === 0) return;
    if (!window.confirm(`确定删除选中的 ${selectedIds.length} 条标注吗？`)) return;
    onBulkDelete(selectedIds);
    clearSelect();
  };

  // 按类别分组统计
  const stats = annotations.reduce((acc, ann, idx) => {
    const key = ann.class;
    if (!acc[key]) acc[key] = [];
    acc[key].push({ ...ann, idx });
    return acc;
  }, {} as Record<string, (SAMAnnotation & { idx: number })[]>);

  return (
    <div style={{
      width: studio ? '100%' : '300px',
      background: bg,
      borderRadius: studio ? 0 : 8,
      boxShadow: studio ? 'none' : '0 1px 3px rgba(0,0,0,0.1)',
      display: 'flex',
      flexDirection: 'column',
      maxHeight: studio ? 'none' : '600px',
      minHeight: studio ? 0 : undefined,
      flex: studio ? 1 : undefined,
      height: studio ? '100%' : undefined,
    }}>
      {/* 头部 */}
      <div style={{
        padding: '12px 16px',
        borderBottom: `1px solid ${line}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <h3 style={{ margin: 0, fontSize: '16px', color: fgTitle }}>Objects</h3>
        <span style={{ fontSize: '12px', color: fgMuted }}>
          {annotations.length} 个对象
        </span>
      </div>

      {/* 批量操作区 */}
      {annotations.length > 0 && (
        <div style={{
          padding: '10px 12px',
          borderBottom: `1px solid ${line}`,
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          background: bg2,
        }}>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button onClick={selectAll} style={{ fontSize: '12px', border: `1px solid ${line}`, borderRadius: '4px', background: surface, color: fgTitle, padding: '4px 8px', cursor: 'pointer' }}>
              全选
            </button>
            <button onClick={clearSelect} style={{ fontSize: '12px', border: `1px solid ${line}`, borderRadius: '4px', background: surface, color: fgTitle, padding: '4px 8px', cursor: 'pointer' }}>
              清空选择
            </button>
            <span style={{ marginLeft: 'auto', fontSize: '12px', color: fgMuted }}>
              已选 {selectedIds.length}
            </span>
          </div>
          <div style={{ display: 'flex', gap: '6px' }}>
            <select
              value={bulkClass}
              onChange={(e) => setBulkClass(e.target.value)}
              style={{ flex: 1, padding: '4px', fontSize: '12px', border: `1px solid ${line}`, borderRadius: '4px', background: surface, color: fgTitle }}
            >
              {classes.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <button
              onClick={applyBulkClass}
              disabled={selectedIds.length === 0}
              style={{
                fontSize: '12px',
                border: 'none',
                borderRadius: '4px',
                background: selectedIds.length ? '#3b82f6' : '#cbd5e1',
                color: '#fff',
                padding: '4px 8px',
                cursor: selectedIds.length ? 'pointer' : 'not-allowed',
              }}
            >
              批量改类
            </button>
            <button
              onClick={applyBulkDelete}
              disabled={selectedIds.length === 0}
              style={{
                fontSize: '12px',
                border: 'none',
                borderRadius: '4px',
                background: selectedIds.length ? '#ef4444' : '#cbd5e1',
                color: '#fff',
                padding: '4px 8px',
                cursor: selectedIds.length ? 'pointer' : 'not-allowed',
              }}
            >
              批量删除
            </button>
          </div>
        </div>
      )}

      {/* 标注列表 */}
      <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
        {annotations.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '32px 16px',
            color: fgMuted,
          }}>
            <div style={{ fontSize: '32px', marginBottom: '8px' }}>📦</div>
            <p style={{ color: fgTitle }}>暂无标注</p>
            <p style={{ fontSize: '12px' }}>
              使用点标注或框选工具开始
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {annotations.map((ann, idx) => (
              <div
                key={idx}
                onClick={(e) => handleRowClick(idx, e)}
                onMouseEnter={() => onHover?.(String(idx))}
                onMouseLeave={() => onHover?.(null)}
                style={{
                  padding: '10px 12px',
                  borderRadius: '6px',
                  border:
                    selectedId === String(idx)
                      ? '2px solid #3b82f6'
                      : hoveredId === String(idx)
                        ? '1px solid #f59e0b'
                        : `1px solid ${line}`,
                  background:
                    selectedId === String(idx)
                      ? rowSelBg
                      : hoveredId === String(idx)
                        ? rowHoverBg
                        : rowIdleBg,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
              >
                <div style={{ marginBottom: '6px' }}>
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(String(idx))}
                    onChange={(e) => {
                      e.stopPropagation();
                      toggleMultiSelect(String(idx));
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 500, fontSize: '14px', color: fgTitle }}>{ann.class}</span>
                  <span style={{ fontSize: '12px', color: fgMuted }}>
                    {ann.confidence != null ? ann.confidence.toFixed(2) : '—'}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: fgMuted, marginTop: '4px' }}>
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
                      border: `1px solid ${line}`,
                      borderRadius: '4px',
                      background: surface,
                      color: fgTitle,
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
          borderTop: `1px solid ${line}`,
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
