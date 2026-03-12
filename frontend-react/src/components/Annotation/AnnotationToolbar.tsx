import React, { useState, useEffect } from 'react';
import type { AnnotationTool, ClassSuggestion } from '../../types';

interface Props {
  tool: AnnotationTool;
  currentClass: string;
  classes: string[];
  suggestions?: ClassSuggestion[];
  onToolChange: (tool: AnnotationTool) => void;
  onClassChange: (className: string) => void;
  onAddClass?: (className: string) => void;
  onAutoLabel: () => void;
  onClear: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onDeleteSelected?: () => void;
  onSave?: () => void;
  loading?: boolean;
  shortcuts?: { key: string; desc: string }[];
}

export const AnnotationToolbar: React.FC<Props> = ({
  tool,
  currentClass,
  classes,
  suggestions = [],
  onToolChange,
  onClassChange,
  onAddClass,
  onAutoLabel,
  onClear,
  onUndo,
  onRedo,
  onDeleteSelected,
  onSave,
  loading = false,
  shortcuts = [
    { key: 'P', desc: '点标注' },
    { key: 'B', desc: '框选' },
    { key: 'S', desc: '选择' },
    { key: 'A', desc: '自动标注' },
    { key: 'Ctrl+Z', desc: '撤销' },
    { key: 'Ctrl+Y', desc: '重做' },
    { key: 'Del', desc: '删除选中' },
  ],
}) => {
  const [showAddClass, setShowAddClass] = useState(false);
  const [newClassName, setNewClassName] = useState('');
  const [showShortcuts, setShowShortcuts] = useState(false);

  const tools: { id: AnnotationTool; label: string; icon: string; shortcut: string }[] = [
    { id: 'select', label: '选择', icon: '👆', shortcut: 'S' },
    { id: 'point', label: '点标注', icon: '📍', shortcut: 'P' },
    { id: 'box', label: '框选', icon: '⬜', shortcut: 'B' },
    { id: 'polygon', label: '多边形', icon: '🔺', shortcut: 'L' },
    { id: 'auto', label: '自动', icon: '🤖', shortcut: 'A' },
  ];

  // 处理添加新类别
  const handleAddClass = () => {
    if (newClassName.trim() && onAddClass) {
      onAddClass(newClassName.trim());
      setNewClassName('');
      setShowAddClass(false);
    }
  };

  // 随机颜色生成
  const getRandomColor = () => {
    const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      padding: '12px 16px',
      background: '#fff',
      borderRadius: '8px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      flexWrap: 'wrap',
    }}>
      {/* 工具选择 */}
      <div style={{ display: 'flex', gap: '4px' }}>
        {tools.map((t) => (
          <button
            key={t.id}
            onClick={() => onToolChange(t.id)}
            disabled={loading}
            title={`${t.label} (${t.shortcut})`}
            style={{
              padding: '8px 12px',
              border: 'none',
              borderRadius: '6px',
              background: tool === t.id ? '#3b82f6' : '#f1f5f9',
              color: tool === t.id ? '#fff' : '#475569',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              position: 'relative',
            }}
          >
            <span>{t.icon}</span>
            <span style={{ fontWeight: 500 }}>{t.label}</span>
            <span style={{
              position: 'absolute',
              top: '-6px',
              right: '-6px',
              fontSize: '10px',
              background: '#1e293b',
              color: '#fff',
              padding: '2px 4px',
              borderRadius: '3px',
            }}>
              {t.shortcut}
            </span>
          </button>
        ))}
      </div>

      {/* 分隔线 */}
      <div style={{ width: '1px', height: '28px', background: '#e2e8f0' }} />

      {/* 类别选择 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <label style={{ fontSize: '13px', color: '#64748b', fontWeight: 500 }}>类别:</label>
        <div style={{ position: 'relative' }}>
          <select
            value={currentClass}
            onChange={(e) => onClassChange(e.target.value)}
            style={{
              padding: '6px 28px 6px 12px',
              border: '1px solid #e2e8f0',
              borderRadius: '6px',
              fontSize: '13px',
              background: '#fff',
              minWidth: '120px',
              cursor: 'pointer',
            }}
          >
            {classes.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          {/* 添加类别按钮 */}
          <button
            onClick={() => setShowAddClass(!showAddClass)}
            style={{
              position: 'absolute',
              right: '2px',
              top: '50%',
              transform: 'translateY(-50%)',
              border: 'none',
              background: 'transparent',
              cursor: 'pointer',
              fontSize: '16px',
              color: '#64748b',
              padding: '4px',
            }}
            title="添加新类别"
          >
            +
          </button>
        </div>

        {/* 添加类别弹窗 */}
        {showAddClass && (
          <div style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            marginTop: '4px',
            padding: '8px',
            background: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
            zIndex: 100,
            display: 'flex',
            gap: '4px',
          }}>
            <input
              type="text"
              value={newClassName}
              onChange={(e) => setNewClassName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddClass()}
              placeholder="新类别名"
              autoFocus
              style={{
                padding: '6px 10px',
                border: '1px solid #e2e8f0',
                borderRadius: '4px',
                fontSize: '13px',
                width: '100px',
              }}
            />
            <button
              onClick={handleAddClass}
              style={{
                padding: '6px 10px',
                border: 'none',
                borderRadius: '4px',
                background: '#3b82f6',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '13px',
              }}
            >
              添加
            </button>
          </div>
        )}
      </div>

      {/* 类别建议 */}
      {suggestions.length > 0 && (
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', color: '#64748b' }}>建议:</span>
          {suggestions.slice(0, 4).map((s) => (
            <button
              key={s.name}
              onClick={() => onClassChange(s.name)}
              style={{
                padding: '4px 8px',
                border: 'none',
                borderRadius: '4px',
                background: s.color + '20',
                color: s.color,
                cursor: 'pointer',
                fontSize: '11px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: s.color }} />
              {s.name} ({s.count})
            </button>
          ))}
        </div>
      )}

      {/* 分隔线 */}
      <div style={{ width: '1px', height: '28px', background: '#e2e8f0' }} />

      {/* 操作按钮 */}
      <div style={{ display: 'flex', gap: '6px' }}>
        <button
          onClick={onUndo}
          disabled={loading}
          title="撤销 (Ctrl+Z)"
          style={{
            padding: '6px 10px',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            background: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '12px',
          }}
        >
          ↩️
        </button>
        <button
          onClick={onRedo}
          disabled={loading}
          title="重做 (Ctrl+Y)"
          style={{
            padding: '6px 10px',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            background: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '12px',
          }}
        >
          ↪️
        </button>
        {onDeleteSelected && (
          <button
            onClick={onDeleteSelected}
            disabled={loading}
            title="删除选中 (Del)"
            style={{
              padding: '6px 10px',
              border: '1px solid #fee2e2',
              borderRadius: '6px',
              background: '#fff',
              color: '#ef4444',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '12px',
            }}
          >
            🗑️
          </button>
        )}
        <button
          onClick={onClear}
          disabled={loading}
          title="清除所有"
          style={{
            padding: '6px 10px',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            background: '#fff',
            color: '#64748b',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '12px',
          }}
        >
          清空
        </button>
      </div>

      {/* 分隔线 */}
      <div style={{ width: '1px', height: '28px', background: '#e2e8f0' }} />

      {/* 自动标注按钮 */}
      <button
        onClick={onAutoLabel}
        disabled={loading}
        style={{
          padding: '8px 16px',
          border: 'none',
          borderRadius: '6px',
          background: loading ? '#94a3b8' : '#10b981',
          color: '#fff',
          cursor: loading ? 'not-allowed' : 'pointer',
          fontSize: '13px',
          fontWeight: 500,
        }}
      >
        {loading ? '⏳ 处理中...' : '🤖 自动标注'}
      </button>

      {/* 保存按钮 */}
      {onSave && (
        <button
          onClick={onSave}
          disabled={loading}
          style={{
            padding: '8px 16px',
            border: 'none',
            borderRadius: '6px',
            background: '#3b82f6',
            color: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '13px',
            fontWeight: 500,
          }}
        >
          💾 保存
        </button>
      )}

      {/* 快捷键帮助 */}
      <button
        onClick={() => setShowShortcuts(!showShortcuts)}
        style={{
          padding: '6px 10px',
          border: '1px solid #e2e8f0',
          borderRadius: '6px',
          background: '#fff',
          cursor: 'pointer',
          fontSize: '12px',
          color: '#64748b',
        }}
        title="快捷键帮助"
      >
        ⌨️
      </button>

      {/* 快捷键弹窗 */}
      {showShortcuts && (
        <div style={{
          position: 'absolute',
          top: '100%',
          right: 0,
          marginTop: '8px',
          padding: '16px',
          background: '#fff',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
          zIndex: 100,
          minWidth: '200px',
        }}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>快捷键</h4>
          <div style={{ display: 'grid', gap: '8px' }}>
            {shortcuts.map((s) => (
              <div key={s.key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                <span style={{ color: '#64748b' }}>{s.desc}</span>
                <span style={{
                  background: '#f1f5f9',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '11px',
                }}>
                  {s.key}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AnnotationToolbar;
