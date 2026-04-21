import React, { useState, useEffect, useRef } from 'react';
import type { AnnotationTool, ClassSuggestion } from '../../types';

/** 各类别稳定配色（接近 Ultralytics Hub / 工装安全数据集标注习惯） */
export function colorForClassName(name: string): string {
  const palette = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
    '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075',
  ];
  let h = 0;
  for (let i = 0; i < name.length; i++) {
    h = name.charCodeAt(i) + ((h << 5) - h);
  }
  return palette[Math.abs(h) % palette.length];
}

interface Props {
  tool: AnnotationTool;
  currentClass: string;
  classes: string[];
  /** 若提供，仅用于工具栏「类别色条」芯片；下拉仍用 classes（可选未在本图出现的类） */
  classPalette?: string[];
  suggestions?: ClassSuggestion[];
  samVersion?: 'sam2_lite' | 'sam2_base' | 'sam2_large' | 'sam3';
  onToolChange: (tool: AnnotationTool) => void;
  onClassChange: (className: string) => void;
  onAddClass?: (className: string) => void;
  onAutoLabel: () => void;
  onDetectAll?: () => void;
  detectAllModel?: string;
  detectAllConfidence?: number;
  onDetectAllModelChange?: (model: string) => void;
  onDetectAllConfidenceChange?: (conf: number) => void;
  availableModels?: string[];
  /** 训练/上传权重（value 为服务器可解析的 .pt 路径） */
  userDetectModels?: { path: string; label: string }[];
  onClear: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onDeleteSelected?: () => void;
  onSave?: () => void;
  /** 为 true 时保存按钮置灰（如未选项目），悬停可看 saveHint */
  saveDisabled?: boolean;
  saveHint?: string;
  onSamVersionChange?: (version: 'sam2_lite' | 'sam2_base' | 'sam2_large' | 'sam3') => void;
  loading?: boolean;
  shortcuts?: { key: string; desc: string }[];
  /** 框选完成后用当前 YOLO 模型推断类别（需模型含对应类，如自建安全帽数据集权重） */
  autoClassifyOnBox?: boolean;
  onAutoClassifyChange?: (enabled: boolean) => void;
  /** Draw=手绘优先排布；Smart=AI 一键/自动标注排前（功能相同，仅顺序） */
  workMode?: 'draw' | 'smart';
  /** studio=暗色三栏工作台（对齐 Ultralytics Hub 视觉层级） */
  variant?: 'card' | 'studio';
  /** 紧凑模式：减少工具栏留白，给画布更多空间 */
  compact?: boolean;
}

export const AnnotationToolbar: React.FC<Props> = ({
  tool,
  currentClass,
  classes,
  classPalette,
  suggestions = [],
  samVersion = 'sam2_base',
  onToolChange,
  onClassChange,
  onAddClass,
  onAutoLabel,
  onDetectAll,
  detectAllModel = 'yolo11n.pt',
  detectAllConfidence = 0.1,
  onDetectAllModelChange,
  onDetectAllConfidenceChange,
  availableModels = ['yolo11n.pt', 'yolo11m.pt', 'yolo26n.pt', 'yolo26m.pt'],
  userDetectModels = [],
  onClear,
  onUndo,
  onRedo,
  onDeleteSelected,
  onSave,
  saveDisabled = false,
  saveHint,
  onSamVersionChange,
  loading = false,
  autoClassifyOnBox = true,
  onAutoClassifyChange,
  workMode = 'draw',
  variant = 'card',
  compact = false,
  shortcuts = [
    { key: 'P', desc: '点标注' },
    { key: 'B', desc: '框选' },
    { key: 'S', desc: '选择' },
    { key: 'A', desc: '自动标注' },
    { key: 'Ctrl+Z', desc: '撤销' },
    { key: 'Ctrl+Y', desc: '重做' },
    { key: 'Del', desc: '删除选中标注' },
    { key: '1-9', desc: '快速切换类别' },
  ],
}) => {
  const chipClasses = classPalette !== undefined ? classPalette : classes;
  const [showAddClass, setShowAddClass] = useState(false);
  const [newClassName, setNewClassName] = useState('');
  const [showShortcuts, setShowShortcuts] = useState(false);
  const classPickerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showAddClass) return;
    const onDocDown = (e: MouseEvent) => {
      const el = classPickerRef.current;
      if (el && !el.contains(e.target as Node)) setShowAddClass(false);
    };
    document.addEventListener('mousedown', onDocDown);
    return () => document.removeEventListener('mousedown', onDocDown);
  }, [showAddClass]);

  const allTools: { id: AnnotationTool; label: string; icon: string; shortcut: string }[] = [
    { id: 'select', label: '选择', icon: '👆', shortcut: 'S' },
    { id: 'point', label: '点标注', icon: '📍', shortcut: 'P' },
    { id: 'box', label: '框选', icon: '⬜', shortcut: 'B' },
    { id: 'polygon', label: '多边形', icon: '🔺', shortcut: 'L' },
    { id: 'auto', label: '自动', icon: '🤖', shortcut: 'A' },
  ];
  const tools = workMode === 'draw' ? allTools.filter((t) => t.id !== 'auto') : allTools;
  const shortcutsForHelp =
    workMode === 'draw' ? shortcuts.filter((s) => s.desc !== '自动标注') : shortcuts;

  const studio = variant === 'studio';
  const line = studio ? '#3c3c3c' : '#e2e8f0';
  const surface = studio ? '#2d2d2d' : '#fff';
  const surface2 = studio ? '#3a3a3c' : '#f1f5f9';
  const fg = studio ? '#e4e4e7' : '#475569';
  const fgMuted = studio ? '#a1a1aa' : '#64748b';
  const compactGap = compact ? '6px' : '10px';
  const compactPad = compact ? (studio ? '6px 10px' : '8px 12px') : (studio ? '10px 12px' : '12px 16px');

  // 处理添加新类别
  const handleAddClass = () => {
    const name = newClassName.trim();
    if (!name) return;
    if (!onAddClass) {
      alert('当前页面未启用「自定义类别」，请联系开发者开启 onAddClass。');
      return;
    }
    onAddClass(name);
    setNewClassName('');
    setShowAddClass(false);
  };

  // 随机颜色生成
  const getRandomColor = () => {
    const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  const manualCluster = (
    <>
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
              background: tool === t.id ? '#3b82f6' : surface2,
              color: tool === t.id ? '#fff' : fg,
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
      <div style={{ width: '1px', height: '28px', background: line }} />

      {/* 类别选择（弹窗必须放在 position:relative 内部，否则 absolute 会相对整页定位，看起来像「加不了类别」） */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <label style={{ fontSize: '13px', color: fgMuted, fontWeight: 500 }}>类别:</label>
        <div ref={classPickerRef} style={{ position: 'relative', zIndex: showAddClass ? 50 : undefined }}>
          <select
            value={classes.includes(currentClass) ? currentClass : ''}
            onChange={(e) => onClassChange(e.target.value)}
            style={{
              padding: '6px 28px 6px 12px',
              border: `1px solid ${line}`,
              borderRadius: '6px',
              fontSize: '13px',
              background: surface,
              color: fg,
              minWidth: '120px',
              cursor: 'pointer',
            }}
          >
            <option value="">请选择类别</option>
            {classes.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setShowAddClass((v) => !v);
            }}
            style={{
              position: 'absolute',
              right: '2px',
              top: '50%',
              transform: 'translateY(-50%)',
              border: 'none',
              background: 'transparent',
              cursor: 'pointer',
              fontSize: '16px',
              color: fgMuted,
              padding: '4px',
            }}
            title="添加新类别"
          >
            +
          </button>

          {showAddClass && (
            <div
              role="dialog"
              aria-label="添加新类别"
              style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                marginTop: '6px',
                padding: '8px',
                background: surface,
                border: `1px solid ${line}`,
                borderRadius: '6px',
                boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
                zIndex: 2000,
                display: 'flex',
                gap: '6px',
                alignItems: 'center',
                whiteSpace: 'nowrap',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="text"
                value={newClassName}
                onChange={(e) => setNewClassName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddClass();
                  }
                }}
                placeholder="新类别名"
                autoFocus
                style={{
                  padding: '6px 10px',
                  border: `1px solid ${line}`,
                  borderRadius: '4px',
                  fontSize: '13px',
                  width: '140px',
                  background: studio ? '#1e1e1e' : '#fff',
                  color: fg,
                }}
              />
              <button
                type="button"
                onClick={() => handleAddClass()}
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
      </div>

      {/* 类别建议 */}
      {suggestions.length > 0 && (
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', color: fgMuted }}>建议:</span>
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
      <div style={{ width: '1px', height: '28px', background: line }} />

      {/* 操作按钮 */}
      <div style={{ display: 'flex', gap: '6px' }}>
        <button
          onClick={onUndo}
          disabled={loading}
          title="撤销 (Ctrl+Z)"
          style={{
            padding: '6px 10px',
            border: `1px solid ${line}`,
            borderRadius: '6px',
            background: surface,
            color: fg,
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
            border: `1px solid ${line}`,
            borderRadius: '6px',
            background: surface,
            color: fg,
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
            title="删除选中标注；未选中时移除当前图片"
            style={{
              padding: '6px 10px',
              border: `1px solid ${studio ? '#7f1d1d' : '#fee2e2'}`,
              borderRadius: '6px',
              background: surface,
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
            border: `1px solid ${line}`,
            borderRadius: '6px',
            background: surface,
            color: fgMuted,
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '12px',
          }}
        >
          清空
        </button>
      </div>
    </>
  );

  const saveBlocked = Boolean(loading || saveDisabled);
  const saveToolbarButton = onSave && (
    <button
      type="button"
      onClick={onSave}
      disabled={saveBlocked}
      title={
        loading
          ? '处理中…'
          : saveDisabled && saveHint
            ? saveHint
            : '将当前图的标注写入已选项目（服务器）'
      }
      style={{
        padding: '8px 16px',
        border: 'none',
        borderRadius: '6px',
        background: saveBlocked ? '#94a3b8' : '#3b82f6',
        color: '#fff',
        cursor: saveBlocked ? 'not-allowed' : 'pointer',
        fontSize: '13px',
        fontWeight: 500,
        opacity: saveBlocked ? 0.92 : 1,
      }}
    >
      💾 保存
    </button>
  );

  /** 一键 / 单类自动标注（仅 Smart 模式展示；Draw 模式只做手绘） */
  const smartAutoCluster = (
    <>
      <div style={{ width: '1px', height: '28px', background: line }} />

      {onDetectAll && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
          <select
            value={detectAllModel}
            onChange={(e) => onDetectAllModelChange?.(e.target.value)}
            disabled={loading}
            style={{ padding: '5px 8px', border: `1px solid ${line}`, borderRadius: '6px', fontSize: '12px', background: surface, color: fg, minWidth: '140px', maxWidth: 'min(260px, 36vw)' }}
            title="选择检测模型（可选自己训练的权重）"
          >
            {userDetectModels.length > 0 && (
              <optgroup label="我的模型">
                {userDetectModels.map((m) => (
                  <option key={m.path} value={m.path}>{m.label}</option>
                ))}
              </optgroup>
            )}
            <optgroup label="预训练">
              {availableModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </optgroup>
          </select>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ fontSize: '11px', color: fgMuted, whiteSpace: 'nowrap' }}>
              置信度: {detectAllConfidence.toFixed(2)}
            </span>
            <input
              type="range"
              min="0.05"
              max="0.9"
              step="0.05"
              value={detectAllConfidence}
              onChange={(e) => onDetectAllConfidenceChange?.(parseFloat(e.target.value))}
              disabled={loading}
              style={{ width: '70px', cursor: 'pointer' }}
            />
          </div>
          <button
            onClick={onDetectAll}
            disabled={loading}
            title="用 YOLO 自动识别图中所有对象并添加标注"
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: '6px',
              background: loading ? '#94a3b8' : '#f59e0b',
              color: '#fff',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '13px',
              fontWeight: 600,
            }}
          >
            {loading ? '⏳ 识别中...' : '✨ 一键标注'}
          </button>
        </div>
      )}

      <button
        onClick={onAutoLabel}
        disabled={loading}
        title="对当前选中类别进行自动标注"
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

      {saveToolbarButton}
    </>
  );

  const trailingCluster = (
    <>
      {/* SAM 版本选择 */}
      {onSamVersionChange && workMode === 'smart' && (
        <>
          <div style={{ width: '1px', height: '28px', background: line }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '12px', color: fgMuted, fontWeight: 500 }}>SAM:</span>
            <select
              value={samVersion}
              onChange={(e) => onSamVersionChange(e.target.value as 'sam2_lite' | 'sam2_base' | 'sam2_large' | 'sam3')}
              style={{
                padding: '4px 8px',
                border: `1px solid ${line}`,
                borderRadius: '6px',
                background: surface,
                color: fg,
                fontSize: '12px',
                cursor: 'pointer',
              }}
              title="选择 SAM 模型档位"
            >
              <option value="sam2_lite">SAM 2.1 Lite（低显存）</option>
              <option value="sam2_base">SAM 2.1 Base（平衡）</option>
              <option value="sam2_large">SAM 2.1 Large（高精度）</option>
              <option value="sam3">SAM 3（新模型）</option>
            </select>
            <span style={{ fontSize: '11px', color: fgMuted }}>
              {samVersion === 'sam2_lite' && '建议显存: >=4GB'}
              {samVersion === 'sam2_base' && '建议显存: >=6GB'}
              {samVersion === 'sam2_large' && '建议显存: >=10GB'}
              {samVersion === 'sam3' && '建议显存: >=12GB'}
            </span>
          </div>
        </>
      )}

      {/* 快捷键帮助 */}
      <button
        onClick={() => setShowShortcuts(!showShortcuts)}
        style={{
          padding: '6px 10px',
          border: `1px solid ${line}`,
          borderRadius: '6px',
          background: surface,
          cursor: 'pointer',
          fontSize: '12px',
          color: fgMuted,
        }}
        title="快捷键帮助"
      >
        ⌨️
      </button>
    </>
  );

  return (
    <div style={{
      position: 'relative',
      display: 'flex',
      flexDirection: 'column',
      gap: compactGap,
      padding: compactPad,
      background: studio ? '#252526' : '#fff',
      borderRadius: studio ? 0 : 8,
      boxShadow: studio ? 'none' : '0 1px 3px rgba(0,0,0,0.1)',
      borderBottom: studio ? '1px solid #2d2d2d' : undefined,
    }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: compact ? '8px' : '12px' }}>
        {workMode === 'smart' ? (
          smartAutoCluster
        ) : (
          <>
            {manualCluster}
            {saveToolbarButton && (
              <>
                <div style={{ width: '1px', height: '28px', background: line }} />
                {saveToolbarButton}
              </>
            )}
          </>
        )}
        {workMode === 'draw' && tool === 'box' && (
          <span
            style={{
              fontSize: '11px',
              color: studio ? '#7dd3fc' : '#0369a1',
              background: studio ? '#164e63' : '#e0f2fe',
              border: studio ? '1px solid #0e7490' : '1px solid #bae6fd',
              borderRadius: '999px',
              padding: '2px 8px',
              fontWeight: 600,
            }}
            title="连续框选模式：可持续拖框标注"
          >
            连续框选中
          </span>
        )}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: compact ? '8px' : '12px' }}>
        {trailingCluster}
      </div>

      {/* 类别色条 + 框选识别（布局参考 Ultralytics Hub） */}
      <div style={{
        width: '100%',
        flexBasis: '100%',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        gap: compact ? '6px' : '8px',
        paddingTop: compact ? '6px' : '10px',
        borderTop: `1px solid ${line}`,
      }}>
        <span style={{ fontSize: '12px', color: fgMuted, fontWeight: 600 }}>类别</span>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '6px',
          flex: 1,
          minWidth: 0,
          maxHeight: compact ? '52px' : '72px',
          overflowY: 'auto',
        }}>
          {chipClasses.map((c, i) => {
            const col = colorForClassName(c);
            const active = c === currentClass;
            return (
              <button
                key={`${c}-${i}`}
                type="button"
                onClick={() => onClassChange(c)}
                title={i < 9 ? `快捷键 ${i + 1}` : c}
                style={{
                  padding: '4px 10px',
                  border: active ? `2px solid ${col}` : `1px solid ${line}`,
                  borderRadius: '6px',
                  background: active ? `${col}18` : surface,
                  cursor: 'pointer',
                  fontSize: '12px',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontWeight: active ? 600 : 500,
                  color: fg,
                }}
              >
                {i < 9 && (
                  <span style={{
                    fontSize: '10px',
                    background: '#1e293b',
                    color: '#fff',
                    borderRadius: '4px',
                    padding: '0 4px',
                    fontFamily: 'monospace',
                  }}>
                    {i + 1}
                  </span>
                )}
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: col }} />
                {c}
              </button>
            );
          })}
        </div>
        {onAutoClassifyChange && (
          <label
            style={{
              fontSize: '12px',
              color: fg,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            <input
              type="checkbox"
              checked={autoClassifyOnBox}
              onChange={(e) => onAutoClassifyChange(e.target.checked)}
            />
            框选后 YOLO 识别类别
          </label>
        )}
      </div>

      {showShortcuts && (
        <div style={{
          position: 'absolute',
          top: '100%',
          right: 0,
          marginTop: '8px',
          padding: '16px',
          background: studio ? '#2d2d2d' : '#fff',
          border: `1px solid ${line}`,
          borderRadius: '8px',
          boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
          zIndex: 3000,
          minWidth: '200px',
        }}>
          <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', color: fg }}>快捷键</h4>
          <div style={{ display: 'grid', gap: '8px' }}>
            {shortcutsForHelp.map((s) => (
              <div key={s.key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                <span style={{ color: fgMuted }}>{s.desc}</span>
                <span style={{
                  background: studio ? '#3a3a3c' : '#f1f5f9',
                  color: fg,
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
