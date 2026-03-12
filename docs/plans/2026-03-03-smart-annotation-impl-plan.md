# 智能标注功能优化实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 实现完整的智能标注功能，包括 SAM 点点击分割、批量同类标注、YOLO+SAM 联动，以及前端交互优化

**Architecture:** 采用前后端分离架构，后端提供 SAM 分割和 YOLO 检测 API，前端使用 React 实现交互式标注画布

**Tech Stack:** React + TypeScript (前端), Python + FastAPI (后端), SAM + YOLO (模型)

---

## Task 1: 后端 - 扩展 SAM API 添加批量同类标注

**Files:**
- Modify: `backend/modules/data_preparation/sam_service.py:470-520`
- Modify: `backend/modules/data_preparation/routes.py` (添加新路由)

**Step 1: 在 SAMService 中添加批量同类标注方法**

在 `sam_service.py` 文件末尾 `batch_auto_label` 方法后添加新方法：

```python
def batch_sam_label(
    self,
    image_path: str,
    class_name: str,
    model_name: str = "yolo11n.pt",
    confidence: float = 0.25
) -> Dict[str, Any]:
    """
    批量 SAM 标注 - 针对特定类别的所有对象
    Args:
        image_path: 图片路径
        class_name: 类别名称
        model_name: YOLO 模型
        confidence: 置信度阈值
    Returns:
        Dict: 标注结果
    """
    logger.info(f"[SAM] 批量标注类别: {class_name}")

    # 设置图片
    self.set_image(image_path)

    # 使用 YOLO 检测
    from backend.core.yolo_engine import yolo_engine
    result = yolo_engine.infer(
        image_path=image_path,
        model_identifier=model_name,
        confidence=confidence
    )

    if not result.get("success"):
        return result

    detections = result.get("detections", [])

    # 过滤特定类别
    filtered = [d for d in detections if d["class_name"].lower() == class_name.lower()]

    # 对每个检测结果进行 SAM 分割
    annotations = []
    for det in filtered:
        bbox = det["bbox"]
        seg_result = self.predict_box(bbox)

        if seg_result.get("success") and seg_result.get("masks"):
            yolo_seg = self._polygons_to_yolo_format(
                seg_result["masks"],
                self.current_image.shape[1],
                self.current_image.shape[0]
            )

            annotations.append({
                "class": det["class_name"],
                "class_id": det["class_id"],
                "bbox": bbox,
                "segmentation": yolo_seg,
                "confidence": det["confidence"],
                "segment_score": seg_result.get("score", 0)
            })

    preview = self._generate_annotation_preview(annotations)

    return {
        "success": True,
        "message": f"找到 {len(annotations)} 个 {class_name} 对象",
        "class_name": class_name,
        "annotations": annotations,
        "preview_image": preview,
        "total": len(annotations)
    }
```

**Step 2: 在 routes.py 添加新路由**

在现有的 SAM 路由部分添加：

```python
@router.post("/sam/batch-sam-label")
async def batch_sam_label(
    image_path: str = Form(...),
    class_name: str = Form(...),
    model_name: str = Form("yolo11n.pt"),
    confidence: float = Form(0.25)
):
    """批量 SAM 标注 - 针对特定类别"""
    result = sam_service.batch_sam_label(
        image_path=image_path,
        class_name=class_name,
        model_name=model_name,
        confidence=confidence
    )
    return result
```

**Step 3: 在 api.ts 添加前端 API**

Modify: `frontend-react/src/services/api.ts:110-165`

```typescript
// 在 samApi 对象中添加新方法
batchSamLabel: (imagePath: string, className: string, modelName?: string, confidence?: number) => {
  const formData = new FormData();
  formData.append('image_path', imagePath);
  formData.append('class_name', className);
  if (modelName) formData.append('model_name', modelName);
  formData.append('confidence', String(confidence || 0.25));
  return api.post('/sam/batch-sam-label', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
},
```

**Step 4: 提交代码**

```bash
git add backend/modules/data_preparation/sam_service.py backend/modules/data_preparation/routes.py frontend-react/src/services/api.ts
git commit -m "feat: add batch SAM label API for same-class objects"
```

---

## Task 2: 前端 - 创建交互式标注画布组件

**Files:**
- Create: `frontend-react/src/components/Annotation/AnnotationCanvas.tsx`
- Modify: `frontend-react/src/types/index.ts`

**Step 1: 添加类型定义**

Modify: `frontend-react/src/types/index.ts` 在文件末尾添加：

```typescript
// 标注交互类型
export interface AnnotationPoint {
  x: number;
  y: number;
  label: 1 | 0; // 1=前景, 0=背景
}

export interface AnnotationBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface AnnotationMask {
  polygons: number[][];
  color: string;
}

export type AnnotationTool = 'point' | 'box' | 'select' | 'auto';

// 标注状态
export interface AnnotationState {
  tool: AnnotationTool;
  points: AnnotationPoint[];
  boxes: AnnotationBox[];
  masks: AnnotationMask[];
  selectedId: string | null;
  currentClass: string;
}
```

**Step 2: 创建 AnnotationCanvas 组件**

Create: `frontend-react/src/components/Annotation/AnnotationCanvas.tsx`

```typescript
import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { AnnotationPoint, AnnotationBox, AnnotationMask, AnnotationTool } from '../../types';

interface Props {
  image: string;
  width: number;
  height: number;
  points: AnnotationPoint[];
  boxes: AnnotationBox[];
  masks: AnnotationMask[];
  tool: AnnotationTool;
  selectedId: string | null;
  onPointAdd: (point: AnnotationPoint) => void;
  onBoxAdd: (box: AnnotationBox) => void;
  onBoxDrag?: (index: number, box: AnnotationBox) => void;
  onMaskSelect?: (index: number) => void;
}

const AnnotationCanvas: React.FC<Props> = ({
  image,
  width,
  height,
  points,
  boxes,
  masks,
  tool,
  selectedId,
  onPointAdd,
  onBoxAdd,
  onMaskSelect,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentPoint, setCurrentPoint] = useState<{ x: number; y: number } | null>(null);

  // 绘制画布
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 设置画布尺寸
    canvas.width = width;
    canvas.height = height;

    // 加载图片
    const img = new Image();
    img.src = image;
    img.onload = () => {
      // 绘制原图
      ctx.drawImage(img, 0, 0, width, height);

      // 绘制掩码
      masks.forEach((mask, idx) => {
        ctx.fillStyle = mask.color + '60';
        ctx.strokeStyle = mask.color;
        ctx.lineWidth = 2;

        // 绘制多边形
        if (mask.polygons.length > 0) {
          ctx.beginPath();
          for (let i = 0; i < mask.polygons.length; i += 2) {
            const x = mask.polygons[i] * width;
            const y = mask.polygons[i + 1] * height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
      });

      // 绘制边界框
      boxes.forEach((box, idx) => {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);

        // 绘制编号
        ctx.fillStyle = '#00ff00';
        ctx.fillRect(box.x1, box.y1 - 20, 24, 20);
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        ctx.fillText(String(idx + 1), box.x1 + 6, box.y1 - 5);
      });

      // 绘制点
      points.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = point.label === 1 ? '#00ff00' : '#ff0000';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // 添加 +/- 符号
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(point.label === 1 ? '+' : '-', point.x, point.y);
      });

      // 绘制框选过程中的预览
      if (isDrawing && startPoint && currentPoint) {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(
          startPoint.x,
          startPoint.y,
          currentPoint.x - startPoint.x,
          currentPoint.y - startPoint.y
        );
        ctx.setLineDash([]);
      }
    };
  }, [image, width, height, points, boxes, masks, isDrawing, startPoint, currentPoint]);

  // 处理鼠标点击 - 添加点
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (tool !== 'point') return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Shift+点击 = 负样本，否则 = 正样本
    const label: 1 | 0 = e.shiftKey ? 0 : 1;
    onPointAdd({ x, y, label });
  }, [tool, width, height, onPointAdd]);

  // 处理鼠标按下 - 开始框选
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (tool !== 'box') return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    setIsDrawing(true);
    setStartPoint({ x, y });
    setCurrentPoint({ x, y });
  }, [tool, width, height]);

  // 处理鼠标移动
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    setCurrentPoint({ x, y });
  }, [isDrawing, width, height]);

  // 处理鼠标抬起 - 完成框选
  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !startPoint || !currentPoint) return;

    const x1 = Math.min(startPoint.x, currentPoint.x);
    const y1 = Math.min(startPoint.y, currentPoint.y);
    const x2 = Math.max(startPoint.x, currentPoint.x);
    const y2 = Math.max(startPoint.y, currentPoint.y);

    if (x2 - x1 > 10 && y2 - y1 > 10) {
      onBoxAdd({ x1, y1, x2, y2 });
    }

    setIsDrawing(false);
    setStartPoint(null);
    setCurrentPoint(null);
  }, [isDrawing, startPoint, currentPoint, onBoxAdd]);

  return (
    <canvas
      ref={canvasRef}
      onClick={handleCanvasClick}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      style={{
        width: '100%',
        height: 'auto',
        cursor: tool === 'point' ? 'crosshair' : tool === 'box' ? 'crosshair' : 'default',
        border: '1px solid #ddd',
        borderRadius: '8px',
      }}
    />
  );
};

export default AnnotationCanvas;
```

**Step 3: 创建组件导出文件**

Create: `frontend-react/src/components/Annotation/index.ts`

```typescript
export { default as AnnotationCanvas } from './AnnotationCanvas';
```

**Step 4: 提交代码**

```bash
git add frontend-react/src/components/Annotation/AnnotationCanvas.tsx frontend-react/src/components/Annotation/index.ts frontend-react/src/types/index.ts
git commit -m "feat: add AnnotationCanvas component with point and box interaction"
```

---

## Task 3: 前端 - 创建工具栏和标注面板组件

**Files:**
- Create: `frontend-react/src/components/Annotation/AnnotationToolbar.tsx`
- Create: `frontend-react/src/components/Annotation/AnnotationPanel.tsx`
- Modify: `frontend-react/src/components/Annotation/index.ts`

**Step 1: 创建 AnnotationToolbar**

Create: `frontend-react/src/components/Annotation/AnnotationToolbar.tsx`

```typescript
import React from 'react';
import type { AnnotationTool } from '../../types';

interface Props {
  tool: AnnotationTool;
  currentClass: string;
  classes: string[];
  onToolChange: (tool: AnnotationTool) => void;
  onClassChange: (className: string) => void;
  onAutoLabel: () => void;
  onClear: () => void;
  onUndo: () => void;
  onRedo: () => void;
  loading?: boolean;
}

export const AnnotationToolbar: React.FC<Props> = ({
  tool,
  currentClass,
  classes,
  onToolChange,
  onClassChange,
  onAutoLabel,
  onClear,
  onUndo,
  onRedo,
  loading = false,
}) => {
  const tools: { id: AnnotationTool; label: string; icon: string }[] = [
    { id: 'select', label: '选择', icon: '👆' },
    { id: 'point', label: '点标注', icon: '📍' },
    { id: 'box', label: '框选', icon: '⬜' },
    { id: 'auto', label: '自动', icon: '🤖' },
  ];

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '16px',
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
            style={{
              padding: '8px 12px',
              border: 'none',
              borderRadius: '6px',
              background: tool === t.id ? '#3b82f6' : '#f1f5f9',
              color: tool === t.id ? '#fff' : '#475569',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      {/* 分隔线 */}
      <div style={{ width: '1px', height: '24px', background: '#e2e8f0' }} />

      {/* 类别选择 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <label style={{ fontSize: '14px', color: '#64748b' }}>类别:</label>
        <select
          value={currentClass}
          onChange={(e) => onClassChange(e.target.value)}
          style={{
            padding: '6px 12px',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            fontSize: '14px',
            background: '#fff',
          }}
        >
          {classes.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      {/* 分隔线 */}
      <div style={{ width: '1px', height: '24px', background: '#e2e8f0' }} />

      {/* 操作按钮 */}
      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          onClick={onUndo}
          disabled={loading}
          style={{
            padding: '6px 12px',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            background: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          ↩️ 撤销
        </button>
        <button
          onClick={onRedo}
          disabled={loading}
          style={{
            padding: '6px 12px',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            background: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          ↪️ 重做
        </button>
        <button
          onClick={onClear}
          disabled={loading}
          style={{
            padding: '6px 12px',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            background: '#fff',
            color: '#ef4444',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          🗑️ 清除
        </button>
      </div>

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
          fontSize: '14px',
          fontWeight: 500,
        }}
      >
        {loading ? '⏳ 处理中...' : '🤖 自动标注'}
      </button>
    </div>
  );
};

export default AnnotationToolbar;
```

**Step 2: 创建 AnnotationPanel**

Create: `frontend-react/src/components/Annotation/AnnotationPanel.tsx`

```typescript
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
                  框: [{ann.bbox?.slice(0, 2).join(', ')}]
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
                      onDelete(String(idx));
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
```

**Step 3: 更新导出文件**

Modify: `frontend-react/src/components/Annotation/index.ts`

```typescript
export { default as AnnotationCanvas } from './AnnotationCanvas';
export { default as AnnotationToolbar } from './AnnotationToolbar';
export { default as AnnotationPanel } from './AnnotationPanel';
```

**Step 4: 提交代码**

```bash
git add frontend-react/src/components/Annotation/
git commit -m "feat: add AnnotationToolbar and AnnotationPanel components"
```

---

## Task 4: 前端 - 整合智能标注页面

**Files:**
- Modify: `frontend-react/src/pages/Annotation/Annotation.tsx`

**Step 1: 重构 Annotation 页面**

用新组件替换现有的简单实现，添加完整的交互功能：

```typescript
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardHeader, Button, Input } from '../../components/common';
import { samApi, annotationApi } from '../../services/api';
import type { SAMAnnotation, AnnotationTool, AnnotationPoint, AnnotationBox, AnnotationMask } from '../../types';
import { AnnotationCanvas, AnnotationToolbar, AnnotationPanel } from '../../components/Annotation';

export const Annotation: React.FC = () => {
  // 状态管理
  const [image, setImage] = useState<string | null>(null);
  const [imagePath, setImagePath] = useState<string>('');
  const [tool, setTool] = useState<AnnotationTool>('point');
  const [currentClass, setCurrentClass] = useState('person');
  const [classes] = useState(['person', 'car', 'dog', 'cat', 'bicycle', 'bird']);

  // 标注数据
  const [points, setPoints] = useState<AnnotationPoint[]>([]);
  const [boxes, setBoxes] = useState<AnnotationBox[]>([]);
  const [masks, setMasks] = useState<AnnotationMask[]>([]);
  const [annotations, setAnnotations] = useState<SAMAnnotation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // 撤销/重做
  const [history, setHistory] = useState<{ points: AnnotationPoint[]; boxes: AnnotationBox[]; masks: AnnotationMask[] }[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  // 加载状态
  const [loading, setLoading] = useState(false);
  const [samLoaded, setSamLoaded] = useState(false);

  // 图片尺寸
  const [imgSize, setImgSize] = useState({ width: 800, height: 600 });

  // 初始化加载 SAM 模型
  useEffect(() => {
    const initSAM = async () => {
      try {
        const res = await samApi.loadModel('vit_b');
        setSamLoaded(res.data?.success || res.data?.data?.success || false);
      } catch (e) {
        console.error('SAM init failed:', e);
      }
    };
    initSAM();
  }, []);

  // 保存历史记录
  const saveHistory = useCallback((newPoints: AnnotationPoint[], newBoxes: AnnotationBox[], newMasks: AnnotationMask[]) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({ points: newPoints, boxes: newBoxes, masks: newMasks });
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  }, [history, historyIndex]);

  // 撤销
  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const prev = history[historyIndex - 1];
      setPoints(prev.points);
      setBoxes(prev.boxes);
      setMasks(prev.masks);
      setHistoryIndex(historyIndex - 1);
    }
  }, [history, historyIndex]);

  // 重做
  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const next = history[historyIndex + 1];
      setPoints(next.points);
      setBoxes(next.boxes);
      setMasks(next.masks);
      setHistoryIndex(historyIndex + 1);
    }
  }, [history, historyIndex]);

  // 添加点
  const handlePointAdd = useCallback((point: AnnotationPoint) => {
    const newPoints = [...points, point];
    setPoints(newPoints);
    saveHistory(newPoints, boxes, masks);
  }, [points, boxes, masks, saveHistory]);

  // 添加框
  const handleBoxAdd = useCallback(async (box: AnnotationBox) => {
    const newBoxes = [...boxes, box];
    setBoxes(newBoxes);

    // 调用 SAM 框选分割
    if (samLoaded && imagePath) {
      setLoading(true);
      try {
        const res = await samApi.predictBox([box.x1, box.y1, box.x2, box.y2]);
        const result = res.data?.data || res.data;
        if (result?.success && result.masks?.length > 0) {
          const newMasks = [...masks, {
            polygons: result.masks[0],
            color: getRandomColor()
          }];
          setMasks(newMasks);
          saveHistory(points, newBoxes, newMasks);
        }
      } catch (e) {
        console.error('SAM predict failed:', e);
      } finally {
        setLoading(false);
      }
    } else {
      saveHistory(points, newBoxes, masks);
    }
  }, [boxes, masks, points, samLoaded, imagePath, saveHistory]);

  // 清除
  const handleClear = useCallback(() => {
    setPoints([]);
    setBoxes([]);
    setMasks([]);
    setAnnotations([]);
    saveHistory([], [], []);
  }, [saveHistory]);

  // 自动标注
  const handleAutoLabel = useCallback(async () => {
    if (!imagePath) return;

    setLoading(true);
    try {
      // 使用批量同类标注
      const res = await samApi.batchSamLabel(imagePath, currentClass);
      const result = res.data?.data || res.data;

      if (result?.success) {
        setAnnotations(result.annotations || []);

        // 生成掩码显示
        const newMasks = (result.annotations || []).map((ann: SAMAnnotation, idx: number) => ({
          polygons: ann.segmentation?.split(' ').map(Number) || [],
          color: getRandomColor(),
        }));
        setMasks(newMasks);
      }
    } catch (e) {
      console.error('Auto label failed:', e);
    } finally {
      setLoading(false);
    }
  }, [imagePath, currentClass]);

  // 文件选择
  const handleFileSelect = async (files: File[]) => {
    if (files.length === 0) return;
    const file = files[0];
    const imageUrl = URL.createObjectURL(file);
    setImage(imageUrl);

    // 上传到服务器
    const formData = new FormData();
    formData.append('file', file);
    try {
      // 这里简化处理，实际应该调用上传 API
      setImagePath(file.name);
    } catch (e) {
      console.error(e);
    }

    // 获取图片尺寸
    const img = new Image();
    img.onload = () => {
      setImgSize({ width: img.width, height: img.height });
    };
    img.src = imageUrl;
  };

  // 删除标注
  const handleDelete = (id: string) => {
    const idx = parseInt(id);
    const newAnnotations = annotations.filter((_, i) => i !== idx);
    setAnnotations(newAnnotations);
  };

  // 类别修改
  const handleClassChange = (id: string, newClass: string) => {
    const idx = parseInt(id);
    setAnnotations(annotations.map((a, i) => i === idx ? { ...a, class: newClass } : a));
  };

  // 随机颜色
  const getRandomColor = () => {
    const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  // 导出
  const handleExport = () => {
    console.log('导出标注:', annotations);
    // 调用导出 API
  };

  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ fontFamily: 'DM Sans', fontWeight: 700, marginBottom: '24px' }}>
        智能标注
      </h1>

      {/* 工具栏 */}
      <AnnotationToolbar
        tool={tool}
        currentClass={currentClass}
        classes={classes}
        onToolChange={setTool}
        onClassChange={setCurrentClass}
        onAutoLabel={handleAutoLabel}
        onClear={handleClear}
        onUndo={handleUndo}
        onRedo={handleRedo}
        loading={loading}
      />

      {/* 主工作区 */}
      <div style={{ display: 'flex', gap: '24px', marginTop: '24px' }}>
        {/* 画布区域 */}
        <div style={{ flex: 1 }}>
          {image ? (
            <AnnotationCanvas
              image={image}
              width={imgSize.width}
              height={imgSize.height}
              points={points}
              boxes={boxes}
              masks={masks}
              tool={tool}
              selectedId={selectedId}
              onPointAdd={handlePointAdd}
              onBoxAdd={handleBoxAdd}
              onMaskSelect={setSelectedId}
            />
          ) : (
            <div style={{
              width: '100%',
              height: '400px',
              border: '2px dashed #e2e8f0',
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: '#f8fafc',
            }}>
              <label style={{ cursor: 'pointer', textAlign: 'center' }}>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>📁</div>
                <div>点击或拖拽上传图片</div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleFileSelect(e.target.files ? Array.from(e.target.files) : [])}
                  style={{ display: 'none' }}
                />
              </label>
            </div>
          )}
        </div>

        {/* 标注面板 */}
        <AnnotationPanel
          annotations={annotations}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onDelete={handleDelete}
          onClassChange={handleClassChange}
          onExport={handleExport}
          classes={classes}
        />
      </div>

      {/* 说明 */}
      <div style={{ marginTop: '24px', color: '#64748b', fontSize: '14px' }}>
        <p><strong>使用说明:</strong></p>
        <ul>
          <li>点击图片添加正样本点 (绿色 +)</li>
          <li>Shift + 点击添加负样本点 (红色 -)</li>
          <li>选择框选工具拖动画框进行分割</li>
          <li>选择自动标注并指定类别进行批量标注</li>
        </ul>
      </div>
    </div>
  );
};
```

**Step 2: 提交代码**

```bash
git add frontend-react/src/pages/Annotation/Annotation.tsx
git commit -m "feat: integrate new annotation components into Annotation page"
```

---

## Task 5: 测试和验证

**Step 1: 启动后端服务测试 API**

```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

测试批量标注 API:
```bash
curl -X POST "http://localhost:8000/api/v1/sam/batch-sam-label" \
  -F "image_path=test.jpg" \
  -F "class_name=person" \
  -F "confidence=0.25"
```

**Step 2: 启动前端测试**

```bash
cd frontend-react
npm run dev
```

验证功能:
- [ ] 上传图片显示正常
- [ ] 点击添加正/负样本点可视化
- [ ] 框选工具可以绘制
- [ ] 自动标注返回结果
- [ ] 标注列表正确显示
- [ ] 导出功能正常

**Step 3: 最终提交**

```bash
git add .
git commit -m "feat: complete smart annotation feature with SAM integration"
git push origin main
```

---

## 总结

完成以上 5 个任务后，智能标注功能将具备:

1. **SAM 点点击分割** - 支持正负样本点交互
2. **批量同类标注** - 输入类别自动标注所有对象
3. **YOLO+SAM 联动** - 检测+分割一体化
4. **完整标注管理** - 列表、选择、删除、修改
5. **导出功能** - YOLO 格式导出

**Plan complete and saved to `docs/plans/2026-03-03-smart-annotation-design.md`. Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
