import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { AnnotationPoint, AnnotationBox, AnnotationMask, AnnotationTool, SAMAnnotation } from '../../types';

interface Props {
  image: string;
  width: number;
  height: number;
  points: AnnotationPoint[];
  boxes: AnnotationBox[];
  masks: AnnotationMask[];
  annotations?: SAMAnnotation[];
  tool: AnnotationTool;
  selectedId: string | null;
  hoveredId?: string | null;
  onPointAdd: (point: AnnotationPoint) => void;
  onBoxAdd: (box: AnnotationBox) => void;
  /** 返回 false 时不闭合多边形（例如尚未选择类别） */
  onMaskAdd?: (mask: AnnotationMask) => void | boolean;
  onBoxDrag?: (index: number, box: AnnotationBox) => void;
  onMaskSelect?: (index: string | number) => void;
  onAnnotationSelect?: (id: string) => void;
  showLabels?: boolean;
  currentClass?: string;
  smartMode?: boolean;
  onCursorMove?: (x: number, y: number) => void;
  /** Smart 模式下双击画布时触发（如全屏预览），不占用单点/框选 */
  onSmartImageDoubleClick?: () => void;
  /** 只读展示：不响应交互（全屏带标注预览等） */
  viewOnly?: boolean;
  /** 覆盖画布的 maxHeight 样式，例如全屏时吃满视口 */
  maxHeightOverride?: string;
  /** 只读预览时让整张画布以 contain 方式塞进父容器，避免大图被视口裁切 */
  fitMode?: 'intrinsic' | 'contain';
}

const AnnotationCanvas: React.FC<Props> = ({
  image,
  width,
  height,
  points,
  boxes,
  masks,
  annotations = [],
  tool,
  selectedId,
  hoveredId = null,
  onPointAdd,
  onBoxAdd,
  onMaskAdd,
  onBoxDrag,
  onMaskSelect,
  onAnnotationSelect,
  showLabels = true,
  currentClass = 'object',
  smartMode = false,
  onCursorMove,
  onSmartImageDoubleClick,
  viewOnly = false,
  maxHeightOverride,
  fitMode = 'intrinsic',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentPoint, setCurrentPoint] = useState<{ x: number; y: number } | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);
  const [hoveredBox, setHoveredBox] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [polygonPoints, setPolygonPoints] = useState<{ x: number; y: number }[]>([]);

  // 颜色生成
  const getColorForClass = (className: string) => {
    const colors: Record<string, string> = {
      person: '#ef4444',
      car: '#3b82f6',
      dog: '#22c55e',
      cat: '#f97316',
      bicycle: '#8b5cf6',
      bird: '#06b6d4',
      default: '#ef4444',
    };
    return colors[className.toLowerCase()] || colors.default || '#' + Math.floor(Math.random()*16777215).toString(16);
  };

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

      // 添加半透明遮罩
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.fillRect(0, 0, width, height);

      // 恢复图片显示（通过清除遮罩区域）
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';

      // 绘制掩码（从下往上绘制，让最新的在最上面）
      masks.forEach((mask, idx) => {
        // 创建临时canvas来绘制掩码
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) return;

        // 绘制多边形
        if (mask.polygons.length > 0) {
          tempCtx.beginPath();
          for (let i = 0; i < mask.polygons.length; i += 2) {
            const x = mask.polygons[i] * width;
            const y = mask.polygons[i + 1] * height;
            if (i === 0) tempCtx.moveTo(x, y);
            else tempCtx.lineTo(x, y);
          }
          tempCtx.closePath();
          tempCtx.fillStyle = 'white';
          tempCtx.fill();
        }

        // 使用globalCompositeOperation来显示掩码区域
        ctx.save();
        ctx.drawImage(tempCanvas, 0, 0);
        ctx.restore();
      });

      ctx.restore();

      // 重新绘制原图（现在只有掩码区域显示）
      ctx.drawImage(img, 0, 0, width, height);

      // 绘制半透明掩码叠加
      masks.forEach((mask, idx) => {
        const isSelected = selectedId === String(idx);

        if (mask.polygons.length > 0) {
          ctx.beginPath();
          for (let i = 0; i < mask.polygons.length; i += 2) {
            const x = mask.polygons[i] * width;
            const y = mask.polygons[i + 1] * height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.closePath();

          // 填充
          ctx.fillStyle = mask.color + '40';
          ctx.fill();

          // 描边
          ctx.strokeStyle = isSelected ? '#fff' : mask.color;
          ctx.lineWidth = isSelected ? 3 : 2;
          ctx.stroke();
        }
      });

      // 交互式边界框（底层）：必须在 annotations 之前绘制，否则会盖住右侧列表选中时的白边/光晕
      boxes.forEach((box, idx) => {
        const isHovered = hoveredBox === idx;
        const isSelected = selectedId === String(idx);
        const ann = annotations[idx];
        const hasAnnBbox = Boolean(ann?.bbox && ann.bbox.length >= 4);
        ctx.strokeStyle = isSelected ? '#86efac' : isHovered ? '#fbbf24' : '#00ff00';
        ctx.lineWidth = isSelected ? 2 : isHovered ? 3 : 2;
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);

        // 有标注层时会画序号角标；仅交互框阶段仍画绿底编号
        if (!isSelected || !hasAnnBbox) {
          ctx.fillStyle = '#00ff00';
          ctx.fillRect(box.x1, box.y1 - 22, 22, 20);
          ctx.fillStyle = '#000';
          ctx.font = 'bold 11px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(String(idx + 1), box.x1 + 6, box.y1 - 6);
        }
      });

      // 绘制边界框（来自 SAM 标注结果，盖在绿色交互框之上；选中时加粗高亮便于与右栏联动）
      annotations.forEach((ann, idx) => {
        if (!ann.bbox) return;

        const [x1, y1, x2, y2] = ann.bbox;
        const isSelected = selectedId === String(idx);
        const isHovered = hoveredId === String(idx);
        const color = getColorForClass(ann.class);

        ctx.strokeStyle = isSelected ? '#ffffff' : isHovered ? '#f59e0b' : color;
        ctx.lineWidth = isSelected ? 4 : isHovered ? 3 : 2;

        if (isSelected || isHovered) {
          ctx.shadowColor = isSelected ? '#3b82f6' : color;
          ctx.shadowBlur = isSelected ? 14 : 6;
        }

        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.shadowBlur = 0;

        if (isSelected) {
          ctx.strokeStyle = 'rgba(59, 130, 246, 0.95)';
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect(x1 - 1, y1 - 1, x2 - x1 + 2, y2 - y1 + 2);
          ctx.setLineDash([]);
        }

        // 绘制标签背景
        if (showLabels) {
          ctx.fillStyle = color;
          const confPct = ann.confidence != null ? (ann.confidence * 100).toFixed(0) : '100';
          const label = `${ann.class} ${confPct}%`;
          ctx.font = 'bold 12px Arial';
          const textWidth = ctx.measureText(label).width;

          ctx.fillRect(x1, y1 - 24, textWidth + 12, 22);

          // 绘制标签文字
          ctx.fillStyle = '#fff';
          ctx.fillText(label, x1 + 6, y1 - 7);

          // 绘制编号
          ctx.fillStyle = '#fff';
          ctx.beginPath();
          ctx.arc(x1 + 10, y1 + 10, 10, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = color;
          ctx.font = 'bold 12px Arial';
          ctx.textAlign = 'center';
          ctx.fillText(String(idx + 1), x1 + 10, y1 + 14);
        }
      });

      // 绘制点（SAM 交互点）
      points.forEach((point, idx) => {
        const isHovered = hoveredPoint === idx;

        // 外圈
        ctx.beginPath();
        ctx.arc(point.x, point.y, isHovered ? 10 : 8, 0, Math.PI * 2);
        ctx.fillStyle = point.label === 1 ? '#22c55e' : '#ef4444';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // 中心点
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();

        // +/- 符号
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

        // 显示框选尺寸
        const w = Math.abs(currentPoint.x - startPoint.x);
        const h = Math.abs(currentPoint.y - startPoint.y);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(startPoint.x, startPoint.y - 24, 80, 20);
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.fillText(`${Math.round(w)} × ${Math.round(h)}`, startPoint.x + 40, startPoint.y - 9);
      }

      // 绘制十字准星（点标注模式）
      if (tool === 'point' && currentPoint) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);

        // 水平线
        ctx.beginPath();
        ctx.moveTo(0, currentPoint.y);
        ctx.lineTo(width, currentPoint.y);

        // 垂直线
        ctx.moveTo(currentPoint.x, 0);
        ctx.lineTo(currentPoint.x, height);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 绘制多边形进行中的预览
      if (tool === 'polygon' && polygonPoints.length > 0) {
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
        for (let i = 1; i < polygonPoints.length; i++) {
          ctx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
        }
        if (currentPoint) {
          ctx.lineTo(currentPoint.x, currentPoint.y);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // 绘制已有顶点
        polygonPoints.forEach((pt, i) => {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, i === 0 ? 7 : 5, 0, Math.PI * 2);
          ctx.fillStyle = i === 0 ? '#ef4444' : '#f59e0b';
          ctx.fill();
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        });
      }
    };
  }, [image, width, height, points, boxes, masks, annotations, isDrawing, startPoint, currentPoint, hoveredPoint, hoveredBox, tool, selectedId, showLabels, polygonPoints]);

  // 获取鼠标位置
  const getMousePos = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }, [width, height]);

  // 检查点是否在框内
  const isPointInBox = useCallback((px: number, py: number, box: AnnotationBox) => {
    return px >= box.x1 && px <= box.x2 && py >= box.y1 && py <= box.y2;
  }, []);

  // 检查点是否在某个标注内
  const findAnnotationAtPoint = useCallback((x: number, y: number) => {
    // 反向遍历，最上面的先检查
    for (let i = annotations.length - 1; i >= 0; i--) {
      const ann = annotations[i];
      if (ann.bbox && isPointInBox(x, y, {
        x1: ann.bbox[0],
        y1: ann.bbox[1],
        x2: ann.bbox[2],
        y2: ann.bbox[3],
      })) {
        return String(i);
      }
    }
    return null;
  }, [annotations, isPointInBox]);

  // 处理鼠标移动
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e);
    if (!pos) return;

    setCurrentPoint(pos);
    onCursorMove?.(pos.x, pos.y);

    // Smart 模式下关闭手动交互（仅保留悬停检测给上层自动标注）
    if (smartMode) return;

    // 悬停检测
    if (tool === 'select') {
      // 检查是否悬停在点上
      let foundPoint = null;
      for (let i = points.length - 1; i >= 0; i--) {
        const p = points[i];
        const dist = Math.sqrt((p.x - pos.x) ** 2 + (p.y - pos.y) ** 2);
        if (dist < 10) {
          foundPoint = i;
          break;
        }
      }
      setHoveredPoint(foundPoint);

      // 检查是否悬停在框上
      let foundBox = null;
      for (let i = boxes.length - 1; i >= 0; i--) {
        if (isPointInBox(pos.x, pos.y, boxes[i])) {
          foundBox = i;
          break;
        }
      }
      setHoveredBox(foundBox);

      // 检查是否悬停在标注上
      if (onAnnotationSelect) {
        const annId = findAnnotationAtPoint(pos.x, pos.y);
        if (annId !== selectedId) {
          onAnnotationSelect(annId || '');
        }
      }
    }

    // 拖拽模式
    if (isDragging && dragStart && selectedId) {
      const dx = pos.x - dragStart.x;
      const dy = pos.y - dragStart.y;

      // 拖拽移动标注框（与右侧列表联动）
      const idx = parseInt(selectedId);
      if (!isNaN(idx) && annotations[idx]?.bbox) {
        const bbox = annotations[idx].bbox;
        const newBbox = [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy];
        if (onBoxDrag) {
          onBoxDrag(idx, {
            x1: newBbox[0],
            y1: newBbox[1],
            x2: newBbox[2],
            y2: newBbox[3],
          });
        }
      }
      setDragStart(pos);
    }

    // 框选模式
    if (isDrawing) {
      // 已经在handleMouseMove中更新currentPoint
    }
  }, [tool, getMousePos, points, boxes, isDragging, dragStart, selectedId, annotations, findAnnotationAtPoint, onAnnotationSelect, isDrawing, onBoxDrag, smartMode, onCursorMove]);

  // 处理鼠标按下
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (smartMode) return;
    const pos = getMousePos(e);
    if (!pos) return;

    if (tool === 'box') {
      setIsDrawing(true);
      setStartPoint(pos);
      setCurrentPoint(pos);
    } else if (tool === 'select') {
      // 检查是否点击了标注
      const annId = findAnnotationAtPoint(pos.x, pos.y);
      if (annId && onAnnotationSelect) {
        onAnnotationSelect(annId);
        setIsDragging(true);
        setDragStart(pos);
      } else if (onAnnotationSelect) {
        onAnnotationSelect('');
      }
    }
  }, [tool, getMousePos, findAnnotationAtPoint, onAnnotationSelect, smartMode]);

  // 处理鼠标抬起
  const handleMouseUp = useCallback(() => {
    if (smartMode) return;
    if (isDrawing && startPoint && currentPoint) {
      const x1 = Math.min(startPoint.x, currentPoint.x);
      const y1 = Math.min(startPoint.y, currentPoint.y);
      const x2 = Math.max(startPoint.x, currentPoint.x);
      const y2 = Math.max(startPoint.y, currentPoint.y);

      // 只有框足够大时才触发
      if (x2 - x1 > 10 && y2 - y1 > 10) {
        onBoxAdd({ x1, y1, x2, y2 });
      }
    }

    setIsDrawing(false);
    setIsDragging(false);
    setDragStart(null);
    setStartPoint(null);
    setCurrentPoint(null);
  }, [isDrawing, startPoint, currentPoint, onBoxAdd, smartMode]);

  // 处理点击
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (smartMode) return;
    const pos = getMousePos(e);
    if (!pos) return;

    if (tool === 'point') {
      const label: 1 | 0 = e.shiftKey ? 0 : 1;
      onPointAdd({ x: pos.x, y: pos.y, label });
      return;
    }

    if (tool === 'polygon') {
      // 双击或点击第一个点 = 完成多边形
      if (polygonPoints.length >= 3) {
        const firstPt = polygonPoints[0];
        const dist = Math.sqrt((pos.x - firstPt.x) ** 2 + (pos.y - firstPt.y) ** 2);
        if (dist < 15 || e.detail === 2) {
          // 完成多边形，转为归一化坐标
          const flatCoords: number[] = [];
          polygonPoints.forEach(pt => {
            flatCoords.push(pt.x / width, pt.y / height);
          });
          if (onMaskAdd) {
            const colors = ['#ef4444','#3b82f6','#22c55e','#f97316','#8b5cf6','#06b6d4'];
            const ok = onMaskAdd({ polygons: flatCoords, color: colors[Math.floor(Math.random() * colors.length)] });
            if (ok !== false) setPolygonPoints([]);
          } else {
            setPolygonPoints([]);
          }
          return;
        }
      }
      // 添加新顶点
      setPolygonPoints(prev => [...prev, pos]);
    }
  }, [tool, getMousePos, onPointAdd, polygonPoints, width, height, onMaskAdd, smartMode]);

  // 鼠标离开
  const handleMouseLeave = useCallback(() => {
    setHoveredPoint(null);
    setHoveredBox(null);
    setCurrentPoint(null);
    if (isDrawing) {
      handleMouseUp();
    }
  }, [isDrawing, handleMouseUp]);

  // 获取光标样式
  const getCursor = () => {
    if (viewOnly) return 'default';
    if (smartMode) return 'crosshair';
    if (tool === 'point') return 'crosshair';
    if (tool === 'box') return isDrawing ? 'crosshair' : 'crosshair';
    if (tool === 'select') return 'default';
    if (tool === 'polygon') return 'crosshair';
    return 'default';
  };

  // 多边形绘制中：Esc 清空；Backspace 撤销上一顶点（捕获阶段，避免父页面把 Backspace 当成删标注）
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (tool !== 'polygon' || polygonPoints.length === 0) return;
      if (e.key === 'Escape') {
        e.preventDefault();
        setPolygonPoints([]);
        return;
      }
      if (e.key === 'Backspace') {
        e.preventDefault();
        e.stopPropagation();
        setPolygonPoints((prev) => prev.slice(0, -1));
      }
    };
    window.addEventListener('keydown', handleKeyDown, true);
    return () => window.removeEventListener('keydown', handleKeyDown, true);
  }, [tool, polygonPoints.length]);

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (smartMode && onSmartImageDoubleClick) {
        e.preventDefault();
        onSmartImageDoubleClick();
      }
    },
    [smartMode, onSmartImageDoubleClick],
  );

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      style={{
        display: 'block',
        width: fitMode === 'contain' ? '100%' : 'auto',
        height: fitMode === 'contain' ? '100%' : 'auto',
        objectFit: fitMode === 'contain' ? 'contain' : undefined,
        maxWidth: '100%',
        maxHeight:
          maxHeightOverride !== undefined
            ? maxHeightOverride
            : 'min(100%, min(92vh, calc(100dvh - 72px)))',
        cursor: getCursor(),
        border: viewOnly ? 'none' : '1px solid #e2e8f0',
        borderRadius: '8px',
        background: '#1e293b',
        pointerEvents: viewOnly ? 'none' : 'auto',
      }}
    />
  );
};

export default AnnotationCanvas;
