import type { SAMAnnotation } from '../../types';

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

/** 像素框 [x1,y1,x2,y2] → YOLO 检测行 class xc yc w h（归一化） */
export function bboxToYoloDetectLine(classId: number, bbox: number[], imgW: number, imgH: number): string {
  const [x1, y1, x2, y2] = bbox;
  const w = Math.max(1, imgW);
  const h = Math.max(1, imgH);
  const xc = ((x1 + x2) / 2) / w;
  const yc = ((y1 + y2) / 2) / h;
  const bw = Math.abs(x2 - x1) / w;
  const bh = Math.abs(y2 - y1) / h;
  return `${classId} ${clamp01(xc).toFixed(6)} ${clamp01(yc).toFixed(6)} ${clamp01(bw).toFixed(6)} ${clamp01(bh).toFixed(6)}`;
}

/**
 * 将当前画布上的 SAM 标注转为 YOLO txt 行（与 Ultralytics 一致：分割行为 class + 归一化多边形点串；仅框为 class + xc yc w h）。
 * 注意：class_id 为标注里原始 id；若需与「仅本图出现的类别」一致，请用 {@link buildYoloSingleImageExport}。
 */
export function buildYoloLabelLines(annotations: SAMAnnotation[], imgW: number, imgH: number): string[] {
  if (!imgW || !imgH) return [];
  const lines: string[] = [];
  for (const ann of annotations) {
    const seg = ann.segmentation?.trim();
    if (seg) {
      lines.push(`${ann.class_id} ${seg}`);
      continue;
    }
    if (ann.bbox && ann.bbox.length >= 4) {
      lines.push(bboxToYoloDetectLine(ann.class_id, ann.bbox, imgW, imgH));
    }
  }
  return lines;
}

export interface YoloSingleImageExport {
  /** 已重映射 class_id 为 0..n-1 的 YOLO 行 */
  lines: string[];
  /** 仅包含本图实际用到的类别名，行下标 = txt 中的 class_id */
  classNames: string[];
}

/**
 * 导出单图：只保留「能写出 YOLO 行」的标注；类别文件只列本图出现的类，并重写 txt 中的 class_id 与之对齐。
 */
export function buildYoloSingleImageExport(
  annotations: SAMAnnotation[],
  samClasses: string[],
  imgW: number,
  imgH: number
): YoloSingleImageExport {
  if (!imgW || !imgH) return { lines: [], classNames: [] };

  const exportable = annotations.filter((ann) => {
    if (ann.segmentation?.trim()) return true;
    return Boolean(ann.bbox && ann.bbox.length >= 4);
  });

  if (!exportable.length) return { lines: [], classNames: [] };

  const usedOldIds = [...new Set(exportable.map((a) => a.class_id))].sort((a, b) => a - b);
  const oldToNew = new Map<number, number>();
  usedOldIds.forEach((oldId, i) => oldToNew.set(oldId, i));

  const classNames = usedOldIds.map((oldId) => {
    const first = exportable.find((a) => a.class_id === oldId);
    const fromAnn = first?.class?.trim();
    if (fromAnn) return fromAnn;
    if (oldId >= 0 && oldId < samClasses.length) return samClasses[oldId];
    return `class_${oldId}`;
  });

  const lines: string[] = [];
  for (const ann of exportable) {
    const newId = oldToNew.get(ann.class_id);
    if (newId === undefined) continue;
    const seg = ann.segmentation?.trim();
    if (seg) {
      lines.push(`${newId} ${seg}`);
    } else if (ann.bbox && ann.bbox.length >= 4) {
      lines.push(bboxToYoloDetectLine(newId, ann.bbox, imgW, imgH));
    }
  }

  return { lines, classNames };
}

export function stemFromImageName(name: string): string {
  const base = name.split(/[/\\]/).pop() || 'export';
  const i = base.lastIndexOf('.');
  return i > 0 ? base.slice(0, i) : base;
}

export function triggerDownload(filename: string, content: string, mime = 'text/plain;charset=utf-8'): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
