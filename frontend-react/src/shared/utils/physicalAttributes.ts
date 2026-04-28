import type { AnnotationBox } from '../../types';

/** 边界框面积（像素） */
export function boxArea(b: AnnotationBox): number {
  const w = Math.abs(b.x2 - b.x1);
  const h = Math.abs(b.y2 - b.y1);
  return w * h;
}

/** 宽高比 (>= 1) */
export function boxAspectRatio(b: AnnotationBox): number {
  const w = Math.abs(b.x2 - b.x1);
  const h = Math.abs(b.y2 - b.y1);
  if (h === 0) return 0;
  return w >= h ? w / h : h / w;
}

/** 中心点 */
export function boxCenter(b: AnnotationBox): { x: number; y: number } {
  return {
    x: (b.x1 + b.x2) / 2,
    y: (b.y1 + b.y2) / 2,
  };
}
