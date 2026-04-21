import React from 'react';

const p = (size: number) => ({
  width: size,
  height: size,
  viewBox: '0 0 24 24',
  fill: 'none',
  xmlns: 'http://www.w3.org/2000/svg',
  'aria-hidden': true as const,
});

/** 系统状态 — 折线趋势 */
export function IconHomeChart({ size = 22 }: { size?: number }) {
  const s = p(size);
  return (
    <svg {...s}>
      <path
        d="M4 18V6M8 18v-6l4-4 4 3 4-5"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M4 18h16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.45" />
    </svg>
  );
}

/** AI 工作流 — 齿轮 */
export function IconHomeWorkflow({ size = 22 }: { size?: number }) {
  const s = p(size);
  return (
    <svg {...s}>
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.65" fill="none" />
      <path
        d="M12 2v2M12 20v2M2 12h2M20 12h2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** 智能方案总览 — 灯泡轮廓 */
export function IconHomeSolutions({ size = 22 }: { size?: number }) {
  const s = p(size);
  return (
    <svg {...s}>
      <path
        d="M9 18h6M10 22h4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M12 2a7 7 0 00-3 13.2V17h6v-1.8A7 7 0 0012 2z"
        stroke="currentColor"
        strokeWidth="1.65"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** 主按钮：智能标注 */
export function IconHomeAnnotation({ size = 20 }: { size?: number }) {
  const s = p(size);
  return (
    <svg {...s}>
      <path
        d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
        fill="currentColor"
        fillOpacity="0.12"
      />
      <path d="M4 20l4-8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/** 主按钮：开始训练 */
export function IconHomeTrain({ size = 20 }: { size?: number }) {
  const s = p(size);
  return (
    <svg {...s}>
      <path
        d="M4.5 16.5l2-6 3 2 4-8 3 5 3.5-1"
        stroke="currentColor"
        strokeWidth="1.65"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M19 11v6M16 14h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
