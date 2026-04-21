import React from 'react';

const svgProps = (size: number) =>
  ({
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    xmlns: 'http://www.w3.org/2000/svg',
    'aria-hidden': true as const,
  });

/** 智能方案卡片 / 详情顶栏：与方案 name 对应的矢量图标（currentColor） */
export function SolutionFeatureIcon({ name, size = 24 }: { name: string; size?: number }) {
  const p = svgProps(size);
  switch (name) {
    case 'object-counting':
      return (
        <svg {...p}>
          <rect x="4" y="12" width="3.5" height="8" rx="1" fill="currentColor" opacity="0.85" />
          <rect x="10.25" y="8" width="3.5" height="12" rx="1" fill="currentColor" />
          <rect x="16.5" y="4" width="3.5" height="16" rx="1" fill="currentColor" opacity="0.75" />
        </svg>
      );
    case 'heatmap':
      return (
        <svg {...p}>
          <path
            d="M12 3c-2 3-4 4.5-4 8a4 4 0 008 0c0-3.5-2-5-4-8z"
            fill="currentColor"
            opacity="0.9"
          />
          <path d="M12 10v11M9 14h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      );
    case 'speed-estimation':
      return (
        <svg {...p}>
          <path
            d="M5 16h14v2.5H5V16zm1.5-2.2 2.5-6.5 3 2.8 2.5-4.3 2.8 4.3H6.5z"
            fill="currentColor"
            opacity="0.9"
          />
          <circle cx="7.5" cy="17.8" r="1.2" fill="var(--gray-700, #374151)" />
          <circle cx="16.5" cy="17.8" r="1.2" fill="var(--gray-700, #374151)" />
        </svg>
      );
    case 'distance-calculation':
      return (
        <svg {...p}>
          <rect x="3" y="10" width="18" height="4" rx="1" fill="currentColor" opacity="0.35" />
          <path d="M5 10V6M9 10V5M13 10V7M17 10V5M21 10V8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      );
    case 'object-blur':
      return (
        <svg {...p}>
          <rect x="5" y="9" width="14" height="10" rx="2" stroke="currentColor" strokeWidth="2" />
          <path d="M8 12h8M8 15h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.6" />
          <circle cx="12" cy="7" r="1.5" fill="currentColor" />
        </svg>
      );
    case 'object-crop':
      return (
        <svg {...p}>
          <circle cx="9" cy="7" r="2" stroke="currentColor" strokeWidth="1.6" fill="none" />
          <circle cx="15" cy="17" r="2" stroke="currentColor" strokeWidth="1.6" fill="none" />
          <path d="M10.5 8.5L6 20M13.5 15.5L18 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      );
    case 'queue-management':
      return (
        <svg {...p}>
          <circle cx="8" cy="9" r="3" stroke="currentColor" strokeWidth="1.8" />
          <circle cx="16" cy="9" r="3" stroke="currentColor" strokeWidth="1.8" />
          <path d="M5 19c0-2.2 1.8-4 4-4h2c2.2 0 4 1.8 4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      );
    case 'parking-management':
      return (
        <svg {...p}>
          <rect x="4" y="4" width="16" height="16" rx="3" stroke="currentColor" strokeWidth="2" fill="none" />
          <path d="M9 7h1.5v10H9V7z" fill="currentColor" />
          <path
            d="M10.5 7h2.5a2.5 2.5 0 010 5h-2.5"
            stroke="currentColor"
            strokeWidth="1.5"
            fill="none"
            strokeLinecap="round"
          />
        </svg>
      );
    case 'vision-eye':
      return (
        <svg {...p}>
          <ellipse cx="12" cy="12" rx="9" ry="5" stroke="currentColor" strokeWidth="1.8" />
          <circle cx="12" cy="12" r="2.5" fill="currentColor" />
        </svg>
      );
    case 'workout-monitoring':
      return (
        <svg {...p}>
          <path d="M6 10h2v4H6V10zm10 0h2v4h-2V10z" fill="currentColor" />
          <path d="M8 11.5h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      );
    default:
      return (
        <svg {...p}>
          <rect x="4" y="4" width="16" height="16" rx="3" stroke="currentColor" strokeWidth="1.8" />
        </svg>
      );
  }
}

export function IconMediaImage({ size = 14 }: { size?: number }) {
  return (
    <svg {...svgProps(size)} style={{ display: 'block', flexShrink: 0 }}>
      <rect x="3" y="5" width="14" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="8.5" cy="9.5" r="1.2" fill="currentColor" opacity="0.7" />
      <path d="M3 14l4-3 3 2 4-5 3 4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function IconMediaVideo({ size = 14 }: { size?: number }) {
  return (
    <svg {...svgProps(size)} style={{ display: 'block', flexShrink: 0 }}>
      <rect x="4" y="6" width="12" height="10" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
      <path d="M16 9l4-2v8l-4-2V9z" fill="currentColor" opacity="0.85" />
    </svg>
  );
}

/** 「处理结果」等区块标题用小图标 */
export function IconResultChart({ size = 18 }: { size?: number }) {
  return <SolutionFeatureIcon name="object-counting" size={size} />;
}
