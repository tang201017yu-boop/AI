import React from 'react';
import { ImageBox } from '../../../shared/components';
import type { CompareAnnotation } from '../../../types';
import styles from './AnnotationCompare.module.css';

export interface AnnotationCompareProps {
  left: CompareAnnotation;
  right: CompareAnnotation;
}

export const AnnotationCompare: React.FC<AnnotationCompareProps> = ({ left, right }) => (
  <div className={styles.grid}>
    <div>
      <div className={styles.label}>{left.label}</div>
      {left.dataUrl ? (
        <ImageBox src={left.dataUrl} alt={left.label} />
      ) : (
        <pre className={styles.fallback}>{left.jsonSummary || '无预览'}</pre>
      )}
    </div>
    <div>
      <div className={styles.label}>{right.label}</div>
      {right.dataUrl ? (
        <ImageBox src={right.dataUrl} alt={right.label} />
      ) : (
        <pre className={styles.fallback}>{right.jsonSummary || '无预览'}</pre>
      )}
    </div>
  </div>
);
