import React from 'react';
import styles from './ImageBox.module.css';

export interface ImageBoxProps {
  src: string;
  alt: string;
  children?: React.ReactNode;
  className?: string;
}

/**
 * 带可选叠层的图像容器，用于标注/对比等场景
 */
export const ImageBox: React.FC<ImageBoxProps> = ({ src, alt, children, className }) => (
  <div className={`${styles.wrap} ${className ?? ''}`}>
    <img className={styles.img} src={src} alt={alt} />
    {children ? <div className={styles.overlay}>{children}</div> : null}
  </div>
);
