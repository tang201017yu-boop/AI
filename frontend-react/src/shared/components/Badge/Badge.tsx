import React from 'react';
import styles from './Badge.module.css';

export type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info';

export interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({ children, variant = 'default', className }) => (
  <span className={`${styles.badge} ${styles[variant]} ${className ?? ''}`}>{children}</span>
);
