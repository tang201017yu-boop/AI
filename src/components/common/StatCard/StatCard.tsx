import React from 'react';
import styles from './StatCard.module.css';

interface StatCardProps {
  value: string | number;
  label: string;
  variant?: 'default' | 'accent' | 'success';
}

export const StatCard: React.FC<StatCardProps> = ({ value, label, variant = 'default' }) => {
  return (
    <div className={`${styles.statCard} ${variant !== 'default' ? styles[variant] : ''}`}>
      <div className={styles.value}>{value}</div>
      <div className={styles.label}>{label}</div>
    </div>
  );
};
