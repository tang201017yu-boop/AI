import React from 'react';
import styles from './Card.module.css';

interface CardProps {
  children: React.ReactNode;
  variant?: 'default' | 'elevated' | 'highlight';
  className?: string;
}

interface CardHeaderProps {
  title: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({ children, variant = 'default', className = '' }) => {
  const classes = [styles.card, variant !== 'default' && styles[variant], className]
    .filter(Boolean)
    .join(' ');
  return <div className={classes}>{children}</div>;
};

export const CardHeader: React.FC<CardHeaderProps> = ({ title, icon, action }) => (
  <div className={styles.header}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
      {icon && <div className={styles.icon}>{icon}</div>}
      <h3 className={styles.title}>{title}</h3>
    </div>
    {action && <div>{action}</div>}
  </div>
);
