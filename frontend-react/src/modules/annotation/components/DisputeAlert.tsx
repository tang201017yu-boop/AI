import React from 'react';
import type { DisputeAlertInfo } from '../../../types';
import styles from './DisputeAlert.module.css';

export interface DisputeAlertProps {
  items: DisputeAlertInfo[];
}

const levelClass: Record<DisputeAlertInfo['level'], string> = {
  info: styles.info,
  warning: styles.warning,
  error: styles.error,
};

export const DisputeAlert: React.FC<DisputeAlertProps> = ({ items }) => {
  if (items.length === 0) return null;
  return (
    <ul className={styles.list}>
      {items.map((a) => (
        <li key={a.id} className={`${styles.item} ${levelClass[a.level]}`}>
          {a.message}
        </li>
      ))}
    </ul>
  );
};
