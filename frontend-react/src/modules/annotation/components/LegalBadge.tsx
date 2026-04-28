import React from 'react';
import { Badge } from '../../../shared/components';
import type { LegalityLabel } from '../../../types';
import styles from './LegalBadge.module.css';

const map: Record<LegalityLabel, { label: string; variant: 'default' | 'success' | 'warning' | 'danger' | 'info' }> = {
  unknown: { label: '待判定', variant: 'default' },
  compliant: { label: '合规', variant: 'success' },
  risk: { label: '风险', variant: 'warning' },
  violation: { label: '违规', variant: 'danger' },
};

export interface LegalBadgeProps {
  legality: LegalityLabel;
  className?: string;
}

export const LegalBadge: React.FC<LegalBadgeProps> = ({ legality, className }) => {
  const m = map[legality];
  return (
    <span className={styles.wrap}>
      <Badge variant={m.variant} className={className}>
        合法性 · {m.label}
      </Badge>
    </span>
  );
};
