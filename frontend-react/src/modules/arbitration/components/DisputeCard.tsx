import React from 'react';
import { Link } from 'react-router-dom';
import { Card } from '../../../components/common';
import { Badge } from '../../../shared/components';
import type { DisputeStatus, DisputeSummary } from '../../../types';
import styles from './DisputeCard.module.css';

const statusMap: Record<DisputeStatus, { label: string; variant: 'warning' | 'info' | 'success' | 'danger' }> = {
  open: { label: '待处理', variant: 'warning' },
  review: { label: '审核中', variant: 'info' },
  resolved: { label: '已解决', variant: 'success' },
  rejected: { label: '已驳回', variant: 'danger' },
};

export interface DisputeCardProps {
  dispute: DisputeSummary;
}

export const DisputeCard: React.FC<DisputeCardProps> = ({ dispute }) => {
  const status = statusMap[dispute.status];
  return (
    <Link to={`/governance/arbitration/${dispute.id}`} className={styles.link}>
      <Card className={styles.card}>
        <div className={styles.row}>
          <h3 className={styles.title}>{dispute.title}</h3>
          <Badge variant={status.variant}>{status.label}</Badge>
        </div>
        {dispute.scenarioLabel ? <p className={styles.meta}>{dispute.scenarioLabel}</p> : null}
        {dispute.triggerIou != null ? <p className={styles.meta}>IoU: {dispute.triggerIou.toFixed(3)}</p> : null}
        {dispute.decisionAction ? <p className={styles.meta}>{dispute.decisionAction}</p> : null}
        <p className={styles.time}>更新于 {new Date(dispute.updatedAt).toLocaleString()}</p>
      </Card>
    </Link>
  );
};
