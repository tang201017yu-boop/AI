import React from 'react';
import { Card } from '../../../components/common';
import type { ArbitrationStats } from '../../../types';
import styles from './StatsPanel.module.css';

export interface StatsPanelProps {
  stats: ArbitrationStats | null;
  loading?: boolean;
}

export const StatsPanel: React.FC<StatsPanelProps> = ({ stats, loading }) => (
  <div className={styles.row}>
    <Card className={styles.cell}>
      <div className={styles.num}>{loading ? '--' : stats?.open ?? 0}</div>
      <div className={styles.lbl}>待处理</div>
    </Card>
    <Card className={styles.cell}>
      <div className={styles.num}>{loading ? '--' : stats?.inReview ?? 0}</div>
      <div className={styles.lbl}>审核中</div>
    </Card>
    <Card className={styles.cell}>
      <div className={styles.num}>{loading ? '--' : stats?.resolved ?? 0}</div>
      <div className={styles.lbl}>已解决</div>
    </Card>
    <Card className={styles.cell}>
      <div className={styles.num}>{loading ? '--' : stats?.total ?? 0}</div>
      <div className={styles.lbl}>总数</div>
    </Card>
  </div>
);
