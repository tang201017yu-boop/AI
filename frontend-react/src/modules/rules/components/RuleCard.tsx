import React from 'react';
import { Link } from 'react-router-dom';
import { Card } from '../../../components/common';
import { Badge } from '../../../shared/components';
import type { RuleSummary, RuleStatus } from '../../../types';
import styles from './RuleCard.module.css';

const statusVariant: Record<RuleStatus, 'default' | 'success' | 'warning' | 'info'> = {
  draft: 'default',
  published: 'success',
  archived: 'warning',
};

const statusLabel: Record<RuleStatus, string> = {
  draft: '草稿',
  published: '已发布',
  archived: '已归档',
};

export interface RuleCardProps {
  rule: RuleSummary;
}

export const RuleCard: React.FC<RuleCardProps> = ({ rule }) => (
  <Link to={`/governance/rules/${rule.id}`} className={styles.link}>
    <Card className={styles.card}>
      <div className={styles.row}>
        <h3 className={styles.title}>{rule.name}</h3>
        <Badge variant={statusVariant[rule.status]}>{statusLabel[rule.status]}</Badge>
      </div>
      {rule.description ? <p className={styles.desc}>{rule.description}</p> : null}
      <div className={styles.meta}>
        <span>v{rule.version}</span>
        <span>{new Date(rule.updatedAt).toLocaleString()}</span>
      </div>
    </Card>
  </Link>
);
