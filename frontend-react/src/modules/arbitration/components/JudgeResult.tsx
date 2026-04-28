import React from 'react';
import { Card } from '../../../components/common';
import { Badge } from '../../../shared/components';
import type { JudgeOutcome } from '../../../types';
import styles from './JudgeResult.module.css';

export interface JudgeResultProps {
  outcome: JudgeOutcome | null;
}

export const JudgeResult: React.FC<JudgeResultProps> = ({ outcome }) => {
  if (!outcome) {
    return (
      <Card className={styles.card}>
        <p className={styles.muted}>暂无裁定</p>
      </Card>
    );
  }
  return (
    <Card className={styles.card}>
      <div className={styles.row}>
        <span className={styles.title}>裁定结果</span>
        <Badge variant={outcome.valid ? 'success' : 'danger'}>
          {outcome.valid ? '通过' : '不通过'}
        </Badge>
      </div>
      {outcome.score != null ? <p className={styles.line}>评分：{outcome.score}</p> : null}
      {outcome.reason ? <p className={styles.reason}>{outcome.reason}</p> : null}
      <p className={styles.meta}>
        {outcome.decidedBy ? `${outcome.decidedBy} · ` : null}
        {outcome.decidedAt ? new Date(outcome.decidedAt).toLocaleString() : null}
      </p>
    </Card>
  );
};
