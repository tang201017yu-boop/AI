import React from 'react';
import { Card } from '../../../components/common';
import type { RuleVersionEntry } from '../../../types';
import styles from './RuleHistory.module.css';

export interface RuleHistoryProps {
  versions: RuleVersionEntry[];
  loading?: boolean;
}

export const RuleHistory: React.FC<RuleHistoryProps> = ({ versions, loading }) => (
  <Card>
    <h4 className={styles.heading}>版本历史</h4>
    {loading ? <p className={styles.muted}>加载中…</p> : null}
    {!loading && versions.length === 0 ? <p className={styles.muted}>暂无历史版本</p> : null}
    <ul className={styles.list}>
      {versions.map((v) => (
        <li key={v.id} className={styles.item}>
          <div className={styles.ver}>{v.version}</div>
          <div className={styles.body}>
            {v.summary ? <div>{v.summary}</div> : null}
            <div className={styles.sub}>
              {v.createdBy ? `${v.createdBy} · ` : null}
              {new Date(v.createdAt).toLocaleString()}
            </div>
          </div>
        </li>
      ))}
    </ul>
  </Card>
);
