import React from 'react';
import { Button, Card } from '../../../components/common';
import type { AgentSuggestionItem } from '../../../types';
import styles from './AgentSuggestion.module.css';

export interface AgentSuggestionProps {
  items: AgentSuggestionItem[];
  onPick?: (item: AgentSuggestionItem) => void;
}

export const AgentSuggestion: React.FC<AgentSuggestionProps> = ({ items, onPick }) => (
  <Card className={styles.card}>
    <h4 className={styles.heading}>智能建议</h4>
    {items.length === 0 ? <p className={styles.muted}>暂无建议</p> : null}
    <ul className={styles.list}>
      {items.map((it) => (
        <li key={it.id} className={styles.item}>
          <div>
            <span className={styles.name}>{it.className}</span>
            <span className={styles.conf}>{(it.confidence * 100).toFixed(0)}%</span>
            {it.reason ? <p className={styles.reason}>{it.reason}</p> : null}
          </div>
          {onPick ? (
            <Button size="sm" variant="secondary" onClick={() => onPick(it)}>
              采用
            </Button>
          ) : null}
        </li>
      ))}
    </ul>
  </Card>
);
