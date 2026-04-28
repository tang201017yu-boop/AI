import React from 'react';
import { Card } from '../../../components/common';
import type { AttributeField } from '../../../types';
import styles from './AttributePanel.module.css';

export interface AttributePanelProps {
  title?: string;
  fields: AttributeField[];
}

export const AttributePanel: React.FC<AttributePanelProps> = ({ title = '属性', fields }) => (
  <Card className={styles.card}>
    <h4 className={styles.heading}>{title}</h4>
    {fields.length === 0 ? <p className={styles.muted}>无属性</p> : null}
    <dl className={styles.dl}>
      {fields.map((f) => (
        <div key={f.key} className={styles.row}>
          <dt className={styles.dt}>{f.label}</dt>
          <dd className={styles.dd}>{f.value}</dd>
        </div>
      ))}
    </dl>
  </Card>
);
