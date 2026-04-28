import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button, Card, CardHeader } from '../../../components/common';
import { RuleCard } from '../components/RuleCard';
import { rulesApi } from '../api/rulesApi';
import type { RuleSummary } from '../../../types';
import styles from './RulesList.module.css';

function unwrapList(d: unknown): RuleSummary[] {
  if (d == null) return [];
  if (Array.isArray(d)) return d as RuleSummary[];
  const obj = d as { items?: RuleSummary[]; data?: { items?: RuleSummary[] } | RuleSummary[] };
  if (Array.isArray(obj.items)) return obj.items;
  if (obj.data) {
    if (Array.isArray(obj.data)) return obj.data;
    if (Array.isArray(obj.data.items)) return obj.data.items;
  }
  return [];
}

const demoRules: RuleSummary[] = [
  {
    id: 'demo-1',
    name: '示例：检测框最小边长',
    description: '未接入后端时展示的示例数据',
    version: '1.0.0',
    status: 'draft',
    updatedAt: new Date().toISOString(),
  },
];

export const RulesList: React.FC = () => {
  const [rules, setRules] = useState<RuleSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [useDemo, setUseDemo] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await rulesApi.list();
        const body = (res.data as { data?: unknown })?.data ?? res.data;
        const list = unwrapList(body);
        if (!cancelled) {
          if (list.length) setRules(list);
          else {
            setRules(demoRules);
            setUseDemo(true);
          }
        }
      } catch {
        if (!cancelled) {
          setRules(demoRules);
          setUseDemo(true);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className={styles.page}>
      <Card>
        <div className={styles.head}>
          <CardHeader
            title="规则库"
            action={
              <Link to="/governance/rules/new">
                <Button variant="primary" size="sm">新建规则</Button>
              </Link>
            }
          />
        </div>
        {useDemo ? (
          <p className={styles.hint}>
            当前为示例或离线数据。接入后端 <code>/api/v1/governance/rules</code> 后将显示真实规则。
          </p>
        ) : null}
        {loading ? <p className={styles.muted}>加载中…</p> : null}
        <div className={styles.grid}>
          {rules.map((r) => (
            <RuleCard key={r.id} rule={r} />
          ))}
        </div>
        {!loading && rules.length === 0 ? <p className={styles.muted}>暂无规则</p> : null}
      </Card>
    </div>
  );
};
