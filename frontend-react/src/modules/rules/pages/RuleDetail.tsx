import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Button, Card, CardHeader } from '../../../components/common';
import { RuleHistory } from '../components/RuleHistory';
import { rulesApi } from '../api/rulesApi';
import type { RuleDetail as RuleDetailType, RuleVersionEntry } from '../../../types';
import styles from './RuleDetail.module.css';

export const RuleDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [rule, setRule] = useState<RuleDetailType | null>(null);
  const [versions, setVersions] = useState<RuleVersionEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [verLoading, setVerLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    let c = false;
    (async () => {
      try {
        const res = await rulesApi.get(id);
        const body = (res.data as { data?: RuleDetailType })?.data;
        if (!c) setRule(body || null);
      } catch {
        if (!c) {
          setRule({
            id,
            name: '示例规则',
            description: '后端未就绪时的占位',
            version: '1.0.0',
            status: 'draft',
            updatedAt: new Date().toISOString(),
            content: '请接入 GET /governance/rules/:id 获取真实数据。',
            tags: ['demo'],
          });
        }
      } finally {
        if (!c) setLoading(false);
      }
    })();
    return () => {
      c = true;
    };
  }, [id]);

  useEffect(() => {
    if (!id) return;
    let c = false;
    (async () => {
      try {
        const res = await rulesApi.history(id);
        const body = (res.data as { data?: { versions?: RuleVersionEntry[] } })?.data;
        if (!c) setVersions(body?.versions ?? []);
      } catch {
        if (!c) {
          setVersions([
            { id: 'v1', version: '1.0.0', createdAt: new Date().toISOString(), summary: '初始版本' },
          ]);
        }
      } finally {
        if (!c) setVerLoading(false);
      }
    })();
    return () => {
      c = true;
    };
  }, [id]);

  if (!id) return null;

  if (loading || !rule) {
    return <p className={styles.muted}>加载中…</p>;
  }

  return (
    <div className={styles.page}>
      <div className={styles.top}>
        <Link to="/governance/rules" className={styles.back}>← 规则列表</Link>
        <div className={styles.actions}>
          <Link to={`/governance/rules/${id}/edit`}>
            <Button variant="primary" size="sm">编辑</Button>
          </Link>
        </div>
      </div>
      <Card>
        <CardHeader title={rule.name} />
        {rule.description ? <p className={styles.desc}>{rule.description}</p> : null}
        {rule.tags?.length ? (
          <div className={styles.tags}>
            {rule.tags.map((t) => (
              <span key={t} className={styles.tag}>{t}</span>
            ))}
          </div>
        ) : null}
        <pre className={styles.content}>{rule.content}</pre>
      </Card>
      <div className={styles.history}>
        <RuleHistory versions={versions} loading={verLoading} />
      </div>
    </div>
  );
};
