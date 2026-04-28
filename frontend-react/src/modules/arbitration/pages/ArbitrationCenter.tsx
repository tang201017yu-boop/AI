import React, { useEffect, useState } from 'react';
import { Card, CardHeader } from '../../../components/common';
import { DisputeCard } from '../components/DisputeCard';
import { StatsPanel } from '../components/StatsPanel';
import { arbitrationApi } from '../api/arbitrationApi';
import type { ArbitrationStats, DisputeSummary } from '../../../types';
import styles from './ArbitrationCenter.module.css';

function unwrapList(data: unknown): DisputeSummary[] {
  if (data == null) return [];
  if (Array.isArray(data)) return data as DisputeSummary[];
  const payload = data as { items?: DisputeSummary[]; data?: { items?: DisputeSummary[] } };
  if (Array.isArray(payload.items)) return payload.items;
  if (Array.isArray(payload.data?.items)) return payload.data.items;
  return [];
}

export const ArbitrationCenter: React.FC = () => {
  const [list, setList] = useState<DisputeSummary[]>([]);
  const [stats, setStats] = useState<ArbitrationStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [listRes, statsRes] = await Promise.all([arbitrationApi.list(), arbitrationApi.stats()]);
        if (cancelled) return;
        setList(unwrapList((listRes.data as { data?: unknown })?.data ?? listRes.data));
        setStats((statsRes.data as { data?: ArbitrationStats })?.data ?? null);
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
      <StatsPanel stats={stats} loading={loading} />
      {stats ? (
        <Card className={styles.tipCard}>
          <CardHeader title="仲裁覆盖情况" />
          <p className={styles.tipText}>
            当前自动裁决覆盖率 {stats.coverage ?? 0}% ，预计专家介入减少 {stats.expertReduction ?? 0}% 。
          </p>
        </Card>
      ) : null}
      <Card>
        <CardHeader title="仲裁案件" />
        {loading ? <p className={styles.muted}>加载中...</p> : null}
        <div className={styles.grid}>
          {list.map((dispute) => (
            <DisputeCard key={dispute.id} dispute={dispute} />
          ))}
        </div>
        {!loading && list.length === 0 ? <p className={styles.muted}>暂无案件</p> : null}
      </Card>
    </div>
  );
};
