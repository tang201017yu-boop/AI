import React, { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Button, Card, CardHeader } from '../../../components/common';
import { AnnotationCompare } from '../components/AnnotationCompare';
import { JudgeResult } from '../components/JudgeResult';
import { arbitrationApi } from '../api/arbitrationApi';
import type { ArbitrationCaseDetail, CompareAnnotation, JudgeOutcome } from '../../../types';
import styles from './DisputeDetail.module.css';

function metricLines(input?: Record<string, unknown>) {
  if (!input) return [];
  return Object.entries(input).map(([key, value]) => `${key}: ${String(value)}`);
}

export const DisputeDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [detail, setDetail] = useState<ArbitrationCaseDetail | null>(null);
  const [outcome, setOutcome] = useState<JudgeOutcome | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await arbitrationApi.get(id);
        const payload = (res.data as { data?: ArbitrationCaseDetail })?.data ?? null;
        if (!cancelled) {
          setDetail(payload);
          setOutcome(payload?.judgement ?? null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id]);

  const left = useMemo<CompareAnnotation>(() => detail?.leftAnnotation ?? {
    id: 'left',
    label: 'Agent 标注',
    jsonSummary: '暂无数据',
  }, [detail]);

  const right = useMemo<CompareAnnotation>(() => detail?.rightAnnotation ?? {
    id: 'right',
    label: '人类标注',
    jsonSummary: '暂无数据',
  }, [detail]);

  if (!id) return null;
  if (loading) return <p className={styles.muted}>加载中...</p>;
  if (!detail) return <p className={styles.muted}>未找到案件</p>;

  return (
    <div className={styles.page}>
      <div className={styles.top}>
        <Link to="/governance/arbitration" className={styles.back}>返回仲裁中心</Link>
        <div className={styles.actions}>
          <Button
            variant="primary"
            size="sm"
            onClick={async () => {
              await arbitrationApi.submitJudgement(id, {
                valid: detail.decisionWinner !== 'expert',
                reason: detail.decisionAction,
              });
              setOutcome({
                valid: detail.decisionWinner !== 'expert',
                reason: detail.decisionAction,
                decidedAt: new Date().toISOString(),
                decidedBy: 'operator',
              });
            }}
          >
            确认裁决
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader title={detail.title} />
        <p className={styles.desc}>{detail.description}</p>
        <div className={styles.metaGrid}>
          <div>
            <span className={styles.label}>缺陷类型</span>
            <strong>{detail.defectType ?? '--'}</strong>
          </div>
          <div>
            <span className={styles.label}>场景</span>
            <strong>{detail.scenarioLabel ?? detail.scenario ?? '--'}</strong>
          </div>
          <div>
            <span className={styles.label}>IoU</span>
            <strong>{detail.triggerIou?.toFixed(3) ?? '--'}</strong>
          </div>
          <div>
            <span className={styles.label}>动作</span>
            <strong>{detail.decisionAction ?? '--'}</strong>
          </div>
        </div>
      </Card>

      <div className={styles.section}>
        <h4 className={styles.subheading}>标注对比</h4>
        <AnnotationCompare left={left} right={right} />
      </div>

      <div className={styles.ruleGrid}>
        <Card className={styles.ruleCard}>
          <h4 className={styles.subheading}>Agent 规则命中</h4>
          <ul className={styles.ruleList}>
            {(detail.ruleHits?.agent ?? []).map((rule) => (
              <li key={rule.id}>
                {rule.label}: {rule.value} {rule.operator} {rule.threshold} {rule.passed ? '通过' : '未通过'}
              </li>
            ))}
          </ul>
          <div className={styles.metricLines}>
            {metricLines(detail.leftAnnotation?.attributes).map((line) => (
              <p key={line}>{line}</p>
            ))}
          </div>
        </Card>
        <Card className={styles.ruleCard}>
          <h4 className={styles.subheading}>人类规则命中</h4>
          <ul className={styles.ruleList}>
            {(detail.ruleHits?.human ?? []).map((rule) => (
              <li key={rule.id}>
                {rule.label}: {rule.value} {rule.operator} {rule.threshold} {rule.passed ? '通过' : '未通过'}
              </li>
            ))}
          </ul>
          <div className={styles.metricLines}>
            {metricLines(detail.rightAnnotation?.attributes).map((line) => (
              <p key={line}>{line}</p>
            ))}
          </div>
        </Card>
      </div>

      <JudgeResult outcome={outcome} />
    </div>
  );
};
