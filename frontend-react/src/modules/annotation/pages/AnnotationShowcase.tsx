import React, { useEffect, useMemo, useState } from 'react';
import { Card, CardHeader } from '../../../components/common';
import { AgentSuggestion } from '../components/AgentSuggestion';
import { AttributePanel } from '../components/AttributePanel';
import { DisputeAlert } from '../components/DisputeAlert';
import { LegalBadge } from '../components/LegalBadge';
import { useDisputeDetection } from '../hooks/useDisputeDetection';
import { usePhysicalJudge } from '../hooks/usePhysicalJudge';
import type { AgentSuggestionItem, ArbitrationEvaluatePayload, AttributeField } from '../../../types';
import styles from './AnnotationShowcase.module.css';

const samples: Record<string, ArbitrationEvaluatePayload> = {
  crack: {
    annotation_id: 'demo-crack-001',
    defect_type: 'crack',
    source: {
      project: 'demo',
      image_name: 'sample-crack.jpg',
    },
    agent_annotation: {
      label: 'crack',
      bbox: [40, 80, 200, 120],
      attributes: {
        area: 2500,
        width_length_ratio: 0.06,
      },
    },
    human_annotation: {
      label: 'crack',
      bbox: [40, 80, 200, 120],
      attributes: {
        area: 5000,
        width_length_ratio: 0.12,
      },
    },
  },
  seepage: {
    annotation_id: 'demo-seepage-001',
    defect_type: 'seepage',
    source: {
      project: 'demo',
      image_name: 'sample-seepage.jpg',
    },
    agent_annotation: {
      label: 'seepage',
      bbox: [60, 40, 210, 180],
      attributes: {
        wetness_index: 0.34,
        connected_components: 7,
      },
    },
    human_annotation: {
      label: 'seepage',
      bbox: [60, 40, 210, 180],
      attributes: {
        wetness_index: 0.62,
        connected_components: 3,
      },
    },
  },
};

function fieldsFromRecord(title: string, input?: Record<string, string | number>): AttributeField[] {
  if (!input) return [];
  return Object.entries(input).map(([key, value]) => ({
    key: `${title}-${key}`,
    label: key,
    value: String(value),
  }));
}

export const AnnotationShowcase: React.FC = () => {
  const [sampleKey, setSampleKey] = useState<'crack' | 'seepage'>('crack');
  const { alerts, evaluation, loading, scan, clear } = useDisputeDetection();
  const { judgement, evaluate, reset } = usePhysicalJudge();

  const payload = samples[sampleKey];

  useEffect(() => {
    void scan(payload);
    const attrs = payload.agent_annotation.attributes;
    evaluate(
      {
        x1: payload.agent_annotation.bbox[0],
        y1: payload.agent_annotation.bbox[1],
        x2: payload.agent_annotation.bbox[2],
        y2: payload.agent_annotation.bbox[3],
      },
      payload.defect_type,
      attrs
    );
  }, [evaluate, payload, sampleKey, scan]);

  const suggestionItems = useMemo<AgentSuggestionItem[]>(() => {
    if (!evaluation) return [];
    const winner = evaluation.decisionWinner;
    const reason = `${evaluation.scenarioLabel}，${evaluation.decisionAction}`;
    return [
      {
        id: 'winner',
        className:
          winner === 'agent'
            ? '采信 Agent'
            : winner === 'human'
              ? '采信人类'
              : winner === 'expert'
                ? '升级专家'
                : '双合法融合',
        confidence: Math.max(0.55, evaluation.iou),
        reason,
      },
    ];
  }, [evaluation]);

  const agentFields = useMemo(
    () => fieldsFromRecord('agent', evaluation?.case.leftAnnotation?.attributes),
    [evaluation]
  );
  const humanFields = useMemo(
    () => fieldsFromRecord('human', evaluation?.case.rightAnnotation?.attributes),
    [evaluation]
  );

  return (
    <div className={styles.page}>
      <Card>
        <CardHeader title="标注增强与物理约束仲裁" />
        <p className={styles.lead}>
          这里接入了真实的 <code>/api/v1/governance/arbitration/*</code> 接口，与专利流程图一致：
          当 IoU ≥ T_iou（默认 0.3）时进入仲裁入口，再结合缺陷类型做物理规则与四场景判决。
        </p>

        <div className={styles.toolbar}>
          <button
            type="button"
            className={sampleKey === 'crack' ? styles.btn : styles.btnGhost}
            onClick={() => setSampleKey('crack')}
          >
            裂缝样例
          </button>
          <button
            type="button"
            className={sampleKey === 'seepage' ? styles.btn : styles.btnGhost}
            onClick={() => setSampleKey('seepage')}
          >
            渗水样例
          </button>
          <button type="button" className={styles.btnGhost} onClick={() => void scan(payload)}>
            重新仲裁
          </button>
          <button
            type="button"
            className={styles.btnGhost}
            onClick={() => {
              clear();
              reset();
            }}
          >
            清空
          </button>
        </div>

        <DisputeAlert items={alerts} />

        <div className={styles.statusGrid}>
          <div className={styles.statusCard}>
            <span className={styles.statusLabel}>当前缺陷</span>
            <strong>{payload.defect_type}</strong>
          </div>
          <div className={styles.statusCard}>
            <span className={styles.statusLabel}>IoU</span>
            <strong>{evaluation ? evaluation.iou.toFixed(3) : '--'}</strong>
          </div>
          <div className={styles.statusCard}>
            <span className={styles.statusLabel}>仲裁场景</span>
            <strong>{evaluation?.scenarioLabel ?? '--'}</strong>
          </div>
          <div className={styles.statusCard}>
            <span className={styles.statusLabel}>动作</span>
            <strong>{evaluation?.decisionAction ?? (loading ? '计算中' : '--')}</strong>
          </div>
        </div>

        <div className={styles.judgeRow}>
          <div className={styles.judgeBox}>
            <div className={styles.inlineHeading}>Agent 标注</div>
            <LegalBadge legality={evaluation?.agentReview.legality ?? judgement?.legality ?? 'unknown'} />
            <p className={styles.note}>{evaluation?.agentReview.notes.join('；') ?? judgement?.notes ?? '等待判定'}</p>
          </div>
          <div className={styles.judgeBox}>
            <div className={styles.inlineHeading}>人类标注</div>
            <LegalBadge legality={evaluation?.humanReview.legality ?? 'unknown'} />
            <p className={styles.note}>{evaluation?.humanReview.notes.join('；') ?? '等待判定'}</p>
          </div>
        </div>
      </Card>

      <div className={styles.split}>
        <AgentSuggestion items={suggestionItems} />
        <AttributePanel title="Agent 物理属性" fields={agentFields} />
      </div>

      <div className={styles.split}>
        <AttributePanel title="人类物理属性" fields={humanFields} />
        <Card className={styles.summaryCard}>
          <h4 className={styles.inlineHeading}>规则命中</h4>
          <div className={styles.ruleGroup}>
            <strong>Agent</strong>
            <ul className={styles.ruleList}>
              {(evaluation?.agentReview.rule_hits ?? []).map((rule) => (
                <li key={rule.id}>
                  {rule.label}: {rule.value} {rule.operator} {rule.threshold} {rule.passed ? '通过' : '未通过'}
                </li>
              ))}
            </ul>
          </div>
          <div className={styles.ruleGroup}>
            <strong>人类</strong>
            <ul className={styles.ruleList}>
              {(evaluation?.humanReview.rule_hits ?? []).map((rule) => (
                <li key={rule.id}>
                  {rule.label}: {rule.value} {rule.operator} {rule.threshold} {rule.passed ? '通过' : '未通过'}
                </li>
              ))}
            </ul>
          </div>
        </Card>
      </div>
    </div>
  );
};
