import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../../../components/common';
import { Badge } from '../../../shared/components';
import type {
  CompareAnnotation,
  DisputeStatus,
  DisputeSummary,
  ScenarioCode,
  SideJudge,
} from '../../../types';
import styles from './DisputeCard.module.css';

/* 状态徽章映射 */
const statusMap: Record<DisputeStatus, { label: string; variant: 'warning' | 'info' | 'success' | 'danger' }> = {
  open: { label: '待处理', variant: 'warning' },
  review: { label: '审核中', variant: 'info' },
  resolved: { label: '已解决', variant: 'success' },
  rejected: { label: '已驳回', variant: 'danger' },
};

/* 物理四场景配色（设计稿要求） */
const scenarioColors: Record<ScenarioCode, string> = {
  1: 'var(--warning)',     // 人类违规
  2: 'var(--info)',        // Agent 违规
  3: 'var(--gray-400)',    // 真实分歧（中性）
  4: 'var(--error)',       // 双方违规
};

const scenarioLabels: Record<ScenarioCode, string> = {
  1: '场景1: 人类违规，Agent 合法',
  2: '场景2: Agent 违规，人类合法',
  3: '场景3: 双方均合法（真实分歧）',
  4: '场景4: 双方均违规',
};

/* 判定结果徽章 */
function judgeBadge(judge?: SideJudge) {
  if (!judge) return <Badge variant="default">未知</Badge>;
  if (judge.result === 'compliant') return <Badge variant="success">合法</Badge>;
  if (judge.result === 'violation') return <Badge variant="danger">违规</Badge>;
  return <Badge variant="default">未知</Badge>;
}

/* 单边预览块（无图时回退为占位 SVG） */
const SidePreview: React.FC<{ title: string; annotation?: CompareAnnotation; judge?: SideJudge }> = ({
  title,
  annotation,
  judge,
}) => {
  const src = annotation?.preview || annotation?.dataUrl;
  return (
    <div className={styles.side}>
      <div className={styles.sideHead}>
        <span className={styles.sideTitle}>{title}</span>
        {judgeBadge(judge)}
      </div>
      <div className={styles.previewBox}>
        {src ? (
          <img src={src} alt={`${title} 预览`} className={styles.previewImg} />
        ) : (
          <div className={styles.previewPlaceholder} aria-label="暂无预览">
            <svg viewBox="0 0 64 64" width="48" height="48" aria-hidden="true">
              <rect x="4" y="10" width="56" height="44" rx="4" fill="#e2e8f0" />
              <circle cx="20" cy="26" r="5" fill="#cbd5e1" />
              <path d="M8 50 L26 32 L40 44 L56 28 L56 50 Z" fill="#cbd5e1" />
            </svg>
            <span className={styles.previewHint}>暂无预览</span>
          </div>
        )}
      </div>
    </div>
  );
};

export interface DisputeCardProps {
  dispute: DisputeSummary;
  /** 自定义"查看详情"行为；不传则默认跳转 /governance/arbitration/:id */
  onViewDetail?: (dispute: DisputeSummary) => void;
}

export const DisputeCard: React.FC<DisputeCardProps> = ({ dispute, onViewDetail }) => {
  const navigate = useNavigate();
  const status = statusMap[dispute.status];
  const code = dispute.scenarioCode;
  const iou = dispute.iou ?? dispute.triggerIou;

  const handleView = () => {
    if (onViewDetail) {
      onViewDetail(dispute);
      return;
    }
    navigate(`/governance/arbitration/${dispute.id}`);
  };

  return (
    <Card className={styles.card}>
      {/* 顶部：ID + IoU + 状态 */}
      <div className={styles.header}>
        <span className={styles.id}>#{dispute.id.slice(0, 8)}</span>
        {iou != null ? <span className={styles.iou}>IoU: {iou.toFixed(2)}</span> : null}
        {status ? <Badge variant={status.variant}>{status.label}</Badge> : null}
      </div>

      {/* 标题 */}
      <h3 className={styles.title}>{dispute.title}</h3>

      {/* 中部：双预览对比 */}
      <div className={styles.body}>
        <SidePreview title="人类标注" annotation={dispute.humanAnnotation} judge={dispute.humanJudge} />
        <SidePreview title="Agent 标注" annotation={dispute.agentAnnotation} judge={dispute.agentJudge} />
      </div>

      {/* 底部：场景彩色横条 */}
      {code ? (
        <div className={styles.scenarioBar} style={{ backgroundColor: scenarioColors[code] }}>
          {scenarioLabels[code]}
        </div>
      ) : dispute.scenarioLabel ? (
        <div className={styles.scenarioBar} style={{ backgroundColor: 'var(--gray-400)' }}>
          {dispute.scenarioLabel}
        </div>
      ) : null}

      {/* 操作区 */}
      <div className={styles.footer}>
        {dispute.decisionAction ? <span className={styles.action}>{dispute.decisionAction}</span> : <span />}
        <button type="button" className={styles.viewBtn} onClick={handleView}>
          查看详情
        </button>
      </div>
    </Card>
  );
};
