export type DisputeStatus = 'open' | 'review' | 'resolved' | 'rejected';
export type ArbitrationScenario = 'LEGAL' | 'ILLEGAL-B' | 'ILLEGAL-A' | 'ILLEGAL-AB';

/**
 * 物理四场景编码（与设计稿一致）
 *  1 = 人类违规，Agent 合法    (后端 ILLEGAL-B)
 *  2 = Agent 违规，人类合法    (后端 ILLEGAL-A)
 *  3 = 双方均合法（真实分歧）  (后端 LEGAL)
 *  4 = 双方均违规              (后端 ILLEGAL-AB)
 */
export type ScenarioCode = 1 | 2 | 3 | 4;

/** 单边裁判结果（来自物理合法性判定） */
export type JudgeResultLabel = 'compliant' | 'violation' | 'unknown';

export interface SideJudge {
  result: JudgeResultLabel;
}

/** 后端 scenario 字符串 → 设计稿数字编码 */
export function mapScenarioToCode(scenario?: ArbitrationScenario | string | null): ScenarioCode | undefined {
  switch (scenario) {
    case 'ILLEGAL-B':
      return 1;
    case 'ILLEGAL-A':
      return 2;
    case 'LEGAL':
      return 3;
    case 'ILLEGAL-AB':
      return 4;
    default:
      return undefined;
  }
}

export interface PhysicalMetricSet {
  width: number;
  height: number;
  area: number;
  width_length_ratio: number;
  wetness_index: number;
  connected_components: number;
}

export interface ArbitrationRuleHit {
  id: string;
  label: string;
  metric: string;
  operator: '<=' | '>=';
  threshold: number;
  value: number;
  passed: boolean;
}

export interface CompareAnnotation {
  id: string;
  label: string;
  /** 预览图（设计稿要求字段，可为 dataURL 或 http URL） */
  preview?: string;
  dataUrl?: string;
  jsonSummary?: string;
  bbox?: [number, number, number, number];
  legality?: 'unknown' | 'compliant' | 'risk' | 'violation';
  notes?: string[];
  attributes?: Record<string, string | number>;
}

export interface DisputeSummary {
  id: string;
  title: string;
  annotationId?: string;
  status: DisputeStatus;
  openedAt: string;
  updatedAt: string;
  defectType?: string;
  scenario?: ArbitrationScenario;
  scenarioLabel?: string;
  decisionAction?: string;
  decisionWinner?: 'either' | 'agent' | 'human' | 'expert';
  triggerIou?: number;
  triggered?: boolean;

  /* ===== 设计稿对齐字段（由 normalizeDispute 派生填充） ===== */
  /** IoU 别名（= triggerIou），便于设计稿组件直接使用 dispute.iou */
  iou?: number;
  /** 物理四场景编码 1-4 */
  scenarioCode?: ScenarioCode;
  /** Agent 标注（对应后端 leftAnnotation） */
  agentAnnotation?: CompareAnnotation;
  /** 人类标注（对应后端 rightAnnotation） */
  humanAnnotation?: CompareAnnotation;
  /** Agent 单边判定 */
  agentJudge?: SideJudge;
  /** 人类单边判定 */
  humanJudge?: SideJudge;
}

export interface ArbitrationCaseDetail extends DisputeSummary {
  description?: string;
  claimantNote?: string;
  respondentNote?: string;
  iouThreshold?: number;
  leftAnnotation?: CompareAnnotation;
  rightAnnotation?: CompareAnnotation;
  metrics?: {
    agent?: PhysicalMetricSet;
    human?: PhysicalMetricSet;
  };
  ruleHits?: {
    agent?: ArbitrationRuleHit[];
    human?: ArbitrationRuleHit[];
  };
  judgement?: JudgeOutcome | null;
  source?: Record<string, unknown>;
}

export interface JudgeOutcome {
  valid: boolean;
  score?: number;
  reason?: string;
  decidedAt?: string;
  decidedBy?: string;
}

export interface ArbitrationStats {
  open: number;
  inReview: number;
  resolved: number;
  total: number;
  coverage?: number;
  expertReduction?: number;
}

export interface ArbitrationEvaluation {
  triggered: boolean;
  iou: number;
  scenario: ArbitrationScenario;
  scenarioLabel: string;
  decisionAction: string;
  decisionWinner: 'either' | 'agent' | 'human' | 'expert';
  agentReview: {
    legality: 'compliant' | 'violation';
    metrics: PhysicalMetricSet;
    rule_hits: ArbitrationRuleHit[];
    notes: string[];
  };
  humanReview: {
    legality: 'compliant' | 'violation';
    metrics: PhysicalMetricSet;
    rule_hits: ArbitrationRuleHit[];
    notes: string[];
  };
  alerts: {
    id: string;
    level: 'info' | 'warning' | 'error';
    message: string;
    disputeId?: string;
  }[];
  case: ArbitrationCaseDetail;
}

export interface ArbitrationEvaluatePayload {
  annotation_id?: string;
  defect_type: string;
  iou_threshold?: number;
  source?: Record<string, unknown>;
  agent_annotation: {
    label: string;
    bbox: [number, number, number, number];
    attributes?: Record<string, number>;
  };
  human_annotation: {
    label: string;
    bbox: [number, number, number, number];
    attributes?: Record<string, number>;
  };
}
