export type DisputeStatus = 'open' | 'review' | 'resolved' | 'rejected';
export type ArbitrationScenario = 'LEGAL' | 'ILLEGAL-B' | 'ILLEGAL-A' | 'ILLEGAL-AB';

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
