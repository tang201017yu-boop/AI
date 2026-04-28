import type { ArbitrationRuleHit, PhysicalMetricSet } from './arbitration';

export type LegalityLabel = 'unknown' | 'compliant' | 'risk' | 'violation';

export interface AgentSuggestionItem {
  id: string;
  className: string;
  confidence: number;
  reason?: string;
}

export interface AttributeField {
  key: string;
  label: string;
  value: string;
}

export interface DisputeAlertInfo {
  id: string;
  level: 'info' | 'warning' | 'error';
  message: string;
  disputeId?: string;
}

export interface PhysicalJudgement {
  legality: LegalityLabel;
  notes?: string;
  metrics?: PhysicalMetricSet;
  ruleHits?: ArbitrationRuleHit[];
}
