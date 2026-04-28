/** 规则库领域类型 */
export type RuleStatus = 'draft' | 'published' | 'archived';

export interface RuleSummary {
  id: string;
  name: string;
  description?: string;
  version: string;
  status: RuleStatus;
  updatedAt: string;
  createdBy?: string;
}

export interface RuleDetail extends RuleSummary {
  content: string;
  tags?: string[];
}

export interface RuleVersionEntry {
  id: string;
  version: string;
  summary?: string;
  createdAt: string;
  createdBy?: string;
}

export interface RuleFormValues {
  name: string;
  description: string;
  content: string;
  tags: string;
}
