import api from '../../../services/api';
import type { ApiResponse, PaginatedResponse } from '../../../types';
import type { RuleDetail, RuleSummary, RuleVersionEntry } from '../../../types';

/** 规则库 HTTP 接口（后端路由就绪后按实际 path 调整） */
export const rulesApi = {
  list: (params?: { page?: number; page_size?: number; q?: string }) =>
    api.get<ApiResponse<PaginatedResponse<RuleSummary> | { items: RuleSummary[] }>>('/governance/rules', { params }),
  get: (id: string) => api.get<ApiResponse<RuleDetail>>(`/governance/rules/${id}`),
  create: (body: { name: string; description?: string; content: string; tags?: string[] }) =>
    api.post<ApiResponse<RuleDetail>>('/governance/rules', body),
  update: (id: string, body: Partial<{ name: string; description: string; content: string; tags: string[] }>) =>
    api.put<ApiResponse<RuleDetail>>(`/governance/rules/${id}`, body),
  history: (id: string) =>
    api.get<ApiResponse<{ versions: RuleVersionEntry[] }>>(`/governance/rules/${id}/versions`),
  publish: (id: string, comment?: string) =>
    api.post<ApiResponse<RuleDetail>>(`/governance/rules/${id}/publish`, { comment }),
  archive: (id: string, comment?: string) =>
    api.post<ApiResponse<RuleDetail>>(`/governance/rules/${id}/archive`, { comment }),
  delete: (id: string) => api.delete<ApiResponse<void>>(`/governance/rules/${id}`),
};
