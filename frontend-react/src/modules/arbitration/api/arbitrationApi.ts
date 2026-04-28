import api from '../../../services/api';
import type {
  ApiResponse,
  PaginatedResponse,
  ArbitrationCaseDetail,
  DisputeSummary,
  ArbitrationStats,
  ArbitrationEvaluation,
  ArbitrationEvaluatePayload,
} from '../../../types';

/** 仲裁中心 HTTP 接口 */
export const arbitrationApi = {
  list: (params?: { page?: number; page_size?: number; status?: string }) =>
    api.get<ApiResponse<PaginatedResponse<DisputeSummary> | { items: DisputeSummary[] }>>(
      '/governance/arbitration/disputes',
      { params }
    ),
  get: (id: string) => api.get<ApiResponse<ArbitrationCaseDetail>>(`/governance/arbitration/disputes/${id}`),
  stats: () => api.get<ApiResponse<ArbitrationStats>>('/governance/arbitration/stats'),
  evaluate: (body: ArbitrationEvaluatePayload) =>
    api.post<ApiResponse<ArbitrationEvaluation>>('/governance/arbitration/evaluate', body),
  submitJudgement: (id: string, body: { valid: boolean; reason?: string }) =>
    api.post<ApiResponse<void>>(`/governance/arbitration/disputes/${id}/judge`, body),
};
