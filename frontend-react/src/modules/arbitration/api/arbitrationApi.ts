import api from '../../../services/api';
import type {
  ApiResponse,
  PaginatedResponse,
  ArbitrationCaseDetail,
  CompareAnnotation,
  DisputeSummary,
  ArbitrationStats,
  ArbitrationEvaluation,
  ArbitrationEvaluatePayload,
  SideJudge,
} from '../../../types';
import { mapScenarioToCode } from '../../../types';

/**
 * 把后端原始 dispute 数据归一化为前端通用形态。
 * - 派生 iou / scenarioCode
 * - 派生 agentAnnotation / humanAnnotation（来自 leftAnnotation / rightAnnotation）
 * - 派生 agentJudge / humanJudge（来自各自 legality）
 *
 * 设计要点：保留所有原字段以兼容现有页面（DisputeDetail 等），仅追加新字段。
 */
export function normalizeDispute<T extends Partial<DisputeSummary> & Record<string, unknown>>(
  raw: T | null | undefined
): (T & DisputeSummary) | null {
  if (!raw || typeof raw !== 'object') return null;

  const left = (raw as { leftAnnotation?: CompareAnnotation }).leftAnnotation;
  const right = (raw as { rightAnnotation?: CompareAnnotation }).rightAnnotation;

  const iou = raw.triggerIou ?? (raw as { iou?: number }).iou;
  const scenarioCode = mapScenarioToCode(raw.scenario as string | undefined);

  const toJudge = (anno?: CompareAnnotation): SideJudge | undefined => {
    if (!anno?.legality) return undefined;
    const legality = anno.legality;
    if (legality === 'compliant' || legality === 'violation') {
      return { result: legality };
    }
    return { result: 'unknown' };
  };

  return {
    ...(raw as object),
    iou,
    scenarioCode,
    agentAnnotation: left,
    humanAnnotation: right,
    agentJudge: toJudge(left),
    humanJudge: toJudge(right),
  } as T & DisputeSummary;
}

function unwrapList(payload: unknown): DisputeSummary[] {
  if (payload == null) return [];
  if (Array.isArray(payload)) return payload as DisputeSummary[];
  const obj = payload as { items?: DisputeSummary[]; data?: { items?: DisputeSummary[] } };
  if (Array.isArray(obj.items)) return obj.items;
  if (Array.isArray(obj.data?.items)) return obj.data!.items!;
  return [];
}

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

  /** 拉取列表并自动归一化为含设计稿字段的形态 */
  async listNormalized(params?: { page?: number; page_size?: number; status?: string }): Promise<DisputeSummary[]> {
    const res = await arbitrationApi.list(params);
    const inner = (res.data as { data?: unknown })?.data ?? res.data;
    const items = unwrapList(inner);
    return items.map((item) =>
      normalizeDispute(item as Partial<DisputeSummary> & Record<string, unknown>) ?? item
    );
  },

  /** 拉取详情并归一化 */
  async getNormalized(id: string): Promise<ArbitrationCaseDetail | null> {
    const res = await arbitrationApi.get(id);
    const payload = (res.data as { data?: ArbitrationCaseDetail })?.data ?? null;
    return normalizeDispute(payload as Partial<ArbitrationCaseDetail> & Record<string, unknown>) as
      | ArbitrationCaseDetail
      | null;
  },
};
