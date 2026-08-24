import api from '../../../services/api';
import type { ApiResponse, ArbitrationEvaluatePayload } from '../../../types';

export type AnnotationEnhancementSamples = Record<string, ArbitrationEvaluatePayload>;

/** 标注增强菜单 HTTP 接口 */
export const annotationEnhancementApi = {
  samples: () =>
    api.get<ApiResponse<AnnotationEnhancementSamples>>('/governance/annotation/samples'),
};
