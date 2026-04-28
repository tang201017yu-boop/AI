import { useCallback, useMemo, useState } from 'react';
import type { AnnotationBox } from '../../../types';
import { boxArea } from '../../../shared/utils/physicalAttributes';
import type { LegalityLabel, PhysicalJudgement } from '../../../types';
import type { ArbitrationRuleHit } from '../../../types';

export interface UsePhysicalJudgeState {
  judgement: PhysicalJudgement | null;
  evaluate: (
    box: AnnotationBox,
    defectType?: string,
    attributes?: Partial<{
      wetness_index: number;
      connected_components: number;
      area: number;
      width_length_ratio: number;
    }>
  ) => void;
  reset: () => void;
}

export function usePhysicalJudge(): UsePhysicalJudgeState {
  const [judgement, setJudgement] = useState<PhysicalJudgement | null>(null);

  const evaluate = useCallback(
    (
      box: AnnotationBox,
      defectType = 'crack',
      attributes?: Partial<{
        wetness_index: number;
        connected_components: number;
        area: number;
        width_length_ratio: number;
      }>
    ) => {
      const width = Math.abs(box.x2 - box.x1);
      const height = Math.abs(box.y2 - box.y1);
      const longSide = Math.max(width, height);
      const shortSide = Math.min(width, height);
      const metrics = {
        width,
        height,
        area: attributes?.area ?? boxArea(box),
        width_length_ratio: attributes?.width_length_ratio ?? (longSide > 0 ? shortSide / longSide : 0),
        wetness_index: attributes?.wetness_index ?? 0.5,
        connected_components: attributes?.connected_components ?? 1,
      };

      const notes: string[] = [];
      const ruleHits: ArbitrationRuleHit[] = [];
      let legality: LegalityLabel = 'compliant';

      const pushRule = (id: string, label: string, passed: boolean, threshold: number, value: number, operator: '<=' | '>=') => {
        ruleHits.push({ id, label, passed, threshold, value, operator, metric: label });
        notes.push(`${label}${passed ? '通过' : '未通过'}`);
        if (!passed) legality = 'violation';
      };

      if (defectType === 'seepage') {
        pushRule('R_seep_001', '湿润指数', metrics.wetness_index >= 0.45, 0.45, metrics.wetness_index, '>=');
        pushRule('R_seep_002', '连通域数量', metrics.connected_components <= 5, 5, metrics.connected_components, '<=');
      } else if (defectType === 'spalling') {
        pushRule('R_spall_001', '脱落面积', metrics.area >= 200, 200, metrics.area, '>=');
        pushRule('R_spall_002', '宽长比', metrics.width_length_ratio <= 0.85, 0.85, metrics.width_length_ratio, '<=');
      } else if (defectType === 'hollow') {
        pushRule('R_hollow_001', '空鼓面积', metrics.area >= 150, 150, metrics.area, '>=');
        pushRule('R_hollow_002', '形状紧致度', metrics.width_length_ratio >= 0.12, 0.12, metrics.width_length_ratio, '>=');
      } else {
        pushRule('R_crack_001', '宽长比', metrics.width_length_ratio <= 0.08, 0.08, metrics.width_length_ratio, '<=');
        pushRule('R_crack_002', '像素面积', metrics.area >= 180, 180, metrics.area, '>=');
      }

      setJudgement({
        legality,
        notes: notes.join('；'),
        metrics,
        ruleHits,
      });
    },
    []
  );

  const reset = useCallback(() => setJudgement(null), []);

  return useMemo(
    () => ({ judgement, evaluate, reset }),
    [judgement, evaluate, reset]
  );
}
