import { useCallback, useMemo, useState } from 'react';
import { arbitrationApi } from '../../arbitration/api/arbitrationApi';
import type { ArbitrationEvaluatePayload, ArbitrationEvaluation, DisputeAlertInfo } from '../../../types';

export interface UseDisputeDetectionState {
  alerts: DisputeAlertInfo[];
  evaluation: ArbitrationEvaluation | null;
  loading: boolean;
  scan: (payload: ArbitrationEvaluatePayload) => Promise<ArbitrationEvaluation | null>;
  clear: () => void;
}

export function useDisputeDetection(): UseDisputeDetectionState {
  const [alerts, setAlerts] = useState<DisputeAlertInfo[]>([]);
  const [evaluation, setEvaluation] = useState<ArbitrationEvaluation | null>(null);
  const [loading, setLoading] = useState(false);

  const scan = useCallback(async (payload: ArbitrationEvaluatePayload) => {
    setLoading(true);
    try {
      const response = await arbitrationApi.evaluate(payload);
      const result = (response.data as { data?: ArbitrationEvaluation })?.data ?? null;
      setEvaluation(result);
      setAlerts(result?.alerts ?? []);
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : '争议检测失败';
      setEvaluation(null);
      setAlerts([
        {
          id: 'dispute-error',
          level: 'error',
          message,
        },
      ]);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setAlerts([]);
    setEvaluation(null);
  }, []);

  return useMemo(
    () => ({ alerts, evaluation, loading, scan, clear }),
    [alerts, evaluation, loading, scan, clear]
  );
}
