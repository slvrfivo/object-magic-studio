import { describe, expect, it } from 'vitest';
import type { RemoteStatus } from '../types';
import { estimateRemoteTrackSeconds, fmtUSD } from './ComputeSelector';

const usage = (likelyWarm: boolean) => ({
  todayBillableS: 0,
  todayCalls: 0,
  todayTrackedFrames: 0,
  monthBillableS: 0,
  monthEstimatedUSD: 0,
  dailyLimitS: 1800,
  monthlyLimitUSD: 5,
  costPerSecondUSD: 0.000208,
  idleWindowS: 120,
  coldStartEstimateS: 45,
  likelyWarm,
});

describe('remote cost estimate', () => {
  it('includes cold start only when the GPU is not running, and always the idle window', () => {
    const cold: RemoteStatus = { configured: true, usage: usage(false) };
    const warm: RemoteStatus = { configured: true, usage: usage(true) };
    expect(estimateRemoteTrackSeconds(cold, 300)).toBeCloseTo(45 + 75 + 120);
    expect(estimateRemoteTrackSeconds(warm, 300)).toBeCloseTo(75 + 120);
    // 300프레임 콜드 ≈ 240초 × $0.000208 ≈ $0.05
    expect(fmtUSD(estimateRemoteTrackSeconds(cold, 300) * 0.000208)).toBe('$0.05');
    expect(fmtUSD(0.0012)).toBe('$0.0012');
  });
});
