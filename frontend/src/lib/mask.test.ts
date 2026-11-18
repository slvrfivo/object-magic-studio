import { describe, expect, it } from 'vitest';
import { maskArea, maskCoverage, mockMaskFromPoints } from './mask';

describe('mockMaskFromPoints', () => {
  it('is empty without positive points', () => {
    expect(maskArea(mockMaskFromPoints([{ x: 0.5, y: 0.5, label: 0 }], 100, 100))).toBe(0);
  });

  it('covers the clicked pixel for a positive point', () => {
    const m = mockMaskFromPoints([{ x: 0.25, y: 0.75, label: 1 }], 200, 100);
    expect(m.data[75 * 200 + 50]).toBe(1);
    expect(m.data[0]).toBe(0);
    expect(maskCoverage(m)).toBeGreaterThan(0);
  });

  it('negative point carves out the positive region', () => {
    const m = mockMaskFromPoints(
      [
        { x: 0.5, y: 0.5, label: 1 },
        { x: 0.5, y: 0.5, label: 0 },
      ],
      100,
      100,
    );
    expect(m.data[50 * 100 + 50]).toBe(0);
    expect(maskArea(m)).toBeGreaterThan(0); // 링 형태로 남음
  });
});
