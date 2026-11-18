import { describe, expect, it } from 'vitest';
import { DEFAULT_EFFECTS, anyEffectEnabled, effectParams } from './effectParams';
import type { EffectSettings } from '../types';

const allOn: EffectSettings = {
  neon: { enabled: true, intensity: 1, color: '#ffffff', thickness: 5 },
  colorPop: { enabled: true, intensity: 0.5 },
  blur: { enabled: true, intensity: 1 },
};

describe('effectParams', () => {
  it('scales pixel sizes proportionally with resolution', () => {
    const preview = effectParams(allOn, 960, 540);
    const full = effectParams(allOn, 1920, 1080);
    expect(full.blurPx / preview.blurPx).toBeCloseTo(2);
    expect(full.neon!.thicknessPx / preview.neon!.thicknessPx).toBeCloseTo(2);
    expect(full.neon!.glowPx / preview.neon!.glowPx).toBeCloseTo(2);
  });

  it('disabled effects produce neutral params', () => {
    const off: EffectSettings = {
      neon: { ...allOn.neon, enabled: false },
      colorPop: { ...allOn.colorPop, enabled: false },
      blur: { ...allOn.blur, enabled: false },
    };
    const p = effectParams(off, 640, 360);
    expect(p).toEqual({ blurPx: 0, grayscale: 0, objectSaturate: 1, neon: null });
    expect(anyEffectEnabled(off)).toBe(false);
    expect(anyEffectEnabled(DEFAULT_EFFECTS)).toBe(true);
  });
});
