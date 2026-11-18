// 이펙트 설정(UI 값) → 실제 렌더링 파라미터 변환
// 해상도에 비례하는 값으로 계산하므로, 저해상도 미리보기와 원본 해상도 export 결과가 같은 모양이 됩니다.
// Phase 4 백엔드(OpenCV) 구현도 이 공식을 그대로 따릅니다.
import type { EffectSettings } from '../types';

export interface NeonParams {
  color: string;
  thicknessPx: number;
  glowPx: number;
  alpha: number;
}

export interface EffectParams {
  blurPx: number;
  grayscale: number; // 0..1, 배경에만 적용
  objectSaturate: number; // 1 = 원본
  neon: NeonParams | null;
}

export const NEON_PRESETS = ['#22d3ee', '#e879f9', '#a3e635', '#fbbf24', '#f43f5e', '#ffffff'] as const;

export const DEFAULT_EFFECTS: EffectSettings = {
  neon: { enabled: true, intensity: 0.75, color: NEON_PRESETS[0], thickness: 3 },
  colorPop: { enabled: false, intensity: 1 },
  blur: { enabled: false, intensity: 0.5 },
};

export function anyEffectEnabled(e: EffectSettings): boolean {
  return e.neon.enabled || e.colorPop.enabled || e.blur.enabled;
}

export function effectParams(e: EffectSettings, width: number, height: number): EffectParams {
  const long = Math.max(width, height);
  return {
    blurPx: e.blur.enabled ? e.blur.intensity * 0.03 * long : 0,
    grayscale: e.colorPop.enabled ? e.colorPop.intensity : 0,
    objectSaturate: e.colorPop.enabled ? 1 + 0.35 * e.colorPop.intensity : 1,
    neon: e.neon.enabled
      ? {
          color: e.neon.color,
          thicknessPx: Math.max(1, (e.neon.thickness * long) / 640),
          glowPx: e.neon.intensity * 0.035 * long,
          alpha: 0.45 + 0.55 * e.neon.intensity,
        }
      : null,
  };
}
