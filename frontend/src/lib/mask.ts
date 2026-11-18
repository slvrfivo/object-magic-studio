// 마스크 관련 순수 함수 (DOM 의존 없음)
import type { Mask, PromptPoint } from '../types';

export function createMask(width: number, height: number): Mask {
  return { width, height, data: new Uint8Array(width * height) };
}

export function maskArea(mask: Mask): number {
  let n = 0;
  const data = mask.data; // 지연 디코딩 마스크: getter 를 한 번만 호출
  for (let i = 0; i < data.length; i++) n += data[i];
  return n;
}

export function maskCoverage(mask: Mask): number {
  const total = mask.width * mask.height;
  return total === 0 ? 0 : maskArea(mask) / total;
}

function stampCircle(mask: Mask, cx: number, cy: number, r: number, value: 0 | 1) {
  const { width: w, height: h, data } = mask;
  const x0 = Math.max(0, Math.floor(cx - r));
  const x1 = Math.min(w - 1, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r));
  const y1 = Math.min(h - 1, Math.ceil(cy + r));
  const r2 = r * r;
  for (let y = y0; y <= y1; y++) {
    const dy = y + 0.5 - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - cx;
      if (dx * dx + dy * dy <= r2) data[y * w + x] = value;
    }
  }
}

/**
 * ⚠️ MOCK 전용: 실제 분할이 아닙니다.
 * positive 점마다 원을 합치고 negative 점 주변을 제거한 가짜 마스크를 만듭니다.
 * Phase 2에서 SAM 2.1 추론 결과로 대체됩니다.
 */
export function mockMaskFromPoints(points: Pick<PromptPoint, 'x' | 'y' | 'label'>[], width: number, height: number): Mask {
  const mask = createMask(width, height);
  const base = Math.min(width, height);
  for (const p of points) {
    if (p.label === 1) stampCircle(mask, p.x * width, p.y * height, base * 0.16, 1);
  }
  for (const p of points) {
    if (p.label === 0) stampCircle(mask, p.x * width, p.y * height, base * 0.08, 0);
  }
  return mask;
}
