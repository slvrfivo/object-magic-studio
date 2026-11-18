import { describe, expect, it } from 'vitest';
import { clientToNormalized, fitContain, normalizedToPixel } from './coords';

describe('fitContain', () => {
  it('letterboxes a 16:9 frame inside a wide container', () => {
    const r = fitContain({ width: 1000, height: 500 }, { width: 1920, height: 1080 });
    expect(r.height).toBeCloseTo(500);
    expect(r.width).toBeCloseTo((1920 * 500) / 1080);
    expect(r.offsetY).toBeCloseTo(0);
    expect(r.offsetX).toBeCloseTo((1000 - r.width) / 2);
  });

  it('pillarboxes a portrait frame', () => {
    const r = fitContain({ width: 800, height: 800 }, { width: 1080, height: 1920 });
    expect(r.height).toBeCloseTo(800);
    expect(r.width).toBeCloseTo(450);
    expect(r.offsetX).toBeCloseTo(175);
  });

  it('returns zero size for empty input', () => {
    expect(fitContain({ width: 0, height: 100 }, { width: 10, height: 10 }).width).toBe(0);
  });
});

describe('clientToNormalized', () => {
  const rect = { left: 100, top: 50, width: 400, height: 200 };

  it('maps rect corners and center', () => {
    expect(clientToNormalized(100, 50, rect)).toEqual({ x: 0, y: 0 });
    expect(clientToNormalized(500, 250, rect)).toEqual({ x: 1, y: 1 });
    expect(clientToNormalized(300, 150, rect)).toEqual({ x: 0.5, y: 0.5 });
  });

  it('rejects clicks outside the frame (letterbox area)', () => {
    expect(clientToNormalized(99, 100, rect)).toBeNull();
    expect(clientToNormalized(200, 251, rect)).toBeNull();
  });
});

describe('normalizedToPixel', () => {
  it('round-trips a click through a downscaled preview to full resolution', () => {
    // 1920x1080 원본을 화면에 480x270으로 표시, 화면 (120, 90) 클릭
    const n = clientToNormalized(120, 90, { left: 0, top: 0, width: 480, height: 270 })!;
    const px = normalizedToPixel(n, { width: 1920, height: 1080 });
    expect(px.x).toBeCloseTo(480);
    expect(px.y).toBeCloseTo(360);
  });

  it('clamps to frame bounds', () => {
    expect(normalizedToPixel({ x: 1.2, y: -0.1 }, { width: 100, height: 50 })).toEqual({ x: 100, y: 0 });
  });
});
