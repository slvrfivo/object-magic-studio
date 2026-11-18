import { describe, expect, it } from 'vitest';
import { computeRanges, formatTime, frameToTime, mediaTimeToFrame, timeToFrame } from './frames';

describe('frame/time conversion', () => {
  it('round-trips every frame', () => {
    const fps = 30;
    const frameCount = 300;
    for (let f = 0; f < frameCount; f++) {
      expect(timeToFrame(frameToTime(f, fps, 10), fps, frameCount)).toBe(f);
    }
  });

  it('clamps out-of-range times', () => {
    expect(timeToFrame(-1, 30, 300)).toBe(0);
    expect(timeToFrame(99, 30, 300)).toBe(299);
  });
});

describe('computeRanges', () => {
  it('merges consecutive frames', () => {
    expect(computeRanges([5, 1, 2, 3, 7, 8])).toEqual([
      [1, 3],
      [5, 5],
      [7, 8],
    ]);
    expect(computeRanges([])).toEqual([]);
  });
});

describe('formatTime', () => {
  it('formats seconds', () => {
    expect(formatTime(3.5)).toBe('0:03.50');
    expect(formatTime(65.25)).toBe('1:05.25');
  });
});

describe('mediaTimeToFrame (playback sync, requestVideoFrameCallback)', () => {
  it('maps the presented frame PTS to its index for exact and NTSC rates', () => {
    for (const [num, den] of [[30, 1], [30000, 1001], [24, 1], [25, 1]]) {
      const fps = num / den;
      for (let k = 0; k < 300; k++) expect(mediaTimeToFrame((k * den) / num, fps, 300)).toBe(k);
    }
  });

  it('tolerates float jitter and an approximate fps (29.97 vs 30000/1001)', () => {
    for (let k = 0; k < 300; k++) {
      const pts = (k * 1001) / 30000;
      expect(mediaTimeToFrame(pts - 1e-9, 29.97, 300)).toBe(k);
      expect(mediaTimeToFrame(pts + 1e-9, 29.97, 300)).toBe(k);
    }
  });

  it('clamps to the video range', () => {
    expect(mediaTimeToFrame(-0.01, 30, 100)).toBe(0);
    expect(mediaTimeToFrame(10, 30, 100)).toBe(99);
  });
});
