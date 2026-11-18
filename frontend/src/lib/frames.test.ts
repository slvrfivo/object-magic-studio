import { describe, expect, it } from 'vitest';
import { computeRanges, formatTime, frameToTime, timeToFrame } from './frames';

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
