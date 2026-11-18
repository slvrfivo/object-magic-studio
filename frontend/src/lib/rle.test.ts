import { describe, expect, it } from 'vitest';
import { rleDecode, rleEncode } from './rle';

describe('RLE', () => {
  it('matches the backend format', () => {
    // backend: rle_encode([0,0,1,1,1,0]) == [2,3,1], rle_encode([1,1,0]) == [0,2,1]
    expect(Array.from(rleDecode({ width: 6, height: 1, counts: [2, 3, 1] }).data)).toEqual([0, 0, 1, 1, 1, 0]);
    expect(Array.from(rleDecode({ width: 3, height: 1, counts: [0, 2, 1] }).data)).toEqual([1, 1, 0]);
  });

  it('round-trips random masks', () => {
    const data = new Uint8Array(37 * 53).map(() => (Math.random() > 0.6 ? 1 : 0));
    const mask = { width: 53, height: 37, data };
    expect(rleDecode({ width: 53, height: 37, counts: rleEncode(mask) }).data).toEqual(data);
  });

  it('rejects malformed input', () => {
    expect(() => rleDecode({ width: 2, height: 2, counts: [3, 2] })).toThrow();
    expect(() => rleDecode({ width: 2, height: 2, counts: [1, 1] })).toThrow();
  });
});
