import { afterEach, describe, expect, it } from 'vitest';
import { RleBackedMask, lazyMaskCache } from './lazyMask';
import { rleEncode } from './rle';

afterEach(() => lazyMaskCache.clear());

function maskFor(i: number) {
  const data = new Uint8Array(20 * 10);
  data.fill(1, i, i + 15);
  return { mask: new RleBackedMask({ width: 20, height: 10, counts: rleEncode({ width: 20, height: 10, data }) }), data };
}

describe('RleBackedMask', () => {
  it('decodes to the same pixels as the source mask', () => {
    const { mask, data } = maskFor(7);
    expect(mask.width).toBe(20);
    expect(mask.data).toEqual(data);
  });

  it('keeps only the most recently used masks decoded', () => {
    lazyMaskCache.setCapacity(3);
    const masks = Array.from({ length: 10 }, (_, i) => maskFor(i).mask);
    masks.forEach((m) => m.data);
    expect(lazyMaskCache.size()).toBe(3);
    expect(lazyMaskCache.decodes()).toBe(10);
    masks[9].data; // 캐시 적중
    expect(lazyMaskCache.decodes()).toBe(10);
    masks[0].data; // 밀려난 마스크는 다시 디코딩
    expect(lazyMaskCache.decodes()).toBe(11);
  });

  it('rejects malformed RLE up front', () => {
    expect(() => new RleBackedMask({ width: 4, height: 4, counts: [3, 2] })).toThrow();
  });
});
