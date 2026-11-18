// RLE 로 보관하고 필요할 때만 디코딩하는 마스크 (최근 사용 N개만 메모리에 유지)
// 300프레임 × 960×540 마스크를 모두 풀어 두면 약 155MB → RLE 로 보관하면 수백 KB.
import type { Mask } from '../types';
import { rleDecode, type RleMask } from './rle';

const DEFAULT_CAPACITY = 16;
const decoded = new Map<RleBackedMask, Uint8Array>(); // 삽입 순서 = LRU 순서
let capacity = DEFAULT_CAPACITY;
let decodeCount = 0;

export class RleBackedMask implements Mask {
  readonly width: number;
  readonly height: number;
  private readonly counts: number[];

  constructor(rle: RleMask) {
    const total = rle.counts.reduce((a, b) => a + b, 0);
    if (total !== rle.width * rle.height) throw new Error(`Invalid RLE mask length ${total}`);
    this.width = rle.width;
    this.height = rle.height;
    this.counts = rle.counts;
  }

  get data(): Uint8Array {
    const hit = decoded.get(this);
    if (hit) {
      decoded.delete(this);
      decoded.set(this, hit);
      return hit;
    }
    const data = rleDecode({ width: this.width, height: this.height, counts: this.counts }).data;
    decodeCount++;
    decoded.set(this, data);
    while (decoded.size > capacity) decoded.delete(decoded.keys().next().value as RleBackedMask);
    return data;
  }
}

/** 테스트/진단용 */
export const lazyMaskCache = {
  size: () => decoded.size,
  decodes: () => decodeCount,
  setCapacity(n: number) {
    capacity = n;
    while (decoded.size > capacity) decoded.delete(decoded.keys().next().value as RleBackedMask);
  },
  clear() {
    decoded.clear();
    decodeCount = 0;
    capacity = DEFAULT_CAPACITY;
  },
};
