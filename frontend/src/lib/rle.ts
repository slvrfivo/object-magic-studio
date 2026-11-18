// 런 길이 인코딩(RLE) 마스크 — backend/app/masks.py 의 rle_encode 와 같은 형식
// 행 우선, 0으로 시작하는 교대 런 길이. 예: [0,0,1,1,1,0] → [2,3,1]
import type { Mask } from '../types';

export interface RleMask {
  width: number;
  height: number;
  counts: number[];
}

export function rleDecode(r: RleMask): Mask {
  const total = r.width * r.height;
  const data = new Uint8Array(total);
  let pos = 0;
  let val = 0;
  for (const c of r.counts) {
    if (c < 0 || pos + c > total) throw new Error('Invalid RLE mask');
    if (val) data.fill(1, pos, pos + c);
    pos += c;
    val ^= 1;
  }
  if (pos !== total) throw new Error(`Invalid RLE mask length ${pos} (expected ${total})`);
  return { width: r.width, height: r.height, data };
}

export function rleEncode(mask: Mask): number[] {
  const runs: number[] = [];
  let val = 0;
  let run = 0;
  const data = mask.data;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] ? 1 : 0;
    if (v === val) run++;
    else {
      runs.push(run);
      val = v;
      run = 1;
    }
  }
  if (data.length) runs.push(run);
  return runs;
}
