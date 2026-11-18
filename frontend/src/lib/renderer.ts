// Canvas 기반 미리보기 렌더러
// 모델 추론과 완전히 분리되어 있으며, 입력은 "프레임 이미지 + 바이너리 마스크 + 이펙트 설정" 뿐입니다.
import type { EffectSettings, Mask, ViewMode } from '../types';
import { effectParams } from './effectParams';

// 최근 사용한 마스크 캔버스만 유지 (960×540 RGBA ≈ 2MB/장 — 300프레임을 모두 두면 600MB)
const MASK_CANVAS_CAPACITY = 8;
const maskCanvasCache = new Map<Mask, HTMLCanvasElement>();

/** 마스크를 흰색 알파 캔버스로 변환 (마스크 객체별 LRU 캐시) */
export function maskToCanvas(mask: Mask): HTMLCanvasElement {
  const cached = maskCanvasCache.get(mask);
  if (cached) {
    maskCanvasCache.delete(mask);
    maskCanvasCache.set(mask, cached);
    return cached;
  }
  const c = document.createElement('canvas');
  c.width = mask.width;
  c.height = mask.height;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(mask.width, mask.height);
  const d = img.data;
  const src = mask.data; // 지연 디코딩 마스크: getter 를 한 번만 호출
  for (let i = 0; i < src.length; i++) {
    if (src[i]) {
      const j = i * 4;
      d[j] = d[j + 1] = d[j + 2] = d[j + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  maskCanvasCache.set(mask, c);
  while (maskCanvasCache.size > MASK_CANVAS_CAPACITY) maskCanvasCache.delete(maskCanvasCache.keys().next().value as Mask);
  return c;
}

export interface RenderOptions {
  mask: Mask | null;
  effects: EffectSettings;
  view: ViewMode;
  /** split 모드에서 Before 영역 비율 (0..1, 왼쪽이 Before) */
  split: number;
  showMask: boolean;
}

function sized(c: HTMLCanvasElement, w: number, h: number) {
  if (c.width !== w) c.width = w;
  if (c.height !== h) c.height = h;
  return c.getContext('2d')!;
}

export class FrameRenderer {
  private base = document.createElement('canvas');
  private out = document.createElement('canvas');
  private obj = document.createElement('canvas');
  private ring = document.createElement('canvas');
  private tint = document.createElement('canvas');

  render(target: CanvasRenderingContext2D, source: CanvasImageSource, o: RenderOptions) {
    const { width: w, height: h } = target.canvas;
    const b = sized(this.base, w, h);
    b.drawImage(source, 0, 0, w, h);

    const after = o.view === 'before' ? this.base : this.composeAfter(o, w, h);

    target.save();
    target.clearRect(0, 0, w, h);
    target.drawImage(after, 0, 0);
    if (o.showMask && o.mask && o.view !== 'before') this.drawMaskTint(target, o.mask, w, h);
    if (o.view === 'split') {
      target.beginPath();
      target.rect(0, 0, o.split * w, h);
      target.clip();
      target.drawImage(this.base, 0, 0);
    }
    target.restore();
  }

  private composeAfter(o: RenderOptions, w: number, h: number): HTMLCanvasElement {
    const p = effectParams(o.effects, w, h);
    const hasBackground = p.blurPx > 0 || p.grayscale > 0;
    if (!o.mask || (!hasBackground && !p.neon)) return this.base;

    const m = maskToCanvas(o.mask);
    const out = sized(this.out, w, h);
    out.save();
    out.clearRect(0, 0, w, h);

    if (hasBackground) {
      // 1) 배경: 흑백/블러 처리 (블러 가장자리 투명화를 막기 위해 아래에 한 번 더 깔아줌)
      out.filter = `grayscale(${p.grayscale})`;
      out.drawImage(this.base, 0, 0);
      if (p.blurPx > 0) {
        out.filter = `blur(${p.blurPx.toFixed(2)}px) grayscale(${p.grayscale})`;
        out.drawImage(this.base, 0, 0);
      }
      out.filter = 'none';

      // 2) 객체: 원본(또는 채도 강화)을 마스크로 잘라 위에 합성
      const ob = sized(this.obj, w, h);
      ob.save();
      ob.clearRect(0, 0, w, h);
      ob.filter = p.objectSaturate !== 1 ? `saturate(${p.objectSaturate})` : 'none';
      ob.drawImage(this.base, 0, 0);
      ob.filter = 'none';
      ob.globalCompositeOperation = 'destination-in';
      ob.imageSmoothingEnabled = true;
      ob.drawImage(m, 0, 0, w, h);
      ob.restore();
      out.drawImage(this.obj, 0, 0);
    } else {
      out.drawImage(this.base, 0, 0);
    }

    // 3) 네온 외곽선: 마스크 팽창(dilate) − 원본 마스크 = 외곽 링, 그림자 블러로 발광
    if (p.neon) {
      this.buildRing(m, w, h, p.neon.thicknessPx, p.neon.color);
      out.globalAlpha = p.neon.alpha;
      out.shadowColor = p.neon.color;
      out.shadowBlur = p.neon.glowPx;
      out.drawImage(this.ring, 0, 0);
      out.drawImage(this.ring, 0, 0);
      out.shadowBlur = 0;
      out.globalAlpha = 1;
      out.drawImage(this.ring, 0, 0);
    }
    out.restore();
    return this.out;
  }

  private buildRing(m: HTMLCanvasElement, w: number, h: number, t: number, color: string) {
    const r = sized(this.ring, w, h);
    r.save();
    r.clearRect(0, 0, w, h);
    const steps = 16;
    for (let i = 0; i < steps; i++) {
      const a = (i / steps) * Math.PI * 2;
      r.drawImage(m, Math.cos(a) * t, Math.sin(a) * t, w, h);
    }
    r.globalCompositeOperation = 'source-in';
    r.fillStyle = color;
    r.fillRect(0, 0, w, h);
    r.globalCompositeOperation = 'destination-out';
    r.drawImage(m, 0, 0, w, h);
    r.restore();
  }

  private drawMaskTint(target: CanvasRenderingContext2D, mask: Mask, w: number, h: number) {
    const t = sized(this.tint, w, h);
    t.save();
    t.clearRect(0, 0, w, h);
    t.drawImage(maskToCanvas(mask), 0, 0, w, h);
    t.globalCompositeOperation = 'source-in';
    t.fillStyle = '#8b5cf6';
    t.fillRect(0, 0, w, h);
    t.restore();
    target.save();
    target.globalAlpha = 0.45;
    target.drawImage(this.tint, 0, 0);
    target.restore();
  }
}
