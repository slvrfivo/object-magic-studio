// 브라우저 <video> 요소를 이용한 메타데이터 읽기 / 썸네일 생성
import type { BasicVideoInfo } from '../types';

/** 미리보기는 긴 변 960px 이하로 렌더링 (성능). 최종 export는 원본 해상도로 백엔드에서 처리 */
export const PREVIEW_MAX_LONG_SIDE = 960;

export function previewSize(width: number, height: number, maxLong = PREVIEW_MAX_LONG_SIDE) {
  const scale = Math.min(1, maxLong / Math.max(width, height));
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

export function loadVideoInfo(url: string, timeoutMs = 10000): Promise<BasicVideoInfo> {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video');
    v.preload = 'metadata';
    v.muted = true;
    const cleanup = () => {
      clearTimeout(timer);
      v.removeAttribute('src');
      v.load();
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Timed out while reading the video metadata.'));
    }, timeoutMs);
    v.onloadedmetadata = () => {
      const info = { width: v.videoWidth, height: v.videoHeight, duration: v.duration };
      cleanup();
      resolve(info);
    };
    v.onerror = () => {
      cleanup();
      reject(new Error('Your browser could not decode this MP4. Please use an H.264-encoded MP4.'));
    };
    v.src = url;
  });
}

export function seekVideo(v: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    if (Math.abs(v.currentTime - time) < 1e-4 && v.readyState >= 2) return resolve();
    const done = () => {
      v.removeEventListener('seeked', done);
      resolve();
    };
    v.addEventListener('seeked', done);
    v.currentTime = time;
  });
}

/** 타임라인용 썸네일을 별도 video 요소로 생성 (메인 미리보기와 간섭 없음) */
export async function generateThumbnails(url: string, duration: number, count: number, height = 64): Promise<string[]> {
  const v = document.createElement('video');
  v.muted = true;
  v.preload = 'auto';
  v.src = url;
  await new Promise<void>((resolve, reject) => {
    v.onloadeddata = () => resolve();
    v.onerror = () => reject(new Error('Thumbnail decode failed'));
  });
  const width = Math.round((v.videoWidth / v.videoHeight) * height);
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  const ctx = c.getContext('2d')!;
  const thumbs: string[] = [];
  for (let i = 0; i < count; i++) {
    await seekVideo(v, Math.min(duration - 0.01, ((i + 0.5) / count) * duration));
    ctx.drawImage(v, 0, 0, width, height);
    thumbs.push(c.toDataURL('image/jpeg', 0.7));
  }
  v.removeAttribute('src');
  v.load();
  return thumbs;
}
