// 프레임 인덱스 ↔ 시간 변환 및 타임라인 유틸리티

/** 프레임 중앙 시각으로 seek하여 경계 반올림 오차를 피합니다 */
export function frameToTime(frame: number, fps: number, duration: number): number {
  const t = (frame + 0.5) / fps;
  return Math.max(0, Math.min(t, duration - 0.001));
}

export function timeToFrame(time: number, fps: number, frameCount: number): number {
  return Math.max(0, Math.min(frameCount - 1, Math.floor(time * fps)));
}

/**
 * 재생 중 화면에 표시된 프레임의 번호 (requestVideoFrameCallback 의 metadata.mediaTime 기준).
 * mediaTime 은 그 프레임의 표시 시작 시각(PTS)이라 k/fps 에 해당합니다. 반올림해서
 * 부동소수 오차나 근사 fps(예: 29.97 vs 30000/1001)로 한 프레임 앞으로 밀리지 않게 합니다.
 */
export function mediaTimeToFrame(mediaTime: number, fps: number, frameCount: number): number {
  return Math.max(0, Math.min(frameCount - 1, Math.round(mediaTime * fps)));
}

/** 정렬되지 않은 프레임 번호 목록 → 연속 구간 [start, end] (양끝 포함) */
export function computeRanges(frames: Iterable<number>): Array<[number, number]> {
  const sorted = [...frames].sort((a, b) => a - b);
  const ranges: Array<[number, number]> = [];
  for (const f of sorted) {
    const last = ranges[ranges.length - 1];
    if (last && f <= last[1] + 1) last[1] = Math.max(last[1], f);
    else ranges.push([f, f]);
  }
  return ranges;
}

export function formatTime(sec: number): string {
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, '0')}`;
}
