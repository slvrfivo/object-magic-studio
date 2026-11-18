// 업로드 파일 검증 (클라이언트 1차 검증 — 백엔드에서도 동일 규칙으로 재검증합니다)

export const LIMITS = {
  maxDurationSec: 10,
  /** 컨테이너 메타데이터 오차 허용치 */
  durationToleranceSec: 0.1,
  maxFileSizeBytes: 100 * 1024 * 1024,
  /** 가로/세로 방향과 무관하게 긴 변 ≤ 1920, 짧은 변 ≤ 1080 */
  maxLongSide: 1920,
  maxShortSide: 1080,
  minSide: 64,
} as const;

export function validateFileBasics(file: { name: string; type: string; size: number }): string | null {
  const isMp4 = file.type === 'video/mp4' || /\.mp4$/i.test(file.name);
  if (!isMp4) return `Unsupported file type. Please upload an MP4 video.`;
  if (file.size === 0) return 'The file is empty.';
  if (file.size > LIMITS.maxFileSizeBytes) {
    return `File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum is ${LIMITS.maxFileSizeBytes / 1024 / 1024} MB.`;
  }
  return null;
}

export function validateVideoInfo(info: { duration: number; width: number; height: number }): string | null {
  if (!Number.isFinite(info.duration) || info.duration <= 0) {
    return 'Could not read the video duration.';
  }
  if (info.duration > LIMITS.maxDurationSec + LIMITS.durationToleranceSec) {
    return `Video is ${info.duration.toFixed(1)}s long. Maximum length is ${LIMITS.maxDurationSec}s.`;
  }
  if (!info.width || !info.height) return 'Could not read the video resolution.';
  const long = Math.max(info.width, info.height);
  const short = Math.min(info.width, info.height);
  if (long > LIMITS.maxLongSide || short > LIMITS.maxShortSide) {
    return `Resolution ${info.width}×${info.height} is too high. Maximum is 1920×1080 (or 1080×1920).`;
  }
  if (short < LIMITS.minSide) {
    return `Resolution ${info.width}×${info.height} is too small. Minimum side is ${LIMITS.minSide}px.`;
  }
  return null;
}
