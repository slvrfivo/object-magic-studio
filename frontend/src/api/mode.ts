export type ApiMode = 'mock' | 'live';

/**
 * API 모드 결정: URL ?mode=mock|live > VITE_API_MODE > 기본값 live.
 * Live 모드에서 백엔드가 실패해도 Mock으로 자동 전환하지 않습니다.
 */
export function resolveApiMode(search: string, envMode: string | undefined): ApiMode {
  const fromUrl = new URLSearchParams(search).get('mode');
  if (fromUrl === 'mock' || fromUrl === 'live') return fromUrl;
  if (envMode === 'mock' || envMode === 'live') return envMode;
  return 'live';
}
