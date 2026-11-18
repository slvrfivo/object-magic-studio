// 클릭 좌표 정규화 유틸리티
// 화면(CSS 픽셀) 좌표 → [0,1] 정규화 좌표 → 원본 프레임 픽셀 좌표 순서로 변환합니다.
// 정규화 좌표를 주고받으면 미리보기 해상도/화면 크기와 무관하게 백엔드에서 정확한 픽셀 위치를 복원할 수 있습니다.

export interface Size {
  width: number;
  height: number;
}

export interface FitResult extends Size {
  offsetX: number;
  offsetY: number;
  scale: number;
}

export interface NormalizedPoint {
  x: number;
  y: number;
}

/** object-fit: contain 과 동일한 계산 (레터박스 포함) */
export function fitContain(container: Size, media: Size): FitResult {
  if (container.width <= 0 || container.height <= 0 || media.width <= 0 || media.height <= 0) {
    return { width: 0, height: 0, offsetX: 0, offsetY: 0, scale: 0 };
  }
  const scale = Math.min(container.width / media.width, container.height / media.height);
  const width = media.width * scale;
  const height = media.height * scale;
  return {
    width,
    height,
    offsetX: (container.width - width) / 2,
    offsetY: (container.height - height) / 2,
    scale,
  };
}

/**
 * 클라이언트 좌표를 표시 영역(rect) 기준 정규화 좌표로 변환합니다.
 * rect는 실제 프레임이 그려진 영역이어야 합니다 (레터박스 제외).
 * 영역 밖 클릭이면 null.
 */
export function clientToNormalized(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
): NormalizedPoint | null {
  if (rect.width <= 0 || rect.height <= 0) return null;
  const x = (clientX - rect.left) / rect.width;
  const y = (clientY - rect.top) / rect.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  return { x, y };
}

/**
 * 정규화 좌표 → 프레임 픽셀 좌표 (연속 좌표계, 이미지가 [0,W]x[0,H]를 차지).
 * SAM 2 predictor가 기대하는 (x, y) 픽셀 좌표 형식과 동일합니다.
 */
export function normalizedToPixel(p: NormalizedPoint, size: Size): NormalizedPoint {
  return {
    x: Math.min(size.width, Math.max(0, p.x * size.width)),
    y: Math.min(size.height, Math.max(0, p.y * size.height)),
  };
}
