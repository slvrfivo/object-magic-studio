"""좌표 변환 및 마스크 직렬화 (순수 함수, 모델 의존 없음)."""

from collections.abc import Sequence

import cv2
import numpy as np


def normalized_to_pixel(points: Sequence[tuple[float, float]], width: int, height: int) -> np.ndarray:
    """[0,1] 정규화 좌표 → 원본 프레임 픽셀 좌표 (연속 좌표계, (x, y) 순서).

    프론트엔드 lib/coords.ts 의 normalizedToPixel 과 동일한 규칙입니다.
    SAM2ImagePredictor.predict(normalize_coords=True) 는 이 원본 이미지 기준 픽셀 좌표를 기대합니다.
    """
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    out = np.empty_like(arr)
    out[:, 0] = np.clip(arr[:, 0], 0.0, 1.0) * width
    out[:, 1] = np.clip(arr[:, 1], 0.0, 1.0) * height
    return out


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """바이너리 마스크를 (width, height)로 리샘플. 축소 시 면적 평균 후 0.5 임계값."""
    m = mask.astype(np.float32)
    if (m.shape[1], m.shape[0]) == (width, height):
        return (m >= 0.5).astype(np.uint8)
    interp = cv2.INTER_AREA if width < m.shape[1] else cv2.INTER_LINEAR
    return (cv2.resize(m, (width, height), interpolation=interp) >= 0.5).astype(np.uint8)


def rle_encode(mask: np.ndarray) -> list[int]:
    """행 우선(row-major) 런 길이 인코딩. 0으로 시작하는 교대 런 길이 목록.

    예: [0,0,1,1,1,0] → [2, 3, 1]. 첫 픽셀이 1이면 첫 값은 0.
    프론트엔드 lib/rle.ts 의 rleDecode 와 짝을 이룹니다.
    """
    flat = (np.asarray(mask).reshape(-1) > 0).astype(np.int8)
    if flat.size == 0:
        return []
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat[0] == 1:
        runs.insert(0, 0)
    return runs


def rle_decode(counts: Sequence[int], width: int, height: int) -> np.ndarray:
    flat = np.zeros(width * height, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            flat[pos : pos + c] = 1
        pos += c
        val ^= 1
    if pos != width * height:
        raise ValueError(f"RLE length {pos} != {width * height}")
    return flat.reshape(height, width)
