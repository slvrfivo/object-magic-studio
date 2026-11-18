"""최종 출력용 시각 효과 렌더러 (OpenCV / NumPy). 모델 추론과 무관합니다.

frontend/src/lib/renderer.ts (Canvas 2D 미리보기) 와 같은 결과를 내도록 연산을 그대로 옮겼습니다.
  - 파라미터 공식: frontend/src/lib/effectParams.ts 의 effectParams() 와 동일 (해상도 비례)
  - CSS grayscale()/saturate(): Filter Effects 명세의 색 행렬 (sRGB 값에 직접 적용, Chrome 동작과 동일)
  - CSS blur(r): 표준편차 r 의 가우시안. Canvas 는 경계 밖을 투명으로 보므로 프리멀티플라이 방식으로 재현
  - Canvas shadowBlur=b: 표준편차 b/2 의 가우시안 그림자
  - 네온 링: 마스크를 반지름 t 로 팽창한 뒤 원래 마스크를 뺀 영역
마스크는 이진(0/1)이며 경계 페더링은 하지 않습니다 (미리보기와 동일).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# BGR 채널 순서로 재배치한 색 행렬을 쓰기 위해 RGB 기준 행렬을 정의
_LUM = np.array([0.2126, 0.7152, 0.0722])


def _grayscale_matrix(amount: float) -> np.ndarray:
    a = 1.0 - max(0.0, min(1.0, amount))
    return np.array(
        [
            [0.2126 + 0.7874 * a, 0.7152 - 0.7152 * a, 0.0722 - 0.0722 * a],
            [0.2126 - 0.2126 * a, 0.7152 + 0.2848 * a, 0.0722 - 0.0722 * a],
            [0.2126 - 0.2126 * a, 0.7152 - 0.7152 * a, 0.0722 + 0.9278 * a],
        ]
    )


def _saturate_matrix(s: float) -> np.ndarray:
    return np.array(
        [
            [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
            [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
            [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s],
        ]
    )


def _apply_rgb_matrix(img_bgr: np.ndarray, m_rgb: np.ndarray) -> np.ndarray:
    """float32 BGR 이미지에 RGB 기준 색 행렬 적용 (결과 0..255 로 클램프)."""
    perm = [2, 1, 0]
    m_bgr = m_rgb[np.ix_(perm, perm)].astype(np.float32)
    out = cv2.transform(img_bgr, m_bgr)
    return np.clip(out, 0, 255)


@dataclass(frozen=True)
class NeonParams:
    color_bgr: tuple[float, float, float]
    thickness_px: float
    glow_px: float
    alpha: float


@dataclass(frozen=True)
class EffectParams:
    blur_px: float
    grayscale: float
    object_saturate: float
    neon: NeonParams | None

    @property
    def has_background(self) -> bool:
        return self.blur_px > 0 or self.grayscale > 0


def hex_to_bgr(color: str) -> tuple[float, float, float]:
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"Invalid color {color!r}")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return float(b), float(g), float(r)


def effect_params(effects: dict, width: int, height: int) -> EffectParams:
    """frontend effectParams() 와 동일한 공식."""
    long = max(width, height)
    neon, pop, blur = effects["neon"], effects["colorPop"], effects["blur"]
    return EffectParams(
        blur_px=blur["intensity"] * 0.03 * long if blur["enabled"] else 0.0,
        grayscale=pop["intensity"] if pop["enabled"] else 0.0,
        object_saturate=1 + 0.35 * pop["intensity"] if pop["enabled"] else 1.0,
        neon=NeonParams(
            color_bgr=hex_to_bgr(neon["color"]),
            thickness_px=max(1.0, neon["thickness"] * long / 640),
            glow_px=neon["intensity"] * 0.035 * long,
            alpha=0.45 + 0.55 * neon["intensity"],
        )
        if neon["enabled"]
        else None,
    )


def any_effect(effects: dict) -> bool:
    return any(effects[k]["enabled"] for k in ("neon", "colorPop", "blur"))


def _gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """경계 밖을 0(투명)으로 보는 가우시안 블러."""
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)


def _over(dst: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return dst * (1 - a) + color * a


def _disk_dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, kernel)


def render_frame(frame_bgr: np.ndarray, mask: np.ndarray | None, p: EffectParams) -> np.ndarray:
    """한 프레임에 효과를 적용. mask: (H, W) 0/1, None 이면 효과 없이 원본 반환 (미리보기와 동일)."""
    if mask is None or (not p.has_background and p.neon is None):
        return frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    if mask.shape != (h, w):
        raise ValueError(f"mask shape {mask.shape} does not match frame {(h, w)}")
    base = frame_bgr.astype(np.float32)
    m = (mask > 0).astype(np.float32)

    if p.has_background:
        # 1) 배경: grayscale 적용본 위에, blur+grayscale 적용본을 겹침 (Canvas 와 같은 순서)
        gray_base = _apply_rgb_matrix(base, _grayscale_matrix(p.grayscale)) if p.grayscale > 0 else base
        out = gray_base
        if p.blur_px > 0:
            ones = np.ones((h, w), np.float32)
            blurred = _gaussian(gray_base, p.blur_px)  # 프리멀티플라이된 색 (알파 1 × 색)
            alpha = _gaussian(ones, p.blur_px)  # 경계 근처에서 알파 < 1
            out = blurred + gray_base * (1 - alpha[..., None])
        # 2) 객체: 원본(채도 강화)을 마스크 영역에 합성
        obj = _apply_rgb_matrix(base, _saturate_matrix(p.object_saturate)) if p.object_saturate != 1 else base
        out = _over(out, obj, m)
    else:
        out = base

    # 3) 네온 외곽선: (팽창 마스크 − 마스크) 링을 그림자(글로우) 두 번 + 선명한 선 한 번
    if p.neon is not None and m.any():
        n = p.neon
        ring = np.clip(_disk_dilate(m, n.thickness_px) - m, 0, 1)
        color = np.array(n.color_bgr, np.float32)
        glow = _gaussian(ring, n.glow_px / 2) if n.glow_px > 0 else ring
        for _ in range(2):
            out = _over(out, color, glow * n.alpha)
            out = _over(out, color, ring * n.alpha)
        out = _over(out, color, ring)

    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
