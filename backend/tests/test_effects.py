"""출력용 효과 렌더러 단위 테스트 (모델/FFmpeg 불필요)."""

import cv2
import numpy as np
import pytest

from app.effects import _grayscale_matrix, effect_params, render_frame


def fx(neon=False, pop=False, blur=False, intensity=1.0, color="#22d3ee", thickness=3):
    return {
        "neon": {"enabled": neon, "intensity": intensity, "color": color, "thickness": thickness},
        "colorPop": {"enabled": pop, "intensity": intensity},
        "blur": {"enabled": blur, "intensity": intensity},
    }


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    img = (rng.random((180, 320, 3)) * 200 + 20).astype(np.uint8)  # 고주파 질감 (블러 확인용)
    img[60:120, 120:200] = (40, 60, 230)  # 빨간 객체 (BGR)
    mask = np.zeros((180, 320), np.uint8)
    mask[60:120, 120:200] = 1
    return img, mask


def test_params_match_frontend_formula():
    # frontend/src/lib/effectParams.test.ts 와 같은 입력 → 같은 값
    p = effect_params(fx(neon=True, pop=True, blur=True, intensity=1.0, color="#ffffff", thickness=5), 960, 540)
    assert p.blur_px == pytest.approx(0.03 * 960)
    assert p.neon.thickness_px == pytest.approx(5 * 960 / 640)
    assert p.neon.glow_px == pytest.approx(0.035 * 960)
    assert p.neon.alpha == pytest.approx(1.0)
    full = effect_params(fx(neon=True, blur=True), 1920, 1080)
    half = effect_params(fx(neon=True, blur=True), 960, 540)
    assert full.blur_px / half.blur_px == pytest.approx(2)
    assert full.neon.thickness_px / half.neon.thickness_px == pytest.approx(2)


def test_no_mask_or_no_effects_is_identity(scene):
    img, mask = scene
    assert np.array_equal(render_frame(img, None, effect_params(fx(pop=True), 320, 180)), img)
    assert np.array_equal(render_frame(img, mask, effect_params(fx(), 320, 180)), img)


def test_grayscale_matrix_full_is_luminance():
    m = _grayscale_matrix(1.0)
    assert np.allclose(m[0], [0.2126, 0.7152, 0.0722]) and np.allclose(m[0], m[1]) and np.allclose(m[1], m[2])
    assert np.allclose(_grayscale_matrix(0.0), np.eye(3))


def test_color_pop_only_outside_mask_is_gray(scene):
    img, mask = scene
    out = render_frame(img, mask, effect_params(fx(pop=True), 320, 180)).astype(int)
    outside = out[mask == 0]
    assert np.abs(outside[:, 0] - outside[:, 2]).max() <= 1  # B≈R → 무채색
    inside = out[mask == 1]
    assert (inside[:, 2] - inside[:, 0]).min() > 150  # 객체는 빨간색 유지 (채도 강화)


def test_blur_keeps_object_pixels_exact(scene):
    img, mask = scene
    out = render_frame(img, mask, effect_params(fx(blur=True, intensity=0.5), 320, 180))
    assert np.array_equal(out[mask == 1], img[mask == 1])
    # 배경 질감(고주파)이 크게 줄어듦
    lap_in, lap_out = cv2.Laplacian(img, cv2.CV_32F), cv2.Laplacian(out, cv2.CV_32F)
    bg = (cv2.erode(1 - mask, np.ones((9, 9), np.uint8)) > 0)
    assert np.abs(lap_out[bg]).mean() < 0.2 * np.abs(lap_in[bg]).mean()


def test_neon_ring_surrounds_mask_only(scene):
    img, mask = scene
    p = effect_params(fx(neon=True, thickness=4), 320, 180)
    out = render_frame(img, mask, p).astype(int)
    t = round(p.neon.thickness_px)
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))) - mask
    neon_bgr = np.array([0xEE, 0xD3, 0x22])
    assert np.abs(out[ring == 1] - neon_bgr).mean() < 10  # 링은 네온 색
    core = cv2.erode(mask, np.ones((15, 15), np.uint8))  # 객체 안쪽 깊은 곳은 글로우 영향 거의 없음
    assert np.abs(out[core == 1] - img.astype(int)[core == 1]).mean() < 25
    far = np.zeros_like(mask)
    far[:20, :20] = 1  # 객체에서 먼 배경은 그대로
    assert np.array_equal(out[far == 1], img.astype(int)[far == 1])


def test_empty_mask_applies_background_effect_everywhere(scene):
    """객체가 보이지 않는 프레임: 미리보기와 같이 배경 효과만 전체에 적용, 네온 없음."""
    img, _ = scene
    empty = np.zeros(img.shape[:2], np.uint8)
    out = render_frame(img, empty, effect_params(fx(pop=True, neon=True), 320, 180)).astype(int)
    assert np.abs(out[..., 0] - out[..., 2]).max() <= 1


def test_mask_shape_mismatch_rejected(scene):
    img, mask = scene
    with pytest.raises(ValueError):
        render_frame(img, mask[:-1], effect_params(fx(pop=True), 320, 180))
