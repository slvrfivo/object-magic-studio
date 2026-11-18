import numpy as np
import pytest

from app.masks import normalized_to_pixel, resize_mask, rle_decode, rle_encode


def test_rle_roundtrip_and_format():
    assert rle_encode(np.array([0, 0, 1, 1, 1, 0])) == [2, 3, 1]
    assert rle_encode(np.array([1, 1, 0])) == [0, 2, 1]
    rng = np.random.default_rng(0)
    m = (rng.random((37, 53)) > 0.6).astype(np.uint8)
    assert np.array_equal(rle_decode(rle_encode(m), 53, 37), m)


def test_rle_decode_rejects_wrong_length():
    with pytest.raises(ValueError):
        rle_decode([3, 2], 2, 2)


def test_normalized_to_pixel_matches_frontend_rule():
    px = normalized_to_pixel([(0.25, 0.5), (1.2, -0.1)], 1920, 1080)
    assert px.tolist() == [[480.0, 540.0], [1920.0, 0.0]]


def test_resize_mask_preserves_shape_location():
    m = np.zeros((1080, 1920), np.uint8)
    m[540:1080, 0:960] = 1  # 왼쪽 아래 사분면
    small = resize_mask(m, 960, 540)
    assert small.shape == (540, 960)
    assert small[400, 100] == 1 and small[100, 800] == 0
    assert abs(small.mean() - 0.25) < 0.01
