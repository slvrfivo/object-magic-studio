"""지연 프레임 소스가 공식 로더와 같은 모델 입력을 만드는지 검증 (모델 불필요)."""

import cv2
import numpy as np
import pytest
import torch

pytest.importorskip("sam2")

from app.inference.frames import LazyVideoFrames, jpeg_frame_paths, lazy_frame_loading  # noqa: E402
from tests.conftest import synthetic_frame  # noqa: E402


@pytest.fixture
def jpeg_dir(tmp_path):
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"{i:05d}.jpg"), synthetic_frame(i / 30))
    return tmp_path


def test_matches_official_loader(jpeg_dir):
    from sam2.utils.misc import load_video_frames_from_jpg_images

    official, h, w = load_video_frames_from_jpg_images(
        str(jpeg_dir), image_size=256, offload_video_to_cpu=True, compute_device=torch.device("cpu")
    )
    lazy = LazyVideoFrames(jpeg_frame_paths(jpeg_dir), image_size=256)
    assert len(lazy) == official.shape[0] == 5
    assert (lazy.video_height, lazy.video_width) == (h, w) == (360, 640)
    for i in range(5):
        assert lazy[i].dtype == torch.float32
        assert torch.allclose(lazy[i], official[i], atol=1e-5)


def test_cache_is_bounded(jpeg_dir):
    lazy = LazyVideoFrames(jpeg_frame_paths(jpeg_dir), image_size=64, cache_size=2)
    for i in range(5):
        lazy[i]
    assert len(lazy._cache) == 2
    lazy[4]  # 캐시 적중
    assert lazy.load_count == 5


def test_patch_is_restored(jpeg_dir):
    import sam2.sam2_video_predictor as vp

    original = vp.load_video_frames
    with lazy_frame_loading(True) as holder:
        frames, h, w = vp.load_video_frames(video_path=str(jpeg_dir), image_size=64, offload_video_to_cpu=False)
        assert isinstance(frames, LazyVideoFrames) and holder["frames"] is frames
    assert vp.load_video_frames is original
    with lazy_frame_loading(False) as holder:
        assert holder is None and vp.load_video_frames is original


def test_frame_order_is_numeric(tmp_path):
    for i in (0, 2, 10, 1):
        cv2.imwrite(str(tmp_path / f"{i:05d}.jpg"), np.zeros((8, 8, 3), np.uint8))
    assert [p.stem for p in jpeg_frame_paths(tmp_path)] == ["00000", "00001", "00002", "00010"]
