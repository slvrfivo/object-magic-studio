"""SAM 2 비디오 예측기용 지연(lazy) 프레임 소스.

공식 init_state() 는 모든 프레임을 1024x1024 텐서로 미리 메모리에 올립니다.
  - 동기 로더: float32, 프레임당 약 12MB  (300프레임 ≈ 3.6GB)
  - 비동기 로더(async_loading_frames): float64로 보관, 프레임당 약 24MB
RAM 7.6GB인 개발 PC에서는 감당할 수 없으므로, init_state 호출 동안만 공식 로더 함수를
이 지연 로더로 교체합니다. 전처리는 공식 _load_img_as_tensor + mean/std 정규화를 그대로 사용하므로
모델 입력은 동일합니다 (tests/test_frames.py 에서 공식 로더와 값 비교).

예측기는 inference_state["images"][frame_idx] 로 한 번에 한 프레임만 읽고,
특징(feature)은 자체 캐시에 보관하므로 프레임 텐서는 최근 몇 개만 유지하면 됩니다.
"""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from pathlib import Path

import torch

# sam2/utils/misc.py 의 load_video_frames 기본값과 동일
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

_patch_lock = threading.Lock()


class LazyVideoFrames:
    def __init__(self, img_paths: list[Path], image_size: int, cache_size: int = 2):
        from sam2.utils.misc import _load_img_as_tensor

        if not img_paths:
            raise RuntimeError("No JPEG frames found")
        self._load = _load_img_as_tensor
        self.img_paths = img_paths
        self.image_size = image_size
        self.img_mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._cache_size = cache_size
        self.load_count = 0
        self[0]  # video_height / video_width 확인

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        img, h, w = self._load(str(self.img_paths[index]), self.image_size)
        img = img.float()  # 공식 동기 로더와 같은 float32
        img -= self.img_mean
        img /= self.img_std
        self.video_height, self.video_width = h, w
        self.load_count += 1
        self._cache[index] = img
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img


def jpeg_frame_paths(video_dir: Path) -> list[Path]:
    paths = [p for p in Path(video_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    return sorted(paths, key=lambda p: int(p.stem))


@contextlib.contextmanager
def lazy_frame_loading(enabled: bool = True):
    """init_state 호출 동안 공식 load_video_frames 를 지연 로더로 교체."""
    if not enabled:
        yield None
        return
    import sam2.sam2_video_predictor as vp

    holder: dict = {}

    def _lazy_load_video_frames(video_path, image_size, **_kwargs):
        frames = LazyVideoFrames(jpeg_frame_paths(Path(video_path)), image_size)
        holder["frames"] = frames
        return frames, frames.video_height, frames.video_width

    with _patch_lock:
        original = vp.load_video_frames
        vp.load_video_frames = _lazy_load_video_frames
        try:
            yield holder
        finally:
            vp.load_video_frames = original
