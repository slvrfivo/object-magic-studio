"""단일 프레임 분할과 영상 추적이 같은 이미지 임베딩(백본 특징)을 공유하기 위한 도구 (Phase 5B-2).

SAM 2.1 은 분할(SAM2ImagePredictor.set_image)과 추적(SAM2VideoPredictor._get_image_feature)에서
같은 이미지 인코더 `forward_image` 를 사용합니다. 차이는 입력 전처리뿐입니다.
  - 이미지 예측기: torchvision Resize(1024) + 정규화
  - 영상 예측기:   PIL resize(1024) + 정규화 (sam2.utils.misc._load_img_as_tensor)

여기서는 **영상 예측기의 전처리**로 프레임을 한 번 인코딩해 캐시에 두고,
  1) 분할: 같은 백본 출력으로 이미지 예측기의 내부 특징을 채운 뒤 predict (set_image 의 후반부와 동일한 계산)
  2) 추적: init_state/add_new_points 가 그 프레임의 특징을 다시 계산하지 않도록 cached_features 에 미리 넣음
→ 추적을 시작할 때 프롬프트 프레임의 인코딩(GPU 에서 수십 ms, CPU 에서 수 초)을 생략하고,
  추적 결과는 특징을 새로 계산한 경우와 같습니다 (tests/real_model/test_remote_worker_real.py 에서 확인).

로컬 CPU 모드의 분할은 기존 set_image 경로를 그대로 사용합니다 (동작 변경 없음).
"""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from pathlib import Path

import torch

from .frames import IMG_MEAN, IMG_STD

#: (영상 키, 프레임 번호) → FrameFeatures
FeatureKey = tuple[str, int]


class FrameFeatures:
    """한 프레임의 백본 출력 (영상 예측기 cached_features 와 같은 형식)."""

    __slots__ = ("image", "backbone_out", "orig_hw")

    def __init__(self, image: torch.Tensor, backbone_out: dict, orig_hw: tuple[int, int]):
        self.image = image  # (1, 3, S, S) float32, 모델 장치
        self.backbone_out = backbone_out
        self.orig_hw = orig_hw  # 원본 프레임 (높이, 너비)


def load_frame_tensor(jpg_path: Path, image_size: int) -> tuple[torch.Tensor, int, int]:
    """영상 예측기와 같은 전처리 (LazyVideoFrames 와 동일): (3, S, S) float32 정규화 텐서."""
    from sam2.utils.misc import _load_img_as_tensor

    img, h, w = _load_img_as_tensor(str(jpg_path), image_size)
    img = img.float()
    img -= torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    img /= torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
    return img, h, w


def encode_frame(video_predictor, jpg_path: Path, device: torch.device) -> FrameFeatures:
    """프레임 1장 인코딩. 호출하는 쪽에서 inference_mode + autocast 를 적용합니다."""
    img, h, w = load_frame_tensor(jpg_path, video_predictor.image_size)
    image = img.to(device).float().unsqueeze(0)
    backbone_out = video_predictor.forward_image(image)
    return FrameFeatures(image, backbone_out, (h, w))


def set_image_from_features(image_predictor, feats: FrameFeatures) -> None:
    """SAM2ImagePredictor.set_image 의 인코딩 이후 단계와 같은 계산으로 내부 특징을 채움."""
    model = image_predictor.model
    image_predictor.reset_predictor()
    image_predictor._orig_hw = [feats.orig_hw]
    _, vision_feats, _, _ = model._prepare_backbone_features(feats.backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    bb_sizes = image_predictor._bb_feat_sizes
    maps = [feat.permute(1, 2, 0).view(1, -1, *size) for feat, size in zip(vision_feats[::-1], bb_sizes[::-1])][::-1]
    image_predictor._features = {"image_embed": maps[-1], "high_res_feats": maps[:-1]}
    image_predictor._is_image_set = True


class FeatureCache:
    """최근 인코딩한 프레임 특징의 작은 LRU (GPU 메모리를 고려해 기본 2장)."""

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self._items: OrderedDict[FeatureKey, FrameFeatures] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: FeatureKey) -> FrameFeatures | None:
        with self._lock:
            f = self._items.get(key)
            if f is not None:
                self._items.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return f

    def put(self, key: FeatureKey, feats: FrameFeatures) -> None:
        with self._lock:
            self._items[key] = feats
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def drop_video(self, video_key: str) -> None:
        with self._lock:
            for k in [k for k in self._items if k[0] == video_key]:
                del self._items[k]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


@contextlib.contextmanager
def preloaded_features(video_predictor, preload: dict[int, FrameFeatures] | None):
    """이 구간 동안 영상 예측기가 preload 에 있는 프레임(로컬 번호)의 특징을 다시 계산하지 않게 함.

    공식 _get_image_feature 는 cached_features 에 없으면 forward_image 를 호출하므로,
    호출 직전에 미리 계산한 특징을 cached_features 에 넣어 줍니다 (인스턴스 속성만 잠시 교체).
    """
    if not preload:
        yield {"reused": 0}
        return
    original = video_predictor._get_image_feature
    info = {"reused": 0}

    def _get_image_feature(inference_state, frame_idx, batch_size):
        cache = inference_state["cached_features"]
        if frame_idx not in cache and frame_idx in preload:
            f = preload[frame_idx]
            inference_state["cached_features"] = {frame_idx: (f.image, f.backbone_out)}
            info["reused"] += 1
        return original(inference_state, frame_idx, batch_size)

    video_predictor._get_image_feature = _get_image_feature
    try:
        yield info
    finally:
        del video_predictor._get_image_feature  # 클래스 메서드로 복구
