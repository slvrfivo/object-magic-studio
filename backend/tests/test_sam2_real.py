"""실제 SAM 2.1 Hiera Tiny 추론 테스트 (느림). 체크포인트/sam2 패키지가 없으면 건너뜁니다.

실행: pytest -m real_model -s
정답(ground truth)을 아는 합성 프레임(원 + 사각형)으로 마스크 품질과 negative 점 반영을 검증합니다.
"""

import cv2
import numpy as np
import pytest

from app.config import Settings
from tests.conftest import synthetic_frame

pytestmark = pytest.mark.real_model

sam2 = pytest.importorskip("sam2")


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max(1, (a | b).sum())


@pytest.fixture(scope="module")
def service():
    from app.inference.engine import Sam2Engine
    from app.inference.sam2_image import Sam2ImageService

    s = Settings()
    if not s.sam2_checkpoint.exists():
        pytest.skip("checkpoint missing")
    engine = Sam2Engine(s)
    svc = Sam2ImageService(engine)
    engine.load(warmup=svc.warmup)
    assert engine.state == "ready", engine.error
    print("\n[status]", svc.status())
    return svc


@pytest.fixture(scope="module")
def frame_and_truth():
    bgr = synthetic_frame(0.0)  # 원 중심 (320, 220), 반지름 55 / 사각형 (40,260)-(200,330)
    circle = np.zeros(bgr.shape[:2], np.uint8)
    cv2.circle(circle, (320, 220), 55, 1, -1)
    rect = np.zeros(bgr.shape[:2], np.uint8)
    cv2.rectangle(rect, (40, 260), (200, 330), 1, -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), circle, rect


def run(service, frame, pts, labels, key=("test", 0)):
    out = service.segment(key, lambda: frame, np.array(pts, np.float32), np.array(labels))
    print(f"  points={pts} labels={labels} -> {out.metrics}")
    return out


def test_single_click_segments_circle(service, frame_and_truth):
    frame, circle, _ = frame_and_truth
    out = run(service, frame, [[320, 220]], [1])
    assert out.mask.shape == circle.shape
    assert iou(out.mask, circle) > 0.85
    assert out.metrics["embeddingCached"] is False


def test_second_click_reuses_embedding(service, frame_and_truth):
    frame, _, rect = frame_and_truth
    out = run(service, frame, [[120, 295]], [1])
    assert out.metrics["embeddingCached"] is True
    assert out.metrics["setImageMs"] == 0
    assert iou(out.mask, rect) > 0.85


def test_negative_point_removes_object(service, frame_and_truth):
    frame, circle, rect = frame_and_truth
    both = run(service, frame, [[320, 220], [120, 295]], [1, 1]).mask
    assert both[220, 320] and both[295, 120], "두 객체 모두 포함되어야 함"
    neg = run(service, frame, [[320, 220], [120, 295], [120, 295]], [1, 1, 0]).mask
    # 같은 위치에 negative를 추가하면 사각형 쪽이 빠지고 원은 유지되어야 함
    assert neg[220, 320] == 1
    assert neg[295, 120] == 0
    assert (neg & rect).sum() < 0.2 * rect.sum()


def test_new_frame_recomputes_embedding(service, frame_and_truth):
    frame, _, _ = frame_and_truth
    out = run(service, frame, [[320, 220]], [1], key=("test", 1))
    assert out.metrics["embeddingCached"] is False
    assert out.metrics["setImageMs"] > 0
