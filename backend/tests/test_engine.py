"""엔진 상태 전환 테스트 (실제 모델 대신 가짜 빌더 사용)."""

import pytest

pytest.importorskip("sam2")

from app.config import Settings  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.inference.engine import Sam2Engine  # noqa: E402


@pytest.fixture
def engine(tmp_path, monkeypatch):
    import sam2.build_sam
    import sam2.sam2_image_predictor

    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(sam2.build_sam, "build_sam2_video_predictor", lambda *a, **k: object())
    monkeypatch.setattr(sam2.sam2_image_predictor, "SAM2ImagePredictor", lambda model: object())
    s = Settings(sam2_checkpoint=ckpt, device_preference="cpu")
    return Sam2Engine(s)


def test_ready_only_after_warmup(engine):
    seen = []

    def warmup():
        seen.append(engine.state)
        return {"setImageMs": 1, "predictMs": 1}

    engine.load(warmup=warmup)
    assert seen == ["loading"]  # 워밍업 중에는 아직 ready 가 아님
    assert engine.state == "ready" and engine.runtime.name == "cpu"


def test_warmup_failure_marks_error(engine):
    def warmup():
        raise RuntimeError("boom")

    engine.load(warmup=warmup)
    assert engine.state == "error" and "boom" in engine.error
    with pytest.raises(AppError) as e:
        engine.require_ready()
    assert e.value.code == "model_unavailable"


def test_exclusive_busy_timeout(engine):
    engine.load()
    with engine.exclusive("tracking"):
        with pytest.raises(AppError) as e:
            with engine.exclusive("segmentation", timeout=0.05):
                pass
    assert e.value.code == "busy" and "tracking" in e.value.message
