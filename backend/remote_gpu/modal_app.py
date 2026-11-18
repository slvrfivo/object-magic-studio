"""Object Magic Studio — Modal 서버리스 GPU 워커 (Phase 5B-2).

**이 파일을 배포하면 요금이 발생할 수 있습니다.** 배포 전 docs/phase5b2/README.md 의 비용 안내를 확인하세요.

구성
- 기존 추론 코드(backend/app)를 이미지에 그대로 넣고 app.remote.worker.GpuWorker 를 실행합니다.
  (Sam2Engine + Sam2ImageService + VideoTracker, SAM 2.1 Hiera Tiny, fp16 autocast)
- 컨테이너당 모델 1개: 분할과 추적이 같은 모델 인스턴스와 프레임 특징 캐시를 공유합니다.
- 공개 웹 엔드포인트 없음: 로컬 서버가 Modal SDK(토큰 인증)로만 호출합니다.
- 비용 제어: max_containers=1 (동시 GPU 1대), scaledown_window (마지막 호출 뒤 유지 시간), 호출당 timeout,
  min_containers=0 (사용하지 않을 때 0원). 워크스페이스 월 예산은 Modal 대시보드에서 설정하세요.
- 임시 파일: 컨테이너 로컬 디스크(/tmp/oms)에만 저장, 영구 Volume 을 쓰지 않음. 컨테이너가 꺼지면 모두 삭제됩니다.

배포 (사용자가 승인한 뒤에만):
    cd backend
    pip install modal            # 로컬 서버 venv 에 설치
    modal token new              # 브라우저에서 Modal 로그인 → 토큰 저장 (~/.modal.toml)
    modal deploy remote_gpu/modal_app.py
그다음 로컬 서버를 OMS_REMOTE=modal 로 실행합니다.

환경 변수 (배포할 때 읽음):
    OMS_MODAL_GPU      GPU 종류 (기본 T4)
    OMS_MODAL_IDLE_S   마지막 호출 뒤 컨테이너 유지 초 (기본 120, 로컬 OMS_REMOTE_IDLE_S 와 같게)
    OMS_MODAL_APP      앱 이름 (기본 object-magic-studio-gpu, 로컬 OMS_MODAL_APP 과 같게)
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = os.getenv("OMS_MODAL_APP", "object-magic-studio-gpu")
GPU = os.getenv("OMS_MODAL_GPU", "T4")
IDLE_S = int(os.getenv("OMS_MODAL_IDLE_S", "120"))
CALL_TIMEOUT_S = 600  # 300프레임 추적(T4 약 60초) + 콜드 스타트 여유
CANCEL_DICT = "oms-gpu-cancel"

# 로컬 서버와 같은 모델 (backend/app/main.py 의 Phase 5B-1 검증과 동일)
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
CKPT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
CKPT_PATH = "/opt/oms/checkpoints/sam2.1_hiera_tiny.pt"

BACKEND = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "curl", "libgl1", "libglib2.0-0")
    # Colab T4 에서 검증한 조합: torch 2.14.1 + CUDA 13.0 (realworld 검증 결과의 environment 참고)
    .pip_install("torch==2.14.1", "torchvision", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install(
        "hydra-core>=1.3.2",
        "iopath>=0.1.10",
        "opencv-python-headless>=4.10",
        "numpy>=1.26",
        "pillow",
        "tqdm",
        "psutil>=5.9",
        "pydantic>=2",
        "fastapi>=0.115",  # app.errors 가 fastapi 타입을 import (서버는 띄우지 않음)
        "python-multipart",
    )
    .run_commands(
        f"SAM2_BUILD_CUDA=0 pip install --no-deps --no-build-isolation 'git+https://github.com/facebookresearch/sam2.git@{SAM2_COMMIT}'",
        f"mkdir -p /opt/oms/checkpoints && curl -fsSL -o {CKPT_PATH} {CKPT_URL}",
        f"echo '{CKPT_SHA256}  {CKPT_PATH}' | sha256sum -c -",
    )
    .env({"OMS_DEVICE": "cuda", "OMS_PRECISION": "fp16", "PYTHONUNBUFFERED": "1"})
    # 기존 추론 코드 그대로 (마지막 단계에 두어 코드만 바뀌면 빠르게 다시 빌드)
    .add_local_dir(BACKEND / "app", remote_path="/root/app", ignore=["**/__pycache__/**", "**/*.pyc"])
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU,
    cpu=2.0,
    memory=8192,
    timeout=CALL_TIMEOUT_S,
    scaledown_window=IDLE_S,
    min_containers=0,
    max_containers=1,
)
@modal.concurrent(max_inputs=4)  # 추적 중에도 업로드/상태/취소 확인 호출이 같은 컨테이너에서 처리되도록
class Sam2Gpu:
    @modal.enter()
    def load(self) -> None:
        import time

        from app.config import Settings
        from app.remote.worker import GpuWorker

        t = time.perf_counter()
        settings = Settings(
            device_preference="cuda",
            precision="fp16",
            offload_state="off",  # T4 16GB: 300프레임도 GPU 메모리 1GB 미만 (realworld 검증)
            sam2_checkpoint=Path(CKPT_PATH),
            max_track_frames_cuda=300,
            sessions_dir=Path("/tmp/oms/sessions"),
        )
        self.worker = GpuWorker(settings, Path("/tmp/oms/worker"))
        self.worker.load()
        self.cancel = modal.Dict.from_name(CANCEL_DICT, create_if_missing=True)
        print(f"SAM 2.1 ready on {self.worker.engine.runtime.name} in {time.perf_counter() - t:.1f}s", flush=True)

    @modal.exit()
    def cleanup(self) -> None:
        self.worker.shutdown()  # 영상 프레임 등 임시 파일 삭제

    @modal.method()
    def info(self) -> dict:
        return self.worker.info()

    @modal.method()
    def has_video(self, video_sha: str) -> dict:
        return self.worker.has_video(video_sha)

    @modal.method()
    def put_video(self, video_sha: str, data: bytes) -> dict:
        return self.worker.put_video(video_sha, data)

    @modal.method()
    def drop_video(self, video_sha: str) -> dict:
        return self.worker.drop_video(video_sha)

    @modal.method()
    def segment(self, video_sha: str, frame_index: int, points_px: list, labels: list) -> dict:
        return self.worker.segment(video_sha, frame_index, points_px, labels)

    @modal.method()
    def track_stream(self, video_sha: str, prompt_frame: int, points_px: list, labels: list, max_frames: int, cancel_key: str, mask_rle: dict | None = None):
        """진행 이벤트를 스트리밍하고 마지막에 결과(마스크 PNG)를 보냄. 취소는 Modal Dict 플래그로 확인."""

        def should_cancel() -> bool:
            return bool(self.cancel.get(cancel_key, False))

        yield from self.worker.track_stream(video_sha, prompt_frame, points_px, labels, max_frames, should_cancel=should_cancel, mask_rle=mask_rle)
