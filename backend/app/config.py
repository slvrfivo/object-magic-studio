"""애플리케이션 설정. 검증 한도는 frontend/src/lib/validation.ts 와 동일하게 유지합니다.

환경 변수:
  OMS_DEVICE      auto | cuda | cpu   (기본 auto: CUDA 호환성 검사 통과 시 cuda, 아니면 cpu)
  OMS_LOAD_MODEL  1 | 0               (0이면 모델을 로드하지 않음 — API 테스트용)
  OMS_REMOTE      "" | modal | inprocess  (원격 GPU, Phase 5B-2. 비어 있으면 로컬 CPU 만)
"""

import os
from pathlib import Path

from pydantic import BaseModel

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # 업로드 검증 한도
    max_duration_sec: float = 10.0
    duration_tolerance_sec: float = 0.1
    max_file_size_bytes: int = 100 * 1024 * 1024
    max_long_side: int = 1920
    max_short_side: int = 1080
    min_side: int = 64
    max_fps: float = 120.0

    # 세션별 중간 파일 격리 루트 (data/sessions/<session_id>/...)
    sessions_dir: Path = BACKEND_ROOT / "data" / "sessions"
    session_ttl_hours: float = 6.0

    # SAM 2.1 Hiera Tiny (공식 체크포인트)
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint: Path = BACKEND_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
    device_preference: str = os.getenv("OMS_DEVICE", "auto").lower()
    load_model: bool = os.getenv("OMS_LOAD_MODEL", "1") == "1"
    #: 추론 정밀도: fp32 | bf16 | fp16 (bf16/fp16 은 CUDA 에서만, 지원 여부를 확인한 뒤 autocast)
    precision: str = os.getenv("OMS_PRECISION", "fp32").lower()
    #: 비디오 추론 상태 CPU 오프로딩: auto (VRAM 8GB 미만이면 켬) | on | off
    offload_state: str = os.getenv("OMS_OFFLOAD_STATE", "auto").lower()

    # 영상 추적 (Phase 3). CPU 한도는 실측 결과로 결정 (README 참고)
    max_track_frames_cpu: int = int(os.getenv("OMS_MAX_TRACK_FRAMES_CPU", "60"))
    max_track_frames_cuda: int = 300
    #: 이 비율보다 작은 마스크는 "객체 소실" 로 표시
    lost_coverage: float = 0.0002
    #: 직전 프레임 대비 면적이 이 배수 이상 변하면 "불안정" 으로 표시
    unstable_jump_ratio: float = 2.0

    # 오프라인 GPU 작업 (Phase 5B-1): Colab 에서 추적한 결과 가져오기
    #: GPU(cuda) 로 만든 결과에 허용하는 최대 프레임 수 (Phase 5A 에서 T4 fp16 300프레임 56초 실측)
    max_gpu_import_frames: int = 300
    gpu_jobs_dir: Path = BACKEND_ROOT / "data" / "gpu_jobs"
    gpu_job_ttl_days: float = 14.0

    # 원격 GPU (Phase 5B-2). 기본은 꺼져 있음 → 로컬 CPU 만 사용
    #: "" (끔) | "modal" (Modal 서버리스 GPU) | "inprocess" (개발·데모용: 원격 경로를 로컬 모델로 실행)
    remote_provider: str = os.getenv("OMS_REMOTE", "").lower()
    remote_gpu: str = os.getenv("OMS_REMOTE_GPU", "T4")
    remote_max_frames: int = int(os.getenv("OMS_REMOTE_MAX_FRAMES", "300"))
    remote_max_upload_bytes: int = int(os.getenv("OMS_REMOTE_MAX_UPLOAD_MB", "100")) * 1024 * 1024
    #: 예산 확인용 프레임당 예상 시간 (Colab T4 fp16 실측 0.19~0.21초)
    remote_expected_frame_s: float = 0.25
    modal_app_name: str = os.getenv("OMS_MODAL_APP", "object-magic-studio-gpu")
    modal_class_name: str = "Sam2Gpu"
    modal_cancel_dict: str = "oms-gpu-cancel"
    #: 사용량·비용 한도 (추정치 기준, 0 이면 한도 없음)
    remote_daily_gpu_seconds: float = float(os.getenv("OMS_REMOTE_DAILY_GPU_SECONDS", "2400"))
    remote_monthly_usd: float = float(os.getenv("OMS_REMOTE_MONTHLY_USD", "5"))
    #: T4 $0.000164/s + CPU 2코어 + 메모리 8GiB ≈ $0.000208/s (Modal 공개 요금, 2026-11 확인)
    remote_cost_per_s: float = float(os.getenv("OMS_REMOTE_COST_PER_S", "0.000208"))
    #: 마지막 호출 뒤 컨테이너 유지 시간 (remote_gpu/modal_app.py 의 scaledown_window 와 같게)
    remote_idle_s: float = float(os.getenv("OMS_REMOTE_IDLE_S", "120"))
    remote_cold_start_s: float = float(os.getenv("OMS_REMOTE_COLD_START_S", "25"))  # 실측 19.1초 (T4, 2026-11-18) + 여유
    remote_usage_file: Path = BACKEND_ROOT / "data" / "remote_usage.json"

    # MP4 내보내기 (Phase 4): libx264, yuv420p
    export_crf: int = 18
    export_preset: str = "medium"
    export_finalize_timeout_s: float = 120.0

    # 프론트엔드에 반환할 마스크 최대 크기
    max_mask_side: int = 4096

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
