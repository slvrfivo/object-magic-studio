"""장치별(CPU / CUDA) 실행 세부사항을 한곳에 모은 런타임 계층.

모델 코드(sam2_image.py, sam2_video.py)는 장치를 직접 분기하지 않고 이 인터페이스만 사용합니다.
GPU 서버로 이전할 때는 CudaRuntime 의 정책(오프로딩, autocast 등)만 조정하면 됩니다.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod

import torch

from .device import DeviceInfo

# torch>=2.5: torch.OutOfMemoryError, 구버전: torch.cuda.OutOfMemoryError
_OOM_ERROR = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def _process_rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except ImportError:
        return None


PRECISIONS = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


class InferenceRuntime(ABC):
    name: str

    def __init__(self, info: DeviceInfo, precision: str = "fp32", offload_state: str = "auto"):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {list(PRECISIONS)} (got {precision!r})")
        if offload_state not in ("auto", "on", "off"):
            raise ValueError(f"offload_state must be auto, on or off (got {offload_state!r})")
        self.info = info
        self.precision = precision
        self.offload_state = offload_state
        self._validate()

    def _validate(self) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return torch.device(self.name)

    @abstractmethod
    def synchronize(self) -> None: ...

    @abstractmethod
    def reset_peak_memory(self) -> None: ...

    @abstractmethod
    def memory_snapshot(self) -> dict: ...

    def empty_cache(self) -> None:
        pass

    def is_oom(self, exc: BaseException) -> bool:
        return isinstance(exc, _OOM_ERROR)

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return PRECISIONS[self.precision]

    def autocast(self):
        """추론 정밀도 컨텍스트. fp32 면 autocast 없음."""
        dtype = self.autocast_dtype
        if dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.name, dtype=dtype)

    @abstractmethod
    def video_state_options(self) -> dict:
        """SAM2VideoPredictor.init_state 에 전달할 오프로딩 옵션."""

    @property
    @abstractmethod
    def lazy_frames(self) -> bool:
        """프레임을 필요할 때만 디스크에서 읽을지 여부 (메모리 절약)."""

    def describe(self) -> dict:
        return {
            "device": self.name,
            "precision": self.precision,
            "lazyFrames": self.lazy_frames,
            "videoStateOptions": self.video_state_options(),
        }


class CpuRuntime(InferenceRuntime):
    name = "cpu"

    def _validate(self) -> None:
        # 개발 PC(Intel 8세대)에는 bf16/fp16 연산 가속이 없으므로 CPU 는 fp32 만 허용
        if self.precision != "fp32":
            raise ValueError(f"CPU runtime supports only fp32 (got {self.precision}).")

    def synchronize(self) -> None:
        pass

    def reset_peak_memory(self) -> None:
        pass

    def memory_snapshot(self) -> dict:
        rss = _process_rss_mb()
        return {} if rss is None else {"processRssMB": rss}

    def video_state_options(self) -> dict:
        # CPU에서는 이미 모든 텐서가 CPU 메모리에 있으므로 오프로딩은 의미 없음 (설정값 무시)
        return {"offload_video_to_cpu": False, "offload_state_to_cpu": False}

    @property
    def lazy_frames(self) -> bool:
        # 공식 로더는 모든 프레임을 1024x1024 float로 미리 올림 (프레임당 12MB+).
        # RAM이 작은 PC에서는 필요한 프레임만 읽도록 함.
        return True


class CudaRuntime(InferenceRuntime):
    name = "cuda"

    #: 이 값 미만의 VRAM이면 추론 상태를 CPU로 오프로딩 (공식 옵션)
    SMALL_VRAM_MB = 8 * 1024

    def _validate(self) -> None:
        cc = self.info.gpu.compute_capability
        major = int(cc.split(".")[0]) if cc else 0
        if self.precision == "bf16" and not self.info.gpu.bf16_supported:
            # Pascal/Turing(T4 등)은 bf16 연산을 하드웨어로 지원하지 않음 → 가정하지 않고 거부
            raise ValueError(f"bf16 is not supported on this GPU ({self.info.gpu.name}, compute capability {cc}). Use fp16 or fp32.")
        if self.precision == "fp16" and major and major < 7:
            raise ValueError(f"fp16 autocast needs Tensor Cores (compute capability >= 7.0); this GPU is {cc}.")

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def reset_peak_memory(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def memory_snapshot(self) -> dict:
        snap = {
            "peakAllocatedMB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peakReservedMB": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "reservedMB": round(torch.cuda.memory_reserved() / 2**20, 1),
        }
        rss = _process_rss_mb()
        if rss is not None:
            snap["processRssMB"] = rss
        return snap

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def video_state_options(self) -> dict:
        if self.offload_state == "auto":
            offload = (self.info.gpu.total_memory_mb or 0) < self.SMALL_VRAM_MB
        else:
            offload = self.offload_state == "on"
        # 영상 프레임은 항상 CPU에 두고(지연 로더, 오버헤드 작음), VRAM이 작으면 추론 상태도 CPU에 둠
        return {"offload_video_to_cpu": True, "offload_state_to_cpu": offload}

    @property
    def lazy_frames(self) -> bool:
        return True

    # 참고: Ampere(8.0+) GPU 서버에서는 공식 예제처럼 bf16 autocast 를 켜면 빨라집니다 (OMS_PRECISION=bf16).
    # Pascal(MX150)·Turing(T4)은 bf16 연산을 지원하지 않으므로 기본값은 fp32 입니다.


def create_runtime(info: DeviceInfo, precision: str = "fp32", offload_state: str = "auto") -> InferenceRuntime:
    cls = CudaRuntime if info.device == "cuda" else CpuRuntime
    return cls(info, precision=precision, offload_state=offload_state)
