"""추론 장치 선택 및 호환성 진단.

원칙:
- CUDA는 "사용 가능 + 현재 PyTorch 빌드에 해당 GPU 아키텍처 커널 포함 + 실제 연산 smoke test 통과" 일 때만 사용합니다.
- bfloat16 지원을 가정하지 않습니다. 이미지 추론은 float32로 실행합니다(autocast 미사용).
- OMS_DEVICE=cuda 로 강제했는데 사용할 수 없으면 조용히 CPU로 바꾸지 않고 오류를 냅니다.
"""

import subprocess
from dataclasses import asdict, dataclass, field

import torch


@dataclass
class GpuInfo:
    name: str | None = None
    driver_version: str | None = None
    total_memory_mb: int | None = None
    compute_capability: str | None = None
    bf16_supported: bool | None = None


@dataclass
class DeviceInfo:
    device: str  # "cuda" | "cpu"
    reason: str
    torch_version: str = torch.__version__
    torch_cuda_build: str | None = torch.version.cuda
    cuda_available: bool = False
    gpu: GpuInfo = field(default_factory=GpuInfo)

    def to_json(self) -> dict:
        d = asdict(self)
        return {
            "device": d["device"],
            "reason": d["reason"],
            "torchVersion": d["torch_version"],
            "torchCudaBuild": d["torch_cuda_build"],
            "cudaAvailable": d["cuda_available"],
            "gpu": {
                "name": self.gpu.name,
                "driverVersion": self.gpu.driver_version,
                "totalMemoryMB": self.gpu.total_memory_mb,
                "computeCapability": self.gpu.compute_capability,
                "bf16Supported": self.gpu.bf16_supported,
            },
        }


def _query_nvidia_smi() -> GpuInfo:
    """PyTorch가 CPU 빌드여도 진단 정보를 얻기 위해 nvidia-smi를 조회합니다."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return GpuInfo()
        name, driver, mem = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
        return GpuInfo(name=name, driver_version=driver, total_memory_mb=int(float(mem)))
    except (OSError, subprocess.SubprocessError, ValueError):
        return GpuInfo()


def _cuda_problem(gpu: GpuInfo) -> str | None:
    """CUDA를 쓸 수 없는 이유를 반환. 사용 가능하면 None."""
    if torch.version.cuda is None:
        hint = f" (NVIDIA driver {gpu.driver_version})" if gpu.driver_version else ""
        return f"Installed PyTorch is a CPU-only build{hint}."
    if not torch.cuda.is_available():
        return f"torch.cuda.is_available() is False (PyTorch built for CUDA {torch.version.cuda}; driver may be too old)."
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    if arch_list and arch not in arch_list and f"compute_{major}{minor}" not in arch_list:
        return f"This PyTorch build has no kernels for {arch} (built for {', '.join(arch_list)})."
    try:
        x = torch.randn(256, 256, device="cuda")
        float((x @ x).sum().item())
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 — 진단 목적
        return f"CUDA smoke test failed: {e}"
    return None


def select_device(preference: str) -> DeviceInfo:
    if preference not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"OMS_DEVICE must be auto, cuda or cpu (got {preference!r})")

    gpu = _query_nvidia_smi()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu.name = props.name
        gpu.total_memory_mb = props.total_memory // (1024 * 1024)
        gpu.compute_capability = f"{props.major}.{props.minor}"
        try:
            gpu.bf16_supported = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:  # 구버전 PyTorch
            gpu.bf16_supported = bool(torch.cuda.is_bf16_supported())

    if preference == "cpu":
        return DeviceInfo("cpu", "Forced by OMS_DEVICE=cpu.", cuda_available=torch.cuda.is_available(), gpu=gpu)

    problem = _cuda_problem(gpu)
    if problem is None:
        return DeviceInfo("cuda", "CUDA compatibility checks passed.", cuda_available=True, gpu=gpu)
    if preference == "cuda":
        raise RuntimeError(f"OMS_DEVICE=cuda but CUDA cannot be used: {problem}")
    return DeviceInfo("cpu", f"CUDA not used: {problem}", cuda_available=torch.cuda.is_available(), gpu=gpu)
