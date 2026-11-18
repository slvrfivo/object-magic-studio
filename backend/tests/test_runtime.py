"""장치별 런타임 정책 테스트 (GPU 없이 DeviceInfo 를 가짜로 만들어 검증)."""

import pytest
import torch

from app.inference.device import DeviceInfo, GpuInfo
from app.inference.runtime import CpuRuntime, CudaRuntime, create_runtime


def gpu(name="Tesla T4", cc="7.5", mem=15360, bf16=False) -> DeviceInfo:
    return DeviceInfo("cuda", "test", cuda_available=True, gpu=GpuInfo(name=name, total_memory_mb=mem, compute_capability=cc, bf16_supported=bf16))


def test_cpu_is_fp32_only():
    info = DeviceInfo("cpu", "test")
    rt = create_runtime(info)
    assert isinstance(rt, CpuRuntime) and rt.autocast_dtype is None
    for p in ("bf16", "fp16"):
        with pytest.raises(ValueError, match="only fp32"):
            create_runtime(info, precision=p)


def test_unknown_options_rejected():
    with pytest.raises(ValueError):
        create_runtime(DeviceInfo("cpu", "t"), precision="int8")
    with pytest.raises(ValueError):
        create_runtime(DeviceInfo("cpu", "t"), offload_state="maybe")


def test_bf16_requires_hardware_support():
    # T4(Turing 7.5)은 bf16 미지원 → 가정하지 않고 거부
    with pytest.raises(ValueError, match="bf16 is not supported"):
        create_runtime(gpu(), precision="bf16")
    rt = create_runtime(gpu(name="A100", cc="8.0", bf16=True), precision="bf16")
    assert isinstance(rt, CudaRuntime) and rt.autocast_dtype is torch.bfloat16


def test_fp16_requires_tensor_cores():
    assert create_runtime(gpu(), precision="fp16").autocast_dtype is torch.float16
    with pytest.raises(ValueError, match="Tensor Cores"):
        create_runtime(gpu(name="GeForce MX150", cc="6.1", mem=2048), precision="fp16")


@pytest.mark.parametrize(
    "mem,mode,expected",
    [(2048, "auto", True), (15360, "auto", False), (15360, "on", True), (2048, "off", False)],
)
def test_state_offload_policy(mem, mode, expected):
    rt = create_runtime(gpu(mem=mem), offload_state=mode)
    assert rt.video_state_options() == {"offload_video_to_cpu": True, "offload_state_to_cpu": expected}


def test_describe_reports_precision():
    d = create_runtime(gpu(), precision="fp16").describe()
    assert d["precision"] == "fp16" and d["device"] == "cuda"
