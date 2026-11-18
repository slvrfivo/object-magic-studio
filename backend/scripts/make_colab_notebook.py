"""colab/phase5a_gpu_benchmark.ipynb 생성 (nbformat v4 JSON).

사용: python scripts/make_colab_notebook.py ../colab/phase5a_gpu_benchmark.ipynb
"""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Object Magic Studio — Phase 5A: SAM 2.1 GPU 추론 검증 (Google Colab)

로컬 CPU에서 측정한 결과와 **같은 체크포인트, 같은 SAM 2 커밋, 같은 입력 프레임(JPEG 300장), 같은 추론 코드(`app/`)**로 GPU 성능과 정확도를 측정합니다.

## 직접 실행할 단계
1. **런타임 → 런타임 유형 변경 → 하드웨어 가속기: `T4 GPU`** (또는 사용 가능한 GPU) 선택 → 저장
2. **런타임 → 모두 실행**을 누릅니다.
3. **2번 셀**에서 파일 선택 창이 열리면 로컬의 `colab/build/oms_phase5a_bundle.zip`을 업로드합니다.
4. 전체 실행에는 T4 기준 **약 10~20분**이 걸릴 것으로 예상합니다(아직 측정 전).
5. 마지막 셀이 `phase5a_results.zip`을 다운로드합니다. 이 파일을 로컬 프로젝트의 **`colab/results/`** 폴더에 넣고 Claude에게 알려 주세요.

> 무료 Colab은 GPU가 항상 배정되지는 않습니다. 1번 셀에서 GPU가 없다고 나오면 런타임 유형을 다시 확인하거나 잠시 후 다시 시도하세요.
> 실행 도중 연결이 끊기면, 이미 끝난 실험의 결과는 `results/`에 남아 있으므로 실패한 셀부터 다시 실행하면 됩니다.
"""),
    md("## 1. GPU / PyTorch / CUDA 호환성 확인"),
    code("""
!nvidia-smi
import torch, platform
print("python", platform.python_version())
print("torch", torch.__version__, "| built with CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "GPU가 없습니다: 런타임 → 런타임 유형 변경에서 GPU를 선택하세요."
p = torch.cuda.get_device_properties(0)
cc = f"{p.major}.{p.minor}"
print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB, compute capability {cc}")
arch = torch.cuda.get_arch_list()
print("arch list in this torch build:", arch)
assert f"sm_{p.major}{p.minor}" in arch or any(a.startswith("compute_") for a in arch), "이 PyTorch 빌드에 이 GPU 아키텍처 커널이 없습니다"
BF16_HW = torch.cuda.is_bf16_supported(including_emulation=False)
FP16_OK = p.major >= 7
print("bf16 hardware support:", BF16_HW, "| fp16 tensor cores:", FP16_OK)
x = torch.randn(1024, 1024, device="cuda"); torch.cuda.synchronize()
print("CUDA smoke test OK:", float((x @ x).sum()))
major, minor = (int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
assert (major, minor) >= (2, 5), "SAM 2 는 PyTorch 2.5.1 이상이 필요합니다"
"""),
    md("## 2. 번들 업로드 (`colab/build/oms_phase5a_bundle.zip`)"),
    code("""
import os, zipfile, json
from google.colab import files
os.chdir("/content")
uploaded = files.upload()  # ← 파일 선택 창에서 oms_phase5a_bundle.zip 선택
name = next(iter(uploaded))
with zipfile.ZipFile(name) as z:
    z.extractall("/content/oms")
os.chdir("/content/oms")
manifest = json.load(open("manifest.json"))
print(json.dumps(manifest, indent=1))
print("frames:", len(os.listdir("data/frames")))
"""),
    md("## 3. 공식 SAM 2 설치 (로컬과 같은 커밋, PyTorch 는 Colab 기본 버전 유지)"),
    code("""
import subprocess, sys
torch_before = torch.__version__
commit = manifest["sam2Commit"]
!pip -q install "hydra-core>=1.3.2" "iopath>=0.1.10" fastapi psutil
# --no-deps / --no-build-isolation: Colab 의 CUDA PyTorch 를 그대로 사용 (pyproject 의 torch 빌드 의존성 때문에
# 격리 빌드 시 torch 를 따로 내려받는 것을 방지). 선택적 CUDA 확장(_C)은 로컬과 같이 빌드하지 않음.
!SAM2_BUILD_CUDA=0 pip -q install --no-deps --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@{commit}"
torch_after = subprocess.run([sys.executable, "-c", "import torch, sam2; print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
print("sam2 installed from commit", commit, "| torch", torch_after)
assert torch_after == torch_before, f"PyTorch 가 바뀌었습니다: {torch_before} -> {torch_after}"
"""),
    md("## 4. 공식 체크포인트 다운로드 + SHA-256 검증 (로컬과 같은 파일인지 확인)"),
    code("""
import hashlib
os.makedirs("checkpoints", exist_ok=True)
ckpt = "checkpoints/sam2.1_hiera_tiny.pt"
url = manifest["checkpointUrl"]
if not os.path.exists(ckpt):
    !wget -q -O {ckpt} {url}
h = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
print("sha256", h)
assert h == manifest["checkpointSha256"], "체크포인트가 로컬과 다릅니다"
print("checkpoint OK (identical to the local CPU run)")
"""),
    md("""
## 5. GPU fp32 — 단일 프레임 분할 + 10/30/60/300 프레임 추적
기본 정밀도(fp32), 상태 오프로딩 끔. 300프레임은 GPU 메모리가 허용하면 완료되고, 실패하면 오류가 기록된 뒤 다음 실험으로 넘어갑니다.
"""),
    code("""
COMMON = "--frames-dir data/frames --reference reference/cpu --device cuda"
FULL = "--segment --track 10 30 60 300"
!python scripts/gpu_benchmark.py run {COMMON} {FULL} --label gpu_fp32 --out results/gpu_fp32 --precision fp32 --offload off
"""),
    md("""
## 6. Mixed precision
- **fp16 autocast**: Tensor Core가 있는 GPU(compute capability 7.0 이상, T4 포함)에서 실행합니다.
- **bf16 autocast**: bf16을 하드웨어로 지원하는 GPU(Ampere 이상: A100, L4 등)에서만 실행합니다. T4는 지원하지 않으므로 건너뜁니다.
"""),
    code("""
if FP16_OK:
    !python scripts/gpu_benchmark.py run {COMMON} {FULL} --label gpu_fp16 --out results/gpu_fp16 --precision fp16 --offload off
else:
    print("skip fp16: compute capability < 7.0")
"""),
    code("""
if BF16_HW:
    !python scripts/gpu_benchmark.py run {COMMON} {FULL} --label gpu_bf16 --out results/gpu_bf16 --precision bf16 --offload off
else:
    print(f"skip bf16: {torch.cuda.get_device_name(0)} has no bf16 hardware support")
"""),
    md("## 7. CPU 오프로딩 (공식 `offload_state_to_cpu`) — 300프레임 메모리/속도 비교"),
    code("""
!python scripts/gpu_benchmark.py run {COMMON} --track 300 --label gpu_fp32_offload --out results/gpu_fp32_offload --precision fp32 --offload on
"""),
    md("## 8. 요약 (CPU 기준 대비)"),
    code("""
!python scripts/gpu_benchmark.py summarize results/* --reference reference/cpu --out results/summary.md
from IPython.display import Markdown, display
display(Markdown(open("results/summary.md", encoding="utf-8").read()))
"""),
    md("## 9. 결과 다운로드 → 로컬 `colab/results/` 폴더에 넣어 주세요"),
    code("""
!cd /content/oms && rm -f /content/phase5a_results.zip && zip -qr /content/phase5a_results.zip results manifest.json
files.download("/content/phase5a_results.zip")
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
