"""colab/phase5b_gpu_tracking.ipynb 생성 (nbformat v4).

사용: python scripts/make_colab_tracking_notebook.py ../colab/phase5b_gpu_tracking.ipynb
"""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
PINNED_SAM2 = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Object Magic Studio — Phase 5B-1: GPU 영상 추적 (Google Colab)

웹앱에서 내려받은 **GPU 작업 파일**(`oms_gpu_job_….zip`)을 이 노트북에서 실행하면, SAM 2.1 Hiera Tiny로 실제 영상 추적을 하고 **결과 파일**(`oms_gpu_result_….zip`)을 만듭니다. 결과 파일을 웹앱의 **Import GPU result**로 가져오면 효과 미리보기와 MP4 내보내기에 그대로 쓸 수 있습니다.

## 직접 실행할 단계
1. **런타임 → 런타임 유형 변경 → `T4 GPU`** 선택 → 저장
2. 아래 **설정 셀**을 확인합니다. 기본값은 작업 파일의 정밀도(fp16)입니다. fp32 비교 결과도 만들려면 `ALSO_RUN_FP32 = True`로 바꿉니다.
3. **런타임 → 모두 실행** → 2번 셀의 파일 선택 창에서 웹앱이 만든 `oms_gpu_job_….zip`을 선택합니다.
4. 마지막 셀에서 결과 zip이 다운로드되면, 웹앱 오른쪽 **GPU tracking (Colab)** 패널의 **Import GPU result**로 가져옵니다.

> 이 노트북은 웹앱과 직접 연결되지 않습니다. 파일을 한 번 주고받는 일회성 작업이며, Colab을 상시 서버로 쓰지 않습니다.
> 작업 파일에는 영상 프레임(JPEG), 클릭 위치, 추론 코드가 들어 있습니다. 결과 파일에는 프레임별 마스크(PNG)와 검증용 해시가 들어 있습니다.
"""),
    md("## 0. 설정"),
    code("""
PRECISION = "job"      # "job" = 작업 파일에 저장된 정밀도(기본 fp16) / "fp16" / "fp32"
ALSO_RUN_FP32 = False  # True 면 fp32 결과도 만들어 웹앱에서 비교할 수 있음 (시간 약 2배)
"""),
    md("## 1. GPU / PyTorch 확인"),
    code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch
assert torch.cuda.is_available(), "GPU가 없습니다: 런타임 → 런타임 유형 변경에서 GPU를 선택하세요."
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | {p.name} {p.total_memory/2**30:.1f} GB | cc {p.major}.{p.minor}")
FP16_OK = p.major >= 7
major, minor = (int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
assert (major, minor) >= (2, 5), "SAM 2 는 PyTorch 2.5.1 이상이 필요합니다"
"""),
    md("## 2. GPU 작업 파일 업로드"),
    code("""
import os, json, zipfile, shutil
from google.colab import files
os.chdir("/content")
uploaded = files.upload()  # ← 웹앱에서 내려받은 oms_gpu_job_….zip
name = next(iter(uploaded))
shutil.rmtree("/content/oms_job", ignore_errors=True)
with zipfile.ZipFile(name) as z:
    for n in z.namelist():
        assert not n.startswith("/") and ".." not in n.split("/"), f"unsafe path {n}"
    z.extractall("/content/oms_job")
job = json.load(open("/content/oms_job/job.json"))
assert job.get("format") == "oms-gpu-job", "GPU 작업 파일이 아닙니다"
v, t = job["video"], job["tracking"]
print(f"job {job['jobId'][:8]}: frames {t['startFrame']}–{t['startFrame'] + t['numFrames'] - 1} ({t['numFrames']} frames) "
      f"of {v['frameCount']}, {v['width']}x{v['height']} @ {v['fps']}, precision {job['precision']}")
"""),
    md("## 3. 공식 SAM 2 설치 + 체크포인트 검증"),
    code(f"""
import subprocess, sys, hashlib
commit = job["model"].get("sam2Commit") or "{PINNED_SAM2}"
torch_before = torch.__version__
!pip -q install "hydra-core>=1.3.2" "iopath>=0.1.10" fastapi psutil
!SAM2_BUILD_CUDA=0 pip -q install --no-deps --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@{{commit}}"
torch_after = subprocess.run([sys.executable, "-c", "import torch, sam2; print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
assert torch_after == torch_before, f"PyTorch 가 바뀌었습니다: {{torch_before}} -> {{torch_after}}"
os.makedirs("/content/checkpoints", exist_ok=True)
ckpt = "/content/checkpoints/sam2.1_hiera_tiny.pt"
if not os.path.exists(ckpt):
    !wget -q -O {{ckpt}} {CKPT_URL}
h = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
assert h == job["model"]["checkpointSha256"], "체크포인트가 웹앱 서버와 다릅니다"
print("sam2", commit[:8], "| checkpoint OK")
"""),
    md("## 4. GPU 추적 실행"),
    code("""
prec = job["precision"] if PRECISION == "job" else PRECISION
if prec == "fp16" and not FP16_OK:
    print("이 GPU는 fp16 Tensor Core 가 없어 fp32 로 실행합니다"); prec = "fp32"
OUT = f"/content/oms_gpu_result_{job['jobId'][:8]}_{prec}.zip"
!cd /content/oms_job && python scripts/colab_track.py --job-dir /content/oms_job --out {OUT} --precision {prec} --device cuda --checkpoint {ckpt}
RESULTS = [OUT]
"""),
    md("## 5. (선택) fp32 비교 결과"),
    code("""
if ALSO_RUN_FP32 and prec != "fp32":
    OUT32 = f"/content/oms_gpu_result_{job['jobId'][:8]}_fp32.zip"
    !cd /content/oms_job && python scripts/colab_track.py --job-dir /content/oms_job --out {OUT32} --precision fp32 --device cuda --checkpoint {ckpt}
    RESULTS.append(OUT32)
else:
    print("skip fp32 comparison")
"""),
    md("## 6. 결과 다운로드 → 웹앱의 **Import GPU result** 로 가져오기"),
    code("""
for r in RESULTS:
    print(r, f"{os.path.getsize(r)/2**20:.1f} MB")
    files.download(r)
"""),
]

nb = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 0,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
