"""colab/realworld_gpu_batch.ipynb 생성 — 실제 영상 GPU 추적 일괄 실행 (oms-gpu-batch 번들 여러 개).

사용: python scripts/make_colab_batch_notebook.py ../colab/realworld_gpu_batch.ipynb
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
# Object Magic Studio — 실제 영상 GPU 추적 일괄 실행 (Google Colab)

웹앱 서버가 만든 **실제 영상 GPU 작업 번들**(`oms_batch_<영상>.zip`, 7개, 합계 약 444MB)을 한 번에 실행합니다.
작업 17개(사람·동물·사물·SA-V 정답 영상)를 fp16으로 추적하고, 정답이 있는 SA-V 작업 5개는 fp32도 실행합니다.
추론 코드는 웹앱 서버와 같은 `app/` + `scripts/colab_track.py`이며, 결과 형식도 웹앱의 **Import GPU result**와 같습니다.

## 직접 실행할 단계
1. **번들 준비 (권장: Google Drive)**: 로컬의 `realworld/work/gpu_bundles/*.zip` 7개를 Google Drive의 **`oms_realworld`** 폴더(내 드라이브 바로 아래)에 올립니다.
   - Drive를 쓰지 않으려면 설정 셀에서 `SOURCE = "upload"`로 바꾸고, 2번 셀의 파일 선택 창에서 7개를 한 번에 고릅니다(업로드에 시간이 오래 걸릴 수 있음).
2. **런타임 → 런타임 유형 변경 → `T4 GPU`** 선택 → 저장
3. **런타임 → 모두 실행** (Drive를 쓰면 접근 허용 창이 뜹니다)
4. 마지막 셀이 `oms_realworld_results.zip`을 다운로드합니다 (Drive를 쓰면 같은 폴더에도 저장). 로컬의 **`colab/results/realworld/`** 폴더에 **압축을 풀어** 결과 zip들을 넣어 주세요.

> 예상 시간(T4): 추적 약 25~30분 + 설치 수 분 (Phase 5A 실측 기반 추정치이며 실제와 다를 수 있음).
> 중간에 끊기면 다시 실행해도 됩니다. 이미 만든 결과 zip은 건너뜁니다.
"""),
    md("## 0. 설정"),
    code("""
SOURCE = "drive"   # "drive" (권장) 또는 "upload"
DRIVE_DIR = "/content/drive/MyDrive/oms_realworld"
RUN_FP32 = True    # False 면 번들에 fp32 가 지정된 작업(SA-V)도 fp16 만 실행
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
    md("## 2. 번들 가져오기 + 압축 해제"),
    code("""
import os, glob, json, zipfile, shutil
os.makedirs("/content/bundles", exist_ok=True)
if SOURCE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    zips = sorted(glob.glob(f"{DRIVE_DIR}/oms_batch_*.zip"))
else:
    from google.colab import files
    os.chdir("/content/bundles")
    uploaded = files.upload()  # ← oms_batch_*.zip 7개를 한 번에 선택
    zips = sorted(os.path.abspath(n) for n in uploaded)
assert zips, f"번들이 없습니다 ({SOURCE})"
BUNDLES = []
for zp in zips:
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            assert not n.startswith("/") and ".." not in n.split("/"), f"unsafe path {n}"
        b = json.loads(z.read("batch.json"))
        assert b.get("format") == "oms-gpu-batch", f"{zp}: not an OMS batch bundle"
        dest = f"/content/bundles/{b['video']}"
        if not os.path.exists(f"{dest}/batch.json"):
            z.extractall(dest)
    BUNDLES.append(dest)
    print(f"{os.path.basename(zp)}: {b['video']} — {len(b['jobs'])} jobs")
"""),
    md("## 3. 공식 SAM 2 설치 + 체크포인트 검증"),
    code(f"""
import subprocess, sys, hashlib
first_job = json.load(open(glob.glob(f"{{BUNDLES[0]}}/jobs/*/job.json")[0]))
commit = first_job["model"].get("sam2Commit") or "{PINNED_SAM2}"
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
assert h == first_job["model"]["checkpointSha256"], "체크포인트가 웹앱 서버와 다릅니다"
print("sam2", commit[:8], "| checkpoint OK")
"""),
    md("## 4. 일괄 추적 실행 (작업마다 별도 프로세스, 실패해도 다음 작업 계속)"),
    code("""
import time
RESULTS = "/content/results"
os.makedirs(RESULTS, exist_ok=True)
log = []
for bdir in BUNDLES:
    b = json.load(open(f"{bdir}/batch.json"))
    for j in b["jobs"]:
        precs = j["precisions"] if RUN_FP32 else ["fp16"]
        for prec in precs:
            if prec == "fp16" and not FP16_OK:
                prec = "fp32"
            name = j["name"].replace("/", "__")
            out = f"{RESULTS}/{name}__{prec}.zip"
            if os.path.exists(out):
                print("skip (done)", os.path.basename(out)); continue
            # 작업 폴더: job.json + 이 작업 구간의 프레임(공유 프레임에 링크, 0번부터 다시 번호)
            jd = f"/content/run/{j['jobId']}"
            shutil.rmtree(jd, ignore_errors=True)
            os.makedirs(f"{jd}/frames")
            shutil.copy(f"{bdir}/jobs/{j['jobId']}/job.json", f"{jd}/job.json")
            for i in range(j["numFrames"]):
                os.symlink(f"{bdir}/video_frames/{j['startFrame'] + i:05d}.jpg", f"{jd}/frames/{i:05d}.jpg")
            t = time.time()
            r = subprocess.run([sys.executable, f"{bdir}/scripts/colab_track.py", "--job-dir", jd, "--out", out,
                                "--precision", prec, "--device", "cuda", "--checkpoint", ckpt], capture_output=True, text=True)
            ok = r.returncode == 0 and os.path.exists(out)
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            log.append({"name": j["name"], "precision": prec, "ok": ok, "wallS": round(time.time() - t, 1), "tail": tail})
            print(("OK  " if ok else "FAIL"), f"{j['name']:24s} {prec}  {time.time() - t:6.1f}s  |", tail[-1] if tail else "")
            shutil.rmtree(jd, ignore_errors=True)
json.dump(log, open(f"{RESULTS}/run_log.json", "w"), indent=1)
"""),
    md("## 5. 요약"),
    code("""
rows = []
for zp in sorted(glob.glob(f"{RESULTS}/*.zip")):
    r = json.loads(zipfile.ZipFile(zp).read("result.json"))
    s = r["stats"]
    rows.append((os.path.basename(zp), len(r["tracking"]["trackedFrames"]), s.get("avgFrameMs"), round((s.get("totalMs") or 0) / 1000, 1), s.get("peakAllocatedMB"), len(s.get("lostFrames") or [])))
print(f"{'result':48s} {'frames':>6} {'ms/frame':>9} {'total s':>8} {'GPU MB':>8} {'lost':>5}")
for row in rows:
    print(f"{row[0]:48s} {row[1]:6d} {row[2]:9.1f} {row[3]:8.1f} {row[4]:8.1f} {row[5]:5d}")
fails = [l for l in log if not l["ok"]] if "log" in dir() else []
print(f"\\n{len(rows)} results, {len(fails)} failed")
"""),
    md("## 6. 결과 다운로드 → 로컬 `colab/results/realworld/` 에 압축을 풀어 넣기"),
    code("""
!cd /content && rm -f oms_realworld_results.zip && zip -qj oms_realworld_results.zip results/*.zip results/run_log.json
print(f"{os.path.getsize('/content/oms_realworld_results.zip')/2**20:.1f} MB")
if SOURCE == "drive":
    shutil.copy("/content/oms_realworld_results.zip", f"{DRIVE_DIR}/oms_realworld_results.zip")
    print("also saved to", DRIVE_DIR)
from google.colab import files
files.download("/content/oms_realworld_results.zip")
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
