"""실제 모델로 오프라인 GPU 작업 파이프라인 전체를 검증 (로컬 CPU, 몇 프레임).

작업 파일 생성(API) → 빈 폴더에 압축 해제 → 패키지 안의 scripts/colab_track.py 를 별도 프로세스로 실행
(Colab 과 같은 방식, 패키지에 든 app/ 코드 사용) → 결과 zip 가져오기(API) → 마스크가 정답 원을 따라가는지 확인.
"""

import io
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.masks import rle_decode

pytestmark = pytest.mark.real_model
pytest.importorskip("sam2")


def test_package_runs_standalone_and_result_imports(isolated_sessions, tmp_path, monkeypatch, clip_2s):
    from app.config import settings

    if not Settings().sam2_checkpoint.exists():
        pytest.skip("checkpoint missing")
    monkeypatch.setattr(settings, "gpu_jobs_dir", tmp_path / "gpu_jobs")
    with TestClient(app) as client:
        with open(clip_2s, "rb") as f:
            sid = client.post("/api/sessions", files={"file": ("clip.mp4", f, "video/mp4")}).json()["sessionId"]
        # clip_2s: 프레임 0 의 원 중심 (320, 220) → 정규화 좌표
        info = client.post(
            f"/api/sessions/{sid}/gpu-jobs",
            json={"frameIndex": 0, "points": [{"x": 0.5, "y": 220 / 360, "label": 1}], "maxFrames": 4, "precision": "fp32"},
        ).json()
        pkg = client.get(info["downloadUrl"]).content
        job_dir = tmp_path / "colab_job"
        zipfile.ZipFile(io.BytesIO(pkg)).extractall(job_dir)

        out = tmp_path / "result.zip"
        proc = subprocess.run(
            [sys.executable, "-W", "ignore", str(job_dir / "scripts" / "colab_track.py"), "--job-dir", str(job_dir), "--out", str(out),
             "--device", "cpu", "--checkpoint", str(Settings().sam2_checkpoint)],
            capture_output=True, text=True, timeout=900, cwd=job_dir,
        )  # fmt: skip
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        assert proc.returncode == 0

        r = client.post(f"/api/sessions/{sid}/gpu-results", files={"file": ("result.zip", out.read_bytes(), "application/zip")})
        assert r.status_code == 200, r.text
        t = r.json()
        assert t["stats"]["device"] == "cpu" and t["stats"]["precision"] == "fp32" and t["stats"]["trackedFrames"] == [0, 1, 2, 3]

        m = client.get(f"/api/sessions/{sid}/tracks/{t['id']}/masks", params={"width": 640, "height": 360}).json()
        for fr in m["frames"]:
            mask = rle_decode(fr["counts"], 640, 360)
            ys, xs = np.nonzero(mask)
            t_s = fr["index"] / 30
            cx = int(120 + 400 * (0.5 + 0.5 * np.sin(t_s * 1.2)))
            cy = 180 + int(40 * np.cos(t_s * 2))
            assert abs(xs.mean() - cx) < 2 and abs(ys.mean() - cy) < 2, (fr["index"], xs.mean(), ys.mean(), cx, cy)
