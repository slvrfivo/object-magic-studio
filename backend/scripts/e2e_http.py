"""HTTP 종단 간 테스트: 브라우저와 같은 경로(Vite 프록시 /api)로 업로드 → 분할 → 추적 → 취소를 검증합니다.

사용: python scripts/e2e_http.py <video.mp4> [--base http://localhost:5173/api] [--frames 10]
make_test_clip.py 로 만든 영상이면 프레임별 정답 원 위치와 비교합니다.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.masks import rle_decode  # noqa: E402


def gt_center(frame: int, fps: float, sx: float, sy: float) -> tuple[float, float]:
    t = frame / fps
    return (120 + 400 * (0.5 + 0.5 * math.sin(t * 1.2))) * sx, (180 + int(40 * math.cos(t * 2))) * sy


def wait_job(c: httpx.Client, job_id: str, label: str, cancel_after: int | None = None) -> dict:
    last = None
    while True:
        j = c.get(f"/jobs/{job_id}").json()
        if j["message"] != last:
            print(f"  [{label}] {j['status']:9s} {j['progress'] * 100:5.1f}%  {j['message']}")
            last = j["message"]
        if cancel_after is not None and j.get("stats", {}).get("framesDone", 0) >= cancel_after:
            print(f"  [{label}] -> cancel requested")
            c.post(f"/jobs/{job_id}/cancel")
            cancel_after = None
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--base", default="http://localhost:5173/api")
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()
    c = httpx.Client(base_url=args.base, timeout=180)

    t0 = time.time()
    while True:  # 서버 시작 + 모델 로드 대기
        r = c.get("/health")
        if r.status_code == 200:
            h = r.json()
            assert h["model"]["state"] != "error", h["model"]["error"]
            if h["model"]["state"] == "ready":
                break
        assert time.time() - t0 < 300, "backend did not become ready"
        time.sleep(2)
    print(f"model ready ({time.time() - t0:.0f}s), device={h['model']['runtime']['device']}, maxFrames={h['tracking']['maxFrames']}")

    with open(args.video, "rb") as f:
        s = c.post("/sessions", files={"file": (Path(args.video).name, f, "video/mp4")}).json()
    sid, meta = s["sessionId"], s["meta"]
    print("session", meta)
    pw, ph = 320, 180
    sx, sy = pw / meta["width"], ph / meta["height"]

    cx, cy = gt_center(0, meta["fps"], 1, 1)
    points = [{"x": cx / meta["width"], "y": cy / meta["height"], "label": 1}]
    seg = c.post(f"/sessions/{sid}/segment", json={"frameIndex": 0, "points": points, "maskWidth": pw, "maskHeight": ph}).json()
    print(f"segment: {seg['metrics']['inferenceMs']} ms, coverage {seg['metrics']['coverage']}")

    # 1) 정상 추적
    job = c.post(f"/sessions/{sid}/propagate", json={"frameIndex": 0, "points": points, "maxFrames": args.frames}).json()
    done = wait_job(c, job["id"], "track")
    assert done["status"] == "succeeded", done
    st = done["stats"]
    print(f"stats: avg {st['avgFrameMs']} ms/frame, median {st['medianFrameMs']}, max {st['maxFrameMs']}, peak RSS {st['peakRssMB']} MB, lost {st['lostFrames']}")
    r = c.get(f"/sessions/{sid}/tracks/{job['id']}/masks", params={"width": pw, "height": ph}).json()
    errs = []
    for fr in r["frames"]:
        m = rle_decode(fr["counts"], pw, ph)
        ys, xs = np.nonzero(m)
        gx, gy = gt_center(fr["index"], meta["fps"], sx, sy)
        errs.append(math.hypot(xs.mean() - gx, ys.mean() - gy))
    first = rle_decode(r["frames"][0]["counts"], pw, ph)
    last = rle_decode(r["frames"][-1]["counts"], pw, ph)
    moved = math.hypot(*(np.argwhere(last).mean(0) - np.argwhere(first).mean(0)))
    print(f"masks: {len(r['frames'])} frames, mask centroid moved {moved:.1f}px (preview), max center error vs GT {max(errs):.2f}px")
    assert len(r["frames"]) == args.frames and max(errs) < 2.0 and moved > 3

    # 2) 취소 → 부분 결과 유지
    job2 = c.post(f"/sessions/{sid}/propagate", json={"frameIndex": 20, "points": [{"x": gt_center(20, meta['fps'], 1, 1)[0] / meta['width'], "y": gt_center(20, meta['fps'], 1, 1)[1] / meta['height'], "label": 1}], "maxFrames": 30}).json()
    done2 = wait_job(c, job2["id"], "cancel-test", cancel_after=3)
    assert done2["status"] == "cancelled", done2
    r2 = c.get(f"/sessions/{sid}/tracks/{job2['id']}/masks", params={"width": pw, "height": ph}).json()
    print(f"cancelled after {done2['stats']['framesDone']} frames; partial masks returned: {[f['index'] for f in r2['frames']]}")
    assert len(r2["frames"]) == done2["stats"]["framesDone"] >= 3

    # 3) 한도 초과 요청은 즉시 거부
    over = c.post(f"/sessions/{sid}/propagate", json={"frameIndex": 0, "points": points, "maxFrames": h["tracking"]["maxFrames"] + 1})
    print("over-limit request:", over.status_code, over.json()["error"]["message"])
    assert over.status_code == 422

    c.delete(f"/sessions/{sid}")
    print("E2E OK")


if __name__ == "__main__":
    main()
