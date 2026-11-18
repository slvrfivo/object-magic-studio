"""실제 영상 검증 도구 (사람·동물·사물 샘플). 실행 중인 백엔드 API(:8000)를 그대로 사용합니다.

하위 명령:
  ingest      realworld/videos/*.mp4 업로드 → 허용/거부 사유, 메타데이터, 처리 시간
  segment     단일 프레임 분할 (CPU 실측) → 마스크, 시간, SA-V 정답 대비 IoU, 오버레이 이미지
  make-jobs   GPU 작업 파일 생성(API) → 영상별 Colab 일괄 번들(프레임은 영상당 한 번만 포함)
  evaluate    Colab 결과 zip 가져오기(API) → 정확도(SA-V)·안정성 지표, 미리보기 시트, 효과별 MP4 내보내기 검증

데이터: realworld/videos (git 제외), 작업 결과: realworld/work, 보고서: docs/realworld
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import httpx
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
PROJECT = BACKEND.parent
VIDEOS = PROJECT / "realworld" / "videos"
WORK = PROJECT / "realworld" / "work"
DOCS = PROJECT / "docs" / "realworld"
sys.path.insert(0, str(BACKEND))
from app.masks import rle_decode  # noqa: E402

API = "http://localhost:8000/api"

# ---------------------------------------------------------------- 시험 대상 (원본 픽셀 좌표, 프레임 0 기준)
# label 1 = 포함, 0 = 제외
PROMPTS: dict[str, dict] = {
    "bedroom/girl_1click": {"video": "bedroom", "frame": 0, "points": [(395, 235, 1)], "note": "사람 · 한 번 클릭(원피스)"},
    "bedroom/girl": {"video": "bedroom", "frame": 0, "points": [(395, 235, 1), (400, 70, 1), (385, 330, 1)], "note": "사람 · 머리/몸통/다리 3클릭"},
    "bedroom/boy": {"video": "bedroom", "frame": 0, "points": [(230, 250, 1), (255, 165, 1), (215, 370, 1)], "note": "사람 · 3클릭, 여자아이에게 가려짐"},
    "01_dog/dog": {"video": "01_dog", "frame": 0, "points": [(620, 430, 1)], "note": "동물 · 한 번 클릭"},
    "01_dog/woman": {"video": "01_dog", "frame": 0, "points": [(330, 320, 1), (310, 440, 1), (420, 250, 1)], "note": "사람 · 3클릭"},
    "02_cups/middle_cup": {"video": "02_cups", "frame": 0, "points": [(655, 450, 1)], "note": "똑같은 컵 섞기 (정체성)"},
    "02_cups/ball": {"video": "02_cups", "frame": 0, "points": [(400, 463, 1)], "note": "작은 공, 컵 아래로 숨음"},
    "03_blocks/right_hand": {"video": "03_blocks", "frame": 0, "points": [(850, 70, 1), (950, 60, 1)], "note": "손 · 변형, 화면 이탈"},
    "03_blocks/block_A": {"video": "03_blocks", "frame": 0, "points": [(600, 350, 1)], "note": "비슷한 블록 사이의 작은 물체"},
    "04_coffee/kettle": {"video": "04_coffee", "frame": 0, "points": [(570, 470, 1)], "note": "반사 금속, 들어 올려 회전"},
    "04_coffee/mug": {"video": "04_coffee", "frame": 0, "points": [(880, 520, 1)], "note": "머그"},
    "05_juggle/ball": {"video": "05_default_juggle_10s", "frame": 0, "points": [(650, 320, 1)], "note": "작고 빠른 공, 모션 블러"},
    "05_juggle/person": {"video": "05_default_juggle_10s", "frame": 0, "points": [(700, 250, 1), (710, 480, 1), (705, 80, 1)], "note": "사람 · 3클릭"},
}
SAV_VIDEO = "sav_000001_10s"
SAV_GT = VIDEOS / "sav_000001_manual.json"
SAV_STEP = 4  # SA-V 는 4프레임마다(6fps) 정답이 있음


# ---------------------------------------------------------------- SA-V 정답 (COCO 압축 RLE)
def coco_rle_decode(rle: dict) -> np.ndarray:
    """pycocotools 없이 COCO 압축 RLE 문자열을 디코딩 (열 우선)."""
    h, w = rle["size"]
    s = rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]
    counts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = s[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in counts:
        flat[pos : pos + c] = val
        pos += c
        val ^= 1
    return flat.reshape(w, h).T


def sav_gt() -> dict:
    g = json.loads(SAV_GT.read_text())
    return {"masklet": g["masklet"], "ids": g["masklet_id"], "first": g["masklet_first_appeared_frame"], "size": g["masklet_size_bucket"]}


def sav_gt_mask(gt: dict, frame: int, obj: int) -> np.ndarray | None:
    if frame % SAV_STEP:
        return None
    rle = gt["masklet"][frame // SAV_STEP][obj]
    return coco_rle_decode(rle) if rle else None


def inner_point(mask: np.ndarray) -> tuple[int, int]:
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(x), int(y)


def sav_prompts(gt: dict) -> dict:
    out = {}
    for obj in range(len(gt["ids"])):
        # masklet_first_appeared_frame 은 '정답이 있는 프레임' 기준 번호 (×4 = 영상 프레임)
        first = int(gt["first"][obj]) * SAV_STEP
        frame = next((f for f in range(first, 240, SAV_STEP) if (m := sav_gt_mask(gt, f, obj)) is not None and m.sum() > 0), None)
        if frame is None:
            continue  # 10초 구간 안에 나타나지 않는 객체 (예: obj3 는 452번 프레임부터)
        x, y = inner_point(sav_gt_mask(gt, frame, obj))
        out[f"sav/obj{obj}"] = {"video": SAV_VIDEO, "frame": frame, "points": [(x, y, 1)], "note": f"SA-V 정답 객체 {obj} ({gt['size'][obj]})", "savObj": obj}
    if "sav/obj4" in out:
        # 한 번 클릭은 아이 전신을 고름(정답은 조끼) → 조끼 포함 + 머리/다리 제외 보정 클릭
        # 제외 점은 한 번 클릭 결과 중 정답 밖 영역(머리, 다리)의 안쪽 점으로 계산 (실험 기록: docs/realworld/README.md)
        out["sav/obj4_refined"] = {"video": SAV_VIDEO, "frame": 0, "points": [(155, 448, 1), (190, 291, 0), (202, 651, 0)],
                                   "note": "SA-V 객체 4 · 보정 클릭 (포함 1 + 제외 2)", "savObj": 4}  # fmt: skip
    return out


# ---------------------------------------------------------------- 공용
def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def boundary_f(pred: np.ndarray, gt: np.ndarray, tol_ratio: float = 0.008) -> float:
    """DAVIS 방식 경계 F (허용 거리 = 대각선 × 0.008)."""
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if gt.sum() == 0 or pred.sum() == 0:
        return 0.0
    tol = max(1, int(round(tol_ratio * math.hypot(*gt.shape))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))

    def edge(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0

    pe, ge = edge(pred), edge(gt)
    prec = (pe & (cv2.dilate(ge.astype(np.uint8), k) > 0)).sum() / max(1, pe.sum())
    rec = (ge & (cv2.dilate(pe.astype(np.uint8), k) > 0)).sum() / max(1, ge.sum())
    return float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0


def overlay(img: np.ndarray, mask: np.ndarray, pts=(), color=(246, 92, 139)) -> np.ndarray:
    out = img.copy()
    m = mask > 0
    out[m] = (0.5 * np.array(color) + 0.5 * out[m]).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (238, 211, 34), 2)
    for x, y, lab in pts:
        cv2.circle(out, (int(x), int(y)), 7, (80, 200, 60) if lab else (60, 60, 230), -1)
        cv2.circle(out, (int(x), int(y)), 7, (255, 255, 255), 2)
    return out


def read_frame(video: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(VIDEOS / f"{video}.mp4"))
    for _ in range(idx + 1):
        ok, f = cap.read()
    cap.release()
    return f


def sessions() -> dict:
    return json.loads((WORK / "sessions.json").read_text(encoding="utf-8"))


def client() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=600)


# ---------------------------------------------------------------- ingest
def cmd_ingest(_a) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    c = client()
    rows, sess = [], {}
    for p in sorted(VIDEOS.glob("*.mp4")):
        t = time.perf_counter()
        with p.open("rb") as f:
            r = c.post("/sessions", files={"file": (p.name, f, "video/mp4")})
        dt = round(time.perf_counter() - t, 2)
        row = {"file": p.name, "sizeKB": p.stat().st_size // 1024, "status": r.status_code, "seconds": dt}
        if r.status_code == 200:
            body = r.json()
            row["meta"] = body["meta"]
            sess[p.stem] = {"sessionId": body["sessionId"], "meta": body["meta"]}
        else:
            row["error"] = r.json().get("error", r.text)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (WORK / "sessions.json").write_text(json.dumps(sess, indent=1), encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "ingest.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- segment
def all_prompts() -> dict:
    p = dict(PROMPTS)
    if SAV_GT.exists():
        p.update(sav_prompts(sav_gt()))
    return p


def cmd_segment(a) -> int:
    c = client()
    ss = sessions()
    gt = sav_gt() if SAV_GT.exists() else None
    out_dir = WORK / "segment"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for name, pr in all_prompts().items():
        if a.only and not any(name.startswith(o) for o in a.only):
            continue
        s = ss[pr["video"]]
        w, h = s["meta"]["width"], s["meta"]["height"]
        pts = [{"x": x / w, "y": y / h, "label": lab} for x, y, lab in pr["points"]]
        for attempt in range(10):  # 다른 클라이언트가 모델을 쓰는 중이면(409 busy) 잠시 후 재시도
            r = c.post(f"/sessions/{s['sessionId']}/segment", json={"frameIndex": pr["frame"], "points": pts, "maskWidth": w, "maskHeight": h})
            if r.status_code != 409:
                break
            print(f"  {name}: model busy, retry {attempt + 1}")
            time.sleep(3)
        if r.status_code != 200:
            rows.append({"name": name, "error": r.text})
            print(name, "ERROR", r.text)
            continue
        body = r.json()
        mask = rle_decode(body["mask"]["counts"], w, h)
        m = body["metrics"]
        row = {"name": name, "note": pr["note"], "frame": pr["frame"], "points": pr["points"], "coverage": m["coverage"],
               "score": m["score"], "encoderMs": m["setImageMs"], "decoderMs": m["predictMs"], "cached": m["embeddingCached"]}
        if gt is not None and "savObj" in pr:
            g = sav_gt_mask(gt, pr["frame"], pr["savObj"])
            row["iouVsGT"] = round(iou(mask, g), 4)
            row["boundaryF"] = round(boundary_f(mask, g), 4)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "points"}, ensure_ascii=False))
        safe = name.replace("/", "__")
        cv2.imwrite(str(out_dir / f"{safe}.png"), mask * 255)
        img = overlay(read_frame(pr["video"], pr["frame"]), mask, pr["points"])
        scale = 300 / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale), 300))
        label = f"{name}  IoU {row['iouVsGT']:.3f}" if "iouVsGT" in row else name
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(img)
    (DOCS / "segment.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    # 시트: 가로 3장씩
    width = 560
    norm = []
    for t in tiles:
        t = t[:, :width] if t.shape[1] >= width else cv2.copyMakeBorder(t, 0, 0, 0, width - t.shape[1], cv2.BORDER_CONSTANT)
        norm.append(t)
    while len(norm) % 3:
        norm.append(np.zeros_like(norm[0]))
    sheet = np.vstack([np.hstack(norm[i : i + 3]) for i in range(0, len(norm), 3)])
    cv2.imwrite(str(DOCS / "segment_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return 0


# ---------------------------------------------------------------- make-jobs
TRACK_SET = [
    "bedroom/girl", "bedroom/boy", "01_dog/dog", "01_dog/woman", "02_cups/middle_cup", "02_cups/ball",
    "03_blocks/right_hand", "03_blocks/block_A", "04_coffee/kettle", "04_coffee/mug", "05_juggle/ball", "05_juggle/person",
]  # fmt: skip


def cmd_make_jobs(a) -> int:
    c = client()
    ss = sessions()
    prompts = all_prompts()
    names = TRACK_SET + sorted(k for k in prompts if k.startswith("sav/"))
    out = WORK / "gpu_bundles"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    by_video: dict[str, list] = {}
    for name in names:
        pr = prompts[name]
        s = ss[pr["video"]]
        w, h = s["meta"]["width"], s["meta"]["height"]
        pts = [{"x": x / w, "y": y / h, "label": lab} for x, y, lab in pr["points"]]
        r = c.post(f"/sessions/{s['sessionId']}/gpu-jobs", json={"frameIndex": pr["frame"], "points": pts, "maxFrames": 300, "precision": "fp16"})
        r.raise_for_status()
        info = r.json()
        # 정답이 있는 SA-V 는 fp32 도 함께 실행해 정밀도 영향을 실제 영상에서 확인
        precisions = ["fp16", "fp32"] if name.startswith("sav/") or a.fp32_all else ["fp16"]
        by_video.setdefault(pr["video"], []).append({"name": name, "jobId": info["jobId"], "numFrames": info["numFrames"], "startFrame": info["startFrame"], "precisions": precisions})
        print(f"{name:24s} job {info['jobId'][:8]} frames {info['startFrame']}+{info['numFrames']} {precisions}")

    jobs_root = BACKEND / "data" / "gpu_jobs"
    code = [(p, p.relative_to(BACKEND).as_posix()) for p in sorted((BACKEND / "app").rglob("*.py"))] + [(BACKEND / "scripts" / "colab_track.py", "scripts/colab_track.py")]
    plan = {}
    for video, jobs in by_video.items():
        sid = ss[video]["sessionId"]
        frames_dir = BACKEND / "data" / "sessions" / sid / "frames"
        start = min(j["startFrame"] for j in jobs)
        end = max(j["startFrame"] + j["numFrames"] for j in jobs)
        zpath = out / f"oms_batch_{video}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            z.writestr("batch.json", json.dumps({"format": "oms-gpu-batch", "version": 1, "video": video, "jobs": jobs}, indent=1))
            for src, arc in code:
                z.write(src, arc)
            for i in range(start, end):  # 영상 프레임은 한 번만 (작업들이 공유)
                z.write(frames_dir / f"{i:05d}.jpg", f"video_frames/{i:05d}.jpg")
            for j in jobs:
                z.write(jobs_root / j["jobId"] / "job.json", f"jobs/{j['jobId']}/job.json")
        plan[video] = {"bundle": zpath.name, "MB": round(zpath.stat().st_size / 2**20, 1), "jobs": jobs}
        print(f"bundle {zpath.name}: {plan[video]['MB']} MB, {len(jobs)} jobs")
    (WORK / "gpu_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- evaluate
def cmd_evaluate(a) -> int:
    c = client()
    ss = sessions()
    prompts = all_prompts()
    plan = json.loads((WORK / "gpu_plan.json").read_text(encoding="utf-8"))
    job_name = {j["jobId"]: (video, j["name"]) for video, p in plan.items() for j in p["jobs"]}
    gt = sav_gt() if SAV_GT.exists() else None
    res_dir = Path(a.results)
    out_dir = DOCS / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    imported: dict[tuple, str] = {}
    for zp in sorted(res_dir.glob("*.zip")):
        r = json.loads(zipfile.ZipFile(zp).read("result.json"))
        video, name = job_name.get(r["jobId"], (None, None))
        if video is None:
            print("skip (unknown job)", zp.name)
            continue
        sid = ss[video]["sessionId"]
        t = time.perf_counter()
        resp = c.post(f"/sessions/{sid}/gpu-results", files={"file": (zp.name, zp.read_bytes(), "application/zip")})
        imp_s = round(time.perf_counter() - t, 2)
        if resp.status_code != 200:
            rows.append({"name": name, "precision": r["precision"], "importError": resp.json()})
            print(name, r["precision"], "IMPORT FAILED", resp.text)
            continue
        job = resp.json()
        imported[(name, r["precision"])] = job["id"]
        st = job["stats"]
        z = zipfile.ZipFile(zp)
        frames = st["trackedFrames"]
        masks = {f: cv2.imdecode(np.frombuffer(z.read(f"masks/{f:05d}.png"), np.uint8), 0) > 127 for f in frames}
        areas = np.array([m.mean() for m in masks.values()])
        consec = [iou(masks[f], masks[f + 1]) for f in frames[:-1]]
        rs = r["stats"]
        row = {
            "name": name, "precision": r["precision"], "frames": len(frames), "range": [frames[0], frames[-1]],
            "gpuAvgMs": rs.get("avgFrameMs"), "gpuTotalS": round((rs.get("totalMs") or 0) / 1000, 1), "gpuPeakMB": rs.get("peakAllocatedMB"),
            "importS": imp_s, "lostFrames": len(st["lostFrames"]), "unstableFrames": len(st["unstableFrames"]),
            "areaMean": round(float(areas.mean()), 4), "areaCV": round(float(areas.std() / max(areas.mean(), 1e-9)), 3),
            "consecutiveIoUMin": round(min(consec), 3), "consecutiveIoUMean": round(float(np.mean(consec)), 4),
        }
        pr = prompts[name]
        if gt is not None and "savObj" in pr:
            js, fs, vis_ok, n_vis = [], [], 0, 0
            for f in frames:
                g = sav_gt_mask(gt, f, pr["savObj"])
                if g is None:
                    continue
                js.append(iou(masks[f], g))
                fs.append(boundary_f(masks[f], g))
                n_vis += 1
                vis_ok += int((g.sum() > 0) == (masks[f].sum() > 0))
            row.update(J=round(float(np.mean(js)), 4), Jmin=round(min(js), 4), F=round(float(np.mean(fs)), 4),
                       JF=round(float((np.mean(js) + np.mean(fs)) / 2), 4), visibilityAgreement=f"{vis_ok}/{n_vis}", gtFrames=n_vis)  # fmt: skip
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        # 미리보기 시트: 12장 균등 샘플
        idx = np.linspace(0, len(frames) - 1, 12).astype(int)
        cap = cv2.VideoCapture(str(VIDEOS / f"{pr['video']}.mp4"))
        tiles, k = [], 0
        want = {frames[i] for i in idx}
        while k <= frames[-1]:
            ok, fr = cap.read()
            if not ok:
                break
            if k in want:
                o = overlay(fr, masks[k])
                s_ = 180 / o.shape[0]
                o = cv2.resize(o, (int(o.shape[1] * s_), 180))
                cv2.putText(o, f"#{k}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                tiles.append(o)
            k += 1
        cap.release()
        sheet = np.vstack([np.hstack(tiles[i : i + 6]) for i in range(0, 12, 6)])
        cv2.imwrite(str(out_dir / f"{name.replace('/', '__')}_{r['precision']}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    (DOCS / "tracking.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    # fp16 vs fp32 비교 (같은 작업)
    cmp = []
    for (name, prec), jid in imported.items():
        if prec == "fp16" and (name, "fp32") in imported:
            video = prompts[name]["video"]
            r = c.get(f"/sessions/{ss[video]['sessionId']}/tracks/compare", params={"a": jid, "b": imported[(name, "fp32")]}).json()
            cmp.append({"name": name, **r})
            print("compare", name, r)
    (DOCS / "fp16_vs_fp32.json").write_text(json.dumps(cmp, indent=1), encoding="utf-8")

    # 효과별 MP4 내보내기 (fp16 결과, 대표 대상 몇 개)
    exports = []
    fx_sets = {
        "neon": {"neon": {"enabled": True, "intensity": 0.75, "color": "#22d3ee", "thickness": 3}, "colorPop": {"enabled": False, "intensity": 1.0}, "blur": {"enabled": False, "intensity": 0.5}},
        "colorPop": {"neon": {"enabled": False, "intensity": 0.75, "color": "#22d3ee", "thickness": 3}, "colorPop": {"enabled": True, "intensity": 1.0}, "blur": {"enabled": False, "intensity": 0.5}},
        "blur": {"neon": {"enabled": False, "intensity": 0.75, "color": "#22d3ee", "thickness": 3}, "colorPop": {"enabled": False, "intensity": 1.0}, "blur": {"enabled": True, "intensity": 0.5}},
    }
    for name in a.export:
        jid = imported.get((name, "fp16"))
        if not jid:
            continue
        video = prompts[name]["video"]
        sid = ss[video]["sessionId"]
        for fx_name, fx in fx_sets.items():
            t = time.perf_counter()
            e = c.post(f"/sessions/{sid}/export", json={"trackJobId": jid, "effects": fx}).json()
            while (j := c.get(f"/jobs/{e['id']}").json())["status"] not in ("succeeded", "failed", "cancelled"):
                time.sleep(1)
            row = {"name": name, "effect": fx_name, "status": j["status"], "wallS": round(time.perf_counter() - t, 1)}
            if j["status"] == "succeeded":
                st = j["stats"]
                row.update({k: st.get(k) for k in ("frameCount", "outputFps", "fpsRational", "duration", "outputWidth", "outputHeight", "hasAudio", "audio", "avgRenderMs", "fileSizeBytes")})
                # resultUrl 은 "/api/..." 절대 경로 → base_url(/api)과 합치지 않도록 서버 주소에 직접 붙임
                dl = c.get(API.rsplit("/api", 1)[0] + j["resultUrl"])
                dl.raise_for_status()
                mp4 = dl.content
                assert mp4[4:8] == b"ftyp", f"not an MP4 ({len(mp4)} bytes)"
                dst = DOCS / "exports" / f"{name.replace('/', '__')}_{fx_name}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(mp4)
            else:
                row["error"] = j.get("error")
            exports.append(row)
            print(json.dumps(row))
    (DOCS / "exports.json").write_text(json.dumps(exports, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- 마스크 후보 선택 데모 (SA-V 객체 4)
def cmd_candidate_demo(a) -> int:
    """한 번 클릭 → 후보 3개 → 조끼 후보 선택 → 추적 → 정답 대비 J&F (서버 API 경유, backend 지정).

    사람이 썸네일을 보고 조끼를 고르는 과정을 대신해, 정답과 가장 많이 겹치는 후보를 고릅니다.
    원격 GPU(--backend remote)는 요금이 발생합니다 (240프레임 1회 약 $0.04).
    """
    gt = sav_gt()
    pr = sav_prompts(gt)["sav/obj4"]
    frame, (x, y, _) = pr["frame"], pr["points"][0]
    c = client()
    with (VIDEOS / f"{SAV_VIDEO}.mp4").open("rb") as f:
        up = c.post("/sessions", files={"file": (f"{SAV_VIDEO}.mp4", f, "video/mp4")}).json()
    sid, meta = up["sessionId"], up["meta"]
    w, h = meta["width"], meta["height"]
    pts = [{"x": x / w, "y": y / h, "label": 1}]
    t = time.perf_counter()
    seg = c.post(f"/sessions/{sid}/segment", json={"frameIndex": frame, "points": pts, "maskWidth": w, "maskHeight": h, "backend": a.backend}).json()
    seg_s = round(time.perf_counter() - t, 2)
    g = sav_gt_mask(gt, frame, 4)
    cands = [rle_decode(cd["mask"]["counts"], w, h) for cd in seg["candidates"]]
    ious = [round(iou(m, g), 4) for m in cands]
    k = int(np.argmax(ious))
    print(f"candidates (SAM pick #{seg['selected']}): scores {[cd['score'] for cd in seg['candidates']]}, IoU vs GT {ious} -> choose #{k}")
    sheet = np.hstack([overlay(read_frame(SAV_VIDEO, frame), m, [(x, y, 1)]) for m in cands])
    cv2.imwrite(str(DOCS / "candidates_demo_sav_obj4.jpg"), cv2.resize(sheet, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])

    n = 240 - frame
    t = time.perf_counter()
    job = c.post(f"/sessions/{sid}/propagate", json={"frameIndex": frame, "points": pts, "maxFrames": n, "backend": a.backend, "maskCandidate": k}).json()
    if "id" not in job:
        print("propagate failed:", job)
        return 1
    while (j := c.get(f"/jobs/{job['id']}").json())["status"] not in ("succeeded", "failed", "cancelled"):
        time.sleep(1)
    track_s = round(time.perf_counter() - t, 1)
    if j["status"] != "succeeded":
        print("tracking failed:", j.get("error"))
        return 1
    mdir = BACKEND / "data" / "sessions" / sid / "tracks" / job["id"] / "masks"
    js, fs = [], []
    for f in j["stats"]["trackedFrames"]:
        gm = sav_gt_mask(gt, f, 4)
        if gm is None:
            continue
        m = cv2.imread(str(mdir / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        js.append(iou(m, gm))
        fs.append(boundary_f(m, gm))
    st = j["stats"]
    row = {
        "case": "SA-V obj4 (vest), single click, chosen mask option", "backend": a.backend, "promptFrame": frame,
        "candidateScores": [cd["score"] for cd in seg["candidates"]], "candidateIoUvsGT": ious, "samPick": seg["selected"], "chosen": k,
        "segmentS": seg_s, "trackS": track_s, "frames": len(st["trackedFrames"]), "promptType": st.get("promptType"),
        "device": st.get("gpuName") or st.get("device"), "avgFrameMs": st.get("avgFrameMs"),
        "J": round(float(np.mean(js)), 4), "Jmin": round(min(js), 4), "F": round(float(np.mean(fs)), 4),
        "JF": round(float((np.mean(js) + np.mean(fs)) / 2), 4), "gtFrames": len(js),
        "baseline": {"singleClickSamPick_JF": 0.0976, "threeClicks_JF": 0.9544},
    }
    (DOCS / "candidate_demo.json").write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False))
    c.delete(f"/sessions/{sid}")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    s = sub.add_parser("segment")
    s.add_argument("--only", nargs="*")
    m = sub.add_parser("make-jobs")
    m.add_argument("--fp32-all", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--results", default=str(PROJECT / "colab" / "results" / "realworld"))
    e.add_argument("--export", nargs="*", default=["01_dog/dog", "bedroom/girl", "04_coffee/kettle"])
    d = sub.add_parser("candidate-demo")
    d.add_argument("--backend", default="remote", choices=["remote", "local"])
    a = ap.parse_args()
    return {"ingest": cmd_ingest, "segment": cmd_segment, "make-jobs": cmd_make_jobs, "evaluate": cmd_evaluate, "candidate-demo": cmd_candidate_demo}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
