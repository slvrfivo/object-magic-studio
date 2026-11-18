"""SAM 2.1 영상 객체 추적 (공식 SAM2VideoPredictor API).

흐름: init_state → add_new_points_or_box(프롬프트 프레임) → propagate_in_video (정방향)
- 추적 범위는 [프롬프트 프레임, 프롬프트 + N) 의 연속 프레임입니다. 프레임 건너뛰기나 마스크 복사는 하지 않습니다.
- 범위의 JPEG 를 작업 폴더에 하드링크해 init_state 에 넘기므로, 범위 밖 프레임은 읽지 않습니다.
- 매 프레임의 추론 시간과 메모리를 측정하고, 마스크가 사라지거나(객체 소실) 면적이 급변하는 프레임을 표시합니다.
- 각 프레임의 원본 해상도 마스크는 PNG 로 저장합니다 (Phase 4 export 에서 사용).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import Settings
from ..errors import AppError, validation_error
from ..jobs import Job
from .engine import Sam2Engine
from .features import preloaded_features
from .frames import lazy_frame_loading

log = logging.getLogger(__name__)

OBJ_ID = 1


@dataclass
class TrackRequest:
    prompt_frame: int
    points_px: np.ndarray  # (N, 2) 원본 해상도 픽셀 좌표
    labels: np.ndarray  # (N,) 1 = positive, 0 = negative
    max_frames: int
    #: 사용자가 고른 마스크 후보 (원본 해상도 0/1). 있으면 클릭 점 대신 이 마스크로 추적 시작 (공식 add_new_mask)
    mask: np.ndarray | None = None


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _snapshot(stats: dict) -> dict:
    """API 스레드가 직렬화하는 동안 워커가 리스트를 수정하지 않도록 복사본을 넘김."""
    return {k: (list(v) if isinstance(v, list) else v) for k, v in stats.items()}


def classify_frames(coverages: list[float], lost_threshold: float, jump_ratio: float) -> tuple[list[int], list[int]]:
    """로컬 인덱스 기준 (소실 프레임, 면적 급변 프레임) 목록."""
    lost, unstable = [], []
    prev = None
    for i, c in enumerate(coverages):
        if c < lost_threshold:
            lost.append(i)
            continue
        if prev is not None and (c / prev > jump_ratio or prev / c > jump_ratio):
            unstable.append(i)
        prev = c
    return lost, unstable


class VideoTracker:
    def __init__(self, engine: Sam2Engine, settings: Settings):
        self.engine = engine
        self.settings = settings

    def max_frames(self) -> int:
        rt = self.engine.runtime
        if rt is not None and rt.name == "cuda":
            return self.settings.max_track_frames_cuda
        return self.settings.max_track_frames_cpu

    def validate(self, req: TrackRequest, frame_count: int) -> int:
        """실제로 추적할 프레임 수를 반환."""
        if req.max_frames < 2:
            raise validation_error("Track at least 2 frames.")
        limit = self.max_frames()
        if req.max_frames > limit:
            raise validation_error(f"At most {limit} frames can be tracked per job on this device.")
        return min(req.max_frames, frame_count - req.prompt_frame)

    def track(self, session, req: TrackRequest, job: Job, prompt_features=None) -> dict:
        """prompt_features: 프롬프트 프레임의 미리 계산된 백본 특징 (분할에서 계산한 것을 재사용, features.py)."""
        self.engine.require_ready()
        n = self.validate(req, session.meta.frame_count)
        start = req.prompt_frame
        work = session.root / "tracks" / job.id
        frames_dir, masks_dir = work / "frames", work / "masks"
        frames_dir.mkdir(parents=True)
        masks_dir.mkdir()
        for i in range(n):
            _link_or_copy(session.frames_dir / f"{start + i:05d}.jpg", frames_dir / f"{i:05d}.jpg")

        s = self.settings
        stats: dict = {
            "rangeStart": start,
            "rangeEnd": start + n - 1,
            "framesRequested": n,
            "framesDone": 0,
            "trackedFrames": [],
            "frameMs": [],
            "coverage": [],
            "lostFrames": [],
            "unstableFrames": [],
        }
        job.update(0.0, f"Waiting for model (frames {start + 1}–{start + n})…", _snapshot(stats))

        runtime = self.engine.runtime
        predictor = self.engine.video_predictor
        t_job = time.perf_counter()
        state = None
        with self.engine.exclusive("tracking"):
            try:
                runtime.reset_peak_memory()
                peak_rss = 0.0
                job.update(0.0, "Initializing video state…")
                t = time.perf_counter()
                preload = {0: prompt_features} if prompt_features is not None else None
                with torch.inference_mode(), runtime.autocast(), preloaded_features(predictor, preload) as reuse:
                    with lazy_frame_loading(runtime.lazy_frames) as lazy:
                        state = predictor.init_state(video_path=str(frames_dir), **runtime.video_state_options())
                    stats["initMs"] = round((time.perf_counter() - t) * 1000, 1)
                    stats["promptFeaturesReused"] = bool(reuse["reused"])

                    t = time.perf_counter()
                    if req.mask is not None:
                        # 한 번 클릭의 후보 중 사용자가 고른 마스크 → 화면에서 본 마스크 그대로 추적 시작
                        predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=OBJ_ID, mask=req.mask.astype(bool))
                        stats["promptType"] = "mask"
                    else:
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=0,
                            obj_id=OBJ_ID,
                            points=req.points_px.astype(np.float32),
                            labels=req.labels.astype(np.int32),
                            normalize_coords=True,
                        )
                        stats["promptType"] = "points"
                    runtime.synchronize()
                    stats["promptMs"] = round((time.perf_counter() - t) * 1000, 1)

                    t_prev = time.perf_counter()
                    gen = predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=n, reverse=False)
                    try:
                        for local_idx, obj_ids, video_res_masks in gen:
                            runtime.synchronize()
                            frame_ms = (time.perf_counter() - t_prev) * 1000
                            mask = (video_res_masks[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
                            frame = start + int(local_idx)
                            cv2.imwrite(str(masks_dir / f"{frame:05d}.png"), mask * 255)

                            mem = runtime.memory_snapshot()
                            peak_rss = max(peak_rss, mem.get("processRssMB", 0.0))
                            stats["trackedFrames"].append(frame)
                            stats["frameMs"].append(round(frame_ms, 1))
                            stats["coverage"].append(round(float(mask.mean()), 5))
                            stats["framesDone"] = len(stats["trackedFrames"])
                            stats["peakRssMB"] = peak_rss
                            stats.update({k: v for k, v in mem.items() if k != "processRssMB"})
                            self._summarize(stats)
                            done = stats["framesDone"]
                            avg = stats.get("avgFrameMs")
                            if avg is None:
                                # 첫 프레임은 프롬프트 프레임(이미 계산됨)이라 속도 추정에 쓰지 않음
                                msg = f"Frame {frame + 1} ({done}/{n}) · prompt frame · estimating speed…"
                            else:
                                eta_s = (n - done) * avg / 1000
                                msg = f"Frame {frame + 1} ({done}/{n}) · {frame_ms / 1000:.1f}s/frame · ETA {eta_s:.0f}s"
                            job.update(done / n, msg, _snapshot(stats))
                            if job.cancelled:
                                job.update(message=f"Cancelled after {done}/{n} frames")
                                break
                            t_prev = time.perf_counter()  # 저장/측정 시간은 프레임 추론 시간에서 제외
                    finally:
                        gen.close()
                if lazy is not None and "frames" in lazy:
                    stats["framesLoadedFromDisk"] = lazy["frames"].load_count
            except AppError:
                raise
            except Exception as e:
                if runtime.is_oom(e):
                    raise AppError("gpu_oom", "GPU out of memory during tracking. Try fewer frames.", 503) from e
                raise
            finally:
                if state is not None:
                    try:
                        predictor.reset_state(state)
                    except Exception:  # noqa: BLE001
                        pass
                del state
                gc.collect()
                runtime.empty_cache()
                shutil.rmtree(frames_dir, ignore_errors=True)

        stats["totalMs"] = round((time.perf_counter() - t_job) * 1000, 1)
        stats["cancelled"] = job.cancelled
        self._summarize(stats)
        (work / "track.json").write_text(json.dumps(stats, indent=1))
        if not job.cancelled:
            lost = len(stats["lostFrames"])
            note = f", object not visible in {lost} frame(s)" if lost else ""
            job.update(1.0, f"Tracked {stats['framesDone']} frames in {stats['totalMs'] / 1000:.1f}s{note}", _snapshot(stats))
        else:
            job.update(stats=_snapshot(stats))
        return stats

    def _summarize(self, stats: dict) -> None:
        # 첫 프레임은 프롬프트 프레임(이미 계산된 출력 재사용)이므로 속도 통계에서 제외
        ms = stats["frameMs"][1:]
        if ms:
            stats["avgFrameMs"] = round(statistics.fmean(ms), 1)
            stats["medianFrameMs"] = round(statistics.median(ms), 1)
            stats["maxFrameMs"] = round(max(ms), 1)
            stats["fps"] = round(1000 / stats["avgFrameMs"], 3)
        lost, unstable = classify_frames(stats["coverage"], self.settings.lost_coverage, self.settings.unstable_jump_ratio)
        offset = stats["rangeStart"]
        stats["lostFrames"] = [offset + i for i in lost]
        stats["unstableFrames"] = [offset + i for i in unstable]
