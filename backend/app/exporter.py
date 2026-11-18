"""MP4 내보내기: 추적 단계에서 저장한 마스크 + 효과 → FFmpeg(libx264) 인코딩.

- SAM 2 추론을 다시 실행하지 않습니다. tracks/<추적 작업>/masks/*.png 만 읽습니다.
- 추적된 연속 구간만 내보냅니다. 추적하지 않은 프레임은 포함하지 않습니다.
- 원본 프레임은 input.mp4 를 OpenCV 로 순차 디코딩해 얻습니다 (업로드 시 프레임 번호를 매긴 방식과 동일).
  세션 JPEG 를 쓰지 않으므로 손실 압축이 한 번 더 들어가지 않습니다.
- 프레임레이트는 ffprobe 로 원본의 정확한 유리수 값(예: 30000/1001)을 읽어 그대로 사용하고,
  원본에 오디오가 있으면 같은 시간 구간을 잘라 AAC 로 넣습니다.
- 작업 폴더: data/sessions/<id>/exports/<job>/  (output.part.mp4 → 성공 시 output.mp4 로 이름 변경)
  취소/실패 시 폴더 전체를 삭제합니다.
"""

from __future__ import annotations

import logging
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .effects import effect_params, render_frame
from .errors import AppError
from .ffmpeg import FFmpegTools, ffprobe_json, find_ffmpeg, has_audio, video_frame_rate
from .jobs import Job

log = logging.getLogger(__name__)


def tracked_range(stats: dict) -> tuple[int, int]:
    frames = sorted(stats.get("trackedFrames") or [])
    if not frames:
        raise AppError("validation", "This tracking result has no frames.", 422)
    start, end = frames[0], frames[-1]
    if frames != list(range(start, end + 1)):
        raise AppError("validation", "Tracked frames are not contiguous.", 422)
    return start, end


def load_mask(masks_dir: Path, frame: int, shape: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(masks_dir / f"{frame:05d}.png"), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise AppError("export_failed", f"Tracking mask for frame {frame} is missing.", 500)
    if m.shape != shape:
        raise AppError("export_failed", f"Mask size {m.shape} does not match video {shape}.", 500)
    return (m > 127).astype(np.uint8)


def read_frames(video: Path, start: int, end: int):
    """input.mp4 에서 [start, end] 프레임을 순서대로 생성 (프레임 번호, BGR)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise AppError("export_failed", "Could not open the source video.", 500)
    try:
        idx = 0
        while idx <= end:
            ok, frame = cap.read()
            if not ok:
                raise AppError("export_failed", f"Source video ended before frame {idx}.", 500)
            if idx >= start:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_frame(video: Path, index: int) -> np.ndarray:
    for _, frame in read_frames(video, index, index):
        return frame
    raise AppError("export_failed", f"Frame {index} not found.", 500)


class Exporter:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def tools(self) -> FFmpegTools | None:
        return find_ffmpeg()

    def require_tools(self) -> FFmpegTools:
        tools = self.tools
        if tools is None:
            raise AppError(
                "export_unavailable",
                "FFmpeg was not found. Run `python scripts/setup_ffmpeg.py` in backend/ (see README).",
                503,
            )
        return tools

    # ------------------------------------------------------------------ 단일 프레임 (미리보기 일치 검증용)
    def render_single(self, session, track_dir: Path, frame: int, effects: dict, size: tuple[int, int] | None) -> np.ndarray:
        meta = session.meta
        mask = load_mask(track_dir / "masks", frame, (meta.height, meta.width))
        img = read_frame(session.root / "input.mp4", frame)
        out = render_frame(img, mask, effect_params(effects, meta.width, meta.height))
        if size and size != (meta.width, meta.height):
            out = cv2.resize(out, size, interpolation=cv2.INTER_AREA)
        return out

    # ------------------------------------------------------------------ 내보내기
    def export(self, session, track_stats: dict, track_dir: Path, effects: dict, job: Job) -> dict:
        tools = self.require_tools()
        meta = session.meta
        start, end = tracked_range(track_stats)
        n = end - start + 1
        src = session.root / "input.mp4"
        fps = video_frame_rate(tools.ffprobe, src, meta.fps)
        audio = has_audio(tools.ffprobe, src)
        params = effect_params(effects, meta.width, meta.height)

        out_dir = session.root / "exports" / job.id
        out_dir.mkdir(parents=True)
        part, final, log_path = out_dir / "output.part.mp4", out_dir / "output.mp4", out_dir / "ffmpeg.log"

        cmd = [
            str(tools.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{meta.width}x{meta.height}",
            "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0",
        ]  # fmt: skip
        if audio:
            t0 = start * fps.denominator / fps.numerator
            dur = n * fps.denominator / fps.numerator
            # 오디오 길이는 입력 옵션 -t 가 추적 구간 길이로 제한합니다.
            # -shortest 는 쓰지 않음: 인터리빙 버퍼 때문에 마지막 영상 프레임을 잘라 버립니다 (실제 영상 검증 P0, 회귀 테스트 있음).
            # 원본 오디오가 영상보다 짧으면 오디오만 먼저 끝나고 영상 프레임은 모두 유지됩니다.
            cmd += ["-ss", f"{t0:.6f}", "-t", f"{dur:.6f}", "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "160k"]
        else:
            cmd += ["-map", "0:v:0"]
        if meta.width % 2 or meta.height % 2:
            # yuv420p 는 짝수 크기가 필요 → 오른쪽/아래에 1px 패딩 (원본 픽셀은 그대로 유지)
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        cmd += [
            "-c:v", "libx264", "-preset", self.settings.export_preset, "-crf", str(self.settings.export_crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(part),
        ]  # fmt: skip

        stats: dict = {
            "rangeStart": start,
            "rangeEnd": end,
            "framesRequested": n,
            "framesDone": 0,
            "fpsRational": f"{fps.numerator}/{fps.denominator}",
            "width": meta.width,
            "height": meta.height,
            "audio": audio,
            "effects": {k: bool(effects[k]["enabled"]) for k in ("neon", "colorPop", "blur")},
            "effectSettings": effects,
            "renderMs": [],
        }
        job.update(0.0, f"Exporting frames {start + 1}–{end + 1}…", dict(stats))
        t_job = time.perf_counter()
        log_file = log_path.open("wb")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_file)
        success = False
        frames = read_frames(src, start, end)
        try:
            for frame_idx, img in frames:
                t = time.perf_counter()
                mask = load_mask(track_dir / "masks", frame_idx, (meta.height, meta.width))
                out = render_frame(img, mask, params)
                render_ms = (time.perf_counter() - t) * 1000
                try:
                    proc.stdin.write(out.tobytes())
                except (BrokenPipeError, OSError):
                    raise AppError("export_failed", f"FFmpeg stopped unexpectedly: {self._log_tail(log_file, log_path)}", 500)
                stats["renderMs"].append(round(render_ms, 1))
                stats["framesDone"] += 1
                done = stats["framesDone"]
                job.update(0.97 * done / n, f"Frame {frame_idx + 1} ({done}/{n}) · effects {render_ms:.0f} ms/frame", _snapshot(stats))
                if job.cancelled:
                    job.update(message=f"Cancelled after {done}/{n} frames")
                    return _snapshot(stats)
            proc.stdin.close()
            job.update(0.98, "Finalizing MP4…")
            code = proc.wait(timeout=self.settings.export_finalize_timeout_s)
            if code != 0:
                raise AppError("export_failed", f"FFmpeg exited with code {code}: {self._log_tail(log_file, log_path)}", 500)
            part.replace(final)
            success = True
        finally:
            frames.close()  # 중간에 멈춰도 원본 영상 핸들을 즉시 해제 (Windows 파일 잠금 방지)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            log_file.close()
            if not success:
                shutil.rmtree(out_dir, ignore_errors=True)

        probe = probe_output(tools, final)
        stats.update(probe)
        stats["totalMs"] = round((time.perf_counter() - t_job) * 1000, 1)
        stats["avgRenderMs"] = round(statistics.fmean(stats["renderMs"]), 1)
        stats["fileSizeBytes"] = final.stat().st_size
        if probe["frameCount"] != n:
            log.warning("Exported frame count %s != expected %s", probe["frameCount"], n)
        job.update(1.0, f"Exported {n} frames ({probe['duration']:.3f}s) in {stats['totalMs'] / 1000:.1f}s", _snapshot(stats))
        return stats

    @staticmethod
    def _log_tail(f, path: Path) -> str:
        try:
            f.flush()
            return path.read_text(errors="replace").strip()[-400:] or "no output"
        except OSError:
            return "no output"


def _snapshot(stats: dict) -> dict:
    return {k: (list(v) if isinstance(v, list) else v) for k, v in stats.items()}


def probe_output(tools: FFmpegTools, path: Path) -> dict:
    info = ffprobe_json(tools.ffprobe, path, count_frames=True)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {
        "codec": v.get("codec_name"),
        "pixFmt": v.get("pix_fmt"),
        "outputWidth": v.get("width"),
        "outputHeight": v.get("height"),
        "frameCount": int(v.get("nb_read_frames") or v.get("nb_frames") or 0),
        "outputFps": round(num / den, 4),
        "duration": float(info["format"].get("duration", 0.0)),
        "hasAudio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }
