"""추적 1프레임당 단계별 시간 측정 (forward hook). 사용: python scripts/profile_tracking.py <video> <frames>"""
import sys, time, tempfile, collections
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from app.config import Settings
from app.inference.engine import Sam2Engine
from app.inference.frames import lazy_frame_loading
from app.media import extract_frames

video, n = sys.argv[1], int(sys.argv[2])
eng = Sam2Engine(Settings()); eng.load(); p = eng.video_predictor
acc = collections.defaultdict(list); starts = {}
def hook(name, mod):
    mod.register_forward_pre_hook(lambda m, a: starts.__setitem__(name, time.perf_counter()))
    mod.register_forward_hook(lambda m, a, o: acc[name].append((time.perf_counter() - starts[name]) * 1000))
for name in ["image_encoder", "memory_attention", "sam_mask_decoder", "memory_encoder", "sam_prompt_encoder"]:
    hook(name, getattr(p, name))
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp) / "f"; meta = extract_frames(Path(video), d)
    for f in sorted(d.glob("*.jpg"))[n:]: f.unlink()
    with torch.inference_mode():
        with lazy_frame_loading(True):
            st = p.init_state(str(d))
        p.add_new_points_or_box(st, 0, 1, points=np.array([[320., 220.]], np.float32), labels=np.array([1], np.int32))
        for k in acc: acc[k].clear()
        t = time.perf_counter(); frames = 0
        for i, _, m in p.propagate_in_video(st):
            frames += 1
        total = (time.perf_counter() - t) * 1000
tracked = frames - 1
print(f"frames tracked (excluding prompt frame): {tracked}, total {total/1000:.1f}s, per frame {total/tracked:.0f} ms")
for k, v in acc.items():
    if v: print(f"  {k:20s} calls={len(v):3d} total={sum(v)/1000:7.1f}s  per-call={np.mean(v):8.0f} ms  ({sum(v)/total*100:4.1f}% of propagate)")
print("torch threads", torch.get_num_threads())
