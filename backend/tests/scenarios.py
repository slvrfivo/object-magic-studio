"""추적 검증용 합성 장면. 프레임별 정답(목표 객체의 보이는 영역)을 정확히 알고 있습니다.

각 장면: (그리기 함수, 프레임 수, 첫 프레임 클릭 위치)
그리기 함수 draw(i) -> (BGR 프레임, 목표 객체의 보이는 영역 마스크, 목표 중심 좌표)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

W, H, FPS = 640, 360, 30.0
RED = (40, 60, 230)


def background() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = np.linspace(40, 120, W, dtype=np.uint8)[None, :, None]
    img[:, :, 0] = 90
    cv2.rectangle(img, (30, 280), (170, 340), (60, 200, 60), -1)  # 고정된 방해 요소
    return img


def circle_mask(cx: int, cy: int, r: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m


def scene_moving(i: int):
    """곡선 경로로 이동하는 원 (scripts/make_test_clip.py 와 같은 궤적)."""
    t = i / FPS
    cx = int(120 + 400 * (0.5 + 0.5 * np.sin(t * 1.2)))
    cy = 180 + int(40 * np.cos(t * 2))
    img = background()
    cv2.circle(img, (cx, cy), 55, RED, -1)
    return img, circle_mask(cx, cy, 55), (cx, cy)


def scene_exit(i: int):
    """오른쪽으로 빠르게 이동해 화면 밖으로 나가는 원 (i>=10 부터 완전히 사라짐)."""
    cx, cy = 400 + 30 * i, 180
    img = background()
    cv2.circle(img, (cx, cy), 50, RED, -1)
    return img, circle_mask(cx, cy, 50), (cx, cy)


def scene_occlusion(i: int):
    """회색 판이 원을 완전히 가렸다가(i=7..10) 지나가는 장면."""
    cx, cy = 300 + 5 * i, 180
    img = background()
    cv2.circle(img, (cx, cy), 50, RED, -1)
    gt = circle_mask(cx, cy, 50)
    rx = 40 * i - 100
    cv2.rectangle(img, (rx, 60), (rx + 220, 300), (150, 150, 150), -1)
    occ = np.zeros((H, W), np.uint8)
    cv2.rectangle(occ, (rx, 60), (rx + 220, 300), 1, -1)
    return img, gt & (1 - occ), (cx, cy)


def scene_distractor(i: int):
    """똑같이 생긴 원 두 개가 교차 (i=8 에서 겹침). 목표는 왼쪽에서 출발한 원."""
    tx, dx, cy = 150 + 20 * i, 490 - 20 * i, 180
    img = background()
    cv2.circle(img, (dx, cy), 45, RED, -1)
    cv2.circle(img, (tx, cy), 45, RED, -1)
    return img, circle_mask(tx, cy, 45), (tx, cy)


@dataclass
class Scenario:
    name: str
    draw: Callable[[int], tuple]
    frames: int
    description: str

    @property
    def click(self) -> tuple[float, float]:
        _, _, c = self.draw(0)
        return float(c[0]), float(c[1])


SCENARIOS = {
    "moving": Scenario("moving", scene_moving, 12, "circle moving along a curved path"),
    "exit": Scenario("exit", scene_exit, 14, "circle leaves the frame (gone from frame 10)"),
    "occlusion": Scenario("occlusion", scene_occlusion, 16, "grey board fully covers the circle at frames 7-10, then passes"),
    "distractor": Scenario("distractor", scene_distractor, 16, "identical circle crosses the target at frame 8"),
}


def write_frames(scn: Scenario, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(scn.frames):
        img, _, _ = scn.draw(i)
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def evaluate(scn: Scenario, masks: dict[int, np.ndarray]) -> list[dict]:
    """프레임별 IoU, 중심 오차, 정답 가시성."""
    rows = []
    for i in range(scn.frames):
        _, gt, (cx, cy) = scn.draw(i)
        pred = masks.get(i)
        if pred is None:
            continue
        pred = pred.astype(bool)
        g = gt.astype(bool)
        union = (pred | g).sum()
        iou = float((pred & g).sum() / union) if union else 1.0  # 둘 다 비어 있으면 정답
        ys, xs = np.nonzero(pred)
        err = math.hypot(xs.mean() - cx, ys.mean() - cy) if len(xs) else None
        rows.append(
            {
                "frame": i,
                "gtVisiblePx": int(g.sum()),
                "predPx": int(pred.sum()),
                "iou": round(iou, 3),
                "centerErrPx": None if err is None else round(err, 1),
            }
        )
    return rows


def contact_sheet(scn: Scenario, masks: dict[int, np.ndarray], path: Path, cols: int = 8) -> None:
    """프레임별 마스크 오버레이 썸네일을 한 장의 이미지로 저장."""
    tw, th = 200, 113
    tiles = []
    for i in range(scn.frames):
        img, gt, _ = scn.draw(i)
        m = masks.get(i)
        over = img.copy()
        if m is not None and m.any():
            over[m > 0] = (0.45 * np.array((246, 92, 139)) + 0.55 * over[m > 0]).astype(np.uint8)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, contours, -1, (238, 211, 34), 3)
        tile = cv2.resize(over, (tw, th), interpolation=cv2.INTER_AREA)
        label = f"#{i}" + ("  EMPTY" if m is not None and not m.any() else "")
        cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
