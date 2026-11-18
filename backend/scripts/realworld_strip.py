"""마스크 오버레이 스트립(확대 크롭) — 추적 결과 육안 검사용.

사용: python scripts/realworld_strip.py video.mp4 result.zip out.jpg start stop step [x0 y0 x1 y1]
"""

import sys, zipfile, numpy as np, cv2
v, zp, out, a, b, s = sys.argv[1:7]; a, b, s = int(a), int(b), int(s)
crop = list(map(int, sys.argv[7:11])) if len(sys.argv) > 7 else None
z = zipfile.ZipFile(zp); cap = cv2.VideoCapture(v); tiles = []; k = 0
while k <= b:
    ok, fr = cap.read()
    if not ok: break
    if k >= a and (k - a) % s == 0:
        m = cv2.imdecode(np.frombuffer(z.read(f"masks/{k:05d}.png"), np.uint8), 0) > 127
        o = fr.copy(); o[m] = (0.45 * o[m] + 0.55 * np.array([255, 80, 200])).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cs, -1, (255, 255, 0), 2)
        if crop: o = o[crop[1]:crop[3], crop[0]:crop[2]]
        o = cv2.resize(o, (int(o.shape[1] * 200 / o.shape[0]), 200))
        cv2.putText(o, f"#{k}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(o)
    k += 1
cols = 6
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
