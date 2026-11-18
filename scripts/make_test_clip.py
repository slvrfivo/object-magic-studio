import sys, cv2, numpy as np
out, secs = sys.argv[1], float(sys.argv[2])
w, h, fps = 640, 360, 30
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
print('opened', vw.isOpened())
for i in range(int(secs*fps)):
    t = i/fps
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(40, 120, w, dtype=np.uint8)[None, :, None]
    img[:, :, 0] = 90
    cx = int(120 + 400*(0.5+0.5*np.sin(t*1.2)) ); cy = 180 + int(40*np.cos(t*2))
    cv2.circle(img, (cx, cy), 55, (40, 90, 240), -1)
    cv2.rectangle(img, (40, 260), (200, 330), (60, 200, 60), -1)
    vw.write(img)
vw.release()
