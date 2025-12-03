import os, cv2, torch, threading, time, numpy as np
from ultralytics import YOLO

RTSP_URL = "rtsp://admin:20117204wm@192.168.1.134:554/stream1"  # 先用子码流
WIN_NAME = "YOLO Low-Latency (Resizable)"
INITIAL_W, INITIAL_H = 1280, 720
KEEP_ASPECT = True

# ==== 关键：尽量去缓冲 + UDP ====
ffmpeg_opts = [
    ("rtsp_transport", "udp"),  # 不稳再改回 tcp
    ("fflags", "nobuffer"),
    ("reorder_queue_size", "0"),
    ("max_delay", "0"),
    ("buffer_size", "1024"),
    ("stimeout", "500000"),     # 连接/读超时(us)，卡很久就尽快断开重连
    ("analyzeduration", "0"),
    ("probesize", "4096"),
]
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ";".join(f"{k};{v}" for k, v in ffmpeg_opts)

def fit_to_window(img, win_w, win_h, keep_aspect=True):
    if win_w <= 0 or win_h <= 0: return img
    if not keep_aspect:
        interp = cv2.INTER_AREA if (win_w < img.shape[1] or win_h < img.shape[0]) else cv2.INTER_LINEAR
        return cv2.resize(img, (win_w, win_h), interpolation=interp)
    h, w = img.shape[:2]
    scale = max(1e-6, min(win_w / w, win_h / h))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x, y = (win_w - new_w) // 2, (win_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

# ==== 打开流（小缓冲） ====
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 有些平台无效，但设上不亏
assert cap.isOpened(), "打开 RTSP 失败，检查 IP/口令/端口 554"

# ==== 只保留“最新帧”的采集线程 ====
latest = {"frame": None, "ts": 0.0}
stop_flag = False
def reader_loop():
    # 抓-取分离：更快丢弃旧帧
    while not stop_flag:
        if not cap.grab():
            time.sleep(0.005)
            continue
        ok, f = cap.retrieve()
        if not ok:
            time.sleep(0.005)
            continue
        latest["frame"] = f
        latest["ts"] = time.time()

t = threading.Thread(target=reader_loop, daemon=True); t.start()

# ==== 加载 YOLO（GPU更低延迟） ====
model = YOLO("runs/train/v10-APConv-AssemFormer-HSFPN-ATFLm_exp/weights/best.pt")
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.to("cuda")

# ==== 可缩放窗口 ====
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow(WIN_NAME, INITIAL_W, INITIAL_H)
is_fullscreen = False

# 动态“跳帧推理”以追实时（比如 N=2 表示每2帧推理1次）
infer_every_n = 1
frame_count = 0

while True:
    frame = latest["frame"]
    if frame is None:
        cv2.waitKey(1)
        continue

    # 仅对最新帧推理（跳过陈旧帧）
    frame_count += 1
    do_infer = (frame_count % infer_every_n == 0)

    if do_infer:
        results = model.predict(frame, imgsz=640, conf=0.65, verbose=False, device=0 if use_cuda else None)
        annotated = results[0].plot()
    else:
        annotated = frame  # 跳帧时直接显示原始帧，进一步降延迟

    _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
    display = fit_to_window(annotated, win_w, win_h, KEEP_ASPECT)
    cv2.imshow(WIN_NAME, display)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    elif key == ord('f'):
        is_fullscreen = not is_fullscreen
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
    elif key == ord('g'):
        # 动态切换“每 N 帧推理一次”，追实时
        infer_every_n = 1 if infer_every_n != 1 else 2
        print(f"[info] infer_every_n -> {infer_every_n}")

stop_flag = True
t.join(timeout=0.5)
cap.release()
cv2.destroyAllWindows()
