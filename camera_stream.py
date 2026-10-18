#!/usr/bin/env python3
"""USB 摄像头实时画面 → 浏览器观看 (MJPEG over HTTP)

运行后访问 http://localhost:8080
摄像头直接输出 MJPG 编码帧,原样转发,CPU 占用极低。
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

DEVICE = 2
PORT = 8080
WIDTH, HEIGHT = 1920, 1080

cap = cv2.VideoCapture(DEVICE)
if not cap.isOpened():
    print(f"无法打开 /dev/video{DEVICE}")
    raise SystemExit(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)  # 直接取 MJPG 字节,不转码
print(f"摄像头已打开: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

_lock = threading.Lock()
_latest = b""

def grab():
    global _latest
    while True:
        ret, frame = cap.read()
        if ret:
            with _lock:
                _latest = frame.tobytes()

threading.Thread(target=grab, daemon=True).start()

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>USB 摄像头实时画面</title>
<style>
  body { margin:0; background:#111; display:flex; flex-direction:column;
         align-items:center; justify-content:center; height:100vh;
         font-family: sans-serif; color:#ccc; }
  img { max-width:95vw; max-height:88vh; border-radius:8px; }
  .hint { margin-top:10px; font-size:14px; }
</style></head>
<body>
  <img src="/stream" alt="camera">
  <div class="hint">实时画面 · 刷新页面可重连 · 关闭此页面即停止观看(服务继续运行)</div>
</body></html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        payload = _latest
                    if payload:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            b"Content-Length: %d\r\n\r\n" % len(payload)
                            + payload + b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # 安静模式

server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
print(f"请在浏览器打开: http://localhost:{PORT}")
server.serve_forever()
