#!/usr/bin/env python3
"""实时显示 USB 摄像头画面 (GP_Flip_Mirror, /dev/video2)
按键: q/Esc 退出, s 保存当前画面截图
"""
import cv2

DEVICE = 2  # /dev/video2

cap = cv2.VideoCapture(DEVICE)
if not cap.isOpened():
    print(f"无法打开 /dev/video{DEVICE}")
    exit(1)

# 尝试用 MJPG 提高分辨率(失败则退回默认)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"分辨率: {w}x{h} | q/Esc 退出, s 截图")

snap = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("读取画面失败")
        break

    cv2.imshow("USB Camera 实时画面", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):  # q 或 Esc
        break
    elif key == ord("s"):
        snap += 1
        name = f"snapshot_{snap}.jpg"
        cv2.imwrite(name, frame)
        print(f"已截图: {name}")

cap.release()
cv2.destroyAllWindows()
