# -*- coding: utf-8 -*-
"""相机小件：开 /dev/video2、抓"追上运动"的新帧、两帧间转角估计。

FRESH_* 判据与流程是从 auto_handeye.py 原样搬过来的（2026-10-15 深夜在真机上量的）：
这台相机在【伺服运动后会先吐 ~10s 旧画面】，追上那一刻相邻两帧均差会跳变（静止时只有 ~2）。
小位移（<26px）对不上跳变判据，就按"一直稳 + 等满 20s"兜底。
改判据时两处一起改（auto_handeye.py 那套还在用，别各改各的）。

rot_between 是给"转 J6 画面该跟着转"这类定量旁证用的：ORB 特征 + 比值筛选 +
estimateAffinePartial2D(RANSAC) → 报出两帧之间的旋转角与平移。
"""

import math
import time

import cv2
import numpy as np

import config

# 与 auto_handeye.py 的 FRESH_* 完全一致（同一批实测定的）
FRESH_JUMP_MEAN = 5.0      # 相邻两抓逐像素均差 ≥ 它 = "画面跳变"（= 流追上了）
FRESH_STILL_MEAN = 3.0     # ≤ 它 = 静止噪声帧
FRESH_STABLE_N = 3         # 跳变后再连稳这么多帧才算追稳
FRESH_STILL_MIN_S = 20.0   # 没跳变但一直稳：至少等这么久才按"已追上"收
FRESH_STILL_N = 20         # …且至少稳住这么多帧
FRESH_WAIT_MAX_S = 50.0    # 硬上限：超了也交帧，但打警告（这帧可能还是旧的）
FRESH_TICK_S = 0.25        # 抓帧循环节拍

# 锁焦（2026-10-16 晚抓到的元凶）：PTZ-6609-CA 默认 AF 开——机械臂每换一个站位它自己
# 重新对焦、镜头物理移动、有效焦距就跟着变（实测板距分半 fx 差 4.87%，近 1446/远 1518）。
# 所有"AF 时期"采集的内参/手眼数据都被它污染（子集不稳、跨会话 fx 漂）。计量用途必须锁：
# 关 AF、焦点固定到 FOCUS_ABS（对 ~0.4m 处板的清晰度峰，实测 200±20 平台顶），回读核验。
FOCUS_ABS = 200            # V4L2 focus_absolute 目标值（对工作距离 ~0.4m 的清晰度峰）
FOCUS_TOL = 5              # 回读容差


class CamError(RuntimeError):
    pass


class Cam:
    """打开 + 抓帧。同一时刻只能有一个进程占着相机（浏览器取流会让这里打不开）。"""

    def __init__(self, dev=None, width=None, height=None):
        self.dev = config.CAMERA_DEV if dev is None else dev
        w = config.CAMERA_W if width is None else width
        h = config.CAMERA_H if height is None else height
        self.cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise CamError(f"/dev/video{self.dev} 打不开——"
                           f"浏览器取流（camera_stream.py）是不是开着？（相机独占）")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(3):                 # 空抓几帧让参数生效
            self.cap.read()
        # 锁焦：关 AF + 定焦点，回读核验（锁不住宁可炸，也别拿漂移焦距的数据）
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, FOCUS_ABS)
        for _ in range(2):
            self.cap.read()
        af = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
        fo = self.cap.get(cv2.CAP_PROP_FOCUS)
        if af != 0 or abs(fo - FOCUS_ABS) > FOCUS_TOL:
            raise CamError(f"锁焦失败：回读 autofocus={af} focus={fo}（目标 {FOCUS_ABS}）"
                           f"——这台相机的 AF 又会动焦距，锁不住就别采数据")
        print(f"      · 锁焦 ✓ autofocus=0 focus={fo:.0f}")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log = []                      # [(等了多少秒, 有没有见跳变, 跳变幅度)]
        self._last = None

    def close(self):
        try:
            self.cap.release()
        except Exception:                  # noqa: BLE001
            pass

    def raw(self, tries=6):
        for _ in range(tries):
            ok, f = self.cap.read()
            if ok and f is not None:
                return f
            time.sleep(0.05)
        raise CamError("相机读帧失败（连读 6 次没成）——相机是不是被别处抢走了？")

    def fresh(self, say=print):
        """运动后等画面追上再交帧（同 auto_handeye._grab_fresh 的判据）。"""
        t0, prev, stable, jumped, mag = time.time(), self._last, 0, False, 0.0
        el = 0.0
        while True:
            f = self.raw()
            if prev is not None:
                d = float(cv2.absdiff(f, prev).mean())
                if d >= FRESH_JUMP_MEAN:
                    jumped, stable, mag = True, 0, max(mag, d)
                elif d <= FRESH_STILL_MEAN:
                    stable += 1
            prev = f
            el = time.time() - t0
            if jumped and stable >= FRESH_STABLE_N:
                say(f"      · 画面追上：等 {el:.1f}s（跳变均差 {mag:.1f}）")
                break
            if el >= FRESH_WAIT_MAX_S:
                say(f"      ⚠ 画面等了 {FRESH_WAIT_MAX_S:.0f}s 没等到跳变——"
                    f"这一帧可能还是运动前的旧画面！")
                break
            if (not jumped) and stable >= FRESH_STILL_N and el >= FRESH_STILL_MIN_S:
                say(f"      · 画面 {el:.1f}s 没见跳变（位移小或流正常）——按已追上继续")
                break
            time.sleep(FRESH_TICK_S)
        self.log.append((el, jumped, mag))
        self._last = prev
        return prev


def rot_between(gray_a, gray_b):
    """两帧之间的旋转角/平移（度/像素）。返回 dict(ok, ang_deg, dx, dy, n_in) 或 ok=False。"""
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(gray_a, None)
    kb, db = orb.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return {"ok": False, "why": "特征太少"}
    ms = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in ms if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return {"ok": False, "why": f"匹配太少({len(good)})"}
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M_, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0)
    if M_ is None:
        return {"ok": False, "why": "仿射拟合失败"}
    return {"ok": True,
            "ang_deg": float(math.degrees(math.atan2(M_[1, 0], M_[0, 0]))),
            "dx": float(M_[0, 2]), "dy": float(M_[1, 2]),
            "n_in": int(inl.sum()) if inl is not None else len(good)}
