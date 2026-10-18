# -*- coding: utf-8 -*-
"""离线扫"画面滚转角"：相机在 J6 转子上，滚转 J6 ≈ 画面绕主点旋转。

问：如果滚 J6 若干度，全板投影框能不能转进 60px 门槛里？（板完全不动）
做法：用 view_sweep 帧的单应 H，把像点绕主点旋转 θ 得 H' = R·H，重算全板框/切边。
⚠ 只是估计：滚 J6 相机还会绕转子轴平移一点（相机不严格在 J6 轴上），
   真实结果以 board_check.py --j6 实测为准。
用法：python3 -I src/probe/roll_scan.py [照片]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/kelly/桌面/1014/src")
from probe import board_report as BR                            # noqa: E402

GATE = 60.0


def bbox_after_roll(Hm, K, theta_deg):
    """把像点绕主点旋转 θ 后的全板框（H' = R(θ)@H）。"""
    pp = (float(K[0, 2]), float(K[1, 2]))
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    T1 = np.array([[1, 0, pp[0]], [0, 1, pp[1]], [0, 0, 1]])
    T0 = np.array([[1, 0, -pp[0]], [0, 1, -pp[1]], [0, 0, 1]])
    Hp = T1 @ R @ T0 @ Hm
    sq = float(BR.config.CHARUCO_SQUARE_MM)
    nx, ny = BR.config.CHARUCO_SQUARES
    W, H = nx * sq, ny * sq
    outline = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    o = cv2.perspectiveTransform(outline.reshape(-1, 1, 2), Hp).reshape(-1, 2)
    x0, y0 = o.min(axis=0)
    x1, y1 = o.max(axis=0)
    cut = (max(0.0, GATE - x0), max(0.0, GATE - y0),
           max(0.0, x1 - (BR.IW - 1 - GATE)), max(0.0, y1 - (BR.IH - 1 - GATE)))
    return (x0, y0, x1, y1), cut


def main():
    jpg = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        BR.config.DATA_DIR / "view_sweep" / "20261016_102725" / "j5_-016.jpg")
    img = cv2.imread(str(jpg))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = BR.analyze(gray)
    if r is None or r.get("bad"):
        print(f"{jpg.name}: 认不出板")
        return
    K = BR.load_K()
    print(f"# {jpg.name}  当前全板框 {tuple(round(v) for v in r['bbox'])}  "
          f"切(含60px门槛口径) {tuple(round(v) for v in bbox_after_roll(r['Hm'], K, 0)[1])}")
    print("# θ(滚J6)  全板框                     切L/T/R/B(px, 超门槛为正)")
    best = []
    for th in range(-180, 180, 5):
        (x0, y0, x1, y1), cut = bbox_after_roll(r["Hm"], K, th)
        worst = max(cut)
        best.append((worst, th, (x0, y0, x1, y1), cut))
        if th % 15 == 0 or worst <= 0:
            tag = "  ← 过门槛!" if worst <= 0 else ""
            print(f"# {th:+4d}°  x[{x0:7.0f},{x1:7.0f}] y[{y0:7.0f},{y1:7.0f}]  "
                  f"{cut[0]:5.0f}/{cut[1]:5.0f}/{cut[2]:5.0f}/{cut[3]:5.0f}{tag}")
    best.sort()
    w, th, bb, cut = best[0]
    print(f"\n# 最优 θ={th:+d}°：全板框 x[{bb[0]:.0f},{bb[1]:.0f}] y[{bb[2]:.0f},{bb[3]:.0f}]、"
          f"最差边超出 {w:.0f}px")


if __name__ == "__main__":
    main()
