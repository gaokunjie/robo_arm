# -*- coding: utf-8 -*-
"""板投影分析核心 + view_sweep 帧的离线量化报告。

分析核心 analyze(gray)：认角点 → 单应（板面 mm→像素）→ 把【整块板外框】投射回画面，
给出全板框、被每条边切掉多少 px（换成板面 mm）、板心/尺度/距离/法向夹角、
以及"若只平移板、要归位需要多少 mm"（板面 x/y 轴）。board_check.py 复用这里。
离线报告：python3 -I src/probe/board_report.py [run目录]（纯读图）
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/kelly/桌面/1014/src")
import calibrate as C          # noqa: E402
import config                  # noqa: E402

IW, IH = config.CAMERA_W, config.CAMERA_H
GATE = 60.0                    # 采集门槛：全板离画面边 ≥60px（auto_handeye）


def load_K():
    fs = cv2.FileStorage(str(config.DATA_DIR / config.CALIB_CAMERA), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    fs.release()
    return K


def analyze(gray, det=None, K=None, gate=GATE):
    """→ dict（认不出板给 None）。纯离线计算，不碰机器人/相机。"""
    if det is None:
        det = C.BoardDetector()
    if K is None:
        K = load_K()
    sq = float(config.CHARUCO_SQUARE_MM)
    nx, ny = config.CHARUCO_SQUARES
    W, H = nx * sq, ny * sq                                   # 板外框 175×125mm
    corners, ids = det.detect(gray)
    if corners is None:
        return None
    pts = corners.reshape(-1, 2).astype(np.float64)
    iid = ids.reshape(-1)
    obj = det.objp[iid][:, :2].astype(np.float64)
    Hm, mask = cv2.findHomography(obj, pts, cv2.RANSAC, 3.0)
    if Hm is None:
        return {"n": len(pts), "ninl": 0, "bad": "单应没解出来"}
    outline = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    o = cv2.perspectiveTransform(outline.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    x0, y0 = o.min(axis=0)
    x1, y1 = o.max(axis=0)
    cut = {"左": max(0.0, -x0), "上": max(0.0, -y0),
           "右": max(0.0, x1 - (IW - 1)), "下": max(0.0, y1 - (IH - 1))}
    c = np.array([[W / 2, H / 2]], np.float64)
    j = cv2.perspectiveTransform(np.vstack([c, c + [10, 0], c + [0, 10]]
                                           ).reshape(-1, 1, 2), Hm)
    sx = np.linalg.norm(j[1] - j[0]) / 10.0                   # px per mm（板 x 轴）
    sy = np.linalg.norm(j[2] - j[0]) / 10.0                   # px per mm（板 y 轴）
    cut_mm = {k: v / (sx if k in ("左", "右") else sy) for k, v in cut.items()}
    Kinv = np.linalg.inv(K)
    r1, r2 = Kinv @ Hm[:, 0], Kinv @ Hm[:, 1]
    lam = 1.0 / ((np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0)
    n = np.cross(r1 * lam, r2 * lam)
    n /= np.linalg.norm(n)
    t = lam * (Kinv @ Hm[:, 2])
    cc = cv2.perspectiveTransform(c.reshape(-1, 1, 2), Hm).reshape(2)
    # 目标：全板框居进 [gate, IW-1-gate]×[gate, IH-1-gate]
    du = (gate + (IW - 1 - gate)) / 2.0 - (x0 + x1) / 2.0
    dv = (gate + (IH - 1 - gate)) / 2.0 - (y0 + y1) / 2.0
    # 像移 → 板面位移：J = ∂(uv)/∂(板xy)，中心处 2×2
    tp = lambda q: cv2.perspectiveTransform(                       # noqa: E731
        np.array(q, np.float64).reshape(-1, 1, 2), Hm).reshape(-1, 2)
    p0 = tp(c)[0]
    J = np.column_stack([(tp(c + [[5, 0]])[0] - p0) / 5.0,
                         (tp(c + [[0, 5]])[0] - p0) / 5.0])
    try:
        dxy = np.linalg.solve(J, np.array([du, dv]))
    except np.linalg.LinAlgError:
        dxy = None
    # 检测到的角点包围盒（与 frame_stats 同口径：板宽=ptp 最大边）
    dp = corners.reshape(-1, 2)
    side = float(max(dp[:, 0].ptp(), dp[:, 1].ptp()))
    return {"n": len(pts), "ninl": int(mask.sum()), "bbox": (x0, y0, x1, y1),
            "cut": cut, "cut_mm": cut_mm, "center": (float(cc[0]), float(cc[1])),
            "sx": sx, "sy": sy, "dist": float(np.linalg.norm(t)),
            "tilt": float(np.degrees(np.arccos(min(1.0, abs(float(n[2])))))),
            "du": du, "dv": dv, "dxy": None if dxy is None else (float(dxy[0]), float(dxy[1])),
            "side": side, "outline": o, "pts": pts, "Hm": Hm}


def verdict(r, gate=GATE, min_side=350.0, need_full=True):
    """采集门槛判定 → (过不过, [原因行])。"""
    bad = []
    if r is None:
        return False, ["0 角点：板不在画面里或认不出"]
    if r.get("bad"):
        return False, [r["bad"]]
    if need_full and r["n"] < 24:
        bad.append(f"角点只有 {r['n']}/24（门槛 24/24）")
    if r["side"] < min_side:
        bad.append(f"板宽 {r['side']:.0f}px < {min_side:.0f}px")
    for k, v in r["cut"].items():
        if v > 0:
            bad.append(f"{k}边被切 {v:.0f}px（≈板面 {r['cut_mm'][k]:.0f}mm）")
    # 距门槛的余量（未切也要 ≥gate）
    m = {"左": r["bbox"][0] - gate, "上": r["bbox"][1] - gate,
         "右": (IW - 1 - gate) - r["bbox"][2], "下": (IH - 1 - gate) - r["bbox"][3]}
    for k, v in m.items():
        if v < 0:
            bad.append(f"{k}边余量不足：差 {-v:.0f}px（门槛留 {gate:.0f}px）")
    return (len(bad) == 0), bad


def format_report(name, r):
    if r is None:
        return f"{name}: 0 角点（板不在画面里）"
    if r.get("bad"):
        return f"{name}: {r['n']} 角点但 {r['bad']}"
    ok, bad = verdict(r)
    x0, y0, x1, y1 = r["bbox"]
    dxy_s = ("板面平移建议 Δ=({:+.0f}, {:+.0f})mm（板 x/y 轴）".format(*r["dxy"])
             if r.get("dxy") else "板面平移建议：退化")
    lines = [
        f"{name}: 角点 {r['n']}/24、内点 {r['ninl']}、板宽 {r['side']:.0f}px",
        f"    全板框 x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}]  "
        f"切: 左{r['cut']['左']:.0f} 上{r['cut']['上']:.0f} 右{r['cut']['右']:.0f} "
        f"下{r['cut']['下']:.0f}px（板面 ≈ 左{r['cut_mm']['左']:.0f} 上{r['cut_mm']['上']:.0f} "
        f"右{r['cut_mm']['右']:.0f} 下{r['cut_mm']['下']:.0f}mm）",
        f"    板心@({r['center'][0]:.0f},{r['center'][1]:.0f})px  尺度 {r['sx']:.2f}×{r['sy']:.2f}px/mm  "
        f"距离≈{r['dist']:.0f}mm  法向vs光轴 {r['tilt']:.1f}°",
        f"    {'✓ 门槛过（24/24 + 离边 ≥60px）' if ok else '✗ 不过：' + '；'.join(bad)}",
        f"    居中所需像移 ({r['du']:+.0f},{r['dv']:+.0f})px  →  {dxy_s}",
    ]
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        run = sorted((config.DATA_DIR / "view_sweep").iterdir())[-1]
    print(f"# 报告目录 {run}")
    det = C.BoardDetector()
    K = load_K()
    lines = []
    for jpg in sorted(run.glob("j5_*.jpg")):
        img = cv2.imread(str(jpg))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines.append(format_report(jpg.name, analyze(gray, det, K)))
    out = run / "board_report.txt"
    text = "\n".join(lines)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n# 已存 {out}")


if __name__ == "__main__":
    main()
