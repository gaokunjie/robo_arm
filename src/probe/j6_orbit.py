# -*- coding: utf-8 -*-
"""J6 滚转的「纯滚 + 离心漂移」模型：一对实测帧标定 + 外推预测下一滚。

背景：相机装在 J6 转子上但不在轴线上。滚 J6=δ 的真实画面变化 =
  ①内容滚 θ = −k·δ（k=1.003，first_motion ORB 实测）
  ②整幅平动 shift(δ) = M·(cos,sin)(φ0 + δ/2)——相机绕 J6 轴走圆（半径 r），
    画面平移 ≈ −(f/d)·弦长（弦长 2r·sin(δ/2)、方向随 δ/2 转）；外加一点尺度变化。
M、φ0 从一对「同 J5、只差 J6=δ0」的实测帧标定：去掉已知的纯滚 θ0 后，角点（按 id 配对）
的质心差 = shift(δ0)。

用法：
  python3 -I src/probe/j6_orbit.py --base <帧0> --cur <帧δ0> --delta0 80 --scan 90,150,10
输出：①标定值（含纯滚假设的残差 RMS / 尺度 / 残余转角） ②每个候选总 δ 的预测全板框/余量。
"""
import argparse
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/kelly/桌面/1014/src")
import calibrate as C                                          # noqa: E402
from probe import board_report as BR                            # noqa: E402

KJ6 = 1.003       # |dθ_content / dδ_J6|（first_motion ORB：J6 +3° → 画面 −3.01°）
IW, IH = BR.IW, BR.IH
GATE = BR.GATE


def roll_H(Hm, K, theta_deg):
    """把像点绕主点旋转 θ 后的单应（内容滚 θ）。"""
    pp = (float(K[0, 2]), float(K[1, 2]))
    th = math.radians(theta_deg)
    R = np.array([[math.cos(th), -math.sin(th), 0],
                  [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    T1 = np.array([[1, 0, pp[0]], [0, 1, pp[1]], [0, 0, 1]])
    T0 = np.array([[1, 0, -pp[0]], [0, 1, -pp[1]], [0, 0, 1]])
    return T1 @ R @ T0 @ Hm


def outline_bbox(Hm, sq, nx, ny):
    W, H = nx * sq, ny * sq
    o = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    o = cv2.perspectiveTransform(o.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    x0, y0 = o.min(axis=0)
    x1, y1 = o.max(axis=0)
    return (x0, y0, x1, y1)


def margins(bb):
    x0, y0, x1, y1 = bb
    return (x0 - GATE, y0 - GATE, (IW - 1 - GATE) - x1, (IH - 1 - GATE) - y1)


def show(tag, bb):
    m = margins(bb)
    worst = min(m)
    print(f"# {tag}  框 x[{bb[0]:6.0f},{bb[2]:6.0f}] y[{bb[1]:6.0f},{bb[3]:6.0f}]  "
          f"余量 L{m[0]:6.0f} T{m[1]:6.0f} R{m[2]:6.0f} B{m[3]:6.0f}  "
          f"{'✓过' if worst >= 0 else f'✗差{-worst:.0f}px'}")


def main():
    ap = argparse.ArgumentParser(description="J6 滚转离心模型：标定 + 外推预测")
    ap.add_argument("--base", required=True, help="帧 0（J6=δ0 之前）")
    ap.add_argument("--cur", required=True, help="帧 δ0（与 base 同 J5、只差 J6=δ0）")
    ap.add_argument("--delta0", type=float, default=80.0, help="两帧间的 J6 差（度）")
    ap.add_argument("--scan", default="90,150,10", help="预测的总 δ 扫描 start,stop,step")
    args = ap.parse_args()

    det = C.BoardDetector()
    K = BR.load_K()
    sq = float(BR.config.CHARUCO_SQUARE_MM)
    nx, ny = BR.config.CHARUCO_SQUARES

    img_b = cv2.imread(args.base)
    img_c = cv2.imread(args.cur)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    gray_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    rb = BR.analyze(gray_b, det, K)
    rc = BR.analyze(gray_c, det, K)
    if rb is None or rc is None:
        print("有一帧认不出板")
        return 1
    show("实测 base", rb["bbox"])
    show(f"实测 cur(δ={args.delta0:g})", rc["bbox"])

    # 角点按 id 配对（base 检出的 id ∩ cur 检出的 id）
    cb, ib = det.detect(gray_b)
    cc, ic = det.detect(gray_c)
    db = {int(i): c for c, i in zip(cb.reshape(-1, 2), ib.reshape(-1))}
    dc_ = {int(i): c for c, i in zip(cc.reshape(-1, 2), ic.reshape(-1))}
    ids = sorted(set(db) & set(dc_))
    print(f"# 两帧共有角点 {len(ids)} 个")
    if len(ids) < 12:
        print("共有角点太少，标定不可靠")
        return 2

    # 纯滚预测：把 base 的角点绕主点滚 θ0 = −k·δ0
    theta0 = -KJ6 * args.delta0
    Hb = roll_H(rb["Hm"], K, theta0)
    obj = np.array([det.objp[i][:2] for i in ids], np.float64)
    pp = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), Hb).reshape(-1, 2)
    pc = np.array([dc_[i] for i in ids], np.float64)

    # 残差诊断：预估 vs 实测（仿射相似拟合：应≈纯平移，s≈1、残余角≈0）
    M2, inl = cv2.estimateAffinePartial2D(pp, pc, method=cv2.LMEDS)
    d = (M2 @ np.hstack([pp, np.ones((len(pp), 1))]).T).T - pc
    rms = float(np.sqrt((d ** 2).sum(axis=1).mean()))
    s = float(np.hypot(M2[0, 0], M2[1, 0]))
    ares = float(np.degrees(math.atan2(M2[1, 0], M2[0, 0])))
    print(f"# 去掉纯滚 θ0={theta0:+.1f}° 后：残余尺度 s={s:.4f}、残余转角 {ares:+.2f}°、"
          f"相似拟合 RMS {rms:.1f}px")

    # 质心差 = 离心漂移 shift(δ0)
    shift0 = pc.mean(axis=0) - pp.mean(axis=0)
    mag0 = float(np.hypot(*shift0))
    dir0 = float(math.degrees(math.atan2(shift0[1], shift0[0])))
    print(f"# 离心漂移标定 shift(δ0={args.delta0:g}) = ({shift0[0]:+.0f}, {shift0[1]:+.0f})px "
          f"（|{mag0:.0f}px|、方向 {dir0:+.1f}°）")
    print(f"#   ⇒ 圆模型：2r·f/d 尺幅 M = {mag0 / math.sin(math.radians(args.delta0 / 2)):.0f}px、"
          f"相位 φ0 = {dir0 - args.delta0 / 2:+.1f}°（r ≈ M/(2·f/d)，f/d≈{1535.0 / rc['dist']:.2f}px/mm "
          f"⇒ r≈{mag0 / math.sin(math.radians(args.delta0 / 2)) / (2 * (1535.0 / rc['dist'])):.0f}mm）")

    # ---------- 外推预测 ----------
    a0, a1, st = (float(v) for v in args.scan.split(","))
    print("\n# 预测（纯滚 + 圆模型外推；对 cur 帧做附加滚 δ−δ0）")
    print("# 总δ   附加θ    预测全板框                              余量")
    for dlt in np.arange(a0, a1 + 1e-9, st):
        add = dlt - args.delta0
        th_add = -KJ6 * add
        H2 = roll_H(rc["Hm"], K, th_add)
        bb = np.array(outline_bbox(H2, sq, nx, ny))
        dphi = math.radians((dlt - args.delta0) / 2.0)
        mag = mag0 / math.sin(math.radians(args.delta0 / 2)) * math.sin(math.radians(dlt / 2))
        direc = math.radians(dir0 - args.delta0 / 2) + math.radians(dlt / 2)
        shift = np.array([mag * math.cos(direc), mag * math.sin(direc)])
        dx, dy = shift - shift0
        bb = bb + np.array([dx, dy, dx, dy])
        m = margins(bb)
        worst = min(m)
        print(f"# {dlt:5.0f}° {th_add:+7.1f}°  x[{bb[0]:6.0f},{bb[2]:6.0f}] y[{bb[1]:6.0f},{bb[3]:6.0f}]  "
              f"L{m[0]:6.0f} T{m[1]:6.0f} R{m[2]:6.0f} B{m[3]:6.0f}  "
              f"{'✓过' if worst >= 0 else f'✗差{-worst:.0f}'}")
    print(f"# （δ={args.delta0:g} 即当前实测位；上表 add=0 行应≈实测 cur）")


if __name__ == "__main__":
    main()
