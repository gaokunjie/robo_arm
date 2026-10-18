# -*- coding: utf-8 -*-
"""用视觉解出的【真实杯轴】重算抓取位姿，并复核可达性与净空。

## 本次独立裁决（2026-10-18，观察位实测）

    码心(基座系) = (−28.7, −552.3, −22.2)   重投影 rms 1.45px
    杯轴（码心在朝相机那侧、退 36.8mm）≈ (−31.8, −589.0, −62.2)
      → |xy| = 589.8mm   vs 用户卷尺 590mm   **差 0.2mm** ⇒ 手眼+FK+PnP 全链可信
      → 码局部Y 与竖直 9.20°                  ⇒ 杯基本直立
      → 推出杯底 z = −62.2  vs 臂座 −85      ⇒ **z 差 22.8mm**（历史 26mm 分歧，
        与 xy 无关：xy 对 z 的灵敏度实测 0.00mm，所以两者可以一好一坏）

## z 差 23mm 怎么处理（这是本脚本要决定的）

环的**竖直位置是相对杯子**定的（插入 = 杯底+40、卡缘 = 杯底+114），而杯底高度我们
**刚刚实测到了**。⇒ 用视觉给的杯底 z 算法兰高度，而不是用卡尺的 TABLE_Z。
    杯底 −62.2 ⇒ 插入法兰 z = −62.2+40+90 = 67.8，卡缘 141.8，抬升 161.8
比卡尺口径（45/119/139）**整体高 23mm**。

台面净空仍按**保守值**算：台面真实高度未知，取两条口径里**较高**的那个当台面，
这样净空不会算多。TABLE_Z 到底多少另立一条待办（本脚本不覆写 config）。
"""
import sys
import numpy as np
sys.path.insert(0, '/home/kelly/桌面/1014/src')
import config, fr3_math
from fr_fkcache import FKCache
from fr_rpc import FRRpc
import verify_table_place as V

MARKER_BASE = np.array([-28.7, -552.3, -22.2])
CAM_BASE = np.array([-11.4, -344.9, 125.6])
R_MARK = 36.8

rpc = FRRpc(ip=config.ROBOT_IP)
fc = FKCache(rpc.fk); fr3_math.attach_cache(fc)
lim = V.set_limits(rpc)

# 杯轴 = 码心 - R * (从轴指向相机的水平单位向量)
u = CAM_BASE[:2] - MARKER_BASE[:2]
u = u / np.linalg.norm(u)
axis = MARKER_BASE - np.array([R_MARK * u[0], R_MARK * u[1], 0.0])
r = float(np.hypot(*axis[:2]))
print(f"码心 {MARKER_BASE.tolist()}")
print(f"杯轴 xy {np.round(axis[:2],1).tolist()}  |xy| = {r:.1f}mm  "
      f"（用户卷尺 590 → 差 {r-590:+.1f}mm）")
cup_bottom_z = MARKER_BASE[2] - config.MARKER_Z_ABOVE_TABLE
print(f"杯底 z = {cup_bottom_z:.1f}mm")
# 台面：两条口径里取较高者（保守），台面 = 杯底
table_hi = cup_bottom_z
print(f"台面净空按保守口径 {table_hi:.1f}（= 杯底，杯直接站台面）\n")

seeds = V.build_seeds(fc, V.Budget(90), (np.degrees(np.arctan2(axis[1], axis[0])),
                                        np.degrees(np.arctan2(axis[1], axis[0])) + 180.),
                      (-91., -72., -54., -38.), np.arange(-150., 151., 25.))
print(f"精确朝下种子 {len(seeds)} 个\n")
print(f"{'位姿':>5s}{'法兰z需求':>10s}{'实测法兰z':>11s}{'残差mm':>8s}{'tilt':>8s}"
      f"{'环底净空':>10s}  判读")
print("-" * 74, flush=True)
ok_all = True
for name, dz in (("插入", 40.), ("卡缘", 114.), ("抬升", 114. + config.LIFT_H)):
    zf = cup_bottom_z + dz + V.RC
    g = V.polish(fc, np.append(axis[:2], zf), seeds)
    if g is None:
        print(f"{name:>5s}{zf:10.1f}   ✗ 无朝下解"); ok_all = False; continue
    q, err, tl, p = g
    T = fc.T(q)
    rc = p + T[:3, :3] @ np.array([0., 0., V.RC])
    rb = rc[2] - config.RING_THICK / 2.
    inl = bool(np.all(q >= lim[:, 0] - 1e-6) and np.all(q <= lim[:, 1] + 1e-6))
    good = err < 0.5 and tl < 0.1 and inl and rb - table_hi > 20
    ok_all &= good
    print(f"{name:>5s}{zf:10.1f}{p[2]:11.1f}{err:8.3f}{tl:7.2f}°{rb-table_hi:+10.1f}  "
          f"{'✓' if good else '✗'}  [{', '.join(f'{v:.2f}' for v in q)}]", flush=True)
fc.save(force=True)
print("\n★ " + ("三个位姿在【真实杯轴】上全部成立" if ok_all else "存在不成立的位姿"))
