# -*- coding: utf-8 -*-
"""搜索一个能真正看见杯上 ArUco 的观察位（只读，不动臂）。

## 为什么"工具竖直正对杯轴"不是观察位（2026-10-18 的几何澄清）

相机光轴在法兰系 ≈ (0,0,1)，即**相机沿工具 z 看**。环心在**相机系**是
    (4.7, 78.7, 131.4)  —— 光轴下方 30.9°、前方 131mm（由 hand_eye 的 R、t 算出）
环的遮挡角范围：环心方向 ± atan(56.5/131.4)=±23.3° ⇒ 覆盖"轴下 7.6°~54.2°"；
**开口（半径 46.5 的孔）覆盖"轴下 7.6° 以内往上的区域"**。
画面半高角 = atan(540/1462) = 20.3°。
⇒ 若相机正对杯轴，**杯轴落在画面中心，而中心正被环的实体挡住**。
   杯子要看见，只能：(a) 让镜头稍微俯视使杯落到开口方向，或 (b) **转 q6 把缺口对准杯子**。

观察位 q6 完全自由（不进抓取序列），所以 (b) 是干净的解。

## 本脚本做的事

对一组候选姿态，逐个：
    ① 由相机位姿反推法兰位姿 → 用带缓存的控制器正解解 IK（限位感知精修）
    ② **解析遮挡判据**：从相机光心向"码心"投射线，与环平面求交，落在孔内/缺口内才算看得见
    ③ 距离判据：码边长投影 ≥ 25px（解码门限）
    ④ 环与杯/台面的净空
全部用控制器 FK 独立复核。不发任何运动指令。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import cup_pose                                             # noqa: E402
import fr3_math                                             # noqa: E402
from fr_fkcache import Budget, FKCache                      # noqa: E402
from fr_rpc import FRRpc                                    # noqa: E402

TABLE_Z = -85.0
MARKER_Z = TABLE_Z + config.MARKER_Z_ABOVE_TABLE
RC = float(config.RING_CENTER_IN_FLANGE[2])
RING_RI, RING_RO = config.RING_ID / 2.0, config.RING_OD / 2.0
RING_GAP_W, RING_GAP_CENTER_DEG = config.RING_GAP, config.RING_GAP_DIR_DEG
FX = 1462.0                                   # camera_intrinsics.yaml（锁焦版 fx）
CUP_RR, CUP_AZ = 590.0, -85.7


def cam_R(center, target, roll_hint):
    z = np.asarray(target, float) - np.asarray(center, float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, -1.0])
    if abs(float(z @ up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def ring_blocks(T_base_flange, T_flange_cam, p_cam_target):
    """射线 相机光心 → p_cam_target 与环平面求交，判断被实体挡否。

    环在法兰系是"中心 (0,0,RC)、平面垂直工具 z、半径 RING_RI..RING_RO、
    缺口中心方向 RING_GAP_CENTER_DEG（工具系，绕工具 z 从工具 x 量）"。
    返回 (是否遮挡, 交点到环心距离, 交点在缺口内的角度差)。
    """
    fl_of_pt = T_base_flange @ T_flange_cam @ np.append(np.asarray(p_cam_target, float).ravel()[:3], 1.0)
    o = T_base_flange[:3, 3]
    d = fl_of_pt[:3] - o
    n = T_base_flange[:3, 2]                      # 工具 z = 环平面法向
    denom = float(n @ d)
    if abs(denom) < 1e-9:
        return False, np.nan, np.nan
    s = float((T_base_flange[:3, 3] + RC * n - o) @ n) / denom
    if s <= 0:
        return False, np.nan, np.nan            # 环在射线反方向
    hit = o + s * d
    c = o + RC * n
    rel = hit - c
    r = float(np.linalg.norm(rel))
    ex = T_base_flange[:3, 0]                    # 工具 x
    ey = T_base_flange[:3, 1]
    ang = float(np.degrees(np.arctan2(rel @ ey, rel @ ex)))
    dang = abs((ang - RING_GAP_CENTER_DEG + 180.0) % 360.0 - 180.0)
    # 缺口张角：以"缺口两尖之间的直线"为准，半角 = atan((W/2)/(R+th/2))
    half_gap = float(np.degrees(np.arctan((RING_GAP_W / 2.0) / RING_RO)))
    in_gap = dang <= half_gap
    blocked = (RING_RI - 1.0) <= r <= (RING_RO + 1.0) and not in_gap
    return blocked, r, dang


def main():
    rpc = FRRpc(ip=config.ROBOT_IP)
    fc = FKCache(rpc.fk)
    K, D, X = cup_pose.load_calib()
    Xinv = np.linalg.inv(X)
    lim = np.asarray(rpc.joint_soft_limits(), float).reshape(6, 2)
    b = Budget(240)

    cup = np.array([CUP_RR * np.cos(np.radians(CUP_AZ)),
                    CUP_RR * np.sin(np.radians(CUP_AZ)), TABLE_Z])
    target = cup.copy()
    target[2] = MARKER_Z
    toward = -cup[:2] / np.linalg.norm(cup[:2])
    print(f"杯位估计 xy={np.round(cup[:2],1).tolist()} r={CUP_RR:.0f}  码心 z={MARKER_Z:.0f}")
    print(f"缺口半角 {np.degrees(np.arctan((RING_GAP_W/2)/RING_RO)):.0f}°  "
          f"画面半高角 {np.degrees(np.arctan(540/FX)):.1f}°\n")

    q_now = np.asarray(rpc.joint_deg(), float)
    T_now = fc.T(q_now)
    fk = lambda q: fc.pose(list(q))

    def solve(T_fl):
        def cost(qq):
            T = fc.T(qq)
            ce = fr3_math.rotvec_deg(T[:3, :3] @ T_fl[:3, :3].T)
            pe = T[:3, 3] - T_fl[:3, 3]
            viol = (np.clip(lim[:, 0] - qq, 0, None)
                    + np.clip(qq - lim[:, 1], 0, None))
            return float(pe @ pe + ce @ ce + 1e4 * float(viol @ viol))
        r = minimize(cost, q_now, method="L-BFGS-B",
                     bounds=[(lim[i, 0], lim[i, 1]) for i in range(6)],
                     options=dict(maxiter=250, eps=0.6, ftol=1e-16, gtol=1e-16))
        T = fc.T(r.x)
        ep = float(np.linalg.norm(T[:3, 3] - T_fl[:3, 3]))
        er = float(np.linalg.norm(fr3_math.rotvec_deg(T[:3, :3] @ T_fl[:3, :3].T)))
        ok = (np.all(r.x >= lim[:, 0] - 1e-6) and np.all(r.x <= lim[:, 1] + 1e-6)
              and ep < 3.0 and er < 1.5)
        return (r.x if ok else None), ep, er

    print(f"{'仰角':>5s}{'距离':>6s}{'光轴瞄点偏移':>12s}{'q6':>6s}{'残差mm/°':>12s}"
          f"{'投影px':>8s}{'码px':>7s}{'环遮挡':>8s}  判读")
    print("-" * 86, flush=True)
    best = None
    for elev in (30.0, 45.0, 60.0):
        for dist in (300.0, 380.0, 460.0):
            for aim_dz in (0.0, -60.0, -110.0):
                if b.left() < 30:
                    break
                el = np.radians(elev)
                aim = target.copy()
                aim[2] += aim_dz                     # 瞄点在码心上方/下方 → 改变杯子落点
                C = aim + np.array([toward[0] * dist * np.cos(el),
                                    toward[1] * dist * np.cos(el),
                                    dist * np.sin(el)])
                T_cam = np.eye(4)
                T_cam[:3, :3] = cam_R(C, aim, None)
                T_cam[:3, 3] = C
                T_fl0 = T_cam @ Xinv
                q, ep, er = solve(T_fl0)
                if q is None:
                    print(f"{elev:5.0f}{dist:6.0f}{aim_dz:12.0f}{'—':>6s}"
                          f"{ep:11.1f}/{er:.1f}{'—':>8s}{'—':>7s}{'—':>8s}  ✗ IK/限位不过",
                          flush=True)
                    continue
                # 转 q6 让缺口对准杯子，再复核遮挡
                T_fl = fc.T(q)
                blk, rr, dang = ring_blocks(T_fl, X, np.append(target, 1.0))
                # 投影
                T_c = T_fl @ X
                p_cam = np.linalg.inv(T_c) @ np.append(target, 1.0)
                import cv2
                uv, _ = cv2.projectPoints(p_cam[:3].reshape(1, 1, 3), np.zeros(3),
                                          np.zeros(3), K, np.asarray(D, float))
                uv = uv.reshape(2)
                dist_c = float(p_cam[2])
                span = config.ARUCO_MARKER_MM / max(dist_c, 1.0) * FX
                inf = 0 <= uv[0] < config.CAMERA_W and 0 <= uv[1] < config.CAMERA_H
                good = (not blk) and inf and span >= 25.0
                print(f"{elev:5.0f}{dist:6.0f}{aim_dz:12.0f}{q[5]:6.0f}"
                      f"{ep:7.2f}/{er:5.2f}{uv[0]:8.0f}{span:7.0f}"
                      f"{'挡✗' if blk else '通✓':>8s}  {'✓ 可用' if good else '✗'}",
                      flush=True)
                if good and (best is None or span > best[0]):
                    best = (span, elev, dist, aim_dz, q.copy(), T_fl.copy())
    fc.save(force=True)
    if best:
        span, elev, dist, aim_dz, q, T_fl = best
        print(f"\n★ 最佳：仰角 {elev:.0f}° 距离 {dist:.0f}mm 瞄点偏移 {aim_dz:.0f}mm "
              f"码投影 {span:.0f}px")
        print(f"  关节 {np.round(q,2).tolist()}")
        print(f"  法兰 {np.round(T_fl[:3,3],1).tolist()}")
        print(f"  当前法兰 {np.round(T_now[:3,3],1).tolist()}  "
              f"最大关节变化 {np.max(np.abs(q-q_now)):.1f}°")
    else:
        print("\n✗ 本轮网格内没有通过全部判据的姿态")
    st = fc.stats()
    print(f"缓存 {st['entries']} 条，命中 {st['hits']}，新查 {st['misses']} 用时 {b.elapsed():.0f}s")


if __name__ == "__main__":
    main()
