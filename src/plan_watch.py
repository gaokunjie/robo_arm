# -*- coding: utf-8 -*-
"""观察位规划（只读）：让工具相机看见杯子上的 ArUco 码。

## 为什么观察位必须【倾斜】而不是"工具竖直俯视"

手眼给的光心位置：法兰系 (83.3, −7.2, **−38.4**)；环心：法兰系 (0, 0, **+90**)。
两者在工具轴上分居法兰两侧 ⇒ **工具竖直时，相机在环上方 38mm、横向偏 84mm**。

算一下视线是否被环挡住（设法兰正对杯轴、法兰 z=z_f，相机偏 x=+84）：
    视线到【杯轴】在环平面（z=z_f−90）处的横坐标 x = 84·(1−t)，t = 128.4/(z_f+83.4)
    要穿过环内孔（半径 46.5）需 x<46.5 ⇒ t>0.554 ⇒ **z_f < 149mm**
    ⇒ 法兰得压到 149mm 以下（环心 59mm、离台面只有 144mm）——贴着桌面拍，太危险。
到【近侧码面】更严：要 z_f < 78mm，环已经低于杯口，物理上会撞杯。

⇒ **观察位必须倾斜**，让环偏到视线一侧。这与 see_cup.py 当初用 20~40° 俯角是一致的。

## 本脚本做什么

    ① 由"相机位姿"反推"法兰位姿"：T_base_flange = T_base_cam · inv(X)
    ② 用【带缓存的正解】解 IK（不动臂，缓存命中免费）
    ③ 三道复核（全部用控制器 FK 独立算，不信任中间量）：
         · 把码心投影回图像，看落在不在画面内、有多大
         · 环心离杯轴的水平距离（要 > 环外/2 + 杯半径 + 余量）
         · 法兰与环底离台面的净空
    ④ 只打印计划，不发运动指令

⚠ 杯位此刻是【用户给的 r=590mm + 上次视觉方位 −85.7°】的估计值，不是测量值。
   定位完真实杯轴后必须重跑本脚本。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import cup_pose                                             # noqa: E402
import fr3_math                                             # noqa: E402
from fr_fkcache import Budget, FKCache                      # noqa: E402
from fr_rpc import FRRpc                                    # noqa: E402

TABLE_Z = -85.0
MARKER_Z = TABLE_Z + config.MARKER_Z_ABOVE_TABLE        # 码心 z = −45
RC = float(config.RING_CENTER_IN_FLANGE[2])               # 90
CUP_R = config.CUP_PROFILE[2][1] / 2.0                    # 码高处杯半径 ≈ 36.5
RING_OUT = config.RING_OD / 2.0                           # 56.5
RING_TH = config.RING_THICK / 2.0                         # 6

# 杯位估计（用户 2026-10-18 报 r=590mm；方位沿用上次视觉解）
CUP_RR, CUP_AZ = 590.0, -85.7
# 观察位几何：相机相对杯的仰角与距离
ELEV_DEG, DIST_MM = 45.0, 420.0


def look_at_cam_R(center, target, up_hint):
    """构造相机系 R（OpenCV：x 右、y 下、z 沿光轴朝 target）。"""
    z = np.asarray(target, float) - np.asarray(center, float)
    z /= np.linalg.norm(z)
    up = np.asarray(up_hint, float)
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-9:                       # 视线与 up 平行 → 换一个参考
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)                  # x×y=z ⇒ y = z×x
    return np.column_stack([x, y, z])


def main():
    rpc = FRRpc(ip=config.ROBOT_IP)
    fc = FKCache(rpc.fk)
    K, D, X = cup_pose.load_calib()
    Xinv = np.linalg.inv(X)

    cup = np.array([CUP_RR * np.cos(np.radians(CUP_AZ)),
                    CUP_RR * np.sin(np.radians(CUP_AZ)), 0.0])
    cup[2] = TABLE_Z
    target = cup.copy()
    target[2] = MARKER_Z
    print(f"杯底 z={TABLE_Z:.0f}  码心 z={MARKER_Z:.0f}  杯轴 xy={np.round(cup[:2],1).tolist()}"
          f"（r={np.hypot(*cup[:2]):.0f}mm，**估计值**）")

    toward_base = -cup[:2] / np.linalg.norm(cup[:2])
    el = np.radians(ELEV_DEG)
    C = (target + np.array([toward_base[0] * DIST_MM * np.cos(el),
                            toward_base[1] * DIST_MM * np.cos(el),
                            DIST_MM * np.sin(el)]))
    R_cam = look_at_cam_R(C, target, up_hint=(0.0, 0.0, -1.0))
    T_cam = np.eye(4)
    T_cam[:3, :3] = R_cam
    T_cam[:3, 3] = C
    T_fl = T_cam @ Xinv
    print(f"\n目标相机位: 光心 {np.round(C,1).tolist()}  距码心 {DIST_MM:.0f}mm  仰角 {ELEV_DEG:.0f}°")
    print(f"反推法兰位: {np.round(T_fl[:3,3],1).tolist()}")

    # 环心在法兰系 (0,0,90) ⇒ 基座系
    ring_c = (T_fl[:3, 3] + T_fl[:3, :3] @ np.array([0.0, 0.0, RC]))
    d_ring_cup = float(np.hypot(ring_c[0] - cup[0], ring_c[1] - cup[1]))
    ring_bottom_z = ring_c[2] - RING_TH
    print(f"环心 {np.round(ring_c,1).tolist()}  离杯轴水平 {d_ring_cup:.0f}mm  "
          f"环底 z={ring_bottom_z:.0f}（台面净空 {ring_bottom_z - TABLE_Z:+.0f}mm）")

    # ---- ② 解 IK（只用带缓存的控制器正解；不发指令）----
    fk = lambda q: fc.pose(list(q))
    b = Budget(150)
    q_now = np.asarray(rpc.joint_deg(), float)
    calls, secs = 2000, 2000 * fr3_math.FK_MIN_THROTTLE_S
    print(f"\nIK 成本预估 {calls} 次 FK ≈ {secs:.0f}s（缓存命中不计费）")
    q = None
    for steps, iters in ((10, 15), (20, 12)):
        if b.left() < 30:
            break
        r = fr3_math.solve_delta(fk, q_now, dp=T_fl[:3, 3] - np.asarray(fc.pose(q_now)[:3]),
                                 drot=fr3_math.rotvec_deg(T_fl[:3, :3]
                                                           @ fc.T(q_now)[:3, :3].T),
                                 steps=steps, iters=iters)
        print(f"  steps={steps}: 残差 {r['err_pos']:.2f}mm/{r['err_ang']:.2f}° "
              f"max|dq|={np.max(np.abs(r['dq'])):.1f}°", flush=True)
        if r["err_pos"] < 1.0 and r["err_ang"] < 0.5:
            q = r["q"]
            break
    if q is None:
        print("✗ IK 未收敛（残差 >1mm / >0.5°）——不发任何运动，先查可达性")
        return
    # ---- 限位感知精修 ----
    # ⚠ solve_delta **不约束关节限位**，本次解出 q3=162.28° 超出 [-160,160]。
    # ⚠⚠ 而且**不能靠"折返"修**：q3 限位跨 320° 而非 360°，把它折到 −157.72°
    #    等于改了 320°，FK 落点随之偏 **185mm**（2026-10-18 实测）。
    #    折返只有在限位跨度是 360° 整数倍时才等价 —— 本项目没有一处满足。
    # ⇒ 正确做法：把限位做成软约束重解，允许位姿残差换限位合规。
    lim = np.asarray(rpc.joint_soft_limits(), float).reshape(6, 2)
    W_LIM, TGT = 1e4, T_fl

    def cost(qq):
        T = fc.T(qq)
        ce = fr3_math.rotvec_deg(T[:3, :3] @ TGT[:3, :3].T)
        pe = T[:3, 3] - TGT[:3, 3]
        viol = np.clip(lim[:, 0] - qq, 0, None) + np.clip(qq - lim[:, 1], 0, None)
        return float(pe @ pe + (ce @ ce) + W_LIM * float(viol @ viol))

    from scipy.optimize import minimize
    res = minimize(cost, q, method="L-BFGS-B",
                   bounds=[(lim[i, 0], lim[i, 1]) for i in range(6)],
                   options=dict(maxiter=400, eps=0.6, ftol=1e-16, gtol=1e-16))
    q = res.x
    T_f = fc.T(q)
    e_pos = float(np.linalg.norm(T_f[:3, 3] - TGT[:3, 3]))
    e_rot = float(np.linalg.norm(fr3_math.rotvec_deg(T_f[:3, :3] @ TGT[:3, :3].T)))
    inlim = bool(np.all(q >= lim[:, 0] - 1e-6) and np.all(q <= lim[:, 1] + 1e-6))
    print(f"  限位感知精修后 {np.round(q,2).tolist()}")
    print(f"  限位内={'是 ✓' if inlim else '否 ✗'}   位姿残差 {e_pos:.3f}mm / {e_rot:.3f}°")
    if not inlim or e_pos > 2.0 or e_rot > 1.0:
        print("✗ 限位不满足或残差过大 —— 不发运动")
        return

    # ---- ③ 三道复核（全部用控制器 FK 独立算）----
    T_f = fc.T(q)
    T_c = T_f @ X                                   # 实际相机位姿（基座←相机）
    # ⚠ 码心在【基座系】⇒ 必须用 inv(T_base_cam) 转到相机系。
    #   （误用 Xinv=inv(T_flange_cam) 会得到天文数字的像素 —— 2026-10-18 实踩）
    p_cam = np.linalg.inv(T_c) @ np.append(target, 1.0)
    uv, _ = cv2_project(K, D, p_cam[:3])
    w, h = config.CAMERA_W, config.CAMERA_H
    inframe = 0 <= uv[0] < w and 0 <= uv[1] < h
    f_mm = DIST_MM
    span_px = config.ARUCO_MARKER_MM / f_mm * K[0, 0]
    print(f"\n=== 复核 ===")
    print(f"  关节解 {np.round(q,2).tolist()}")
    print(f"  法兰实测 {np.round(T_f[:3,3],1).tolist()}  （目标 {np.round(T_fl[:3,3],1).tolist()}）")
    print(f"  码心投影像素 {np.round(uv,0).tolist()}  画面 {w}×{h}  "
          f"{'✓ 在画面内' if inframe else '✗ 出画面'}")
    print(f"  码边长投影 ≈ {span_px:.0f}px（解码需要 ≥20px，余量 "
          f"{'充足' if span_px > 60 else '偏小' if span_px > 25 else '不够'}）")
    print(f"  环心离杯轴 {d_ring_cup:.0f}mm（需 >{RING_OUT + CUP_R + 15:.0f}mm 才不蹭杯）"
          f"  {'✓' if d_ring_cup > RING_OUT + CUP_R + 15 else '✗ 会蹭杯'}")
    print(f"  环底离台面 {ring_bottom_z - TABLE_Z:+.0f}mm  "
          f"{'✓' if ring_bottom_z - TABLE_Z > 40 else '✗ 太低'}")
    fc.save(force=True)
    st = fc.stats()
    print(f"\n缓存 {st['entries']} 条，命中 {st['hits']}，新查 {st['misses']}  用时 {b.elapsed():.0f}s")
    print("\n★ 本脚本只读，未发任何运动指令。")


def cv2_project(K, D, p):
    import cv2
    uv, _ = cv2.projectPoints(np.asarray(p, float).reshape(1, 1, 3),
                              np.zeros(3), np.zeros(3), K, np.asarray(D, float))
    return uv.reshape(2), p


if __name__ == "__main__":
    main()
