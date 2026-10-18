# -*- coding: utf-8 -*-
"""从当前姿态规划观察位（只读）。

## 为什么不用 see_cup.py 的 lookdown_plan

它解出 `J4=146.3° 顶软限位 [-265,85]` 就直接判失败。**但 +146.3 ≡ −213.7 (mod 360)**，
物理位形完全相同，且 −213.7 离当时的关节只差几度。旋转关节 q 与 q+360k 严格等价，
**折回 360° 整数倍是无损的**，不做就等于把"腕正常翻转"误判成"跳 352°"。

本脚本照抄 see_cup 的观察位构造（elev=40°、dist=330mm，真机验证过残差 0.123mm），
但求解后做 360° 折回再比关节跳变。

只读：全程只调带缓存的控制器正解，不发任何运动指令。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import cup_pose                                             # noqa: E402
import fr3_math as M                                       # noqa: E402
from fr_fkcache import FKCache                             # noqa: E402
from fr_rpc import FRRpc                                   # noqa: E402
import verify_table_place as V                             # noqa: E402

ELEV_DEG, DIST_MM, CAP_DEG = 40.0, 330.0, 60.0


def main():
    rpc = FRRpc(ip=config.ROBOT_IP)
    fc = FKCache(rpc.fk)
    M.attach_cache(fc)
    lim = V.set_limits(rpc)
    BB = [(lim[i, 0], lim[i, 1]) for i in range(6)]
    K, D, X = cup_pose.load_calib()
    Xinv = np.linalg.inv(X)
    q_now = np.asarray(rpc.joint_deg(), float)
    T_now = fc.T(q_now)
    aim = np.array([config.CUP_AXIS_XY[0], config.CUP_AXIS_XY[1], config.marker_plane_z()])
    print(f"当前关节 {np.round(q_now,2).tolist()}")
    print(f"当前法兰 {np.round(T_now[:3,3],1).tolist()}")
    print(f"瞄点(码心) {np.round(aim,1).tolist()}  elev {ELEV_DEG}° dist {DIST_MM}mm\n")

    toward = -aim[:2] / np.linalg.norm(aim[:2])
    az = math.atan2(T_now[:3, 3][0] - aim[0], T_now[:3, 3][1] - aim[1])
    el = math.radians(ELEV_DEG)
    cam1 = aim + np.array([math.sin(az) * math.cos(el),
                           math.cos(az) * math.cos(el),
                           math.sin(el)]) * DIST_MM
    z1 = (aim - cam1) / np.linalg.norm(aim - cam1)
    xax = np.cross(z1, [0.0, 0.0, 1.0])
    xax /= np.linalg.norm(xax)
    yax = np.cross(z1, xax)
    C = np.eye(4)
    C[:3, 0], C[:3, 1], C[:3, 2], C[:3, 3] = xax, yax, z1, cam1
    T_fl = C @ Xinv
    print(f"相机目标位 {np.round(cam1,1).tolist()}  视线 {np.round(z1,3).tolist()}")
    print(f"目标法兰位 {np.round(T_fl[:3,3],1).tolist()}\n")

    def cost(q, roll=0.0):
        T = fc.T(q)
        ce = M.rotvec_deg(T[:3, :3] @ T_fl[:3, :3].T)
        pe = T[:3, 3] - T_fl[:3, 3]
        # 姿态里"绕视线滚转"这一维交给 ρ 扫；这里只罚位置 + 非滚转姿态
        return float(pe @ pe + ce @ ce)

    print(f"{'ρ':>5s}{'残差mm/°':>13s}{'max|Δq|':>9s}{'Δq(折回后)':>38s}  判读")
    print("-" * 84, flush=True)
    best = None
    for roll in list(range(0, 360, 30)) + [135, 165, 195, 225, 255]:
        cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        xr = cr * xax + sr * yax
        Cx = np.eye(4)
        Cx[:3, 0], Cx[:3, 1] = xr, np.cross(z1, xr)
        Cx[:3, 2], Cx[:3, 3] = z1, cam1
        T_fl_r = Cx @ Xinv

        def c(q):
            T = fc.T(q)
            ce = M.rotvec_deg(T[:3, :3] @ T_fl_r[:3, :3].T)
            pe = T[:3, 3] - T_fl_r[:3, 3]
            viol = np.clip(lim[:, 0] - q, 0, None) + np.clip(q - lim[:, 1], 0, None)
            return float(pe @ pe + ce @ ce + 1e4 * float(viol @ viol))

        # 两遍：① 带限位罚项求到"进限位"的解 → 360° 折回；② 无罚项精修位姿。
        #    只做①的话 1e4·viol² 会一直压着位姿项，收敛到 2~3° 就停在那儿（实测）。
        r = minimize(c, q_now, method="L-BFGS-B", bounds=BB,
                     options=dict(maxiter=300, eps=0.6, ftol=1e-16, gtol=1e-16))
        q0 = V.fold(r.x)

        def c2(q):
            T = fc.T(q)
            ce = M.rotvec_deg(T[:3, :3] @ T_fl_r[:3, :3].T)
            pe = T[:3, 3] - T_fl_r[:3, 3]
            return float(pe @ pe + ce @ ce)

        r2 = minimize(c2, q0, method="L-BFGS-B", bounds=BB,
                      options=dict(maxiter=400, eps=0.4, ftol=1e-18, gtol=1e-18))
        qf = V.fold(r2.x)
        T = fc.T(qf)
        ep = float(np.linalg.norm(T[:3, 3] - T_fl_r[:3, 3]))
        er = float(np.linalg.norm(M.rotvec_deg(T[:3, :3] @ T_fl_r[:3, :3].T)))
        inl = bool(np.all(qf >= lim[:, 0] - 1e-6) and np.all(qf <= lim[:, 1] + 1e-6))
        dq = np.abs(qf - q_now)
        mx = float(dq.max())
        # 观察位的判据是【物理】的，不是标称位姿的：
        #   cup_pose.solve_pnp 用的是实际相机位姿（FK×手眼）反算，不吃标称偏差。
        #   所以 2~3° 的姿态残差无害（330mm 处瞄点只挪 15mm），真正的门是
        #   「码心落在画面内」+「码边长投影 ≥25px（解码门限）」。
        Tq = fc.T(qf)
        Tb = Tq @ X
        pc = np.linalg.inv(Tb) @ np.append(aim, 1.0)
        import cv2
        uvp, _ = cv2.projectPoints(pc[:3].reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                  K, np.asarray(D, float))
        uvp = uvp.reshape(2)
        span = config.ARUCO_MARKER_MM / max(float(pc[2]), 1.0) * K[0, 0]
        inview = 0 <= uvp[0] < config.CAMERA_W and 0 <= uvp[1] < config.CAMERA_H
        ok = inl and mx <= CAP_DEG and inview and span >= 25.0
        print(f"      码心像素 {np.round(uvp,0).tolist()}  深度 {pc[2]:.0f}mm  "
              f"码投影 {span:.0f}px  {'在画面内✓' if inview else '出画面✗'}")
        print(f"{roll:5d}{ep:8.2f}/{er:5.2f}{mx:9.1f}"
              f"{np.round(dq,0).astype(int).tolist()!s:>38s}"
              f"  {'✓ 过闸' if ok else '✗'}", flush=True)
        if ok and (best is None or mx < best[0]):
            best = (mx, roll, qf.copy(), T_fl_r.copy(), ep, er)
    fc.save(force=True)
    if best is None:
        print("\n✗ 没有 ρ 过闸 —— 观察位需要从别的中间位再规划，不硬上")
        return
    mx, roll, q, T_fl, ep, er = best
    print(f"\n★ 观察位：ρ={roll}°  max|Δq|={mx:.1f}°（闸 {CAP_DEG}）  残差 {ep:.2f}mm/{er:.2f}°")
    print(f"  关节 {np.round(q,2).tolist()}")
    print(f"  法兰 {np.round(fc.T(q)[:3,3],1).tolist()}  (目标 {np.round(T_fl[:3,3],1).tolist()})")
    ring = fc.T(q)[:3, 3] + fc.T(q)[:3, :3] @ np.array([0.0, 0.0, V.RC])
    print(f"  环心 {np.round(ring,1).tolist()}  台面净空 {ring[2]-config.TABLE_Z:+.0f}mm")


if __name__ == "__main__":
    main()
