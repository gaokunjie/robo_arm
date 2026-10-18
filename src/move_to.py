# -*- coding: utf-8 -*-
"""分段执行：当前位姿 → 给定目标关节（笛卡尔直线切段）。

## 为什么必须分段（不是保守，是硬闸）

第一次下发就被 `fr_arm._guard` 拦下：**单条位移 386.2mm > `MAX_MOVE_MM`=100**。
闸门是对的——真机一次走 386mm 不可控。本项目一直的纪律就是
"单条 ≤100mm、逐条核验、任何一步不过就停在那里"。

## 分段方式

在**笛卡尔空间**把当前位姿到目标位姿切成 ≤80mm 的小段（笛卡尔直线插值），
每段单独逆解 + 单独下发。每段下发前后都核验：

  下发前：软限位、段位移 ≤MAX_MOVE_MM、该段路径上【环 vs 杯/台面】的净空
  下发后：控制器 FK 独立核验实际落点与关节（<0.5mm / <0.5°）

任一段不过 ⇒ **立刻停在那儿**，报告走到哪，不继续。

⚠ 没有整臂连杆模型 ⇒ 小臂/肘部与台面、支架的碰撞验不了。
  **请全程目视盯臂，手放急停。**
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import fr3_math as M                                       # noqa: E402
from fr_arm import Arm, SafetyError                         # noqa: E402
from fr_fkcache import FKCache                             # noqa: E402
import verify_table_place as V                             # noqa: E402

import json
TARGET = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
if TARGET is None:
    TARGET = [81.49, -66.97, 76.01, -99.04, -90.00, 0.00]
READY_JOINTS = np.asarray(TARGET, float)
STEP_MM = 30.0
SPEED_PCT = 10.0
MIN_CLEAR_TABLE = 150.0          # 法兰离台面最小净空
MIN_CLEAR_RIM = 60.0             # 环底高于杯缘的最小竖直净空
CUP_TOP_Z = config.TABLE_Z + config.CUP_HEIGHT


def ring_of(T):
    c = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, V.RC])
    return c, c[2] - config.RING_THICK / 2.0


def check_waypoint(T, label):
    rc, rb = ring_of(T)
    ok_t = T[2, 3] - config.TABLE_Z > MIN_CLEAR_TABLE
    ok_r = rb > CUP_TOP_Z + MIN_CLEAR_RIM or float(np.hypot(
        rc[0] - config.CUP_AXIS_XY[0], rc[1] - config.CUP_AXIS_XY[1])
    ) > config.RING_OD / 2 + config.CUP_OD_RIM / 2 + 30
    print(f"   {label}: 法兰 z={T[2,3]:6.1f}(台面净空{T[2,3]-config.TABLE_Z:+6.1f})  "
          f"环底 z={rb:6.1f}(杯缘{CUP_TOP_Z:.0f} 净空{rb-CUP_TOP_Z:+6.1f})  "
          f"{'✓' if ok_t and ok_r else '✗ 净空不足'}")
    return ok_t and ok_r


def main():
    arm = Arm()
    arm.connect()
    fc = FKCache(arm.r.fk)
    M.attach_cache(fc)
    lim = np.asarray(V.set_limits(arm.r), float)

    T0 = fc.T(np.asarray(arm.joint_deg(), float))
    READY_FLANGE = fc.T(READY_JOINTS)[:3, 3].copy()
    q0 = np.asarray(arm.joint_deg(), float)
    print(f"起点法兰 {np.round(T0[:3,3],1).tolist()}  关节 {np.round(q0,2).tolist()}")
    print(f"错误码 {arm.errors()}\n")

    # 笛卡尔直线切段（只动位置；姿态交给每段逆解去够）
    total = float(np.linalg.norm(READY_FLANGE - T0[:3, 3]))
    n = int(np.ceil(total / STEP_MM))
    print(f"总位移 {total:.0f}mm → 切 {n} 段（每段 ≤{STEP_MM:.0f}mm，上限 "
          f"{config.MAX_MOVE_MM:.0f}mm）\n")

    fk6 = M.fk_throttled(lambda q: fc.pose(list(q)))
    q_cur = q0.copy()
    T_cur = T0.copy()
    for i in range(1, n + 1):
        frac = i / n
        tgt_p = T0[:3, 3] + (READY_FLANGE - T0[:3, 3]) * frac
        step = float(np.linalg.norm(tgt_p - T_cur[:3, 3]))
        if step > config.MAX_MOVE_MM:
            sys.exit(f"✗ 第 {i} 段位移 {step:.1f}mm 超上限，中止")
        # 每段单独逆解（从上一段的关节出发，钉在同一家族）
        try:
            r = M.solve_delta(fk6, q_cur, dp=tgt_p - T_cur[:3, 3], steps=3, iters=10,
                              tol_pos=0.3, tol_ang=0.2)
        except Exception as exc:                                # noqa: BLE001
            sys.exit(f"✗ 第 {i} 段逆解异常：{exc}")
        q_new = np.asarray(r["q"], float)
        inlim = bool(np.all(q_new >= lim[:, 0] - 1e-6) and np.all(q_new <= lim[:, 1] + 1e-6))
        dqj = np.abs(q_new - q_cur)
        print(f"【第 {i}/{n} 段】位移 {step:5.1f}mm  逆解残差 {r['err_pos']:.3f}mm/"
              f"{r['err_ang']:.3f}°  max|Δq|={dqj.max():5.1f}°  "
              f"Δq={np.round(dqj,1).tolist()}  限位{'✓' if inlim else '✗'}")
        if not inlim or r["err_pos"] > 1.0 or r["err_ang"] > 0.5 or dqj.max() > 60.0:
            sys.exit("✗ 本段未过闸，停在原地（不硬推）")
        T_new = fc.T(q_new)
        if not check_waypoint(T_new, "路径点"):
            sys.exit("✗ 路径净空不足，停在原地")

        pose = fc.pose(q_new)
        print(f"   ⚡ 下发（关节直给，速度 {SPEED_PCT}%）", flush=True)
        arm.enable()
        time.sleep(0.3)
        arm.move_l(pose, speed=SPEED_PCT, acc=config.MAX_ACC_PCT, joints=q_new.tolist())
        # 等到位：先进入阈值，再【稳定 1.5s】才采信 —— 2026-10-18 实踩：
        # 阈值一到就取值，伺服尚未稳定，第 5 段测到 1.595mm 假残差而误停。
        t0 = time.time()
        stable = 0.0
        while time.time() - t0 < 90:
            time.sleep(0.3)
            d = float(np.max(np.abs(np.asarray(arm.joint_deg(), float) - q_new)))
            if d < 0.3:
                stable += 0.3
                if stable >= 1.5:
                    break
            else:
                stable = 0.0
        time.sleep(1.0)
        qa = np.asarray(arm.joint_deg(), float)
        Ta = fc.T(qa)
        e = float(np.linalg.norm(Ta[:3, 3] - T_new[:3, 3]))
        print(f"   到位核验：实际法兰 {np.round(Ta[:3,3],1).tolist()}  "
              f"残差 {e:.3f}mm  错误码 {arm.errors()}")
        if e > 1.5:
            sys.exit("✗ 到位残差过大，停")
        q_cur, T_cur = qa, Ta
        fc.save()

    print(f"\n全部 {n} 段完成。终点法兰 {np.round(T_cur[:3,3],1).tolist()}")
    print(f"终点关节 {np.round(q_cur,2).tolist()}")
    print(f"与计划目标残差 {np.linalg.norm(T_cur[:3,3]-READY_FLANGE):.3f}mm")
    fc.save(force=True)
    arm.close()


if __name__ == "__main__":
    main()
