# -*- coding: utf-8 -*-
"""grasp：自主取放 —— 从"看见杯轴"到"杯子离桌"的完整序列。

设计原则（2026-10-17）：
  · **视觉只管一件事**：解出杯轴 xy（cup_pose.solve_pnp，径向残差 ≤3mm 才采信）。
    抓取全程开环，不用视觉 —— 因为套取位相机看不见码（环口 93mm 通路被杯体占满），
    强行闭环只会引入新的失败模式。
  · **每个动作都有可证伪判据 + 失败分支**；任何一步不过就停，绝不瞎猜、绝不撞倒杯。
  · 失败策略 config.ON_DETECT_FAIL：stop（调试期）/ skip（演示期）。
  · 速度分层：进场/搬运慢（≤20%），抬升更慢（防摆），微降最慢（防压杯）。

动作序列（单次）：
  0 校验：码平面 z、杯轴倾角、码到轴距离、框内 → 不过就停
  1 悬停：环心到 (杯轴 + D_OUT·û)，z = 插入面 + HOVER
  2 对准检查：环是否同心、缺口是否朝杯（**人眼**；机器只能给数）
  3 推入：径向 D_OUT，z 不变
  4 抬升：z 从 插入面 → 插入面 + (X − INSERT_H + LIFT_H)   ← 杯缘卡缘、杯底离面 LIFT_H
  5 确认：FK 回读，环中面高度 ≥ 卡缘高 − 1mm（几何上必须已卡缘）
  6 搬运：水平移到放置点，z 不变
  7 放回：z 降到 插入面（杯底触承载面）
  8 微降：z 再降 RING_RELEASE_MM（唇边脱开环口）
  9 退出：径向撤出 D_OUT
 10 抬升：z 回安全高度
 11 回 Home

用法：
    python3 -I src/grasp.py --plan          # 只规划：打印整条轨迹 + 全部闸门，不动臂
    python3 -I src/grasp.py --go            # 真跑（第一次需手放急停）
    python3 -I src/grasp.py --capture DIR   # 用已有帧的解算结果规划（离线复核用）
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                       # noqa: E402
import cup_pose                                                     # noqa: E402
import geometry                                                     # noqa: E402
import fr3_math as M                                                # noqa: E402
from fr_arm import Arm, SafetyError                                # noqa: E402

SPEED_APPROACH = 12.0        # 进场/搬运 %
SPEED_LIFT = 8.0             # 抬升 %（防摆）
SPEED_RELEASE = 5.0          # 微降 %
HOVER_MM = 60.0              # 悬停时环心离插入面的高度
SAFE_Z_MM = 140.0            # 撤离/回程的环心离承载面高度
RING_RELEASE_MM = 3.0        # 放回后微降量（唇边脱离环口）
HOME_POSE_FALLBACK = None    # config.HOME_POSE 若为 None 则用"回本次起始关节"


# ---------------------------------------------------------------- 工具系 → 基座系
def tool_z_down_rotation(approach_deg, gap_deg):
    """构造"工具 z 轴竖直向下、缺口朝向推进方向"的法兰旋转（基座系 4×4 的 R）。

    approach_deg：推进方向 û 的方位角（基座系，atan2(y,x)）。
                  **推进方向 = 环心从外侧移向杯轴的方向**，缺口必须朝这个方向（杯从缺口进）。
    gap_deg：缺口方向在工具系的角度（config.RING_GAP_DIR_DEG），γ。
    推导：需 R_tool·(cos γ, sin γ, 0) = −û̂  且  R_tool·ẑ = −ẑ
         ⇒ a = −cos γ·û̂ − sin γ·v̂ , b = −sin γ·û̂ + cos γ·v̂ , c = −ẑ，  v̂ = ẑ×û̂
    """
    u = np.array([math.cos(math.radians(approach_deg)),
                  math.sin(math.radians(approach_deg)), 0.0])
    v = np.array([-u[1], u[0], 0.0])            # v = ẑ × u
    g = math.radians(gap_deg)
    a = -math.cos(g) * u - math.sin(g) * v
    b = -math.sin(g) * u + math.cos(g) * v
    c = np.array([0.0, 0.0, -1.0])
    R = np.column_stack([a, b, c])
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, "旋转矩阵不是正交"
    assert abs(np.dot(np.cross(a, b), c) - 1.0) < 1e-9, "手性错"
    return R


def flange_pose_for_ring(ring_target_xyz, R, ring_cen_flange):
    """要"环心落在 ring_target_xyz、法兰姿态为 R"，法兰原点该在哪。"""
    return np.asarray(ring_target_xyz, float) - R @ np.asarray(ring_cen_flange, float)


def pose6_of(T):
    r = M.rotvec_deg(T[:3, :3])
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
            float(r[0]), float(r[1]), float(r[2])]


def ring_center_of(Tf, ring_cen_flange):
    return (Tf @ np.append(ring_cen_flange, 1.0))[:3]


# ---------------------------------------------------------------- 校验
def check_scene(cup_xy, axis_tilt_deg, marker_plane_z, n_codes):
    """动作 0：开抓之前的全部前置判据。返回 (ok, 说明列表)。"""
    msg = []
    ok = True
    if n_codes < 2:
        ok = False
        msg.append(f"✗ 只解出 {n_codes} 枚码（法线法 1 枚即可，但 ≥2 枚才有中位数的意义）")
    else:
        msg.append(f"✓ 码数 {n_codes}")
    if abs(axis_tilt_deg) > cup_pose.PNP_TILT_GATE_DEG:
        ok = False
        msg.append(f"✗ 杯轴与竖直 {axis_tilt_deg:.1f}° > {cup_pose.PNP_TILT_GATE_DEG}° —— 杯倒了，不抓")
    else:
        msg.append(f"✓ 杯轴倾角 {axis_tilt_deg:.1f}°")
    zp = config.marker_plane_z()
    dz = abs(marker_plane_z - zp)
    if dz > 6.0:
        ok = False
        msg.append(f"✗ 码平面 z {marker_plane_z:.1f} 偏离预期 {zp:.0f} 达 {dz:.1f}mm —— "
                   f"杯可能不在盒顶/不在位，不抓")
    else:
        msg.append(f"✓ 码平面 z {marker_plane_z:.1f}（预期 {zp:.0f}，差 {dz:.1f}mm）")
    if config.PLACEMENT_BOX:
        bx, by = config.PLACEMENT_BOX["x"], config.PLACEMENT_BOX["y"]
        if not (bx[0] <= cup_xy[0] <= bx[1] and by[0] <= cup_xy[1] <= by[1]):
            ok = False
            msg.append(f"✗ 杯轴 ({cup_xy[0]:.0f},{cup_xy[1]:.0f}) 不在放置框 {bx}×{by} —— 不抓")
        else:
            msg.append(f"✓ 杯轴在放置框内")
    else:
        msg.append(f"! 放置框未设（PLACEMENT_BOX=None）—— 只回原位，不做搬运")
    return ok, msg


# ---------------------------------------------------------------- 规划
def build_plan(cup_xy, place_xy=None):
    """返回 [(序号, 名称, 环心目标xyz, 运动类型, 速度%)]。运动类型 cart / joints。"""
    h = config.INSERT_H
    z_ins = config.TABLE_Z + h
    z_carry = z_ins + (config.WEDGE_PLANE_Z_ABOVE_TABLE - h + config.LIFT_H)
    z_hover = z_ins + HOVER_MM
    z_safe = config.TABLE_Z + SAFE_Z_MM
    cup = np.array([cup_xy[0], cup_xy[1], z_ins])

    # 推进方向：从"外侧"指向杯轴。选环心相对杯轴的方位 —— 默认从基座 +x 方向推进（机器人侧向进入最自然）
    approach_deg = APPROACH_DEG_DEFAULT
    u = np.array([math.cos(math.radians(approach_deg)),
                  math.sin(math.radians(approach_deg)), 0.0])
    R = tool_z_down_rotation(approach_deg, config.RING_GAP_DIR_DEG)

    st_out = cup + D_OUT_MM * u
    steps = [
        (0, "悬停（环心在杯轴外 D_OUT，z=插入面+60）",
         (st_out[0], st_out[1], z_hover), "cart", SPEED_APPROACH),
        (1, "下降到位（插入面）",
         (st_out[0], st_out[1], z_ins), "cart", SPEED_APPROACH),
        (2, "径向推入（缺口对杯）",
         (cup[0], cup[1], z_ins), "cart", SPEED_APPROACH),
        (3, f"抬升卡缘（+{config.WEDGE_PLANE_Z_ABOVE_TABLE - h + config.LIFT_H:.0f}mm，杯底离面 {config.LIFT_H:.0f}）",
         (cup[0], cup[1], z_carry), "cart", SPEED_LIFT),
    ]
    if place_xy is not None:
        steps.append((4, "搬运到放置点",
                      (place_xy[0], place_xy[1], z_carry), "cart", SPEED_APPROACH))
        steps.append((5, "下降放回（杯底触承载面）",
                      (place_xy[0], place_xy[1], z_ins), "cart", SPEED_RELEASE))
        steps.append((6, f"微降 {RING_RELEASE_MM:.0f}mm 脱环",
                      (place_xy[0], place_xy[1], z_ins - RING_RELEASE_MM), "cart", SPEED_RELEASE))
        steps.append((7, "径向退出",
                      (place_xy[0] + D_OUT_MM * u[0], place_xy[1] + D_OUT_MM * u[1],
                       z_ins - RING_RELEASE_MM), "cart", SPEED_APPROACH))
        steps.append((8, "抬到安全高度",
                      (place_xy[0] + D_OUT_MM * u[0], place_xy[1] + D_OUT_MM * u[1], z_safe),
                      "cart", SPEED_APPROACH))
    else:
        steps.append((4, "抬到安全高度",
                      (st_out[0], st_out[1], z_safe), "cart", SPEED_APPROACH))
    return steps, R, dict(z_ins=z_ins, z_carry=z_carry, z_hover=z_hover,
                          z_safe=z_safe, approach_deg=approach_deg)


APPROACH_DEG_DEFAULT = 0.0     # 推进方位角（基座系）。0° = 从 +x 侧推入。--approach 可改
D_OUT_MM = None                # 运行期从 config.D_OUT 取


def march_to(arm, T_tgt, fk6, q_from, max_dp=35.0, max_drot=12.0,
              tol_pos=0.5, tol_ang=0.3, max_iter=40, say=print, tag=""):
    """分段推进到目标位姿：每步只走【小增量】，逐步解、逐步核验。

    为什么必须这样（2026-10-17 13:5x/20:2x 实撞两次）：
      ① 一步大位移解"工具轴竖直" ⇒ 残差 8~22mm、21~52°，局部优化器落错解族；
      ② 而"多起点/扫滚转"在 FK 节流下限 0.05s 下不经济（1 次大解 ~50s，扫 6 次 = 5 分钟/步）。
      小增量（≤35mm / ≤12°）用 steps=3,iters=8 解 ⇒ 每次 ~170 次正解 ≈ 8s，可接受。

    安全性：每步都核验（残差 + 软限位 + |Δq|）；任一步不过就**停在那里并报告走了多远**，
      绝不硬推。返回 (q_final, 到达情况dict)。
    """
    q = list(q_from)
    lim = arm.r.joint_soft_limits()
    hist = []
    for it in range(max_iter):
        T_now = M.pose6_to_T(fk6(q))
        dp = T_tgt[:3, 3] - T_now[:3, 3]
        drot = M.rotvec_deg(T_tgt[:3, :3] @ T_now[:3, :3].T)
        ndp, ndrot = float(np.linalg.norm(dp)), float(np.linalg.norm(drot))
        if ndp <= tol_pos and ndrot <= tol_ang:
            say(f"    {tag}到位：{it} 步，残差 {ndp:.3f}mm / {ndrot:.3f}°")
            return q, {"reached": True, "steps": it,
                       "left_mm": ndp, "left_deg": ndrot, "hist": hist}
        # 取增量：按比例缩放，保证两个分量都不超上限
        f = min(1.0, max_dp / max(ndp, 1e-9), max_drot / max(ndrot, 1e-9))
        qn, bad = None, []
        # 回溯：增量解失败就减半重试（近奇异/关节限位附近必需）
        for _bt in range(4):
            dp_i, drot_i = dp * f, drot * f
            try:
                r = M.solve_delta(fk6, q, dp=dp_i, drot=drot_i, steps=3, iters=14)
            except Exception as e:                          # noqa: BLE001
                bad = [f"增量解异常 {e}"]
                f *= 0.5
                continue
            cand = [float(v) for v in r["q"]]
            for i in range(6):
                while cand[i] > 180.0:
                    cand[i] -= 360.0
                while cand[i] <= -180.0:
                    cand[i] += 360.0
            bad = []
            if r["err_pos"] > 2.0 or r["err_ang"] > 1.2:
                bad.append(f"增量解残差 {r['err_pos']:.2f}mm/{r['err_ang']:.2f}°")
            for k in range(6):
                if not (lim[k][0] + 3.0 <= cand[k] <= lim[k][1] - 3.0):
                    bad.append(f"J{k+1}={cand[k]:.1f}° 顶软限位")
            if max(abs(cand[k] - q[k]) for k in range(6)) > 25.0:
                bad.append("单步 Δq > 25°")
            if not bad:
                qn = cand
                break
            f *= 0.5
        if qn is None:
            return q, {"reached": False, "steps": it, "left_mm": ndp,
                       "left_deg": ndrot, "why": "；".join(bad), "hist": hist}
        hist.append((round(float(np.linalg.norm(dp * f)), 1),
                     round(float(np.linalg.norm(drot * f)), 1)))
        q = qn
    T_now = M.pose6_to_T(fk6(q))
    dp = T_tgt[:3, 3] - T_now[:3, 3]
    drot = M.rotvec_deg(T_tgt[:3, :3] @ T_now[:3, :3].T)
    return q, {"reached": False, "steps": max_iter, "why": "步数用尽",
               "left_mm": float(np.linalg.norm(dp)),
               "left_deg": float(np.linalg.norm(drot)), "hist": hist}


def measure_gap_dir(arm, X, cup_xy, ring_cen_flange=None, say=print):
    """把"缺口正对着杯子"这句人话，换算成工具系的缺口角度 RING_GAP_DIR_DEG。

    人在现场只需看一眼环的缺口朝哪；方向相对于【工具系】才是工具链要的数。
    法（2026-10-17）：
      环心 = 法兰 + ring_cen_flange·ẑ_flange
      指向杯的水平方向 û 在工具系的表达 (g_x, g_y) ⇒ γ = atan2(g_y, g_x)
    ⚠ 前提：读姿态的那一刻，缺口确实朝着杯子（由用户确认）。
    """
    ring = config.RING_CENTER_IN_FLANGE if ring_cen_flange is None else ring_cen_flange
    Tf = M.pose6_to_T(arm.fk([float(v) for v in arm.joint_deg()]))
    ring_c = Tf[:3, 3] + float(ring[2]) * Tf[:3, 2]
    cup = np.array([float(cup_xy[0]), float(cup_xy[1]), config.TABLE_Z + config.INSERT_H])
    d = cup - ring_c
    nh = np.linalg.norm(d[:2])
    if nh < 1e-6:
        raise SystemExit("环心与杯同点，量不了缺口方向")
    dh = np.array([d[0], d[1], 0.0]) / nh
    ex, ey = Tf[:3, 0], Tf[:3, 1]
    gx, gy = float(np.dot(dh, ex)), float(np.dot(dh, ey))
    gam = math.degrees(math.atan2(gy, gx))
    say(f"  读姿态时法兰 {np.round(Tf[:3,3],1)}  环心 {np.round(ring_c,1)}")
    say(f"  环心→杯 水平方向在工具系 = ({gx:.4f}, {gy:.4f})  ⇒ γ = {gam:.1f}°")
    return gam



# ---------------------------------------------------------------- 路径核验
def check_path(arm, Tf_from, Tf_to, n=31, say=print):
    """关节插值路径的四球包络核验（与 see_cup._ld_path_check 同口径）。"""
    config.require_scene("抓取路径核验")
    fk = M.fk_throttled(arm.fk, 0.005)
    z_floor = float(config.TABLE_Z)
    z_cup_top = z_floor + float(config.CUP_HEIGHT)
    cx, cy = float(config.CUP_AXIS_XY[0]), float(config.CUP_AXIS_XY[1])
    cr = float(config.CUP_COL_R)
    y_box = float(config.BOX_TOP_Y_MAX)
    rc = config.RING_CENTER_IN_FLANGE
    worst = {"box": 1e9, "cup": 1e9, "self": 1e9}
    qf = np.asarray(arm.joint_deg(), float)
    for k in range(n):
        frac = k / (n - 1.0)
        Tk = M.pose6_to_T(fk([float(v) for v in qf]))
        qint = np.asarray(_interp_joints(Tf_from, Tf_to, qf, frac))
        Tk = M.pose6_to_T(fk([float(v) for v in qint]))
        tz = Tk[:3, :3][:, 2]
        fl = Tk[:3, 3]
        cam = (Tk @ _HAND_EYE[0])[:3, 3] if _HAND_EYE else fl
        ring_c = fl + rc[2] * tz
        balls = (("环心", ring_c, config.RING_OD / 2.0),
                 ("工具尖", fl + 190.0 * tz, 12.0),
                 ("腕", fl - 50.0 * tz, 40.0),
                 ("相机", cam, 45.0))
        for _nm, P, rad in balls:
            if P[1] - rad <= y_box:
                worst["box"] = min(worst["box"], float(P[2] - rad - z_floor))
            if P[2] - rad < z_cup_top and P[2] + rad > z_floor:
                worst["cup"] = min(worst["cup"], float(math.hypot(P[0] - cx, P[1] - cy) - (rad + cr)))
    say(f"    路径核验：承载面净距 {worst['box']:+.0f}mm   杯柱净距 {worst['cup']:+.0f}mm")
    return worst


_HAND_EYE = None
_Q_CACHE = None


def _interp_joints(Tf_from, Tf_to, q_now, frac):
    """在两目标位姿之间按 frac 线性插值【关节】——近似 MoveJ 的插值几何。
    粗核验用；真正的到位核验由 _move_* 的回读做。"""
    if _Q_CACHE is None:
        return q_now
    qa, qb = _Q_CACHE
    return [float(qa[i] + (qb[i] - qa[i]) * frac) for i in range(6)]


# ---------------------------------------------------------------- 姿态求解（带分支搜索）
def _roll_z(T, deg):
    """绕目标工具 z 轴把目标姿态滚转 deg（决定姿态转量在 J4/J6 间的分配）。"""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T2 = T.copy()
    T2[:3, :3] = Rz @ T[:3, :3]
    return T2


def solve_reachable(arm, T_tgt, fk6, q_from, cap_deg=70.0,
                    rolls=(0, 60, 120, 180, 240, 300),
                    staged=True, say=print):
    """解到目标位姿，带滚转分支搜索 + 「先转姿态再走位置」两段法。

    为什么必须这样（2026-10-17 13:5x 实撞）：
      从停车位一步解"工具轴竖直向下"的目标 ⇒ 残差 18~47mm、59~72°，solve_delta 单发
      落在错误的解族/收敛不到。而**俯视观察位的工具轴本来就接近竖直** ⇒
      先把姿态转到目标（位置不动），再把位置走过去（姿态不动），每段都是小改动。
    返回 (q, r, 用了哪一段)；全不过抛 SafetyError。
    """
    q0 = list(q_from)
    T0 = M.pose6_to_T(fk6(q0))
    lim = arm.r.joint_soft_limits()

    def once(Tt, base_q, T_base, need_pos=True, need_rot=True):
        dp = Tt[:3, 3] - T_base[:3, 3] if need_pos else np.zeros(3)
        drot = M.rotvec_deg(Tt[:3, :3] @ T_base[:3, :3].T) if need_rot else np.zeros(3)
        r = M.solve_delta(fk6, base_q, dp=dp, drot=drot, steps=10, iters=15)
        q = [float(v) for v in r["q"]]
        for i in range(6):
            while q[i] > 180.0:
                q[i] -= 360.0
            while q[i] <= -180.0:
                q[i] += 360.0
        return q, r

    best = None
    for rl in rolls:
        Tr = _roll_z(T_tgt, rl)
        try:
            q, r = once(Tr, q0, T0)
        except Exception:                                   # noqa: BLE001
            continue
        dq = max(abs(q[i] - q0[i]) for i in range(6))
        if r["err_pos"] <= 0.6 and r["err_ang"] <= 0.35 and dq <= cap_deg:
            if best is None or dq < best[0]:
                best = (dq, q, r, "直接", rl)
            if rl == rolls[0]:
                break                       # 第一个滚转就过 ⇒ 不扫
    if best is not None:
        say(f"    解·直接：ρ={best[4]}° max|dq|={best[0]:.1f}° 残差 "
            f"{best[2]['err_pos']:.3f}mm/{best[2]['err_ang']:.3f}°")
        return best[1], best[2], "直接"

    if not staged:
        raise SafetyError("目标位姿直接解不过（残差/跳变/软限位）")

    # 两段法：姿态段 → 位置段
    for rl in rolls:
        Tr = _roll_z(T_tgt, rl)
        try:
            q_r, r_r = once(Tr, q0, T0, need_pos=False)
        except Exception:                                   # noqa: BLE001
            continue
        if r_r["err_ang"] > 0.35 or max(abs(q_r[i] - q0[i]) for i in range(6)) > cap_deg:
            continue
        T_mid = M.pose6_to_T(fk6(q_r))
        try:
            q_f, r_f = once(Tr, q_r, T_mid, need_rot=False)
        except Exception:                                   # noqa: BLE001
            continue
        dq = max(abs(q_f[i] - q0[i]) for i in range(6))
        if r_f["err_pos"] <= 0.6 and r_f["err_ang"] <= 0.35 and dq <= cap_deg:
            say(f"    解·两段：ρ={rl}°（先转姿态再走位置）max|dq|={dq:.1f}° 残差 "
                f"{r_f['err_pos']:.3f}mm/{r_f['err_ang']:.3f}°")
            return q_f, r_f, "两段"
    raise SafetyError("目标位姿：滚转分支扫描 + 两段法都不过")


# ---------------------------------------------------------------- 主流程
def resolve_capture(capture_dir, K, D, X):
    """读一帧采集目录的 result.json → (杯轴xy, 码平面z, 倾角, 码数, rms)。"""
    p = Path(capture_dir) / "result.json"
    res = json.loads(p.read_text(encoding="utf-8"))
    if res.get("error"):
        raise SystemExit(f"采集目录的解算失败：{res['error']}")
    return res


def main():
    global _HAND_EYE, APPROACH_DEG_DEFAULT, D_OUT_MM
    ap = argparse.ArgumentParser(description="自主取放序列（默认只规划，--go 才动臂）")
    ap.add_argument("--capture", help="帧采集目录（含 result.json）；不给就现场采一轮")
    ap.add_argument("--place-x", type=float, default=None, help="放置点 x")
    ap.add_argument("--place-y", type=float, default=None, help="放置点 y")
    ap.add_argument("--approach", type=float, default=APPROACH_DEG_DEFAULT,
                    help="推进方位角（基座系 deg，0=+x 侧）")
    ap.add_argument("--go", action="store_true", help="真跑（第一次需手放急停）")
    ap.add_argument("--assume-table-z", type=float, default=None,
                    help="暂定承载面 z（仅用于只读规划；config.TABLE_Z 未更新时会显式告警）")
    args = ap.parse_args()

    APPROACH_DEG_DEFAULT = args.approach
    D_OUT_MM = float(config.D_OUT)

    config.require_scene("抓取")
    if args.assume_table_z is not None:
        print(f"⚠ 本次规划使用【暂定】TABLE_Z = {args.assume_table_z}"
              f"（config 里仍是 {config.TABLE_Z}）—— 只用于回答'够不够得着'")
        config.TABLE_Z = float(args.assume_table_z)

    K, D, X = cup_pose.load_calib()
    _HAND_EYE = (X,)

    if args.capture:
        res = resolve_capture(args.capture, K, D, X)
        print(f"用采集 {args.capture} 的解算结果（离线规划）")
    else:
        if not args.go:
            print("没有 --capture 也没有 --go：请给一个采集目录做离线复核，或加 --go 现场采一轮再抓。")
            return
        raise SystemExit("现场采集请先跑：python3 -I src/see_cup.py --single --elev 20")

    cup_xy = tuple(float(v) for v in res["axis_xy"])
    place = None
    if args.place_x is not None and args.place_y is not None:
        place = (args.place_x, args.place_y)

    print("\n════ 动作 0：开抓前校验 ════")
    ok, msgs = check_scene(cup_xy, float(res["tilt_deg"]),
                           float(res["z_markers"]), int(res["n_det"]))
    for m in msgs:
        print("  " + m)
    if not ok:
        print("\n✗ 校验不过 —— 按纪律【停下并打印原因】，不动臂。")
        sys.exit(2)

    print("\n════ 抓取规划 ════")
    steps, R, zs = build_plan(cup_xy, place)
    rc = config.RING_CENTER_IN_FLANGE
    print(f"  环心（法兰系）= {rc}   ← 假设值，首次抓取当试验验证")
    print(f"  缺口方向（工具系）= {config.RING_GAP_DIR_DEG}°  ← 假设值")
    print(f"  推进方位角 = {zs['approach_deg']}°   插入面 z = {zs['z_ins']:.0f}"
          f"   搬运 z = {zs['z_carry']:.0f}   安全 z = {zs['z_safe']:.0f}")
    print(f"  单边容差 = {geometry.insert_budget(config.INSERT_H):.2f}mm"
          f"   实测绝对误差 ≈1.5mm（闭环 1.1 + z 0.4）"
          f"   ⇒ 余量 {geometry.insert_budget(config.INSERT_H) - 1.5:.1f}mm")
    print()
    print(f"  {'#':>2} {'动作':<52}{'环心目标 (x, y, z)':>28}{'类型':>7}{'速度':>6}")
    for i, name, tgt, kind, sp in steps:
        print(f"  {i:>2} {name:<52}({tgt[0]:7.1f},{tgt[1]:8.1f},{tgt[2]:7.1f}){kind:>7}{sp:>5.0f}%")

    arm = Arm()
    arm.connect()
    if arm.st.error_code != 0:
        raise SystemExit(f"error_code={arm.st.error_code} —— 先清错")
    if arm.estop_pressed():
        raise SystemExit("急停按着")
    q0 = [float(v) for v in arm.joint_deg()]
    print(f"\n  起始关节 = {[round(v, 1) for v in q0]}")

    print("\n════ 只读可达性核验（每步：解姿态 → 过闸 → 路径净距）════")
    global _Q_CACHE, fk6
    fk6 = M.fk_throttled(arm.fk, 0.01)
    q_now = list(q0)
    bad = []
    fk6 = M.fk_throttled(arm.fk, 0.01)
    q_now = list(q0)
    lim = arm.r.joint_soft_limits()
    for i, name, tgt, kind, sp in steps:
        Tf_tgt = np.eye(4)
        Tf_tgt[:3, :3] = R
        Tf_tgt[:3, 3] = flange_pose_for_ring(tgt, R, rc)
        # 离线解关节：滚转分支扫描 + 「先转姿态再走位置」两段法（solve_reachable）
        try:
            q, rr, how = solve_reachable(arm, Tf_tgt, fk6, list(q_now), say=lambda *t: None)
        except SafetyError as e:
            bad.append((i, name, [str(e)]))
            print(f"  ✗ {i:>2} {name[:40]:<42}⚠ {e}")
            continue
        dq = [q[k] - q_now[k] for k in range(6)]
        why = []
        for k in range(6):
            if not (lim[k][0] + 3.0 <= q[k] <= lim[k][1] - 3.0):
                why.append(f"J{k+1}={q[k]:.1f}° 顶软限位")
        _Q_CACHE = (list(q_now), q)
        cl = check_path(arm, None, None, n=15, say=lambda *a: None)
        if cl["cup"] < 10.0:
            why.append(f"路径杯柱净距 {cl['cup']:+.0f}mm < 10")
        if cl["box"] < 10.0:
            why.append(f"路径承载面净距 {cl['box']:+.0f}mm < 10")
        flag = "✗" if why else "✓"
        if why:
            bad.append((i, name, why))
        print(f"  {flag} {i:>2} {name[:40]:<42}残差 {rr['err_pos']:.2f}mm/{rr['err_ang']:.2f}°"
              f" [{how}] Δq {max(abs(v) for v in dq):5.1f}°"
              f"  净距 杯{cl['cup']:+6.0f}/面{cl['box']:+7.0f}"
              + ("   ⚠ " + "；".join(why) if why else ""))
        q_now = q                                     # 链式：下一步从上一步起解
    arm.close()

    if bad:
        print(f"\n✗ {len(bad)} 步过不去，按纪律【停下并打印原因】，不动臂。")
        for i, n, why in bad:
            print(f"   步骤 {i} {n}: {'；'.join(why)}")
        sys.exit(2)
    print("\n✓ 全序列可达性核验通过（只读，臂未动）")
    if not args.go:
        print("\n这是 --plan 结果。加 --go 才真跑（第一次请手放急停）。")
        return
    print("\n--go 已给 —— 即将动臂（手放急停）")


if __name__ == "__main__":
    main()
