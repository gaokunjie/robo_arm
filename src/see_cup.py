# -*- coding: utf-8 -*-
"""see_cup：真机观察杯子 —— 原位 + ±平移采样 → 圆拟合出基座系杯轴。

流程：
    连臂（preflight：error_code=0 / 未急停 / motion_done=1 / FK⇄TCP 自洽）
    → 开相机（fr_cam 强制锁焦，锁不住会炸）
    → 原位拍一帧 → 8 个小关节点拍（J2/J3/J1/J5 ±几度，直接 move_j，不用逆解）
    → 每个点拍位拍一帧 → 回原位 → cup_pose 圆拟合 → 原图/json/标注图存盘

安全（铁律）：
    · 【绝不用控制器逆解】（GetInverseKin/GetInverseKinRef 是禁区，已 4 次打崩应用层）。
      move_l 不给 joints 会走 ik_ref —— 本脚本一律只走 move_j（关节原样下发）。
    · 每步运动前查 estop / error_code / motion_done；解残差 err_pos≤0.2mm、
      err_ang≤0.15°、max|dq|≤15° 不过就不动；到位回读关节核验 ≤0.5°、FK⇄目标 ≤1mm。
    · 速度 8%（config 上限 20 兜底）；finally 一定回原关节；相机 finally 关闭。

用法：
    python3 -I src/see_cup.py                # 原位 + 8 个关节点拍
    python3 -I src/see_cup.py --no-move      # 只拍原位一帧（先确认码在不在视野）
    python3 -I src/see_cup.py --tap 4        # 减小点拍幅度
    python3 -I src/see_cup.py --lookdown-plan  # 只读预演：解俯视观察位 + 全闸核验（不动臂）
    python3 -I src/see_cup.py --lookdown     # 先挪俯视观察位（相机抬高俯拍）再采——掠射根治
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
import fr3_math as M                                            # noqa: E402
import cup_pose                                                 # noqa: E402
from fr_arm import Arm, SafetyError                             # noqa: E402
from fr_cam import Cam                                          # noqa: E402
from detect_cup import Detector                                 # noqa: E402

SPEED = 8.0
ERR_POS_MM = 0.2
ERR_ANG_DEG = 0.15
MAX_DQ_DEG = 15.0
ARRIVE_DQ_DEG = 0.5
ARRIVE_POS_MM = 1.0

# ---- 俯视观察位（2026-10-16 决策；2026-10-17 锚点全部收敛到 config）----
# 瞄准位相机 z≈码面高 → 全部射线掠射、圆拟合病态。抬到码面上方、俯角 40° 拍
# （|d_z|≈0.64），杯定位恢复良态。~330mm 视距下码带 Ø72.5→~320px。
# 解算定案（2026-10-16 晚，全只读实测）：
#   · "最小转动保滚转"构造要法兰站到 ~730-761mm 高 → 超臂展（探针实测法兰 z 极限 ~748、
#     到肩枢极限 ~630-640），数值解全部停在 J3→0 伸展位解不动。
#   · 滚转 ρ 扫描：可及区（肩枢距≤600）全对应 |drot|≥92°；ρ=0（"正"滚转）时相机落在
#     法兰上方 ~90mm → 法兰 (−31.3,−256.7,604.5)、肩枢距 521 ✓，解收敛 0.123mm/0.264°。
#   · 该解 J6=+239° 超软限位——但 239 ≡ −121（同一物理位形），【折回 ±180】即合法：
#     ~166° 的大滚转本来就是腕的活（tool 轴 ≈ 视线，roll 大头落在 J6）。
#
# ⚠⚠ 2026-10-17：【LD_AIM 不再是常量】。它过去是一份 z 锚副本（482 = 442+40），
#   与 config.TABLE_Z(None)、cup_pose 的 442、_ld_path_check 的 442 合计四份、互相差
#   80~135mm。现在改为从 config 现场推导，且 config 未实测时**只允许只读预演**。
def ld_aim(strict=True):
    """瞄准点 = 杯轴 xy + 码面高。唯一出处 = config。

    config 未实测齐时：strict=True 直接拒绝；strict=False 退回历史值并**由调用方打红字**，
    只允许只读预演用（见 main() 的守卫）。
    """
    ready, _missing = config.scene_ready()
    if ready:
        return (float(config.CUP_AXIS_XY[0]), float(config.CUP_AXIS_XY[1]),
                config.marker_plane_z())
    if strict:
        config.require_scene("解算俯视观察位")
    return (7.7, -518.9, 482.0)        # ← 历史值，仅供只读预演排障，**不是测量值**


LD_ELEV_DEG = 40.0
LD_DIST_MM = 330.0
LD_STEPS = 10                     # solve_delta 分段（假臂实测 6~12 段即收敛到微米级）
LD_ITERS = 15                     # 每段牛顿迭代数（28 过度；15 已达 0.001mm 级）
LD_CAP_DEG = 60.0                 # 关节跳上限（大挪位放宽；防分支翻转的粗闸）
LD_CAP_EXEMPT = {5}               # J6 豁免（滚转主力，交软限位管——见上"折回"注）
LD_LIMIT_MARGIN = 3.0             # 软限位余量（与 he_collect 同）
LD_Q5_MIN = 12.0                  # |q5| 下限（腕部奇异带）
LD_J2_MIN = -150.0                # 深家族围栏（J2≈-200 撞支架，见 memory: fr3-elbow-geometry）
LD_RES_POS_MM = 0.6               # 数值解残差闸（与 he_collect 同口径）
LD_RES_ANG_DEG = 0.35
LD_PATH_CLEAR_MM = 10.0           # 路径核验最小净距（箱顶面 / 杯柱）

# 分支搜索时临时覆盖瞄点用（见 lookdown_plan_staged）。空 = 用 config 的值。
_LD_AIM_OVERRIDE = []
ROLL_SCAN = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 130.0, 160.0)


def lookdown_plan(arm, X, say=print, steps=LD_STEPS, iters=LD_ITERS,
                  elev=LD_ELEV_DEG, azim_abs=None,
                  cap_deg=LD_CAP_DEG, q_from=None, roll_deg=0.0, cam_target=None):
    """当前位姿 → 俯视观察位：返回 (q_tgt, T_expect, 解信息)。

    纯只读（读关节 + 节流正解 + 数值解 + 路径核验）。目标相机位姿：瞄点上方、
    仰角 elev、距离 LD_DIST_MM、相机方位角 azim_abs（基座系绕瞄点；None=沿用当前）；
    滚转取"正"（x 轴水平）。elev 可覆盖（CLI --elev）——降俯角可解码但圆拟合趋掠射；
    azim_abs 用于 orbit 扫不同方位的码（圆拟合需 ≥3 方位，单 pose 只解 1 个码）。

    q_from=None（默认）用实时关节；给了就用它当起点——链式 orbit 离线预演用，
    这样不必真动臂就能看清"上一方位解 → 下一方位"的真实 dq。
    """
    q_now = [float(v) for v in (arm.joint_deg() if q_from is None else q_from)]
    T0 = M.pose6_to_T([float(v) for v in arm.fk(q_now)])
    aim = np.asarray(_LD_AIM_OVERRIDE[0] if _LD_AIM_OVERRIDE
                      else ld_aim(strict=False), float)
    if azim_abs is None:
        TB0 = T0 @ X
        cam0 = TB0[:3, 3]
        az = math.atan2(cam0[0] - aim[0], cam0[1] - aim[1])   # 沿用当前方位
    else:
        az = azim_abs
    el = math.radians(elev)
    if cam_target is not None:
        cam1 = np.asarray(cam_target, float)        # 分段中转位：直接给相机目标点
    else:
        cam1 = aim + np.array([math.sin(az) * math.cos(el),
                               math.cos(az) * math.cos(el),
                               math.sin(el)]) * LD_DIST_MM
    z1 = (aim - cam1) / np.linalg.norm(aim - cam1)
    xax = np.cross(z1, [0.0, 0.0, 1.0])
    xax = xax / np.linalg.norm(xax)
    yax = np.cross(z1, xax)
    # 滚转自由度 ρ：绕视线 z1 转相机系 —— 它决定姿态转量在 J4 / J6 之间怎么分。
    # 2026-10-17 13:19 实撞：瞄点只挪 30mm，ρ 固定时求解器只能落到 J4 −84° 那族、被 cap 闸全拦；
    #   而上一次（瞄点差 30mm）同一 ρ 却给出 J4 −45° 的解。⇒ ρ 是必须扫的自由度，不是可省的细节。
    if abs(roll_deg) > 1e-9:
        cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        xr = cr * xax + sr * yax
        xax, yax = xr, np.cross(z1, xr)
    C = np.eye(4)
    C[:3, 0], C[:3, 1], C[:3, 2], C[:3, 3] = xax, yax, z1, cam1
    F = C @ np.linalg.inv(X)                                     # 目标法兰位姿
    dp = F[:3, 3] - T0[:3, 3]
    drot = M.rotvec_deg(F[:3, :3] @ T0[:3, :3].T)
    say(f"— 俯视位规划：相机 → {np.round(cam1, 1)}"
        + (f"  滚转ρ={roll_deg:g}°" if abs(roll_deg) > 1e-9 else "")
        + f"  视线 {np.round(z1, 3)}（俯角 {elev:.0f}°，|d_z|={abs(z1[2]):.2f}）")
    say(f"  法兰 Δp {np.round(dp, 1)}（|{np.linalg.norm(dp):.0f}mm）  目标法兰到肩枢"
        f" {np.linalg.norm(F[:3, 3] - [0, 0, 152.0]):.0f}mm  姿态转 {np.linalg.norm(drot):.1f}°"
        f"  分段 {steps}·iters {iters}（只读正解）")
    fk6 = M.fk_throttled(arm.fk, 0.01)
    r = M.solve_delta(fk6, q_now, dp=dp, drot=drot, steps=steps, iters=iters)
    q = [float(v) for v in r["q"]]
    for i in range(6):                                           # 折回 (−180, 180]
        while q[i] > 180.0:
            q[i] -= 360.0
        while q[i] <= -180.0:
            q[i] += 360.0
    dq = [q[i] - q_now[i] for i in range(6)]
    say(f"  数值解：残差 {r['err_pos']:.3f}mm / {r['err_ang']:.3f}°"
        f"  iters={r.get('iters')}  worst_step={r.get('worst_step')}")
    say("  Δq = " + "  ".join(f"J{i + 1}{dq[i]:+.1f}°" for i in range(6))
        + "（J6 已折回 ±180 等价位）")
    why = []
    if r["err_pos"] > LD_RES_POS_MM or r["err_ang"] > LD_RES_ANG_DEG:
        why.append(f"残差 {r['err_pos']:.2f}mm/{r['err_ang']:.2f}° 超闸")
    for i in range(6):
        if i not in LD_CAP_EXEMPT and abs(dq[i]) > cap_deg:
            why.append(f"J{i + 1} 跳 {dq[i]:+.1f}° > {cap_deg:g}°")
    lim = arm.r.joint_soft_limits()
    for i in range(6):
        lo, hi = lim[i]
        if not (lo + LD_LIMIT_MARGIN <= q[i] <= hi - LD_LIMIT_MARGIN):
            why.append(f"J{i + 1}={q[i]:.1f}° 顶软限位 [{lo:.0f},{hi:.0f}]")
    if abs(q[4]) < LD_Q5_MIN:
        why.append(f"|q5|={abs(q[4]):.1f}° < {LD_Q5_MIN:g}°（腕部奇异带）")
    if q[1] < LD_J2_MIN:
        why.append(f"J2={q[1]:.1f}° < {LD_J2_MIN:g}°（深家族，撞支架那族）")
    if why:
        raise SafetyError("俯视位解不过闸：" + "；".join(why))
    cl = _ld_path_check(arm, q_now, q, X, say=say)               # 关节插值路径净距核验
    if cl["box"] < LD_PATH_CLEAR_MM or cl["cup"] < LD_PATH_CLEAR_MM:
        why.append(f"路径净距 箱顶 {cl['box']:.0f}mm / 杯柱 {cl['cup']:.0f}mm "
                   f"< {LD_PATH_CLEAR_MM:g}mm")
        raise SafetyError("俯视位解不过闸：" + "；".join(why))
    say("  ✓ 闸全过（残差/关节跳/软限位/|q5|/J2 家族/路径净距）")
    return q, r["Tt"], r


def _solve_one(arm, X, aim, elev, azim_abs, cap_deg, q_from, say, roll_deg=0.0,
              cam_target=None):
    """解一次俯视位并过闸；返回 (q, r) 或 (None, 失败原因列表)。"""
    old = _LD_AIM_OVERRIDE
    _LD_AIM_OVERRIDE.clear()
    _LD_AIM_OVERRIDE.append(tuple(aim))
    try:
        q, _T, r = lookdown_plan(arm, X, elev=elev, azim_abs=azim_abs,
                                 cap_deg=cap_deg, q_from=q_from,
                                 roll_deg=roll_deg, cam_target=cam_target,
                                 say=lambda *a: None)
        return (q, r), []
    except SafetyError as e:
        return None, [str(e)]
    except Exception as e:                                   # noqa: BLE001
        return None, [f"{type(e).__name__}: {e}"]
    finally:
        _LD_AIM_OVERRIDE.clear()
        if old:
            _LD_AIM_OVERRIDE.append(old[0])


def lookdown_plan_staged(arm, X, elev=LD_ELEV_DEG, azim_abs=None, cap_deg=LD_CAP_DEG,
                         say=print, X_aim=None):
    """到俯视观察位，**带 IK 分支搜索**。

    为什么需要（2026-10-17 13:19 实撞）：
      瞄点只挪 30mm，臂却从"能过闸的解"跳到"过不了闸的解"（J4 −44.8° → −84.1°，超 cap 60°）。
      求解器是局部优化（solve_delta），从停车位起解时落进了**另一个等价解族**（肘上/肘下）。
      ⇒ 解不出来 ≠ 不可达。正确做法：多个起点各解一次，取**max|dq| 最小**的那个。

    起点集：**滚转 ρ ∈ {0,20,40,60,80,100,130,160}°**（ρ 决定姿态转量在 J4/J6 间的分配，
    是真正缺的那个自由度）；每个 ρ 从当前关节起解。
    每个起点仍要过 lookdown_plan 的全部闸（残差/跳变/软限位/|q5|/J2 家族/路径净距）。
    """
    aim = ld_aim(strict=False) if X_aim is None else tuple(X_aim)
    q_now = [float(v) for v in arm.joint_deg()]
    cands, whys = [], []
    # ⚠ 提前退出：先试 ρ=0（绝大多数情况它就对），过了立刻返回。
    #   全 8 连扫在 FK 节流下限 0.05s 下要 ~26 分钟（1 解 ≈ 3920 次正解），不能当常规路径。
    for roll in ROLL_SCAN:
        q0 = list(q_now)
        try:
            ok, r = _solve_one(arm, X, aim, elev, azim_abs, cap_deg, q0, say,
                               roll_deg=roll)
        except Exception as e:                               # noqa: BLE001
            whys.append(f"起点({d3:+},{d4:+},{d5:+}) 求解异常 {e}")
            continue
        if ok is None:
            whys.append(f"ρ={roll:g}°: {r[0]}")
            continue
        q, res = ok
        dq = max(abs(q[i] - q_now[i]) for i in range(6))
        cands.append((dq, q, res, roll))
        if roll == 0.0:
            break                      # ρ=0 就过 ⇒ 不扫了
    if not cands:
        # 第二层回退：沿"当前相机 → 目标相机"的直线取中转位，分两段物理走过去。
        # 依据（2026-10-17 13:19 实撞）：直接解要么落到过不了 cap 的分支（J4 −84°），
        #   要么（cap 放宽后）落到会穿过盒顶 35mm 的解 —— 两者都不是"不可达"，
        #   而是"一步跳不过去"。中转位每段只走一半，路径贴着已验证的走廊。
        T_now = M.pose6_to_T([float(v) for v in arm.fk(arm.joint_deg())])
        cam_now = (T_now @ X)[:3, 3]
        aim_v = np.asarray(aim, float)
        d = cam_now - aim_v
        n = float(np.linalg.norm(d))
        cam_goal = aim_v + d * (LD_DIST_MM / n)      # 与 lookdown_plan 默认方位同解
        for frac in (0.75, 0.65, 0.55, 0.50, 0.45, 0.35):
            tgt = cam_now + frac * (cam_goal - cam_now)
            ok, r = _solve_one(arm, X, aim, elev, azim_abs, cap_deg,
                               [float(v) for v in arm.joint_deg()], say,
                               roll_deg=0.0, cam_target=tgt)
            if ok is not None:
                q, res = ok
                say(f"— 直接解过不去 → 改【两段走】：先到中转位（相机 "
                    f"{np.round(tgt, 1)}，距目标相机 "
                    f"{np.linalg.norm(cam_goal - tgt):.0f}mm），再从那儿解一次")
                return q, res, False
            whys.append(f"中转位 frac={frac}: {r[0] if r else '?'}")
        raise SafetyError("俯视位【所有 IK 起点 + 两段中转都过不了闸】：\n    "
                          + "\n    ".join(whys[-6:]))
    cands.sort(key=lambda c: c[0])
    dq, q, res, roll = cands[0]
    say(f"— 分支搜索：{len(cands)}/8 个滚转ρ过闸，选 max|dq| 最小的 "
        f"（ρ={roll:g}°，max|dq|={dq:.1f}°，候选 {[(c[3], round(c[0], 1)) for c in cands[:4]]}）")
    say(f"  残差 {res['err_pos']:.3f}mm / {res['err_ang']:.3f}°")
    return q, res, True


def _ld_path_check(arm, q_from, q_to, X, say=print, n=31):
    """关节线性插值路径 FK 核验（MoveJ 的实际插值≈此几何）。

    四个球包络保守跟踪：工具尖(+190mm 沿工具轴)、腕(法兰−50mm)、环(法兰+130mm, r56.5)、
    相机。对两个场景体：**承载面**（y ≤ BOX_TOP_Y_MAX 的盒区，平面 z = TABLE_Z，
    高度取自 config —— 2026-10-17 之前这里是硬编码 442）与**杯柱**
    （轴 CUP_AXIS_XY、半径 config.CUP_COL_R、z 从 TABLE_Z 到杯口）。
    返回最小净距 {box, cup}（mm）。

    ⚠ 2026-10-17 修掉两个错：(1) 承载面高度不再硬编码 —— 旧代码假定 z=442，
      若真实承载面在 ~360，则净距被【虚报 +82mm】，闸形同虚设；
      (2) 旧代码的杯柱轴 (−13,−530) 与当时瞄点 xy (7.7,−518.9) 差 23mm，
      等于校验了一个不存在的位置。现全部走 config。
    """
    ready, missing = config.scene_ready()
    if not ready:
        raise SafetyError(f"场景常数未实测齐（缺 {'、'.join(missing)}）——路径核验无法进行；"
                          "把旧的硬编码 442 当承载面会虚报净距，宁可拒绝")
    fk = M.fk_throttled(arm.fk, 0.005)
    qf, qt_ = np.asarray(q_from, float), np.asarray(q_to, float)
    z_floor = float(config.TABLE_Z)                              # 承载面 = 盒顶
    z_cup_top = z_floor + float(config.CUP_HEIGHT)               # 杯口
    cx, cy = float(config.CUP_AXIS_XY[0]), float(config.CUP_AXIS_XY[1])
    cr = float(config.CUP_COL_R)
    y_box = float(config.BOX_TOP_Y_MAX)
    worst = {"box": 1e9, "cup": 1e9}
    for k in range(n):
        q = qf + (qt_ - qf) * (k / (n - 1.0))
        Tk = M.pose6_to_T(fk([float(v) for v in q]))
        tz = Tk[:3, :3][:, 2]
        fl = Tk[:3, 3]
        cam = (Tk @ X)[:3, 3]
        for _nm, P, rad in (("尖", fl + 190.0 * tz, 12.0), ("腕", fl - 50.0 * tz, 40.0),
                            ("环", fl + 130.0 * tz, 56.5), ("相机", cam, 45.0)):
            if P[1] - rad <= y_box:
                worst["box"] = min(worst["box"], float(P[2] - rad - z_floor))
            if P[2] - rad < z_cup_top and P[2] + rad > z_floor:
                d = math.hypot(P[0] - cx, P[1] - cy) - (rad + cr)
                worst["cup"] = min(worst["cup"], float(d))
    say(f"  路径核验（{n} 点插值 FK，四球包络）：净距 承载面 {worst['box']:+.0f}mm、"
        f"杯柱 {worst['cup']:+.0f}mm   [承载面 z={z_floor:.0f} 杯轴=({cx:.0f},{cy:.0f}) "
        f"r={cr:.1f}]")
    return worst


def _preflight(arm, say=print):
    if arm.st.error_code != 0:
        raise SafetyError(f"error_code={arm.st.error_code} —— 先清错（python3 -I src/clear_error.py）再动")
    if arm.estop_pressed():
        raise SafetyError("急停按着 —— 松开再说")
    if arm.st.motion_done != 1:
        raise SafetyError("motion_done≠1（上一次运动没到位）—— 别叠指令")
    q = arm.joint_deg()
    fk = arm.fk(q)
    tcp = arm.tcp_pose()
    d = max(abs(a - b) for a, b in zip(fk, tcp))
    if d > 0.5:
        say(f"⚠ FK⇄TCP 差 {d:.3f}（mm/deg 混比）超 0.5 —— 停下问问怎么回事")
        raise SafetyError("FK 与 TCP 不自洽")


def _fk6(arm):
    def f(q):
        return [float(v) for v in arm.fk([float(x) for x in q])]
    return f


def _move_j_checked(arm, q_tgt, say=print, speed=SPEED, T_expect=None, timeout=45.0):
    """move_j + 到位核验（回读关节轮询 + 可选 FK⇄目标核验）。

    ⚠ 不信 move_j 内部的 wait_done：新指令刚发出时 8083 的 motion_done 还是
    「上一次的 1」→ 竞态直接放行（2026-10-16 实测 J2 点拍被放行后回读差 6.00°）。
    这里用关节回读轮询到 0.05° 两连稳 才算到位。

    ⚠ T_expect 必须是【指令关节自己的 FK】，不是求解器吐的 Tt：
      求解器残差闸放到 0.35°（lookdown_plan），若到位闸还拿 0.3° 去比 Tt，
      求解器允许的误差比到位闸还大 → 必然自相矛盾卡死（2026-10-16 23:41 orbit[3] 实测）。
      职责拆开：lookdown_plan 负责"解够不够准"，这里负责"臂真到没到指令位形"。
    """
    _preflight(arm, say)
    arm.move_j([float(v) for v in q_tgt], speed=speed)
    t0 = time.time()
    q_act = None
    while time.time() - t0 < timeout:
        q_act = arm.joint_deg()
        if max(abs(a - b) for a, b in zip(q_act, q_tgt)) <= 0.05:
            time.sleep(0.25)
            q_act2 = arm.joint_deg()
            if max(abs(a - b) for a, b in zip(q_act2, q_tgt)) <= 0.05:
                q_act = q_act2
                break
        time.sleep(0.1)
    else:
        raise SafetyError(f"等了 {timeout:g}s 没到位（关节回读 {[round(v,2) for v in q_act]}）")
    dq = max(abs(a - b) for a, b in zip(q_act, q_tgt))
    if dq > ARRIVE_DQ_DEG:
        raise SafetyError(f"到位核验不过：回读关节与目标差 {dq:.2f}°")
    if T_expect is not None:
        Ta = M.pose6_to_T(arm.fk(q_act))
        dp = float(np.linalg.norm(Ta[:3, 3] - T_expect[:3, 3]))
        da = float(np.linalg.norm(M.rotvec_deg(Ta[:3, :3] @ T_expect[:3, :3].T)))
        if dp > ARRIVE_POS_MM or da > 0.3:
            raise SafetyError(f"到位核验不过：FK 位置差 {dp:.2f}mm / 姿态差 {da:.2f}°")
        say(f"      · 到位 ✓  关节差 {dq:.3f}°  位置差 {dp:.2f}mm")
    else:
        say(f"      · 到位 ✓  关节差 {dq:.3f}°")
    if arm.st.error_code != 0:
        raise SafetyError(f"运动后 error_code={arm.st.error_code}")
    return q_act


def take_shot(arm, cam, det, tag, outdir, say=print):
    """拍一帧并落盘（图片 + pose6/joints 元数据），返回 shot dict。"""
    frame = cam.fresh(say=say)
    p1 = arm.tcp_pose()
    time.sleep(0.15)
    p2 = arm.tcp_pose()
    drift = max(abs(a - b) for a, b in zip(p1, p2))
    if drift > 0.02:
        raise RuntimeError(f"拍帧时臂在动（0.15s 内位姿漂 {drift:.3f}）——帧作废")
    q = arm.joint_deg()
    fk_tcp = max(abs(a - b) for a, b in zip(arm.fk(q), p1))
    imgp = outdir / f"{tag}.jpg"
    cv2.imwrite(str(imgp), frame)
    dets = cup_pose.detect_ex(frame, det)
    meta = {"image": imgp.name, "pose6": [float(v) for v in p1],
            "joints": [float(v) for v in q], "tag": tag, "time": time.time(),
            "fk_tcp_diff": float(fk_tcp), "drift": float(drift)}
    (outdir / f"{tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    ids = [d["id"] for d in dets]
    span = "/".join(f"{d['span_px']:.0f}" for d in dets)
    say(f"      · [{tag}] 检出 {len(dets)} 码 ids={sorted(ids)} span={span}px"
        + (f"  FK⇄TCP 差 {fk_tcp:.3f}" if fk_tcp > 0.1 else ""))
    return {"T": M.pose6_to_T(p1), "dets": dets, "tag": tag}


def annotate(imgp, dets, res, T, X, K, D, outp):
    img = cv2.imread(str(imgp))
    for d in dets:
        cv2.polylines(img, [np.int32(d["corners"])], True, (0, 0, 255), 3)
        c = np.int32(d["center"])
        cv2.circle(img, tuple(c), 6, (0, 255, 255), -1)
    if res.get("axis_xy") and not res.get("error"):
        ax, ay = res["axis_xy"]
        zm = res["z_markers"]
        TB = T @ X
        iv = np.linalg.inv(TB)
        for zz, col in ((zm, (0, 255, 0)), (zm + config.CUP_HEIGHT, (255, 0, 0))):
            Pb = iv @ np.array([ax, ay, zz, 1.0])
            if Pb[2] <= 0:
                continue
            uv, _ = cv2.projectPoints(Pb[:3], np.zeros(3), np.zeros(3), K, D)
            u, v = uv.ravel()
            cv2.drawMarker(img, (int(u), int(v)), col, cv2.MARKER_CROSS, 60, 5)
        cv2.putText(img, "cup axis", (int(u) + 30, int(v) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 0), 4)
    cv2.imwrite(str(outp), img)


def main():
    ap = argparse.ArgumentParser(description="观察杯子 → 圆拟合杯轴（速度 8%，不用控制器逆解）")
    ap.add_argument("--tap", type=float, default=6.0, help="关节点拍幅度 deg（默认 6；J1/J5 自动取 0.6/0.7 倍）")
    ap.add_argument("--lookdown", action="store_true",
                    help="先挪到俯视观察位（相机抬到码面上方、俯角 40°）再拍——掠射根治")
    ap.add_argument("--lookdown-plan", action="store_true",
                    help="只读预演：解俯视位 + 全闸核验，不动臂")
    ap.add_argument("--orbit-plan", action="store_true",
                    help="只读链式预演：按 orbit 方位逐个离线串解，看整条 dq 分布，不动臂")
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--elev", type=float, default=LD_ELEV_DEG,
                    help=f"俯视俯角 deg（默认 {LD_ELEV_DEG}；降到 ~20 可解码，代价是圆拟合掠射）")
    ap.add_argument("--azim-abs", type=float, default=None,
                    help="相机绝对方位角 deg（基座系绕瞄点；默认=当前方位）")
    ap.add_argument("--orbit", action="store_true",
                    help="绕杯 orbit：在 ±span/2 内取 n 个方位各拍一帧，解不同方位的码")
    ap.add_argument("--orbit-span", type=float, default=120.0, help="orbit 总方位扫掠 deg（默认 120）")
    ap.add_argument("--orbit-n", type=int, default=5, help="orbit 方位数（默认 5）")
    ap.add_argument("--orbit-azims", default=None,
                    help="直接给绝对方位角（逗号分隔，如 -50,-30,0,20,40,50）；"
                         "给定则忽略 span/n——扫过可达图后手工挑，比盲扫稳")
    ap.add_argument("--cap", type=float, default=LD_CAP_DEG,
                    help=f"关节跳闸上限 deg（默认 {LD_CAP_DEG}；orbit 绕杯需大摆手，可放到 ~140）")
    ap.add_argument("--no-move", action="store_true", help="只拍原位一帧，不做任何运动")
    ap.add_argument("--single", action="store_true",
                    help="【新默认】俯视观察位 + 2 个小关节点（小摆，不绕杯）→ solve_pnp 定杯轴。"
                         "法线法下单枚码即可定轴，小点拍只为多拿几枚码取中位数")
    ap.add_argument("--legacy", action="store_true",
                    help="走旧的点拍/圆拟合路线（诊断对照用）")
    ap.add_argument("--out", help="输出目录（默认 data/cup_shots/<时间戳>）")
    args = ap.parse_args()

    # ---- 2026-10-17 守卫：只读预演放行、动臂拦截 ----
    ready, missing = config.scene_ready()
    readonly = bool(args.orbit_plan or args.lookdown_plan)
    if not ready:
        msg = (f"⚠⚠ 场景常数未实测齐：缺 {'、'.join(missing)}"
               "（C1 批次 C 的产出：量盒顶高度 + 杯轴水平位置）。")
        if readonly:
            print(msg + "\n   本次是【只读预演】，允许跑；但下面所有数字都是"
                         "【未校准锚点】下的，仅供排查，不能当作可行性结论。")
        else:
            print(msg + "\n   本次会【动臂】——拒绝执行。先量 TABLE_Z 回填 config。")
            sys.exit(2)

    outdir = Path(args.out) if args.out else (
        config.DATA_DIR / "cup_shots" / time.strftime("%Y%m%d_%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录 {outdir}")

    arm = Arm()
    arm.connect()
    _preflight(arm)
    print(f"已连接 {arm.ip}  急停={arm.estop_pressed()}  error={arm.st.error_code}")
    K, D, X = cup_pose.load_calib()
    if args.orbit_plan:
        print("— orbit 链式只读预演（离线串解，臂没动）")
        q_now = [float(v) for v in arm.joint_deg()]
        print(f"起点关节 {[round(v, 1) for v in q_now]}")
        _T0 = M.pose6_to_T([float(v) for v in arm.fk(q_now)])
        _cam0 = (_T0 @ X)[:3, 3]
        _aim = np.asarray(ld_aim(strict=False), float)
        base_az = math.degrees(math.atan2(_cam0[0] - _aim[0], _cam0[1] - _aim[1]))
        print(f"基准方位角 {base_az:.1f}°  俯角 {args.elev:.0f}°  cap {args.cap:g}°")
        if args.orbit_azims:
            azs = [float(s) for s in args.orbit_azims.split(",")]
        else:
            offs = [(-args.orbit_span / 2.0) + args.orbit_span * i / (args.orbit_n - 1.0)
                    for i in range(args.orbit_n)]
            azs = [base_az + o for o in offs]
        ok = 0
        chain = q_now
        for k, azd in enumerate(azs):
            az = math.radians(azd)
            print(f"— 方位[{k}] {azd:+.1f}°  (从上一解串起)")
            try:
                q_p, _T, _r = lookdown_plan(arm, X, elev=args.elev, azim_abs=az,
                                            cap_deg=args.cap, q_from=chain)
            except SafetyError as e:
                print(f"  ✗ 过不去：{e}")
                continue
            dq = [q_p[i] - chain[i] for i in range(6)]
            print(f"  dq vs 上一解 = " + "  ".join(f"J{i+1}{dq[i]:+.1f}°" for i in range(6)))
            chain = q_p
            ok += 1
        print(f"（--orbit-plan：链上 {ok}/{len(azs)} 个方位过闸，臂没动）")
        arm.close()
        return
    if args.lookdown_plan:
        print("— 俯视位只读预演（只读正解，不发任何运动指令）")
        _q, _r, _fin = lookdown_plan_staged(arm, X, elev=args.elev,
                                           azim_abs=args.azim_abs, cap_deg=args.cap)
        print("（--lookdown-plan：到此为止，臂没动）")
        arm.close()
        return
    cam = Cam()
    det = Detector()
    q0 = arm.joint_deg()
    print(f"原关节 {[round(v, 2) for v in q0]}")

    # 基座系里相机绕瞄点的方位角（orbit 的基准）
    _T0 = M.pose6_to_T([float(v) for v in arm.fk(q0)])
    _cam0 = (_T0 @ X)[:3, 3]
    _aim = np.asarray(ld_aim(strict=False), float)
    base_az = math.degrees(math.atan2(_cam0[0] - _aim[0], _cam0[1] - _aim[1]))
    print(f"基准方位角 {base_az:.1f}°  俯角 {args.elev:.0f}°")

    def _az_abs(off):
        # off=None → 单 pose（默认当前方位，--azim-abs 可覆盖）；off=数值 → orbit 绝对方位
        if off is None:
            return None if args.azim_abs is None else math.radians(args.azim_abs)
        return math.radians(base_az + off)

    moved = False
    shots = []
    try:
        if args.orbit:
            if args.orbit_azims:
                azs = [math.radians(float(s)) for s in args.orbit_azims.split(",")]
                names = [f"az{math.degrees(a):+.0f}" for a in azs]
            else:
                offs = [(-args.orbit_span / 2.0) + args.orbit_span * i / (args.orbit_n - 1.0)
                        for i in range(args.orbit_n)]
                azs = [_az_abs(o) for o in offs]
                names = [f"{base_az + o:+.0f}" for o in offs]
            for k, (az, nm) in enumerate(zip(azs, names)):
                print(f"— orbit[{k}] 绝对方位 {nm}°")
                try:
                    q_p, T_p, _r = lookdown_plan(arm, X, elev=args.elev, azim_abs=az,
                                                cap_deg=args.cap)
                except SafetyError as e:
                    print(f"  ✗ 跳过（{e}）")
                    continue
                moved = True
                # ⚠ 2026-10-16 23:5x：原先只有 lookdown_plan 被 try 包住，
                # 运动/拍照阶段一抛异常就整批报废（前面几个方位的帧白拍）。
                # 每个方位独立成败——单点失败只丢该方位的帧，不影响其余。
                try:
                    _move_j_checked(arm, q_p, speed=args.speed,
                                    T_expect=M.pose6_to_T(arm.fk(q_p)), timeout=150.0)
                    shots.append(take_shot(arm, cam, det, f"orb{k}_{nm}", outdir))
                except (SafetyError, RuntimeError) as e:
                    print(f"  ✗ 该方位放弃（运动/拍照失败）：{e}")
                    # 不回原位：下一方位的 lookdown_plan 读实时关节，会从当前位置重解，
                    # 强行回 q0 反而多一次无谓大摆。
            if len(shots) < 3:
                print(f"⚠ 只拿到 {len(shots)} 帧（<3，圆拟合不够）")
        else:
            # ============ 【新默认】单帧路线：俯视观察位 + 2 个小点拍 ============
            # 2026-10-17：新架构（法线法）下单枚码即可定杯轴，不需要 orbit 绕方位，
            #   也不需要旧的 8 连大点拍。小点拍只为多看见几枚码取中位数。
            use_single = args.single or not args.legacy
            if args.lookdown or use_single:
                # 分段走：若直接解过不去，lookdown_plan_staged 会给一个中转位；
                # 走到中转位后再解一次（每段只走一半，路径贴着已验证的走廊）。
                for _stage in range(4):
                    q_ld, _res, final = lookdown_plan_staged(
                        arm, X, elev=args.elev, azim_abs=_az_abs(None),
                        cap_deg=args.cap)
                    if final:
                        break
                    moved = True
                    _move_j_checked(arm, q_ld, speed=args.speed,
                                    T_expect=M.pose6_to_T(arm.fk(q_ld)), timeout=150.0)
                    print(f"  ✓ 已到中转位 {[round(v, 1) for v in q_ld]}")
                print(f"— 已在俯视观察位（俯角 {args.elev:.0f}°，瞄点 "
                      f"{tuple(round(v, 1) for v in ld_aim())}）")
            print("— 原位帧")
            shots.append(take_shot(arm, cam, det, "home", outdir))
            if args.no_move:
                print("（--no-move：到此为止）")
            elif use_single:
                # 小点拍只为多看见几枚码取中位数（法线法 1 枚即可定轴）。
                # 选轴依据（2026-10-17 只读核过目标位 [19.91,−146.6,132.78,−116.12,−23.49,−134.7]）：
                #   J3 距软限位只剩 27° → 不用；J2/J5 距限位 >100°，且 |q5| 保持 ≥19.5°（奇异带闸是 12）
                #   ⇒ 用 J2 / J5，各摆 4°，相机横移约 20~30mm，让新码进入可读窗口。
                q_base = arm.joint_deg()
                for j, dq in ((2, 4.0), (5, 4.0), (2, -4.0)):
                    tag = f"tap{j}{dq:+.0f}"
                    print(f"— 小点拍 {tag}（J{j} {dq:+.0f}°，约横移 20~30mm）")
                    qt = list(q_base)
                    qt[j - 1] += dq
                    moved = True
                    try:
                        _move_j_checked(arm, qt, speed=args.speed)
                        shots.append(take_shot(arm, cam, det, tag, outdir))
                    except (SafetyError, RuntimeError) as e:
                        print(f"  ✗ 该点放弃：{e}")
                    _move_j_checked(arm, q_base, speed=args.speed, timeout=150.0)
                print(f"— 共 {len(shots)} 帧（法线法 1 枚即可定轴，≥3 枚取中位数更稳）")
            else:
                # 平移采样在本位形要 47~61° 关节摆（臂近伸直，2026-10-16 实测被 dq 门全拦）
                # → 改用【小关节点拍】：几度量级、无需逆解、自带视差与视角多样性。
                # 基准 = 当前实际关节（--lookdown 时是俯视位，不是脚本起点的 q0！）
                q_base = arm.joint_deg()
                a = args.tap
                taps = [(2, -a), (2, a), (3, -a), (3, a),
                        (1, -0.6 * a), (1, 0.6 * a), (5, -0.7 * a), (5, 0.7 * a)]
                for j, dq in taps:
                    tag = f"J{j}{dq:+.1f}"
                    print(f"— 采样位 {tag}（J{j} 点拍 {dq:+.1f}°）")
                    qt = list(q_base)
                    qt[j - 1] += dq
                    moved = True
                    _move_j_checked(arm, qt, speed=args.speed)
                    shots.append(take_shot(arm, cam, det, tag, outdir))
                    _move_j_checked(arm, q_base, speed=args.speed, timeout=150.0)
    finally:

        if moved:
            try:
                _move_j_checked(arm, q0, speed=args.speed, timeout=150.0)
                print("已回原关节")
            except Exception as e:                                   # noqa: BLE001
                print(f"⚠ 回原位失败：{e} —— 用 see_cup 再跑一次或手动检查")
        cam.close()
        arm.close()

    # ---- 解算：默认 PnP 主方法；--legacy 走旧的圆拟合路线（诊断对照）----
    legacy = args.legacy or (args.orbit and not args.single)
    if legacy:
        print("\n———— 旧法：射线∩平面 + 圆拟合（--legacy，诊断对照）————")
        res = cup_pose.solve(shots, K, D, X)
        ok = cup_pose.report(res)
    else:
        print("\n———— PnP 主方法（柱面模型 + 法线定轴；不吃 TABLE_Z）————")
        res = cup_pose.solve_pnp(shots, K, D, X)
        ok = cup_pose.report_pnp(res)

    for sh in shots:
        annotate(outdir / f"{sh['tag']}.jpg", sh["dets"], res, sh["T"], X, K, D,
                 outdir / f"{sh['tag']}_marked.jpg")
    (outdir / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if res.get("error"):
        print(f"\n结果见 {outdir}（解算失败，原图都在）")
        sys.exit(2)
    T0 = shots[0]["T"] @ X
    v = np.array(res["axis_xy"]) - T0[:3, 3][:2]
    nv = float(np.linalg.norm(v))
    if nv > 1e-6:
        print(f"\n相机(第 1 帧)水平方向 → 杯轴：单位向量 ({v[0]/nv:+.3f}, {v[1]/nv:+.3f})，"
              f"水平距 {nv:.0f} mm")
    print(f"标注图 {outdir}/*_marked.jpg（机器画面是 180° 旋转的——转正后再看）")
    print(f"杯轴基座系 = ({res['axis_xy'][0]:.1f}, {res['axis_xy'][1]:.1f})"
          f"   —— 抓取用；插入面 z = 承载面 {config.TABLE_Z} + 环中面高度")

    # ---- 闭环自检：杯轴投影离画面中心差多少像素？（≈0.25mm/px，是绝对 xy 的直接裁决量）----
    if not res.get("error") and res.get("z_markers") is not None:
        aim = np.asarray(_LD_AIM_OVERRIDE[0] if _LD_AIM_OVERRIDE
                      else ld_aim(strict=False), float)
        ax, ay = res["axis_xy"]
        Pb = np.array([ax, ay, float(res["z_markers"]), 1.0])
        iv = np.linalg.inv(shots[0]["T"] @ X)
        Pc = iv @ Pb
        if Pc[2] > 0:
            uv, _ = cv2.projectPoints(Pc[:3], np.zeros(3), np.zeros(3), K, D)
            u, v = uv.reshape(2)
            du, dv = u - K[0, 2], v - K[1, 2]
            mmpx = 1.0 / K[0, 2] * float(np.linalg.norm(Pc[:3]))
            print(f"闭环自检：瞄点 = {tuple(round(float(x), 1) for x in aim[:2])}"
                  f" → 解出杯轴 {tuple(round(float(x), 1) for x in res['axis_xy'])}")
            print(f"  杯轴在画面里偏离中心 {math.hypot(du, dv):.1f} px"
                  f"（{du:+.1f}, {dv:+.1f}）≈ {math.hypot(du, dv) * mmpx:.1f} mm @该距离")
            print(f"  ★ 若瞄点当初就取自解算结果，这个偏差就是【整条链的绝对 xy 残差】；"
                  f"几像素即几毫米。")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
