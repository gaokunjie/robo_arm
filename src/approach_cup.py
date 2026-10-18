# -*- coding: utf-8 -*-
"""把环带到杯边并插入：**φ=90 版**（环面含工具轴，2026-10-18 重写）。

## 新几何（旧版「环面⊥工具轴」假设已作废）
    环安装：环面法线 = 法兰 x；缺口沿 +工具 z（远端）；环心在法兰 [0,0,90]。
    要水平套住竖直杯 ⇒ **法兰 x 竖直、工具轴水平指向杯**。
    u = 基座→杯轴的水平单位向量；插入位环心 = 杯轴 xy + 插入高度 z = TABLE_Z+40
    （当前 方案 D：TABLE_Z=+6 ⇒ 插入位 z=+46；方案 C 曾为 TABLE_Z=−85 ⇒ −45）；
    法兰 = 环心 − 90·u（环心在法兰 +工具z 方向 90mm 处）。

## 五段（每段都过「环实体 vs 杯面」点云净空闸，不过闸停在原地）
    ① 竖直抬到 Z_RAISE（保持当前姿态，环整体抬离杯）
    ② 原地把工具轴由竖直转到水平（法兰不动；环在高处扫 ~127mm，离杯远）
    ③ 平移到进场位上方（环心在杯外侧 D_OUT、高度 Z_RAISE）
    ④ 竖直下降到插入高度（环心离杯 D_OUT）
    ⑤ 径向推入到杯心
到位【停】。绝不下发抬升。

## 闸门（任一不过 → sys.exit，停原地、不硬推）
    限位 / 残差 / 单段位移 / 单段关节变化 / 净空。MIN_CLEAR 见下。
⚠ 没有整臂连杆模型 ⇒ 小臂/肘部/腕壳与台面、支架的碰撞验不了。请全程目视盯臂、手放急停。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import fr3_math as M                                       # noqa: E402
from fr_arm import Arm                                      # noqa: E402
from fr_fkcache import FKCache                             # noqa: E402
import verify_table_place as V                             # noqa: E402

# ---------------- 场景常数 ----------------
#   2026-10-19：config 值已由仓库内 cup_pose 直线交点法【精确复现】(45.01,−566.09)=config 值，
#   且仍有 22.8mm 的【z 绝对锚】悬账（见 config.TABLE_Z 注释）⇒ 每次动臂前仍建议观察位实拍复核。
CUP_XY = np.array(config.CUP_AXIS_XY, float)   # 已实测+仓库内可复现；动臂前仍复核
RC = config.RING_CENTER_IN_FLANGE[2]           # 90
Z_INS = config.TABLE_Z + config.INSERT_PLANE_Z_ABOVE_TABLE   # 方案 D = 6+40 = 46
D_OUT = config.D_OUT                            # 115
Z_RAISE = 350.0                                # ① 抬到的法兰 z（高位安全区）
STEP_MM = 25.0
ANG_STEP = 4.0                                  # 每段姿态最多转多少度
SPEED_PCT = 8.0
MIN_CLEAR = 3.0                                 # 环实体对杯面的最低净空（插入段固有 ~8.4mm）
MAX_DQ = 45.0

RIN, RON, GAPW = config.RING_ID / 2, config.RING_OD / 2, config.RING_GAP
PROFILE = np.array(config.CUP_PROFILE)
MARKER_THICK = config.MARKER_THICK
M_FR = np.asarray(config.M_FLANGE_RING, float)
GD = np.radians(90.0)                           # 缺口沿 +环 y（= +工具 z）；新模型下固定 90°


def unit_xy(v):
    return v / np.linalg.norm(v)


def insert_axes():
    """返回 (u, R) —— u=基座→杯水平单位向量；R=插入位法兰姿态（法兰x=+Z、法兰z=+u）。"""
    u = unit_xy(CUP_XY)
    xax = np.array([0.0, 0.0, 1.0])            # 法兰 x 竖直（环面水平）
    zax = np.array([u[0], u[1], 0.0])         # 工具轴水平朝杯
    yax = np.cross(zax, xax)
    return u, np.column_stack([xax, yax, zax])


def build_models():
    """环实体点云（**环系**：原点=环心，z=法线）+ 杯面点云（基座系）。"""
    rad = np.arange(RIN, RON + 0.5, 1.0)
    zz = np.arange(-config.RING_THICK / 2, config.RING_THICK / 2 + 0.5, 1.5)
    ang = np.linspace(0.0, 2 * np.pi, 240, endpoint=False)
    pts = []
    for r in rad:
        x, y = r * np.cos(ang), r * np.sin(ang)
        along = x * np.cos(GD) + y * np.sin(GD)
        perp = -x * np.sin(GD) + y * np.cos(GD)
        keep = ~((along > 0) & (np.abs(perp) < GAPW / 2))
        for z in zz:
            pts.append(np.column_stack([x[keep], y[keep], np.full(int(keep.sum()), z)]))
    ring = np.vstack(pts)
    h = np.linspace(0.0, config.CUP_HEIGHT, 81)
    rr = np.interp(h, PROFILE[:, 0], PROFILE[:, 1]) / 2 + MARKER_THICK
    th = np.linspace(0.0, 2 * np.pi, 240, endpoint=False)
    cup = np.stack([
        (CUP_XY[0] + rr[:, None] * np.cos(th)).ravel(),
        (CUP_XY[1] + rr[:, None] * np.sin(th)).ravel(),
        (config.TABLE_Z + np.repeat(h, 240).reshape(-1, 240)).ravel(),
    ], axis=1)
    return cKDTree(ring), cup


def clearance(T, tree, cup):
    """环实体穿入杯面的净空（mm，正=安全）。T=法兰位姿；内部取 T_环 = T @ M_FLANGE_RING。"""
    Tr = T @ M_FR
    R, rc = Tr[:3, :3], Tr[:3, 3]
    loc = (R.T @ (cup - rc).T).T
    z = loc[:, 2]
    d = np.hypot(loc[:, 0], loc[:, 1])
    along = loc[:, 0] * np.cos(GD) + loc[:, 1] * np.sin(GD)
    perp = -loc[:, 0] * np.sin(GD) + loc[:, 1] * np.cos(GD)
    inz = (z >= -config.RING_THICK / 2) & (z <= config.RING_THICK / 2)
    inslot = (along > 0) & (np.abs(perp) < GAPW / 2)
    inring = (d >= RIN) & (d <= RON) & inz & ~inslot
    if inring.any():
        dd = d[inring]
        return -float(np.minimum(np.abs(dd - RIN), np.abs(dd - RON)).min())
    return float(tree.query(loc, k=1)[0].min())


def seg_clearance(p0, p1, R0, R1, tree, cup, n=16):
    """位姿沿直线段（位置线性、姿态 slerp）的最小净空。"""
    w = 999.0
    sl = Slerp([0, 1], Rot.from_matrix(np.stack([R0, R1])))
    for t in np.linspace(0.0, 1.0, n):
        T = np.eye(4)
        T[:3, :3] = sl(t).as_matrix()
        T[:3, 3] = p0 + t * (p1 - p0)
        w = min(w, clearance(T, tree, cup))
    return w


def staged(arm, fc, fk6, lim, q_cur, p_target, R_target, tree, cup, label, dry, force_ori=False):
    """段执行：位置线性 + 姿态 slerp 切分。返回最后关节。"""
    T0 = fc.T(q_cur)
    p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
    dpos = float(np.linalg.norm(p_target - p0))
    dang = float(np.degrees(Rot.from_matrix(R_target @ R0.T).magnitude()))
    n = max(1, int(np.ceil(dpos / STEP_MM)), int(np.ceil(dang / ANG_STEP)))
    print(f"\n【{label}】{np.round(p0,1).tolist()} → {np.round(p_target,1).tolist()}  "
          f"位移{dpos:.0f}mm 转角{dang:.0f}° → {n}段", flush=True)
    sl = Slerp([0, 1], Rot.from_matrix(np.stack([R0, R_target])))
    for i in range(1, n + 1):
        s = i / n
        pt = p0 + s * (p_target - p0)
        Rt = sl(s).as_matrix()
        Tcu = fc.T(q_cur)
        dp = pt - Tcu[:3, 3]
        drot = M.rotvec_deg(Rt @ Tcu[:3, :3].T)
        seg_min = seg_clearance(Tcu[:3, 3], pt, Tcu[:3, :3], Rt, tree, cup)
        if dry:
            print(f"  [{i}/{n}] 段净空 {seg_min:+6.1f}mm", flush=True)
            q_cur = M.solve_delta(fk6, q_cur, dp=dp, drot=drot, steps=3, iters=12,
                                  tol_pos=0.01, tol_ang=0.01)["q"]
            q_cur = np.asarray(q_cur, float)
            continue
        if float(np.linalg.norm(dp)) > config.MAX_MOVE_MM:
            sys.exit(f"✗ 第{i}段位移超上限，中止")
        r = M.solve_delta(fk6, q_cur, dp=dp, drot=drot, steps=3, iters=12,
                          tol_pos=0.01, tol_ang=0.01)
        q_new = np.asarray(r["q"], float)
        inlim = bool(np.all(q_new >= lim[:, 0] - 1e-6) and np.all(q_new <= lim[:, 1] + 1e-6))
        dq = float(np.max(np.abs(q_new - q_cur)))
        print(f"  第{i}/{n}段 位移{np.linalg.norm(dp):5.1f}mm 残差{r['err_pos']:.3f}mm/"
              f"{r['err_ang']:.3f}° max|Δq|={dq:5.1f}° 段净空{seg_min:+6.1f}mm "
              f"限位{'✓' if inlim else '✗'}", flush=True)
        if not inlim or r["err_pos"] > 0.5 or r["err_ang"] > 0.3 or dq > MAX_DQ:
            sys.exit(f"✗ 第{i}段未过闸，停在原地")
        if seg_min < MIN_CLEAR:
            sys.exit(f"✗ 第{i}段净空 {seg_min:.1f} < {MIN_CLEAR}，停在原地")
        arm.enable()
        arm.move_l(fc.pose(q_new), speed=SPEED_PCT, acc=config.MAX_ACC_PCT, joints=q_new.tolist())
        if not wait_move(arm, 120):
            sys.exit(f"✗ 第{i}段等待到位超时，停在原地（别再发下一条！）")
        time.sleep(0.8)
        qa = np.asarray(arm.joint_deg(), float)
        e = float(np.linalg.norm(fc.T(qa)[:3, 3] - fc.T(q_new)[:3, 3]))
        print(f"    到位 实际法兰 {np.round(fc.T(qa)[:3,3],1).tolist()} 残差 {e:.3f}mm "
              f"错误码 {arm.errors()}", flush=True)
        if e > 1.5:
            sys.exit("✗ 到位残差过大，停")
        q_cur = qa
    return q_cur


def wait_move(arm, timeout=180.0, start_win=4.0):
    """等运动【真正完成】：先确认已启动（motion_done→0 或关节开始动），再等关节连续稳定 1s。

    ⚠ 不能用 arm.wait_done()/fr_state.wait_motion_done：它只看 motion_done==1，指令刚下发
      一瞬该字段仍是上次的 1 ⇒ 立刻返回 ⇒ 下一条指令撞进正在跑的 MoveJ（错误码 6，只能断电）。
    """
    st = arm.st
    t0 = time.time()
    started = False
    prev = None
    while time.time() - t0 < start_win:
        q = np.asarray(st.joints, float)
        if int(st.motion_done) == 0 or (prev is not None and float(np.max(np.abs(q - prev))) > 0.05):
            started = True
            break
        prev = q
        time.sleep(0.1)
    stable = 0.0
    prev = None
    while time.time() - t0 < timeout:
        q = np.asarray(st.joints, float)
        if prev is not None and float(np.max(np.abs(q - prev))) < 0.02:
            stable += 0.2
            # ⚠ 快速小段 MoveJ 可能在首次轮询前就跑完 ⇒ 永远观察不到 motion_done==0
            #   ⇒ started 保持 False，旧逻辑死等 120s 超时。加 0.8s 宽限：只要稳定
            #   1s + motion_done==1，即便没抓到启动也算到位（0.8s 宽限已越过"刚下发
            #   瞬间 motion_done 仍是上一次的 1"这个竞态窗口）。
            if int(st.motion_done) == 1 and stable >= 1.0 and (started or time.time() - t0 > 0.8):
                return True
        else:
            stable = 0.0
        prev = q
        time.sleep(0.2)
    return False


def main():
    dry = "--dry" in sys.argv
    arm = Arm()
    arm.connect()
    fc = FKCache(arm.r.fk)
    M.attach_cache(fc)
    lim = np.asarray(V.set_limits(arm.r), float)
    fk6 = M.fk_throttled(lambda q: fc.pose(list(q)))
    tree, cup = build_models()

    u, Rh = insert_axes()
    p_ins = np.array([CUP_XY[0], CUP_XY[1], Z_INS]) - RC * np.array([u[0], u[1], 0.0])
    p_hover = p_ins - D_OUT * np.array([u[0], u[1], 0.0])
    ring_ins = p_ins + RC * np.array([u[0], u[1], 0.0])

    q_now = np.asarray(arm.joint_deg(), float)
    T0 = fc.T(q_now)
    p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
    print(f"当前关节 {np.round(q_now,2).tolist()}  错误码 {arm.errors()}")
    print(f"杯轴 {CUP_XY.tolist()} |xy|={np.linalg.norm(CUP_XY):.1f}  "
          f"u={np.round(u,3).tolist()}")
    print(f"插入位环心 {np.round(ring_ins,1).tolist()} (z=TABLE_Z+40)  法兰 {np.round(p_ins,1).tolist()}")
    print(f"进场位法兰 {np.round(p_hover,1).tolist()}")

    p_raise = np.array([p0[0], p0[1], Z_RAISE])
    p_high = np.array([p_hover[0], p_hover[1], Z_RAISE])

    stages = [
        (p_raise, R0, "① 竖直抬到高位（保持现姿态）"),
        (p_raise, Rh, "② 原地转工具轴→水平（法兰不动）"),
        (p_high, Rh, "③ 平移到进场位上方"),
        (p_hover, Rh, "④ 竖直下降到插入高度"),
        (p_ins, Rh, "⑤ 径向推入到杯心"),
    ]

    q_cur = q_now.copy()
    for pt, Rt, lab in stages:
        if dry:
            w = seg_clearance(fc.T(q_cur)[:3, 3], pt, fc.T(q_cur)[:3, :3], Rt, tree, cup)
            print(f"\n[dry]【{lab}】整段最小净空 {w:+6.2f}mm  "
                  f"{'✓' if w > MIN_CLEAR else '✗ 不足'}")
        q_cur = staged(arm, fc, fk6, lim, q_cur, pt, Rt, tree, cup, lab, dry)
        if dry:
            q_cur = np.asarray(q_cur, float)

    # 插入位最终复核
    Tins = fc.T(q_cur)
    Tr = Tins @ M_FR
    rc = Tr[:3, 3]
    print(f"\n★ 到位停（{'dry-run，未动臂' if dry else '已到插入位'}）。"
          f"环心 {np.round(rc,1).tolist()}  离杯轴 {np.linalg.norm(rc[:2]-CUP_XY):.2f}mm")
    print(f"  插入位净空 {clearance(Tins, tree, cup):+.2f}mm")
    qf = np.asarray(q_cur, float)
    qw = qf.copy()
    for i in range(6):
        while qw[i] > lim[i, 1]:
            qw[i] -= 360
        while qw[i] < lim[i, 0]:
            qw[i] += 360
    oob = [f"j{i+1}={qf[i]:.0f}∉[{lim[i,0]:.0f},{lim[i,1]:.0f}]"
           for i in range(6) if not (lim[i, 0] - 1e-6 <= qw[i] <= lim[i, 1] + 1e-6)]
    print(f"  终态关节 {np.round(qf,1).tolist()}  限位{'✓' if not oob else '✗ ' + ', '.join(oob)}")
    if not dry:
        print("  请目视：环心与杯同心？缺口正对杯？没蹭到？")
    fc.save(force=True)


if __name__ == "__main__":
    main()
