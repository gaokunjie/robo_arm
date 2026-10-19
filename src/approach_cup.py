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

## ⚠⚠ 2026-10-19 崩溃真凶（rblog 取证定案）：Ry=±90° 万向锁
    早期误判是"FK 打崩控制器"。**真凶是姿态落在欧拉万向锁上**：法兰 x 竖直 ⇒ ry 精确
    = −90° ⇒ 控制器内部 FK 与 GetForwardKin 对同一旋转的 (rx,rz) 拆分不同（矩阵只差
    1.4e-7），其 MoveJ 校验按欧拉数值比对 goalPosT/goalPosC 判"不符" → 应用层崩溃
    （ping✓/80✓ 但 20003/8083 全关，只能断电）。所以：① 把 ② 及以后的目标姿态偏
    TILT_DEG° 离开奇异（见 insert_axes）；② 仍保留"规划/执行分离"，好处是规划可反复
    重试、执行期零 FK、状态更干净：
         python3 -u -I src/approach_cup.py --plan=/tmp/ac_plan.json    # 只算不动
         python3 -u -I src/approach_cup.py --replay=/tmp/ac_plan.json  # 只动不算
         python3 -u -I src/approach_cup.py --check=/tmp/ac_plan.json   # 离线自检，不连臂
     回放用 MoveJ（关节空间：控制器内部不做笛卡尔逆解），位姿随航点存好、直接下发。
    ③ P0 硬护栏（2026-10-19 加）：任何要下发的姿态都先过 guard_gym / guard_pose6——
       |ry| 距 90° 小于 GYM_HARD_DEG(0.5°) 就拒绝，不再只靠 TILT 一个常量自觉。
       规划段 / 实时段 / 回放段三处都挂了闸；--check 可离线体检任意计划文件。
"""

from __future__ import annotations

import json
import math
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
STEP_MM = 15.0                   # 2026-10-19：MoveJ 走关节直线 ≠ 笛卡尔直线，步长收紧
ANG_STEP = 2.0                   # 每段姿态最多转多少度
TILT_DEG = 2.0                   # 法兰 x 离开竖直的角度，逃开 Ry=±90° 万向锁（见 insert_axes）
SPEED_PCT = 8.0
# ---- P0 万向锁硬护栏（本地、下发前拦截；不依赖控制器）----
#   rblog 实测：崩溃发生在 ry=−89.999939°（距 90° 仅 0.00006°）；而 ry=−89.013°
#   （距 0.987°）稳定通过 ⇒ 危险带极窄但代价是整机掉线。取 0.5° 为硬边界：既远高于
#   数值噪声、又放过已知安全的 ①（0.987°）。距 90° 小于 GYM_WARN_DEG 只告警。
GYM_HARD_DEG = 0.5               # 目标姿态 |ry| 距 90° 小于此值 → 拒绝下发（sys.exit）
GYM_WARN_DEG = 3.0               # 距 90° 小于此值 → 打印告警
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
    """返回 (u, R) —— u=基座→杯水平单位向量；R=插入位法兰姿态（法兰z=+u）。

    ⚠⚠ 法兰 x **故意偏离竖直 TILT_DEG°**（绕工具轴 zax 滚）——这是 2026-10-19 的**保命项**：
      法兰 x 竖直 ⇒ R[2,0]=1 ⇒ 欧拉 ry 精确 = −90°，正落在 **Ry=±90° 万向锁**上。
      此时控制器的内部 FK 与 GetForwardKin 对**同一个物理旋转**解出不同的 (rx,rz) 拆分
      （实测旋转矩阵只差 1.4e-7，但欧拉三元组差 1.6°~7.5°），其 MoveJ 校验按欧拉
      **数值**比对 goalPosT/goalPosC → 判"不符" → 应用层崩溃（rblog 铁证）。
      绕工具轴滚 TILT_DEG° ⇒ xax_z=cos(TILT) ⇒ ry=−90+TILT≈−88°，远离奇异、欧拉唯一。
      只把环面滚 2°（缺口仍朝杯），净空闸会复核是否仍安全。
    """
    u = unit_xy(CUP_XY)
    zax = np.array([u[0], u[1], 0.0])                              # 工具轴水平朝杯
    Rx = Rot.from_rotvec(np.radians(TILT_DEG) * zax).as_matrix()
    xax = Rx @ np.array([0.0, 0.0, 1.0])                           # 竖直的 x 绕工具轴滚一点
    yax = np.cross(zax, xax)
    return u, np.column_stack([xax, yax, zax])


def gym_lock(R):
    """返回 (ry_deg, off_deg)。off = 90 − |ry|，越小越贴近 Ry=±90° 万向锁。

    用旋转矩阵直接取（R=Rz·Ry·Rx ⇒ R[2,0] = −sin ry），避免欧拉表示的边角歧义。
    """
    R = np.asarray(R, float)
    sy = -float(R[2, 0])
    ry = math.degrees(math.asin(max(-1.0, min(1.0, sy))))
    return ry, 90.0 - abs(ry)


def guard_gym(R, label, hard=GYM_HARD_DEG):
    """P0 护栏：目标姿态太贴近万向锁就返回拒绝理由（否则 None）。R=法兰姿态矩阵。

    这是 2026-10-19 反复掉线的**直接病因拦截**——凡是喂给控制器的姿态都先过这里；
    任何未来改动（改 TILT、改路径、手填姿态）一旦把 ry 推回 ±90° 都会被当下拦下。
    """
    ry, off = gym_lock(R)
    if off < hard:
        return (f"✗ {label}：目标姿态 ry={ry:+.4f}° 距 ±90° 仅 {off:.4f}° < {hard}° "
                f"——万向锁禁区，拒绝下发（控制器会在此崩溃并整机掉线）")
    if off < GYM_WARN_DEG:
        print(f"  ⚠ {label}：ry={ry:+.3f}° 距 ±90° {off:.3f}°（偏近，警惕欧拉拆分放大）")
    return None


def guard_pose6(pose6, label, hard=GYM_HARD_DEG):
    """同 guard_gym，但输入是 [x,y,z,rx,ry,rz]（度）。"""
    return guard_gym(M.pose6_to_T([float(v) for v in pose6])[:3, :3], label, hard)


def check_plan(path):
    """离线自检一个 --plan 文件：逐航点报 ry 与距 90° 的距离，标出禁区。无需连臂。"""
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    way = plan["waypoints"]
    print(f"自检 {path}：{len(way)} 航点　万向锁硬闸 {GYM_HARD_DEG}°／告警 {GYM_WARN_DEG}°")
    nbad = 0
    for idx, w in enumerate(way):
        ry, off = gym_lock(M.pose6_to_T([float(v) for v in w["pose"]])[:3, :3])
        flag = ""
        if off < GYM_HARD_DEG:
            flag = "  ✗ 万向锁禁区（回放会被拒）"
            nbad += 1
        elif off < GYM_WARN_DEG:
            flag = "  ⚠ 偏近"
        print(f"  [{idx+1:2d}/{len(way)}] {w['label']:28s} "
              f"ry={ry:+9.4f}°  距90°={off:8.4f}°{flag}")
    print(f"\n{'✗ 有 %d 个航点在禁区。' % nbad if nbad else '✓ 全部航点远离万向锁。'}")
    return nbad


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


def staged(arm, fc, fk6, lim, q_cur, p_target, R_target, tree, cup, label, dry, trace=None):
    """段执行：位置线性 + 姿态 slerp 切分。返回最后关节。

    trace 非 None 时（配合 dry=True）把每个子航点 (关节, 位姿) 记进 trace —— 供
    「规划/回放分离」的 --plan 用：回放阶段不再需要任何 FK。
    """
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
            if trace is not None:
                trace.append({"label": f"{label} [{i}/{n}]",
                              "q": q_cur.tolist(), "pose": list(fc.pose(q_cur)),
                              "seg_clear": round(seg_min, 2)})
            continue
        if float(np.linalg.norm(dp)) > config.MAX_MOVE_MM:
            sys.exit(f"✗ 第{i}段位移超上限，中止")
        r = M.solve_delta(fk6, q_cur, dp=dp, drot=drot, steps=3, iters=12,
                          tol_pos=0.01, tol_ang=0.01)
        q_new = np.asarray(r["q"], float)
        inlim = bool(np.all(q_new >= lim[:, 0] - 1e-6) and np.all(q_new <= lim[:, 1] + 1e-6))
        dq = float(np.max(np.abs(q_new - q_cur)))
        # MoveJ 走【关节直线】≠ 笛卡尔直线 ⇒ 在实际关节路径上补采净空（插入段尤其要紧）
        wj = 999.0
        for t in np.linspace(0.0, 1.0, 6):
            wj = min(wj, clearance(fc.T((1 - t) * q_cur + t * q_new), tree, cup))
        print(f"  第{i}/{n}段 位移{np.linalg.norm(dp):5.1f}mm 残差{r['err_pos']:.3f}mm/"
              f"{r['err_ang']:.3f}° max|Δq|={dq:5.1f}° 净空 段{seg_min:+6.1f}/关节{wj:+6.1f}mm "
              f"限位{'✓' if inlim else '✗'}", flush=True)
        if not inlim or r["err_pos"] > 0.5 or r["err_ang"] > 0.3 or dq > MAX_DQ:
            sys.exit(f"✗ 第{i}段未过闸，停在原地")
        if seg_min < MIN_CLEAR or wj < MIN_CLEAR:
            sys.exit(f"✗ 第{i}段净空 段{seg_min:.1f}/关节{wj:.1f} < {MIN_CLEAR}，停在原地")
        bad = guard_pose6(fc.pose(q_new), f"第{i}段")
        if bad:
            sys.exit(bad)
        arm.enable()
        # 用 MoveJ（关节空间）执行：控制器内部不做笛卡尔逆解 ⇒ 绕开 MoveL 的崩溃触发点
        arm.move_j(q_new.tolist(), speed=SPEED_PCT, blocking=False)
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


def staged_gen(fc, fk6, q_cur, p_target, R_target, tree, cup, label):
    """【流式规划生成器】——与 staged(dry=True) 同一算法，但不执行、不打印，逐点 yield。

    next() 得到一个子航点 dict（字段同 trace 项：label/q/pose/seg_clear）；消耗完即整段航点。
    供【流水线】用：执行上一段的同时，用本生成器规划下一段 —— 单连接、单线程安全，
    因为运动只走 :20003 下发、到位靠 :8083 轮询，运动期间 :20003 空闲可发 FK 查询。

    ⚠ FK 调用序列与 staged 完全一致（fc.T 一次 + solve_delta），否则拿不到缓存命中。
    """
    T0 = fc.T(q_cur)
    p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
    p_target = np.asarray(p_target, float)
    dpos = float(np.linalg.norm(p_target - p0))
    dang = float(np.degrees(Rot.from_matrix(R_target @ R0.T).magnitude()))
    n = max(1, int(np.ceil(dpos / STEP_MM)), int(np.ceil(dang / ANG_STEP)))
    sl = Slerp([0, 1], Rot.from_matrix(np.stack([R0, R_target])))
    for i in range(1, n + 1):
        s = i / n
        pt = p0 + s * (p_target - p0)
        Rt = sl(s).as_matrix()
        Tcu = fc.T(q_cur)
        dp = pt - Tcu[:3, 3]
        drot = M.rotvec_deg(Rt @ Tcu[:3, :3].T)
        seg_min = seg_clearance(Tcu[:3, 3], pt, Tcu[:3, :3], Rt, tree, cup)
        q_cur = np.asarray(M.solve_delta(fk6, q_cur, dp=dp, drot=drot, steps=3, iters=12,
                                         tol_pos=0.01, tol_ang=0.01)["q"], float)
        yield {"label": f"{label} [{i}/{n}]", "q": q_cur.tolist(),
               "pose": list(fc.pose(q_cur)), "seg_clear": round(seg_min, 2)}


def wait_move(arm, timeout=180.0, start_win=4.0, pump=None):
    """等运动【真正完成】：先确认已启动（motion_done→0 或关节开始动），再等关节连续稳定 1s。

    ⚠ 不能用 arm.wait_done()/fr_state.wait_motion_done：它只看 motion_done==1，指令刚下发
      一瞬该字段仍是上次的 1 ⇒ 立刻返回 ⇒ 下一条指令撞进正在跑的 MoveJ（错误码 6，只能断电）。

    pump：可选的「空档回调」。运动期间 :20003 空闲（运动只走 :20003 下发、到位靠 :8083 轮询），
       循环每轮调用一次 pump()，供【流水线】在等运动时推进下一段的规划（单连接、单线程安全）。
    """
    st = arm.st
    t0 = time.time()
    started = False
    prev = None
    while time.time() - t0 < start_win:
        if pump is not None:
            pump()
        q = np.asarray(st.joints, float)
        if int(st.motion_done) == 0 or (prev is not None and float(np.max(np.abs(q - prev))) > 0.05):
            started = True
            break
        prev = q
        time.sleep(0.1)
    stable = 0.0
    prev = None
    while time.time() - t0 < timeout:
        if pump is not None:
            pump()
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


def replay(arm, lim, plan, start=0, upto=None, times=None, pump=None, quiet=False):
    """【零 FK 回放】把一个算好的关节航点序列纯执行下去。

    —— 绕开控制器崩溃的要点：位姿随航点存好、直接喂 move_j（否则 fr_arm.move_j
       内部会再调一次 FK）；本阶段绝不再查正解。
    start/upto：只回放 [start, upto) 区间的航点（0 基、不含 upto）。upto=None 跑到头。
       用途：先 --upto=2 只走 ①→②（万向锁崩点）验证不崩，再 --from=2 续跑。
    times：可选 list；每个航点 append `(idx, 该航点耗时s)`（= move_j 下发 → wait_move
       到位，不含 0.4s 整定）。给调用方做「规划 vs 运动」计时用。
    pump：可选回调，透传 wait_move；供流水线在等运动时推进下一段规划。
    quiet：True 时不打印段末提示（流水线逐段回放时用）。
    """
    way = plan["waypoints"]
    end = len(way) if upto is None else min(int(upto), len(way))
    if not quiet:
        print(f"回放航点 [{start}, {end}) / 共 {len(way)} —— 零 FK 模式")
    if start == 0:
        arm.enable()
        time.sleep(0.3)
    for idx in range(start, end):
        w = way[idx]
        q = np.asarray(w["q"], float)
        qw = q.copy()
        for i in range(6):
            while qw[i] > lim[i, 1]:
                qw[i] -= 360
            while qw[i] < lim[i, 0]:
                qw[i] += 360
        oob = [f"j{i+1}={q[i]:.0f}∉[{lim[i,0]:.0f},{lim[i,1]:.0f}]"
               for i in range(6) if not (lim[i, 0] - 1e-6 <= qw[i] <= lim[i, 1] + 1e-6)]
        if oob:
            sys.exit(f"✗ 航点 #{idx} 超限位：{', '.join(oob)}，停")
        bad = guard_pose6(w["pose"], f"航点 #{idx}")
        if bad:
            sys.exit(bad)
        t_send = time.time()
        arm.r.move_j([float(x) for x in q], [float(x) for x in w["pose"]],
                     config.TOOL_ID, config.USER_ID, SPEED_PCT,
                     config.MAX_ACC_PCT, 100.0, 0.0)
        if not wait_move(arm, 120, pump=pump):
            sys.exit(f"✗ 航点 #{idx} 等待到位超时，停在原地（别再发下一条！）")
        if times is not None:
            times.append((idx, time.time() - t_send))
        time.sleep(0.4)
        qa = np.asarray(arm.joint_deg(), float)
        dj = float(np.max(np.abs(qa - q)))
        print(f"  [{idx+1}/{len(way)}] {w['label']:28s} 到位关节差 {dj:.3f}°  "
              f"错误码 {arm.errors()}", flush=True)
        if dj > 0.5:
            sys.exit(f"✗ 航点 #{idx} 到位关节差 {dj:.3f}° 过大，停")
    if quiet:
        return
    if end < len(way):
        print(f"\n■ 已按 --upto 停在航点 #{end}（未到插入位）。"
              f"目视无误后继续：--replay=... --from={end}")
    else:
        print("\n★ 回放完成：已到插入位。请目视——环心与杯同心？缺口正对杯？没蹭到？")


def main():
    argv = sys.argv
    dry = "--dry" in argv
    check_file = next((a.split("=", 1)[1] for a in argv if a.startswith("--check=")), None)
    if check_file:                       # 纯离线：无需连臂，控制器挂着也能跑
        sys.exit(1 if check_plan(check_file) else 0)
    plan_file = next((a.split("=", 1)[1] for a in argv if a.startswith("--plan=")), None)
    replay_file = next((a.split("=", 1)[1] for a in argv if a.startswith("--replay=")), None)
    start_idx = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--from=")), "0"))
    upto_idx = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--upto=")), None)

    # ---- 回放模式：只连臂、零 FK、把算好的航点执行下去 ----
    if replay_file:
        arm = Arm()
        arm.connect()
        lim = np.asarray(V.set_limits(arm.r), float)
        plan = json.loads(Path(replay_file).read_text(encoding="utf-8"))
        print(f"当前关节 {[round(v, 2) for v in arm.joint_deg()]}  错误码 {arm.errors()}")
        replay(arm, lim, plan, start_idx, upto_idx)
        arm.close()
        return

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

    # P0 护栏：规划段的目标姿态在下发前先过万向锁闸（Rh 若因任何改动回到竖直 x，这里立刻炸）
    for lab, R in (("①现姿态 R0", R0), ("②~⑤插入姿态 Rh", Rh)):
        bad = guard_gym(R, lab)
        if bad:
            sys.exit(bad)

    stages = [
        (p_raise, R0, "① 竖直抬到高位（保持现姿态）"),
        (p_raise, Rh, "② 原地转工具轴→水平（法兰不动）"),
        (p_high, Rh, "③ 平移到进场位上方"),
        (p_hover, Rh, "④ 竖直下降到插入高度"),
        (p_ins, Rh, "⑤ 径向推入到杯心"),
    ]

    trace = [] if plan_file else None
    q_cur = q_now.copy()
    for pt, Rt, lab in stages:
        if dry or plan_file:
            w = seg_clearance(fc.T(q_cur)[:3, 3], pt, fc.T(q_cur)[:3, :3], Rt, tree, cup)
            print(f"\n[dry]【{lab}】整段最小净空 {w:+6.2f}mm  "
                  f"{'✓' if w > MIN_CLEAR else '✗ 不足'}")
        q_cur = staged(arm, fc, fk6, lim, q_cur, pt, Rt, tree, cup, lab,
                       dry or bool(plan_file), trace)
        q_cur = np.asarray(q_cur, float)

    # 插入位最终复核
    Tins = fc.T(q_cur)
    Tr = Tins @ M_FR
    rc = Tr[:3, 3]
    print(f"\n★ 到位停（{'dry-run，未动臂' if (dry or plan_file) else '已到插入位'}）。"
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
    if not dry and not plan_file:
        print("  请目视：环心与杯同心？缺口正对杯？没蹭到？")
    fc.save(force=True)

    if plan_file:
        plan = {"start_q": q_now.tolist(), "cup_axis": CUP_XY.tolist(),
                "table_z": config.TABLE_Z, "stages": [s[2] for s in stages],
                "waypoints": trace}
        Path(plan_file).write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(f"\n✔ 已写规划 {plan_file}：{len(trace)} 个航点。"
              f"下一步：python3 -u -I src/approach_cup.py --replay={plan_file}")


if __name__ == "__main__":
    main()
