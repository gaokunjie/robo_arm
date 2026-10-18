# -*- coding: utf-8 -*-
"""内参倾角扫掠采集 · 关节空间版（相机在 J6 转子上，围绕冻结瞄姿做倾角/方位扫掠）。

为什么：D 批手眼判出「存的内参（10-15 旧标定，fx=1535）与当前镜头不符」——
用 D 批自己的内参解（fx≈1426-1460）重解手眼，E max 0.90°→0.41°、Δ max 6.3→3.1mm。
当前镜头真 fx 有 ±1.5% 摆幅（D 批倾角中位 15° 定不死），E1 需要 <0.5%。
本采集器专为内参补倾角多样性：板固定不动、相机走位姿，倾角推到 26~31°（含方位变体）。

内参标定不吃运动学（不进 X、不怕姿态相关柔性）：机器人只是定位台，
板只要能看见就有效——所以门比手眼采集放宽（允许部分板、允许切边）。

为什么全程不碰控制器逆解：GetInverseKin / GetInverseKinRef 是本项目禁区
（4 次把控制器应用层打崩，见 memory: fairino-controller-ik-ban）。旋转候选 =
关节直加；平移笔画 = 自研数值运动学 `fr3_math.solve_delta`（只调 GetForwardKin），
**残差先核验、过了才允许发这条运动**（铁律）。

门（本版放宽，专为倾角）：≥20/24 角点 + 板宽 ≥300px + 角点离画面边 ≥4px。

流程（--go）：
  0. 只读预演：读关节/位姿 → 当前关节必须 ≈ 冻结瞄姿 → 全议程解算+五道闸
     （软限位/|q5|/家族围栏/位移上限/数值解残差）→ 打印全表
  1. 前检五闸（急停/错误码/不在动/自动模式/相机开）→ 输 GO
  2. 逐条：MoveJ（非阻塞+状态制到位）→ 回读关节 → 抓"追上运动"的新帧 → 过门 →
     存 jpg + flange_poses.csv；不过 → 试备用小幅度 → 再不过跳过。
  3. 收尾：回瞄姿核验 → 解算命令。
     臂保持使能、绝不切模式、绝不自动下电。

用法（自己的终端；相机独占——先关浏览器取流）：
    python3 -I src/he_intr_collect.py --dry    # 只读预演：连上读关节、全部解算核验、打印议程
    python3 -I src/he_intr_collect.py --go     # 真跑：手放急停，程序会等 GO
    python3 -I src/he_intr_collect.py --selftest  # 纯离线自测（假臂），不连机器人

存证：data/intr_shots2/he_<时间>_<序号>.jpg + flange_poses.csv（留档）
      collect_log.csv（每次尝试一行，含失败原因）+ run_<时间>.txt + collect_<时间>.rpc.log（黑匣子）

第二轮（2026-10-16 晚 · 本文件）：第一轮 41 张的解算判出「焦距子集不稳 2.96%——
缺倾斜方位多样性、左列取景为 0、距离没拉开」。本版专攻这些缺口：右族/上族/左族深倾角、
画面四向平移、距离阶梯。见下方 PLAN 上方注释（含第一轮逐帧覆盖实测的映射常数）。
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrate as C                                          # noqa: E402
import fr3_math as M                                           # noqa: E402
import fr_cam as FC                                            # noqa: E402
import view_sweep as VS                                        # noqa: E402 （arrival/Tee 复用）
from fr_arm import Arm                                         # noqa: E402
from fr_cam import Cam, CamError                               # noqa: E402
from probe import board_report as BR                           # noqa: E402

# ---------------------------------------------------------------- 参数

SPEED = VS.SPEED               # 8%（与 view_sweep 同档，侦察级）
AIM_JOINTS = [13.459, -129.064, 82.882, -71.318, -21.607, 46.851]   # 冻结瞄姿（11:36 过门那帧）
AIM_ABORT_DEG = 2.0            # 当前关节偏离瞄姿超过它 → 拒绝（先回瞄姿再跑）
AIM_WARN_DEG = 0.5
LIMIT_MARGIN = 3.0             # 软限位余量（度）
Q5_MIN = 12.0                  # |q5| 下限（腕部奇异带，见 memory: fairino-wrist-singularity-q5-zero）
J2_FENCE = (-150.0, -105.0)    # 家族围栏：深家族（J2≈-200）就是撞支架那族，见 memory: fr3-elbow-geometry
J3_FENCE = (60.0, 105.0)
# 平移解的关节跳上限（2026-10-16 真机 dry 实测定的）：冻结姿态做平移，腕/底座要大幅补偿——
# ±x58 走 J4∓34°/J6±30°（+x 还带 J1 +18.2° 协调）、+z40 走 J3 14°，「大」本身不是罪；
# 【整臂甩/腕翻位】才是（分支翻转）。实测反面样本：+y 解成 J4 +122°/J3 −77°（残差 5.9mm；
# 先被残差闸拦）；−y40 = J1 +19.9° + J4 −46.6°（腕闸拦）。J1 +18.2° 那条：位姿端点解准
#（0.146mm/0.225°、冻结姿态成立）、8% 慢速、且基座转轴是独立第三轴——是协调解不是甩臂。
CAP_J1 = 20.0                  # 底座：+x58 实测 J1 +18.2° 放行；≥25° 的整臂甩仍被拦
CAP_J2 = 10.0                  # 肩：翻转必带 J2 跳（实测 +14°）——单独收紧
CAP_J3 = 22.0                  # 肘：抬落 z 的正当协调（+z40 实测 14.3°）
CAP_WRIST = 45.0               # 腕（J4~J6）：±x58 实测 34°/30° 都是正当补偿；≥46° 的翻转被卡
MOVE_MAX_MM = 95.0             # 单条位移上限（config.MAX_MOVE_MM=100 再留 5mm）
# 数值解残差闸（发指令前核验，铁律）。别拿它当"精度指标"——采集数据里用的是【实测】
# 法兰位姿，笔画终点偏 1mm 对解算毫无影响。它真正的职责：① 挡"根本没解出来"的解
#（实测反面样本：+y 泊在 16mm/10.6°）；② 保证笔画的"姿态冻结"还成立——定尺度对要求
# 两帧相对转角 ≤0.7°（解算端判据），闸放 0.35° 仍有两倍余量（±x58 实测 0.24° 压线过）。
RES_POS_MM = 0.6
RES_ANG_DEG = 0.35
J_VERIFY_STOP = 1.5            # 到位后关节回读 vs 目标的停车线（度）
J_VERIFY_WARN = 0.3
CHECK_EVERY = 12               # 每存 N 张回瞄姿核验"板没动过"（内参不吃板位，核验只为现场安全）
BOARD_MOVED_MM = 3.0           # 板动过判据（内参宽容些：回位滞回单独就 ~2mm，见 relax_probe）
BOARD_MOVED_DEG = 0.5

# 议程：(标签, 类型, 主幅度, 备用幅度)
#   类型 "j"：{关节索引(0基): 增量度}，直接加在（实时）瞄姿上——旋转不用解算
#   类型 "t"：(dp_x, dp_y, dp_z, 段数)，基座系纯平移（姿态冻结）——数值解 + 残差核验
# 第二轮（2026-10-16 晚）为什么这么排——依据【第一轮 41 帧 + D 批 33 帧】解算工具的原话：
#   「焦距没被钉住（子集 2.96%）；缺倾斜——绕横竖轴各 ±30~40° 拍 ≥8 张」「板宽目标 ≥600px」
#   「距离没拉开」「3 格取景为 0（左列）」。第一轮覆盖实测（/tmp/intr_cov.py 逐帧 PnP）：
#   高倾角（>25°）只有"下"一个方位族（az 45-135°，13/14 张）；板心从没进过画面左 1/3；
#   板宽 492-573px（距离 372-435）；平移映射实测：tool +x→画面左 4.2px/mm、−y→更近
#   （20mm≡19mm）、+z→画面上 3.7px/mm。
#   本轮补：① 右族/组合深倾角（J4+ 与 J4−J5+ 加码）；② 跨正视的"上族"（J5 直推 +45 →
#   q5=+23.4，端点在奇异带外；MoveJ 途经 q5=0 已实证可行，见 view_sweep 10:27 第 4 跑；
#   上族倾角受杠杆 ~0.6× 限制，够多少算多少）；③ 左族加码（AZ4 系，第一轮只到 12.3°）；
#   ④ 画面左域平移（+x 60/78）；⑤ 距离阶梯（−y35 → ~371mm、+y16 → ~422）；⑥ 画面下/上域。
#   主/备幅度全部互不相同（防 dup 白扔；第一轮教训）。
PLAN = [
    ("J5m22", "j", {4: -22.0}, {4: -19.0}),                          # 下族加码（~34°，切边风险交给检测门）
    ("J5m26", "j", {4: -26.0}, {4: -23.0}),                          # 下族更深
    ("J4p30", "j", {3: +30.0}, {3: +26.0}),                          # 右族加码
    ("J4p38", "j", {3: +38.0}, {3: +33.0}),
    ("AZ1b", "j", {3: -18.0, 4: +40.0}, {3: -14.0, 4: +34.0}),       # 右下深组合
    ("AZ1c", "j", {3: -24.0, 4: +45.0}, {3: -20.0, 4: +38.0}),
    ("AZ4b", "j", {3: +20.0, 4: -42.0}, {3: +15.0, 4: -35.0}),       # 左族加码
    ("AZ4c", "j", {3: +26.0, 4: -45.0}, {3: +20.0, 4: -38.0}),
    ("UP1", "j", {4: +45.0}, {4: +42.0}),                            # 跨正视·上族（端点 q5=+23.4/+20.4）
    ("UP2", "j", {3: -18.0, 4: +42.0}, {3: -13.0, 4: +36.0}),        # 上-右组合方位
    ("TxP60", "t", (+60.0, 0.0, 0.0, 14), (+48.0, 0.0, 0.0, 11)),    # 画面左域
    ("TxP78", "t", (+78.0, 0.0, 0.0, 18), (+62.0, 0.0, 0.0, 14)),
    ("TyM35", "t", (0.0, -35.0, 0.0, 10), (0.0, -28.0, 0.0, 8)),     # 更近 ~371mm（板更大）
    ("TyP16", "t", (0.0, +16.0, 0.0, 6), (0.0, +12.0, 0.0, 5)),      # 更远 ~422
    ("TzM30", "t", (0.0, 0.0, -30.0, 9), (0.0, 0.0, -23.0, 7)),      # 画面下域
    ("TzP30", "t", (0.0, 0.0, +30.0, 12), (0.0, 0.0, +23.0, 9)),     # 画面上域
]


class SkipStep(RuntimeError):
    """这条候选跳过（不发/已停）。"""


# ---------------------------------------------------------------- 规划

def dq_str(dq, top=2):
    """关节增量 → "J5 -20.4° / J4 +3.1°"（按 |Δq| 取前 top 个）。"""
    idx = sorted(range(6), key=lambda i: -abs(dq[i]))[:top]
    return " / ".join(f"J{i + 1} {dq[i]:+.1f}°" for i in idx)


def make_target(fk, fk6, q_aim, p_aim, lim, kind, spec):
    """算一条候选的目标关节/位姿 + 全部安全闸。返回 dict（why 非空 = 该跳过）。"""
    if kind == "j":
        q = list(q_aim)
        for k, dv in spec.items():
            q[k] += dv
        p6 = [float(v) for v in fk(q)]
        res = None
    else:
        dp, steps = spec[0:3], int(spec[3])
        r = M.solve_delta(fk6, q_aim, dp=dp, steps=steps, iters=22)
        q = [float(v) for v in r["q"]]
        p6 = M.T_to_pose6(r["Tt"])
        res = (r["err_pos"], r["err_ang"])
    why = []
    if res is not None and (res[0] > RES_POS_MM or res[1] > RES_ANG_DEG):
        why.append(f"数值解残差 {res[0]:.2f}mm/{res[1]:.2f}° 超闸 {RES_POS_MM:g}/{RES_ANG_DEG:g}")
    dq = [q[i] - q_aim[i] for i in range(6)]
    if res is not None:
        caps = (CAP_J1, CAP_J2, CAP_J3, CAP_WRIST, CAP_WRIST, CAP_WRIST)
        viol = [f"J{i + 1} {dq[i]:+.1f}°>{caps[i]:g}°" for i in range(6) if abs(dq[i]) > caps[i]]
        if viol:
            why.append("关节跳超上限：" + "、".join(viol))
    for i in range(6):
        if not (lim[i][0] + LIMIT_MARGIN <= q[i] <= lim[i][1] - LIMIT_MARGIN):
            why.append(f"J{i + 1}={q[i]:.1f}° 顶软限位 [{lim[i][0]:.0f},{lim[i][1]:.0f}]")
    if abs(q[4]) < Q5_MIN:
        why.append(f"|q5|={abs(q[4]):.1f}° < {Q5_MIN:g}°（腕部奇异带）")
    if not (J2_FENCE[0] <= q[1] <= J2_FENCE[1]):
        why.append(f"J2={q[1]:.1f}° 出家族围栏 {J2_FENCE}")
    if not (J3_FENCE[0] <= q[2] <= J3_FENCE[1]):
        why.append(f"J3={q[2]:.1f}° 出围栏 {J3_FENCE}")
    d = float(np.linalg.norm(np.asarray(p6[:3]) - np.asarray(p_aim[:3])))
    if d > MOVE_MAX_MM:
        why.append(f"TCP 位移 {d:.0f}mm > {MOVE_MAX_MM:g}mm")
    return dict(q=q, p6=p6, res=res, d=d, why=why, dq=dq)


def build_plan(fk, fk6, q_aim, p_aim, lim, say):
    """全议程解算+核验。返回 [(tag, tier, kind, tgt或None)]，tgt=None 已跳过（原因已打印）。"""
    say("\n[预演] 全部候选解算+核验中（平移笔画要数值解，约 1 分钟，全是只读正解）…")
    out = []
    for tag, kind, spec, fbspec in PLAN:
        for tier, sp in (("主", spec), ("备", fbspec)):
            t = make_target(fk, fk6, q_aim, p_aim, lim, kind, sp)
            out.append((tag + ("" if tier == "主" else "fb"), tier, kind, t, sp))
        t0, t1 = out[-2][3], out[-1][3]
        if t0["why"] and t1["why"]:
            say(f"  ⚠ {tag}：主/备都不行（{t0['why'][0]}）——整条跳过")
    return out


def table_lines(q_aim, plan):
    lines = []
    for tag, tier, kind, t, sp in plan:
        if t is None:
            continue
        amp = (f"{max(abs(v) for v in sp.values()):.1f}°" if kind == "j"
               else f"{math.sqrt(sum(float(v) ** 2 for v in sp[:3])):.0f}mm")
        res = "—" if t["res"] is None else f"{t['res'][0]:.3f}/{t['res'][1]:.3f}"
        dq = max(abs(t["q"][i] - q_aim[i]) for i in range(6))
        mark = "✓" if not t["why"] else "✗ " + t["why"][0]
        tagp = tag if tier == "主" else "  └" + tag
        extra = ("　" + dq_str(t["dq"])) if kind == "t" else ""
        lines.append(f"  {tagp:<10} {amp:>7}  TCP移 {t['d']:>5.1f}mm  "
                     f"max|Δq|{dq:>5.1f}°  解残差 {res:>13}{extra}  {mark}")
    return lines


# ---------------------------------------------------------------- 运行件

class Runner:
    def __init__(self, say, shots, ts, det, K, D):
        self.say, self.shots, self.ts = say, shots, ts
        self.det, self.K, self.D = det, K, D
        self.saved = []            # [(name, pose, M(R_cb.T) or None, R_cb or None, t_cb or None, r)]
        self.logrows = []

    def log_row(self, tag, tier, t, r, accepted, reason, extra=""):
        cut = r["cut"] if (r and not r.get("bad")) else {k: 0.0 for k in "左上右下"}
        row = [time.strftime("%H:%M:%S"), tag, tier] + \
              [f"{v:.3f}" for v in (t["q"] if t else [0] * 6)] + \
              [f"{v:.3f}" for v in (t["p6"] if t else [0] * 6)] + \
              [(r["n"] if r else 0), f"{r['side']:.0f}" if r and not r.get("bad") else "0",
               *[f"{cut[k]:.0f}" for k in "左上右下"],
               f"{r['dist']:.0f}" if r and not r.get("bad") else "0",
               f"{r['tilt']:.1f}" if r and not r.get("bad") else "0",
               ("1" if accepted else "0"), reason, extra]
        self.logrows.append(row)

    def flush_log(self):
        p = self.shots / "collect_log.csv"
        head = ("time,tag,tier,q1,q2,q3,q4,q5,q6,x,y,z,rx,ry,rz,corners,side_px,"
                "cut_l,cut_t,cut_r,cut_b,dist_mm,tilt_deg,accepted,reason,extra\n")
        with p.open("w", encoding="utf-8") as fh:
            fh.write(head)
            for r in self.logrows:
                fh.write(",".join(str(v) for v in r) + "\n")

    def save_shot(self, tag, frame, r, say):
        pose = C._flange_pose()                       # 原始法兰位姿（相机在转子上：不做定子修正）
        for n_old, p_old, *_ in self.saved:           # 备用力撞主幅度（如 J5m16fb == J5m10）出重复帧
            dp, dr = diff_pose(p_old, pose)
            if dp < C.HE_DUP_MM and dr < C.HE_DUP_DEG:
                say(f"      ⏭ 与已存 {n_old} 几乎同帧（{dp:.3f}mm/{dr:.3f}°）——不重复存")
                return None
        idx = len(self.saved)
        name = f"he_{self.ts}_{idx:03d}.jpg"
        cv2.imwrite(str(self.shots / name), frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.det.detect(gray)
        R_cb, t_cb = C._solve_board(self.det, corners, ids, self.K, self.D)
        Mx = None if R_cb is None else R_cb.T
        kind_note = ""
        if Mx is not None and self.saved and self.saved[-1][2] is not None:
            kd, ang = C._classify_motion(self.saved[-1][2], Mx)
            if ang < C.HE_SIMILAR_DEG:
                kind_note = f"（与上一张几乎同向 {ang:.1f}°）"
            else:
                kind_note = f"（相对上一张：{kd} {ang:.0f}°）"
        self.saved.append((name, pose, Mx, R_cb, t_cb, r))
        C._he_write_csv(self.shots / C.HE_POSES_CSV, [(n, p) for n, p, *_ in self.saved])
        say(f"      ✓ 存 {name}（第 {idx + 1} 张）  板宽 {r['side']:.0f}px  距 {r['dist']:.0f}mm  "
            f"倾 {r['tilt']:.1f}°  {kind_note}")
        return name


def fmt_r(name, r):
    """内参版一行简报（不复用 BR.format_report——它内置严格门，会误报 ✗）。"""
    if r is None:
        return f"{name}: 0 角点（板不在画面里）"
    if r.get("bad"):
        return f"{name}: {r['n']} 角点但 {r['bad']}"
    ct = r["cut"]
    return (f"{name}: {r['n']}/24 角点 · 板宽 {r['side']:.0f}px · 距 {r['dist']:.0f}mm · "
            f"倾 {r['tilt']:.1f}° · 切 左/上/右/下 "
            f"{ct['左']:.0f}/{ct['上']:.0f}/{ct['右']:.0f}/{ct['下']:.0f}px")


def diff_pose(p, q):
    dp = float(np.linalg.norm(np.asarray(p[:3]) - np.asarray(q[:3])))
    dr = float(np.linalg.norm(M.rotvec_deg(M.pose6_to_T(p)[:3, :3] @ M.pose6_to_T(q)[:3, :3].T)))
    return dp, dr


def nearest_saved(saved, pose):
    best = None
    for n, p, *_ in saved:
        dp, dr = diff_pose(p, pose)
        if best is None or dp < best[0]:
            best = (dp, dr, n)
    return best


def main():
    ap = argparse.ArgumentParser(
        description="内参倾角扫掠采集（关节空间）：--dry 只读预演 / --go 真跑 / --selftest 离线自测")
    ap.add_argument("--dry", action="store_true", help="只读预演：全部解算核验+打印议程，不发指令、不开相机")
    ap.add_argument("--go", action="store_true", help="真跑：手放急停，程序会等 GO")
    ap.add_argument("--selftest", action="store_true", help="纯离线自测（假臂），不连机器人")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not (args.dry or args.go):
        sys.exit("要 --dry（只读预演）或 --go（真跑）——见文件头用法")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shots = C.HE_SHOTS_DIR.parent / "intr_shots2"
    shots.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FR_RPC_LOG", str(shots / f"collect_{ts}.rpc.log"))
    say = VS.Tee(shots / f"run_{ts}.txt")

    say("=" * 74)
    say(f"内参倾角扫掠采集（关节空间，相机在 J6 转子）· {ts} · "
        f"{'【DRY 只读预演】' if args.dry else '【GO 真跑】'}")
    say("=" * 74)

    # ---- 目标目录干净闸 ----
    old_jpg = sorted(shots.glob("*.jpg"))
    csv_path = shots / C.HE_POSES_CSV
    if old_jpg or csv_path.exists():
        n_csv = len(C._he_read_csv(csv_path)) if csv_path.exists() else 0
        say(f"\n✗ {shots} 里有旧数据（{len(old_jpg)} 张 jpg、{n_csv} 行 csv）——"
            f"这轮要全新开始。先挪走：")
        say(f"    mv {shots} {shots}_废旧_{ts}")
        return 2

    K, D = C._try_load_K(C.OUT_YAML)
    if K is None:
        say(f"\n✗ 读不到内参 {C.OUT_YAML} —— 手眼必须有内参。")
        return 2

    # ---- 连上（只读）----
    arm = Arm()
    arm.connect()
    st = arm.st
    q0 = [float(v) for v in arm.joint_deg()]
    p0 = [float(v) for v in arm.fk(q0)]
    lim = arm.r.joint_soft_limits()
    say(f"\n当前关节 {[round(v, 3) for v in q0]}")
    say(f"当前位姿 {[round(v, 3) for v in p0]}")
    d_aim = [q0[i] - AIM_JOINTS[i] for i in range(6)]
    worst = max(abs(v) for v in d_aim)
    if worst > AIM_ABORT_DEG:
        say(f"\n✗ 当前关节偏离冻结瞄姿 {worst:.2f}°（> {AIM_ABORT_DEG:g}°）——"
            f"不在瞄姿上取景就废了。先把臂回到瞄姿：")
        say(f"    {[round(v, 3) for v in AIM_JOINTS]}")
        say("    （手动点动/拖动都行；差一点点我先说清楚再跑）")
        arm.close()
        return 3
    if worst > AIM_WARN_DEG:
        say(f"  ⚠ 偏离瞄姿 {worst:.2f}°（{['%.2f' % v for v in d_aim]}）——"
            f"小幅可接受，取景会略偏，以每帧的门为准")
    else:
        say(f"  ✓ 在瞄姿上（最大差 {worst:.2f}°）")

    pre = [("急停未按下", st.estop == 0, f"estop={st.estop}"),
           ("无错误码", st.error_code == 0, f"error_code={st.error_code}"),
           ("当前不在运动中", st.motion_done == 1, f"motion_done={st.motion_done}"),
           ("自动模式", st.robot_mode == 0, f"robot_mode={st.robot_mode}")]
    bad = [c for c in pre if not c[1]]
    for name, ok, detail in pre:
        say(f"  {'✓' if ok else '✗'} {name}（{detail}）")
    if bad and args.go:
        say(f"\n✗ 前置不满足：{'、'.join(c[0] for c in bad)}——不发任何指令。")
        if st.error_code != 0:
            say("  error_code≠0：python3 -I src/clear_error.py 看故障原文（--clear 清除）。")
        if st.robot_mode != 0:
            say("  手动模式下不跑（约定：真动在【自动模式】做）。")
        arm.close()
        return 4
    if bad and args.dry:
        say("  （DRY 只读，这几项不拦；真跑时会在这一步停下。）")

    # ---- 全议程解算 + 核验 ----
    fk6 = M.fk_throttled(arm.fk, 0.01)
    plan = build_plan(arm.fk, fk6, q0, p0, lim, say)
    n_ok = sum(1 for _tag, _t, _k, t, _sp in plan if t is not None and not t["why"])
    say("\n  议程核验结果（按执行顺序）：")
    for ln in table_lines(q0, plan):
        say(ln)
    say(f"\n  可用候选 {n_ok} 条（主+备），目标 ≥12 张、舒适 16~18 张。")
    if n_ok < C.HE_HARD_SHOTS:
        say(f"  ⚠ 可用候选不足硬底线 {C.HE_HARD_SHOTS} —— 别跑，把上面的数给我看。")
        arm.close()
        return 3
    est_min = (len(PLAN) * 30 + 240) / 60.0
    say(f"  预计用时 ~{est_min:.0f} 分钟（每条一次移动+等画面追上；含平移间的回瞄姿）")

    if args.dry:
        say("\n[DRY] 全部只读完成，没发任何指令、没开相机、没存帧没写 csv（只有本 log）。")
        say("  真跑（手放急停）：python3 -I src/he_collect.py --go")
        arm.close()
        return 0

    # ---- GO：相机 + 前检 + 开跑 ----
    say(f"\n【真跑】速度 {SPEED:g}%、逐条 MoveJ、每帧过门才存。急停握好，浏览器取流关掉。")
    if input("  输 GO 回车 = 开跑；其他 = 取消：").strip().upper() != "GO":
        say("已取消，未发任何指令。")
        arm.close()
        return 1

    det = C.BoardDetector()
    cam = None
    runner = Runner(say, shots, ts, det, K, D)
    try:
        ret = arm.enable()
        say(f"\n使能返回 {ret}（成功码 1）")
        if int(ret) != 1:
            raise RuntimeError("使能失败")
        time.sleep(0.5)
        q_now = [float(v) for v in arm.joint_deg()]
        arm.move_j(q_now, speed=SPEED)
        say("[签名自检] 原地 MoveJ 通过（零位移）")

        cam = Cam()
        say(f"相机已开 /dev/video{cam.dev} {cam.w}×{cam.h}")

        def gate(frame):
            """内参版放宽门：≥20/24 角点 + 板宽 ≥300px + 角点离边 ≥4px（允许切边/部分板）。"""
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            r = BR.analyze(gray, det, K)
            if r is None:
                return None, False, ["0 角点：板不在画面里或认不出"]
            if r.get("bad"):
                return r, False, [r["bad"]]
            bad = []
            if r["n"] < 20:
                bad.append(f"角点只有 {r['n']}/24（门槛 20）")
            if r["side"] < 300.0:
                bad.append(f"板宽 {r['side']:.0f}px < 300px")
            pts = r["pts"]
            h, w = frame.shape[:2]
            m = min(pts[:, 0].min(), pts[:, 1].min(), w - 1 - pts[:, 0].max(), h - 1 - pts[:, 1].max())
            if m < 4.0:
                bad.append(f"角点贴边（离边 {m:.0f}px < 4px）")
            return r, len(bad) == 0, bad

        # ---- 首帧（基准；含"画面是否已追上"自检）----
        a = cam.raw()
        time.sleep(0.7)
        b = cam.raw()
        if float(cv2.absdiff(a, b).mean()) > FC.FRESH_STILL_MEAN:
            say("  （开场画面在动——按追赶逻辑等稳）")
            b = cam.fresh(say)
        r, ok, why = gate(b)
        say("\n[基准帧]")
        say("    " + fmt_r("基准帧", r))
        if not ok:
            say(f"✗ 基准帧都不过门（{'；'.join(why)}）——取景退化了，别开跑，先给我看。")
            raise RuntimeError("基准帧不过门")
        runner.save_shot("aim", b, r, say)
        R0, t0 = runner.saved[0][3], runner.saved[0][4]

        def aim_checkpoint(why):
            """回瞄姿做"板没动过"核验（铁律①）。返回实测位姿。"""
            q_now = [float(v) for v in arm.joint_deg()]
            p_now = [float(v) for v in arm.fk(q_now)]
            if float(np.linalg.norm(np.asarray(p_now[:3]) - np.asarray(p0[:3]))) > 1.0:
                arm.move_j([float(v) for v in q0], speed=SPEED)
                ok_a, dp_a, da_a, _ = VS.arrival(arm, p0, timeout=45.0)
                if not ok_a:
                    raise RuntimeError(f"回瞄姿没到位（{dp_a:.2f}mm/{da_a:.2f}°）")
            fr = cam.fresh(say)
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            corners, ids = det.detect(gray)
            rr = BR.analyze(gray, det, K)
            if R0 is None or corners is None or len(corners) < 12:
                say(f"      ⚠ 核验帧认不出板（{0 if corners is None else len(corners)} 角点）"
                    f"{'、且基准帧板姿就没解出' if R0 is None else ''}——停下来看一眼现场（板还在吗？）。")
                raise RuntimeError("核对帧认不出板")
            R_cb, t_cb = C._solve_board(det, corners, ids, K, D)
            dp = float(np.linalg.norm(np.asarray(t_cb) - np.asarray(t0)))
            dr = float(np.linalg.norm(M.rotvec_deg(R_cb.T @ np.asarray(R0))))
            okm = dp <= BOARD_MOVED_MM and dr <= BOARD_MOVED_DEG
            say(f"      [核验{why}] 板位姿漂移 {dp:.2f}mm/{dr:.2f}°  "
                f"{'✓ 板没动' if okm else '✗✗ 板动过了！'}")
            runner.log_row(f"CHECK@{why}", "-", None, rr, int(okm),
                           f"板漂移 {dp:.2f}mm/{dr:.2f}°")
            runner.flush_log()
            if not okm:
                raise RuntimeError(f"板动过了（{dp:.2f}mm/{dr:.2f}°）——整批作废，停")
            return t_cb

        n_check = len(runner.saved)      # 基准帧不算（它就在瞄姿上）
        for i, (tag, tier, kind, t, sp) in enumerate(plan, 1):
            if t is None or t["why"]:
                say(f"\n[{i}/{len(plan)}] {tag}：预演已判跳过（{'；'.join(t['why']) if t else '—'}）")
                continue
            if kind == "t":
                span_mm = math.sqrt(sum(float(v) ** 2 for v in sp[:3]))
                say(f"\n[{i}/{len(plan)}] {tag}（平移 {span_mm:.0f}mm）——先回瞄姿再走（顺带核验板没动）")
                aim_checkpoint(tag)
                n_check = len(runner.saved)
            say(f"\n[{i}/{len(plan)}] {tag}")
            try:
                p_tgt = [float(v) for v in arm.fk(t["q"])]
                d_now = float(np.linalg.norm(np.asarray(p_tgt[:3]) - np.asarray(st.tcp[:3])))
                if d_now > MOVE_MAX_MM:
                    raise SkipStep(f"本条离当前位 {d_now:.0f}mm > {MOVE_MAX_MM:g}mm（链上过远）")
                arm.move_j(t["q"], speed=SPEED)
                ok_arr, dp_a, da_a, t_arr = VS.arrival(arm, p_tgt, timeout=45.0)
                say(f"      到位 {'✓' if ok_arr else '✗'}（距目标 {dp_a:.3f}mm/{da_a:.3f}°、{t_arr:.1f}s）")
                if not ok_arr:
                    raise RuntimeError(f"没到位（error_code={st.error_code}）——"
                                       f"先 clear_error 看故障原文，不继续")
                dq_live = max(abs(float(v) - t["q"][k]) for k, v in enumerate(st.joints))
                if dq_live > J_VERIFY_STOP:
                    raise RuntimeError(f"关节回读与目标差 {dq_live:.2f}°（> {J_VERIFY_STOP:g}°）——"
                                       f"停下查（error_code={st.error_code}）")
                if dq_live > J_VERIFY_WARN:
                    say(f"      ⚠ 关节回读差 {dq_live:.2f}°（位姿到位、关节小残差，继续）")
                fr = cam.fresh(say)
                r, ok, why = gate(fr)
                if r is None:
                    say("      ✗ 认不出板（0 角点）")
                elif not ok:
                    say(f"      ✗ 不过门：{'；'.join(why)}")
                if ok:
                    nm = runner.save_shot(tag, fr, r, say)
                    if nm is None:
                        runner.log_row(tag, tier, t, r, 0, "重复帧（与已存几乎同帧）")
                    else:
                        runner.log_row(tag, tier, t, r, 1, "ok")
                        if len(runner.saved) - n_check >= CHECK_EVERY:
                            say(f"      （已连存 {CHECK_EVERY} 张——回瞄姿核验一次板没动过）")
                            aim_checkpoint(f"每{CHECK_EVERY}张")
                            n_check = len(runner.saved)
                else:
                    runner.log_row(tag, tier, t, r, 0,
                                   ("认不出板" if r is None else "；".join(why)))
                runner.flush_log()
            except SkipStep as e:
                say(f"      ⏭ 跳过：{e}")
                runner.log_row(tag, tier, t, None, 0, f"跳过：{e}")
                runner.flush_log()
        # ---- 收尾 ----
        say("\n[收尾] 回瞄姿 + 最终核验")
        aim_checkpoint("final")
    except SkipStep as e:                                    # noqa: BLE001
        say(f"\n⏭ {e}")
    except Exception as e:                                   # noqa: BLE001
        say(f"\n✗✗ 中断：{type(e).__name__}: {e}")
        if isinstance(e, CamError):
            say("   （相机打不开——先关掉浏览器里的取流（相机独占），再重跑；已存的帧还在。）")
        elif "error_code" in str(e) or int(getattr(st, "error_code", 0)) != 0:
            say("   （控制器有错误码——python3 -I src/clear_error.py 看故障原文。）")
        try:
            if int(getattr(st, "motion_done", 1)) != 1:
                say("   （控制器仍在动——等停稳，最多 60s）")
                arm.wait_done(timeout=60.0)
        except Exception:                                    # noqa: BLE001
            pass
    finally:
        if cam is not None:
            cam.close()
        runner.flush_log()
        saved = runner.saved
        say("\n" + "=" * 74)
        say(f"采集结束：{len(saved)} 张 → {shots}")
        say(f"  csv {C.HE_POSES_CSV}（原始法兰位姿）+ collect_log.csv（每次尝试明细）")
        if len(saved) >= 16:
            say("\n  解算（与 D 批合起来解，判据不过不写 yaml，放心跑；先别覆盖正式内参）：")
            say(f"    python3 -I src/calibrate.py --solve --shots {shots} {C.HE_SHOTS_DIR} "
                f"--out /tmp/intr_v2.yaml")
        else:
            say(f"  ⚠ 只有 {len(saved)} 张（目标 ≥40）——把上面的日志给我看再定")
        say(f"  臂【保持使能】，实测停位姿 {[round(float(v), 2) for v in st.tcp]}")
        arm.close()
    return 0


# ---------------------------------------------------------------- 离线自测

def _selftest():
    """假臂（6R DH 链）冒烟：解算/闸/表格全走一遍，不连机器人。"""
    DH = [(30.0, 90.0, 152.0), (400.0, 0.0, 0.0), (30.0, 90.0, 0.0),
          (0.0, -90.0, 380.0), (0.0, 90.0, 0.0), (0.0, 0.0, 90.0)]

    def fake_fk(q):
        T = np.eye(4)
        for qi, (a, al, d) in zip(np.asarray(q, float), DH):
            Rz, Tz, Tx, Rx = (np.eye(4) for _ in range(4))
            Rz[:3, :3] = M.rot_z(qi)
            Tz[2, 3] = d
            Tx[0, 3] = a
            Rx[:3, :3] = M.rot_x(al)
            T = T @ Rz @ Tz @ Tx @ Rx
        return M.T_to_pose6(T)

    q0 = [13.459, -129.064, 82.882, -71.318, -21.607, 46.851]
    p0 = fake_fk(q0)
    lim = [(-175.0, 175.0)] * 6
    say = print
    plan = build_plan(fake_fk, fake_fk, q0, p0, lim, say)
    for ln in table_lines(q0, plan):
        say(ln)
    n = sum(1 for _t, _u, _k, tt, _s in plan if tt is not None and not tt["why"])
    print(f"\n自测：可用候选 {n} 条 / 共 {len(plan)} 条（假臂只是冒烟，数字不必好看）")
    assert n >= 5, "假臂下可用候选过少——规划逻辑有毛病"
    print("自测通过（纯离线，未连机器人、未开相机）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
