# -*- coding: utf-8 -*-
"""手眼补采 · append 版 v2：往 16:11 批（35 帧）后面补 5 帧绕 miss 的复合腕部动作。

为什么补：35 帧批解算除 ③ 全绿；③ 的根因不是数据脏，是**转动轴云第三方向缺口**
（λ3=0.044，miss(法兰系)=[−0.620,+0.757,−0.206]）——X 沿 miss 的旋转分量只被
钉到 ~1°（五解法旋转散 1.252° 同证）。绕 miss 转 ±θ 的帧对能把该分量钉死。
（③ 的"旋转 flag"本身已定性为噪声误报——补采是为了让判据过得干净，不是救火。）

v1（2026-10-16 17:00 跑）5 帧全废，v2 的三处改动：
  1) **取帧闸重做**：v1 的"跳变/稳帧"判据被相机流的"慢放回放"骗过——038/039 拍到
     移动中途（外框错配 112/121px、闭环 26mm）、035~037 也带 1.3~2.3mm 滞后尾。
     v2 双闸：①相邻取样板姿自洽（X 无关；流还在回放就会超差）②宽松绝对闸（vs
     X_v4 预测，兜"停滞回放"）。见 SAMPLE_*/AGREE_*/GATE_* 常量。
  2) **板位变了**：v1 跑步中 17:03 板被外力移动 19.15mm/2.94°（臂够不到板：相机-板
     距 ≥360mm；重拍帧与批内帧差 19.16mm/2.94°，两条独立路子对账一致）。本批以现
     板位为新基准 B_new = T_aim @ X_v4 @ T_cb(基准帧)。旧 35 帧仍在旧板位——解算
     端组内配平（AX=XB 只在同一板位组内成立，整批丢进去必跨组污染）。
  3) **候选重解**：5 条 Δq 按"板位移补偿"重解（dp 加基座系 (−13.2,−5.7,+12.7)mm
     的负向平移 → 相机-板相对几何复原 → 取景回到原设计余量）。复现自检：原 5 条
     不带补偿重解 ≤0.05° 全部复现（配方核对✓）。新 5 条账目：残差 ≤0.10/0.14、
     外框最紧 ≥78px、J3 51.3~121.5 ∈ [50,130]、|q5| ≥25.4、肘 z 342~405 全 ≥328、
     TCP ≤72mm。P16 额外扫过补偿系数×微调才把 J3 拉回 ≥50（α0.9 + −y 6mm）。

与 he_collect.py（母本）的差异，仅六处：
  1) clean-dir 闸 → append 闸：要求目录恰好 35 帧、jpg 与 csv 逐一对齐（多/少/错位都停）；
  2) Runner 预载 35 行 saved → idx 从 35 起、文件名 he_<新ts>_035..039、dup 检查覆盖旧帧；
  3) 基准帧不存新帧：现帧解出新板位 B_new（本批全部闭环闸/核验的锚，v1 的"vs 批内第 0
     帧"核验已删——板位移是已知事实，不再是异常）；
  4) PLAN = 5 条【离线预解好的复合 Δq】（"j" 型），含板位移补偿；数字/账目见 PLAN 注释；
  5) J3_FENCE 60→50、105→130（肘部实测记账见常量注释）；
  6) 备用幅度条件化：只有主不过门才跑（FB_SCALE=0.75），主过了就不浪费一次移动。

其余照旧：全程不发控制器逆解（禁区）、MoveJ 8%、每帧过门才存、到位判据用位姿、
"链上过远先回瞄姿"（顺带板核验）、收尾自洽性/多样性/定尺度统计。臂保持使能。

用法（相机独占——先关浏览器取流；手放急停）：
    python3 -I src/he_collect_extra.py --dry   # 只读预演：append 闸+解算核验+取景预检
    python3 -I src/he_collect_extra.py --go    # 真跑：输 GO 开跑（5 帧 ≈ 5~9 分钟）

存证：data/handeye_shots/ 追加 he_<新ts>_035..039.jpg + flange_poses.csv 续写；
      collect_log.csv、run_<新ts>.txt、collect_<新ts>.rpc.log（黑匣子）。
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrate as C                                          # noqa: E402
import config                                                  # noqa: E402
import fr3_math as M                                           # noqa: E402
import fr_cam as FC                                            # noqa: E402
import view_sweep as VS                                        # noqa: E402 （arrival/Tee 复用）
from fr_arm import Arm                                         # noqa: E402
from fr_cam import Cam, CamError                               # noqa: E402
from probe import board_report as BR                           # noqa: E402

# v2：fresh() 的兜底时间收紧（取样自洽闸在后面兜底）——50s→30s、20s→15s。
# fresh 只负责"过掉伺服后的旧画面期"，真静止判定在 sample_frame 里。
FC.FRESH_WAIT_MAX_S = 20.0
FC.FRESH_STILL_MIN_S = 8.0

# ---------------------------------------------------------------- 参数

SPEED = VS.SPEED               # 8%（与 view_sweep 同档，侦察级）
AIM_JOINTS = [13.459, -129.064, 82.882, -71.318, -21.607, 46.851]   # 冻结瞄姿（16:11 批的基准）
AIM_ABORT_DEG = 2.0            # 当前关节偏离瞄姿超过它 → 拒绝（先回瞄姿再跑）
AIM_WARN_DEG = 0.5
LIMIT_MARGIN = 3.0             # 软限位余量（度）
Q5_MIN = 12.0                  # |q5| 下限（腕部奇异带，见 memory: fairino-wrist-singularity-q5-zero）
J2_FENCE = (-150.0, -105.0)    # 家族围栏：深家族（J2≈-200）就是撞支架那族，见 memory: fr3-elbow-geometry
J3_FENCE = (50.0, 130.0)
# ↑ 2026-10-16 晚放宽（母本原值 60/105）。记账：
#   ① 肘部位置只由 J1/J2 决定——圆拟合探针实测（固定 J1/J2/J4/J5/J6，J3 扫 8 点 → 平面+圆
#      拟合，圆心 = J3 轴点；瞄姿复现 z=357.4/半径 331.8，与 10-15 历史扫描表逐点吻合）；
#   ② 5 条新候选的肘 z 全部 328~398（比瞄姿 357 高或持平；碰撞实锤 z≈40，报险线 z<120）；
#   ③ 历史先例：10-15"回家路线②"在 J2≈−129 把 J3 从 +82.9 扫到 −78.7，干净无碰；
#   ④ 新增区 J3 109~125 仅 −dp 取景补偿（相机在板前下移/后退）触达，肘部只会更远离桌。
#   控制器碰撞检测全程 armed、8% 档——真碰上也是慢速停，不是撞废。
CAP_J1 = 20.0                  # 平移解的关节跳上限（母本口径，沿用；本批无 t 型，运行期不触发）
CAP_J2 = 10.0
CAP_J3 = 22.0
CAP_WRIST = 45.0
MOVE_MAX_MM = 95.0             # 单条位移上限（config.MAX_MOVE_MM=100 再留 5mm）
RES_POS_MM = 0.6               # 数值解残差闸（本批 Δq 是离线解好的，运行时不再解算；
RES_ANG_DEG = 0.35             #   这两个值只用于记录那段的历史口径，运行期不触发）
J_VERIFY_STOP = 1.5            # 到位后关节回读 vs 目标的停车线（度）
J_VERIFY_WARN = 0.3
CHECK_EVERY = 8                # 每存 N 张回瞄姿核验（本批只加 5 张，不会触发）
BOARD_MOVED_MM = 2.0           # 板动过判据（与 auto_handeye 同）
BOARD_MOVED_DEG = 0.5
N_OLD = 40                     # 已有帧数（append 底板；jpg 与 csv 行必须逐一同名对齐）
#   ↑ 2026-10-16 17:32 批 35→40（035–039 那 5 帧先于本批存在；A/B 复核同目标重访用）
FB_SCALE = 0.75                # 备用幅度 = 主的 0.75×（复合动作等比缩角，取景更保险）

# ---- v2 取帧双闸（035~039 被"流慢放回放"污染后的重做）----
GATE_MM = 18.0                 # 绝对闸：|t_cb 观测 − X_v4 预测|。宽松（X_v4 自身误差同量级），
GATE_DEG = 4.0                 #   只为兜"停滞回放"（画面冻在中途帧上不动）
AGREE_MM = 0.7                 # 自洽闸（X 无关）：相邻取样板姿平移差——流还在回放就超它
AGREE_DEG = 0.15               #   连续 AGREE_N 对（=连稳 N+1 帧）才算流静止
AGREE_N = 3                    # 2026-10-16 17:52 加强：2→3（慢回放尾巴能以 <0.23mm/s
                               #   骗过 2 对；3 对+SETTLE 12s 让采样点必然落在"回放已追平"之后）
SAMPLE_S = 3.0                 # 取样间隔（s）
SAMPLE_MAX = 8                 # 每个位姿最多取样次数；超了跳过该位姿不存（宁少勿污）
SETTLE_S = 12.0                # fresh() 返回后再静置这么久才开取样（17:52 加强：6→12）
X_NPZ = "/tmp/X_v4.npz"        # 35 帧手眼解（闭环预测/板位基准都要它）
INTR_LOCK = "/tmp/intr_v4.yaml"  # 锁焦内参——闭环闸的 K 必须与解算同源，否则有常数偏
FRAME_MIN_MARGIN_PX = 55.0     # 跑前取景预检：任何目标外框边距低于它 → 停下报告（别硬跑）
IMG_W, IMG_H = 1920, 1080      # 取流尺寸（fr_cam 实测 1920×1080）

# 整板外框（175×125mm）——取景口径与运行门 BR 的"全板框"一致（不是检出角点内框！）
_nx, _ny = config.CHARUCO_SQUARES
OUTLINE = np.array([[0.0, 0.0, 0.0], [_nx * config.CHARUCO_SQUARE_MM, 0.0, 0.0],
                    [_nx * config.CHARUCO_SQUARE_MM, _ny * config.CHARUCO_SQUARE_MM, 0.0],
                    [0.0, _ny * config.CHARUCO_SQUARE_MM, 0.0]])

# 议程：5 条【离线预解好的复合腕部动作】（"j" 型：{关节索引: 增量度}，运行时直接加在
# 实时瞄姿关节上——数值解只在桌面上做，臂上只做 MoveJ）。
# 目的：绕 X 旋转的缺口方向 miss（转动轴云第三特征向量；法兰系 [−0.620,+0.757,−0.206]，
#   基座系 [+0.538,−0.044,+0.842]）转 ±θ——配对最大杠杆 36°，把 X 沿 miss 的分量钉死。
# 来源（v2）：/tmp/xres2.py + xres3.py —— solve_delta(drot=±θ·miss_base,
#   dp=原取景补偿 + 板位移补偿, steps=6, iters=26)。板位移补偿 = B_old_R @ t(B_old⁻¹B_new)
#   = 基座系 (−13.2,−5.7,+12.7)mm（相机跟着板走同样一段 → 相对几何复原 → 取景回原余量）。
#   复现自检：原 5 条（不带补偿）重解 ≤0.05° 全复现 → 配方与原设计同源 ✓
# 每条账目（v2 新解）：残差 ≤0.10mm/0.14°、∥轴·miss∥=1.000、外框最紧 ≥78px、
#   TCP 位移 ≤72mm、|q5| ≥25.4°、J3 51.3~121.5 ∈ [50,130]、肘 z 342~405（报险线 120）。
#   （P16 的 J3 在纯平移补偿下掉到 46.4 破围栏 → 扫 α×微调，取 α0.9 + 基座 −y 6mm：
#     J3 51.3、外框 78px——与原设计账 (51.6/80px) 持平。）
# 不查 CAP_* 是刻意的：大角度复合动作本就会超腕 cap，但每条都是逐条验证过的干净分支
#   （残差+取景双过）；且 MoveJ 走关节直插，没有平移解那种"分支翻转"的风险面。
# 运行期安全：每条 MoveJ 前重算"当前位 → 目标位"距离，超 95mm 先回瞄姿（顺带板核验）。
_PLAN_RAW = [
    ("P12", "j", {0: +2.6, 1: +13.8, 2: -23.6, 3: +32.6, 4: -3.8, 5: -22.5}),
    # +12° 绕 miss，dp=μ；外框 min 96；J3 59.3；肘 393；TCP 19.1；q5 −25.5
    ("P16", "j", {0: +5.8, 1: +18.8, 2: -31.6, 3: +36.9, 4: -5.8, 5: -23.0}),
    # +16° 绕 miss + z+12 取景 + μ + (0,−6,0)；外框 min 78；J3 51.3；肘 403；TCP 28.5；q5 −27.4
    ("M12", "j", {0: +2.9, 1: -3.3, 2: +26.5, 3: -56.3, 4: -4.8, 5: +33.7}),
    # −12° 绕 miss + z−50 取景 + μ；外框 min 110；J3 109.4；肘 347；TCP 40.0；q5 −26.4
    ("M16", "j", {0: +10.1, 1: -4.8, 2: +35.2, 3: -74.6, 4: -13.7, 5: +45.5}),
    # −16° 绕 miss + x+22 z−72 取景 + μ；外框 min 110；J3 118.1；肘 342；TCP 60.3；q5 −35.3
    ("M20", "j", {0: +19.2, 1: -3.5, 2: +38.6, 3: -86.2, 4: -25.4, 5: +53.0}),
    # −20° 绕 miss + x+38 z−80 取景 + μ；外框 min 90；J3 121.5；肘 346；TCP 72.0；q5 −47.0
]
PLAN = [(tag, kind, spec, {k: FB_SCALE * v for k, v in spec.items()})
        for tag, kind, spec in _PLAN_RAW]


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
    say("\n[预演] 全部候选核验中（本批是预解好的关节增量，只跑正解，很快）…")
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

    def preload(self, rows):
        """把已有 35 行当成已存帧：idx 从 35 起、dup 检查覆盖旧帧、csv 续写靠整表重写。"""
        for name, pose in rows:
            self.saved.append((name, pose, None, None, None, None))

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
        for n_old, p_old, *_ in self.saved:           # 含预载的 35 帧（新帧不许与旧帧同框）
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


def diff_pose(p, q):
    dp = float(np.linalg.norm(np.asarray(p[:3]) - np.asarray(q[:3])))
    dr = float(np.linalg.norm(M.rotvec_deg(M.pose6_to_T(p)[:3, :3] @ M.pose6_to_T(q)[:3, :3].T)))
    return dp, dr


def main():
    ap = argparse.ArgumentParser(
        description="手眼补采（append）：--dry 只读预演 / --go 真跑（5 帧）")
    ap.add_argument("--dry", action="store_true", help="只读预演：append 闸+全部核验+打印议程，不发指令、不开相机")
    ap.add_argument("--go", action="store_true", help="真跑：手放急停，程序会等 GO")
    args = ap.parse_args()
    if not (args.dry or args.go):
        sys.exit("要 --dry（只读预演）或 --go（真跑）——见文件头用法")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shots = C.HE_SHOTS_DIR
    shots.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FR_RPC_LOG", str(shots / f"collect_{ts}.rpc.log"))
    say = VS.Tee(shots / f"run_{ts}.txt")

    say("=" * 74)
    say(f"手眼补采·append（绕 miss 复合腕部动作 ×5）· {ts} · "
        f"{'【DRY 只读预演】' if args.dry else '【GO 真跑】'}")
    say("=" * 74)

    # ---- append 闸（纯文件操作，先做——目录不对根本不连臂）----
    # 必须在 16:11 批的原状上续：jpg 数、csv 行数都恰好 35 且逐一同名对齐。
    # 任何多/少/错位都说明目录被动过——停，宁可不采也不污染可倒推的数据链。
    old_jpg = sorted(shots.glob("*.jpg"))
    csv_path = shots / C.HE_POSES_CSV
    rows = C._he_read_csv(csv_path) if csv_path.exists() else []
    names_jpg = [p.name for p in old_jpg]
    names_csv = [n for n, _p in rows]
    if len(names_jpg) != N_OLD or len(names_csv) != N_OLD or names_jpg != names_csv:
        say(f"\n✗ {shots} 不是预期的原状（应有 {N_OLD} 帧、jpg/csv 逐一同名）："
            f"jpg {len(names_jpg)} 张、csv {len(names_csv)} 行"
            f"{'，且名字对不齐' if len(names_jpg) == len(names_csv) and names_jpg != names_csv else ''}。")
        say(f"  第一张应 ≈ he_20261016_161142_000.jpg，现第一张是 "
            f"{names_jpg[0] if names_jpg else '（无）'}。")
        say("  补采只允许在 16:11 批自己的 35 帧上追加。先查这个目录（谁动过？），别急着跑。")
        return 2
    say(f"\n✓ append 闸：{len(names_jpg)} 帧 jpg/csv 逐一对齐（{names_jpg[0]} … {names_jpg[-1]}）")

    K, D = C._try_load_K(Path(INTR_LOCK))
    if K is None:
        say(f"\n✗ 读不到锁焦内参 {INTR_LOCK} —— 闭环闸的 K 必须与 X_v4 同源（fx=1462），停。")
        return 2
    xp = Path(X_NPZ)
    if not xp.exists():
        say(f"\n✗ 缺 {X_NPZ}（35 帧手眼解）——闭环预测/新板位基准都要它，停。")
        return 2
    zx = np.load(xp)
    X_SOL, B_OLD = zx["X"], zx["T_b_board"]
    say(f"✓ 锁焦内参 {INTR_LOCK}（fx={float(K[0, 0]):.0f}）+ 35 帧解 {X_NPZ}")

    def framing_table(B, fk, q_base, say, abort_below=None):
        """跑前取景预检：按整板外框投影核验 5 条目标（门 ≥60px/边）。
        返回最紧边距(px)。abort_below 非空且低于它 → RuntimeError（别硬跑）。"""
        say("  取景预检（整板外框投影；门：每边 ≥60px）：")
        worst = None
        for tag, _kind, spec, _fb in PLAN:
            q = [q_base[i] + spec.get(i, 0.0) for i in range(6)]
            T_f = M.pose6_to_T([float(v) for v in fk(q)])
            T_cmb = np.linalg.inv(T_f @ X_SOL) @ B
            pc = (T_cmb[:3, :3] @ OUTLINE.T).T + T_cmb[:3, 3]
            uv = (K @ (pc / pc[:, 2:3]).T).T[:, :2]
            m_l = float(uv[:, 0].min()); m_r = IMG_W - float(uv[:, 0].max())
            m_t = float(uv[:, 1].min()); m_b = IMG_H - float(uv[:, 1].max())
            mm = min(m_l, m_r, m_t, m_b)
            say(f"    {tag:<5} 边距 左{m_l:4.0f} 右{m_r:4.0f} 上{m_t:4.0f} 下{m_b:4.0f}px  "
                f"最紧 {mm:3.0f}px  {'✓' if mm >= 60 else '✗ 不过门'}")
            worst = mm if worst is None else min(worst, mm)
        if abort_below is not None and worst < abort_below:
            raise RuntimeError(f"取景预检最紧边距 {worst:.0f}px < {abort_below:.0f}px —— 先别跑，"
                               f"把上面的表给我看（板又动了？）")
        return worst

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

    # ---- 全议程核验 ----
    fk6 = M.fk_throttled(arm.fk, 0.01)
    plan = build_plan(arm.fk, fk6, q0, p0, lim, say)
    n_main = sum(1 for _tag, tier, _k, t, _sp in plan
                 if tier == "主" and t is not None and not t["why"])
    say("\n  议程核验结果（按执行顺序）：")
    for ln in table_lines(q0, plan):
        say(ln)
    say(f"\n  可用主候选 {n_main}/5（备 = 主的 0.75×，仅主不过门时跑）。"
        f"补采目标 = 5 帧（主全过即达）。")
    if n_main < 5:
        say("  ⚠ 有主候选不过预检——先别跑，把上面的数给我看"
            "（备只是救援，救不出主要的设计缺口）。")
        arm.close()
        return 3
    est_min = (len(PLAN) * 30 + 240) / 60.0
    say(f"  预计用时 ~{est_min:.0f} 分钟（每条一次移动+等画面追上）")

    if args.dry:
        # 取景预检（离线板位 = 17:10 重拍帧；真跑会用现场基准帧重算并硬拦）
        xref = Path("/tmp/xreframe_last.jpg")
        if xref.exists():
            det_x = C.BoardDetector()
            cx, ix = det_x.detect(cv2.imread(str(xref), cv2.IMREAD_GRAYSCALE))
            R_x, t_x = C._solve_board(det_x, cx, ix, K, D)
            if R_x is not None:
                T_x = np.eye(4); T_x[:3, :3] = R_x; T_x[:3, 3] = t_x
                B_off = M.pose6_to_T(rows[0][1]) @ X_SOL @ T_x
                say("\n[取景预检]（离线板位=17:10 重拍帧；真跑用现场基准帧）")
                framing_table(B_off, arm.fk, [float(v) for v in q0], say, abort_below=None)
            else:
                say("\n[取景预检] 重拍帧板姿解不出——跳过（真跑时现场核）")
        else:
            say("\n[取景预检] 缺 /tmp/xreframe_last.jpg——跳过（真跑时现场核）")
        say("\n[DRY] 全部只读完成，没发任何指令、没开相机、没存帧没改 csv（只有本 log）。")
        say("  真跑（手放急停）：python3 -I src/he_collect_extra.py --go")
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
    runner.preload(rows)                      # ★ 预载 35 行：idx 从 35 起、csv 续写
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

        def flange_T():
            return M.pose6_to_T(C._flange_pose())

        def sample_frame(say, pred=None, what="", hard=False):
            """v2 取帧闸：取"流已静止"的一帧。每 SAMPLE_S 一抓：
            ① 每次都要过 BR 门（24/24 角点 + 边距）；② 相邻两次板姿差 ≤AGREE_MM/DEG
            连续 AGREE_N 对（=连稳 N+1 帧）→ 认定流静止；③ 给了 pred 再查绝对闸 ≤GATE_MM/DEG
            （兜"画面冻在中途帧"的停滞回放）。返回 (frame, r, R_cb, t_cb, 连稳数)。
            等不到：hard=True 抛 RuntimeError（基准/核验用），否则抛 SkipStep（单条跳过）。"""
            prev = None; agree = 0; frame = None; r = None
            last_abs = "—"
            t0s = time.time()
            for k in range(1, SAMPLE_MAX + 1):
                frame = cam.raw()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                r = BR.analyze(gray, det, K)
                ok, why = BR.verdict(r)
                if not ok:
                    say(f"      · 取样{k}：不过门（{'；'.join(why)}）")
                    prev = None; agree = 0
                else:
                    corners, ids = det.detect(gray)
                    R_cb, t_cb = C._solve_board(det, corners, ids, K, D)
                    if R_cb is None:
                        say(f"      · 取样{k}：板姿解不出")
                        prev = None; agree = 0
                    elif prev is None:
                        prev = (np.asarray(R_cb), np.asarray(t_cb))
                        say(f"      · 取样{k}（{time.time() - t0s:.0f}s）：首帧，开始连稳")
                    else:
                        dp = float(np.linalg.norm(np.asarray(t_cb) - prev[1]))
                        dr = float(np.linalg.norm(M.rotvec_deg(R_cb @ prev[0].T)))
                        agree = agree + 1 if (dp <= AGREE_MM and dr <= AGREE_DEG) else 0
                        msg = (f"      · 取样{k}（{time.time() - t0s:.0f}s）  "
                               f"与上次差 {dp:.2f}mm/{dr:.2f}°  连稳{agree}")
                        abs_ok = True
                        if pred is not None:
                            dg = float(np.linalg.norm(np.asarray(t_cb) - pred[1]))
                            rg = float(np.linalg.norm(M.rotvec_deg(R_cb @ pred[0].T)))
                            abs_ok = (dg <= GATE_MM and rg <= GATE_DEG)
                            last_abs = f"{dg:.1f}mm/{rg:.1f}°"
                            msg += f"  绝对 {last_abs}"
                        say(msg)
                        prev = (np.asarray(R_cb), np.asarray(t_cb))
                        if agree >= AGREE_N and abs_ok:
                            return frame, r, R_cb, t_cb, agree
                        if agree >= AGREE_N and not abs_ok:
                            say("        ⚠ 连稳但绝对闸不过——不存，继续等"
                                "（停滞回放的典型形态）")
                            agree = 0
                time.sleep(SAMPLE_S)
            msg = (f"{what}取样 {SAMPLE_MAX} 次仍未取到静止帧"
                   f"（流没静止/不过闸；最后绝对差 {last_abs}）")
            raise (RuntimeError if hard else SkipStep)(msg)

        # ---- 首帧（基准；append 模式不存新帧）----
        # v2：板在 17:03 被外部动过——现板位成为本批的新基准：
        # B_new = T_aim @ X_v4 @ T_cb(基准帧)。闭环闸与收尾核验都以它为锚。
        cam.fresh(say)                     # 先过掉伺服后的旧画面期（本轮上限 30s）
        time.sleep(SETTLE_S)
        say("\n[基准帧] 取静止确认帧，标定现板位…")
        b, r, R_cb, t_cb, ag = sample_frame(say, pred=None, what="基准", hard=True)
        say(f"  ✓ 基准帧取定（连稳 {AGREE_N + 1} 帧）：")
        for ln in BR.format_report("基准帧", r).splitlines():
            say("    " + ln)
        T_cb0 = np.eye(4); T_cb0[:3, :3] = R_cb; T_cb0[:3, 3] = t_cb
        R_cb0_ref, t_cb0_ref = np.asarray(R_cb), np.asarray(t_cb)
        B_NEW = flange_T() @ X_SOL @ T_cb0
        dB = np.linalg.inv(B_OLD) @ B_NEW
        dpm = float(np.linalg.norm(dB[:3, 3]))
        drm = float(np.linalg.norm(M.rotvec_deg(dB[:3, :3])))
        say(f"  [板位] 现板位 vs 批内旧板位：{dpm:.2f}mm / {drm:.2f}°"
            f"（预期 ~19mm/2.9°；差得多说明板又动了或臂不在瞄姿——停下来给我看）")
        runner.log_row("BASE", "-", None, r, 1, f"新板位 Δ{dpm:.2f}mm/{drm:.2f}°")
        runner.flush_log()
        framing_table(B_NEW, arm.fk, [float(v) for v in q0], say,
                      abort_below=FRAME_MIN_MARGIN_PX)
        say(f"  ✓ 取景预检通过（最紧 ≥{FRAME_MIN_MARGIN_PX:g}px）")

        def aim_checkpoint(why):
            """回瞄姿做"板没动过"核验（铁律①）。v2：取样器取静止帧，与【本次】基准帧比
            （旧口径 vs 批内第 0 帧作废——板位移 19mm 是已知事实，锚必须是新基准）。
            旋转公式用 R_cb @ R_cb0_ref.T（相对旋转用 A·Bᵀ，不能转置混用）。"""
            q_now = [float(v) for v in arm.joint_deg()]
            p_now = [float(v) for v in arm.fk(q_now)]
            if float(np.linalg.norm(np.asarray(p_now[:3]) - np.asarray(p0[:3]))) > 1.0:
                arm.move_j([float(v) for v in q0], speed=SPEED)
                ok_a, dp_a, da_a, _ = VS.arrival(arm, p0, timeout=45.0)
                if not ok_a:
                    raise RuntimeError(f"回瞄姿没到位（{dp_a:.2f}mm/{da_a:.2f}°）")
            cam.fresh(say)
            time.sleep(SETTLE_S)
            T_f_now = flange_T()
            T_pred = np.linalg.inv(T_f_now @ X_SOL) @ B_NEW
            fr, rr, R_cb, t_cb, ag = sample_frame(
                say, pred=(T_pred[:3, :3], T_pred[:3, 3]), what=f"核验{why}", hard=True)
            dp = float(np.linalg.norm(np.asarray(t_cb) - t_cb0_ref))
            dr = float(np.linalg.norm(M.rotvec_deg(np.asarray(R_cb) @ R_cb0_ref.T)))
            okm = dp <= BOARD_MOVED_MM and dr <= BOARD_MOVED_DEG
            say(f"      [核验{why}] 板位姿漂移 {dp:.2f}mm/{dr:.2f}°  "
                f"{'✓ 板没动' if okm else '✗✗ 板动过了！'}")
            runner.log_row(f"CHECK@{why}", "-", None, rr, int(okm),
                           f"板漂移 {dp:.2f}mm/{dr:.2f}°  连稳{ag}")
            runner.flush_log()
            if not okm:
                raise RuntimeError(f"板动过了（{dp:.2f}mm/{dr:.2f}°）——整批作废，停")
            return t_cb

        n_check = len(runner.saved)
        done_main = set()                 # ★ 主已过门的 tag（备用条件化的依据）
        for i, (tag, tier, kind, t, sp) in enumerate(plan, 1):
            if t is None or t["why"]:
                say(f"\n[{i}/{len(plan)}] {tag}：预演已判跳过（{'；'.join(t['why']) if t else '—'}）")
                continue
            if tier == "备" and tag[:-2] in done_main:
                say(f"\n[{i}/{len(plan)}] {tag}：主 {tag[:-2]} 已过门——备用不跑（同目标重试无意义）")
                runner.log_row(tag, tier, t, None, 0, "主已过门，备用跳过")
                runner.flush_log()
                continue
            say(f"\n[{i}/{len(plan)}] {tag}")
            try:
                p_tgt = [float(v) for v in arm.fk(t["q"])]
                d_now = float(np.linalg.norm(np.asarray(p_tgt[:3]) - np.asarray(st.tcp[:3])))
                if d_now > MOVE_MAX_MM:
                    say(f"      链上过远（当前位距目标 {d_now:.0f}mm > {MOVE_MAX_MM:g}mm）"
                        f"——先回瞄姿再走（顺带核验板没动）")
                    aim_checkpoint(tag)
                    n_check = len(runner.saved)
                    d_now = float(np.linalg.norm(np.asarray(p_tgt[:3]) - np.asarray(st.tcp[:3])))
                    if d_now > MOVE_MAX_MM:
                        raise SkipStep(f"回瞄姿后仍 {d_now:.0f}mm > {MOVE_MAX_MM:g}mm")
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
                cam.fresh(say)
                time.sleep(SETTLE_S)
                T_f_now = flange_T()
                T_pred = np.linalg.inv(T_f_now @ X_SOL) @ B_NEW
                fr, r, R_o, t_o, ag = sample_frame(
                    say, pred=(T_pred[:3, :3], T_pred[:3, 3]), what=tag, hard=False)
                dg = float(np.linalg.norm(np.asarray(t_o) - T_pred[:3, 3]))
                rg = float(np.linalg.norm(M.rotvec_deg(np.asarray(R_o) @ T_pred[:3, :3].T)))
                nm = runner.save_shot(tag, fr, r, say)
                if nm is None:
                    runner.log_row(tag, tier, t, r, 0, "重复帧（与已存几乎同帧）")
                else:
                    runner.log_row(tag, tier, t, r, 1, "ok",
                                   extra=f"闭环 {dg:.2f}mm/{rg:.2f}° 连稳{ag}")
                    if tier == "主":
                        done_main.add(tag)
                    if len(runner.saved) - n_check >= CHECK_EVERY:
                        say(f"      （已连存 {CHECK_EVERY} 张——回瞄姿核验一次板没动过）")
                        aim_checkpoint(f"每{CHECK_EVERY}张")
                        n_check = len(runner.saved)
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
        say(f"采集结束：共 {len(saved)} 张（旧 {N_OLD} + 新 {len(saved) - N_OLD}） → {shots}")
        say(f"  csv {C.HE_POSES_CSV}（原始法兰位姿）+ collect_log.csv（每次尝试明细）")
        if len(saved) >= 5:
            poses = [p for _n, p, *_ in saved]
            pairs = [(p, Mx) for _n, p, Mx, *_ in saved if Mx is not None]
            if len(pairs) >= 5:
                # 方向按 calibrate._he_axis_match 文档：要传【相机→板 = _solve_board 返回值的转置】，
                # 即这里的 Mx 本身（Mx 就是 R_cb.T）。传 Mx.T 会顺时针转置——好数据也报 10°+。
                am = C._he_axis_match([p for p, _ in pairs], [Mx for _, Mx in pairs])
                if am is not None:
                    med, mx, npr = am
                    tag_ = ("  ← 数据不自洽！" if med > C.HE_AXIS_BAD_DEG else
                            ("  ← 偏大" if med > C.HE_AXIS_GOOD_DEG else " ✓"))
                    say(f"  数据自洽性（轴对应残差，θ>15° 共 {npr} 对）：中位 {med:.2f}° "
                        f"最大 {mx:.2f}°{tag_}")
                else:
                    say("  数据自洽性：θ>15° 的对不足（旋转幅度对少）")
            axes = C._he_axis_eigs(poses)
            if axes is not None:
                w, V = axes
                say(f"  转动轴多样性：轴 {w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}"
                    f"{' ✓' if w[2] >= C.AXIS_MIN_EIG else '  ← 退化！'}")
                say(f"    （补采前 λ3=0.044——这条就是要看它升没升）")
            # 定尺度对（k 自检前提）：姿态几乎不变 + 位移 > 40mm（姿态冻结时相机位移=法兰位移）
            pins, span60 = 0, 0.0
            Ts = [M.pose6_to_T(p) for p in poses]
            for a in range(len(Ts)):
                for b2 in range(a + 1, len(Ts)):
                    dt = float(np.linalg.norm(Ts[a][:3, 3] - Ts[b2][:3, 3]))
                    th = float(C._rot_axis_angle(Ts[a][:3, :3].T @ Ts[b2][:3, :3])[0])
                    if th <= 0.7 and dt > 40.0:
                        pins += 1
                        span60 = max(span60, dt)
            say(f"  定尺度对（姿态不变+位移>40mm）：{pins} 对，最大基线 {span60:.0f}mm"
                f"{' ✓' if pins >= 2 else '  ← 不足 2 对：板尺寸自检看不出（解算端会提示）'}")
            h = sorted(float(p[2]) for p in poses)
            say(f"  法兰 z 区间 {h[0]:.0f}~{h[-1]:.0f}mm")
        n_new = len(saved) - N_OLD
        if n_new >= 3:
            say(f"\n  解算：本批跨两个板位（旧 {N_OLD} 在旧板位 + 新 {n_new} 在新板位）——")
            say("    AX=XB 只在同一板位组内成立，【不能】直接整批丢给 --handeye-solve（跨组两两配对全污染）。")
            say("    先跑组内配平（把新帧 T_f 换算进旧板位坐标系：T_f'=B_old·T_cb⁻¹·X⁻¹，迭代 X），再解：")
            say("      python3 -I src/calibrate.py --handeye-solve --intr /tmp/intr_v4.yaml")
        else:
            say(f"  ⚠ 新帧只有 {n_new} 张——先看上面日志，还差就告诉我再跑一轮")
        say(f"  臂【保持使能】，实测停位姿 {[round(float(v), 2) for v in st.tcp]}")
        arm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
