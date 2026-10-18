# -*- coding: utf-8 -*-
"""B3 · 首次真动仪式：零位移 → J6 原地 ±3° → 竖直 +5mm 上/回落。

为什么这么小心：本项目从搭起到现在，真实运动【一次都没发生过】——此前所有 MoveL
都是"零位移探针"（线格式验过、臂没动过）。本脚本是那道闸的第一开。每一步都做成
"小到看得见（≤3°/5mm）、算得出（只用正解）、退得回（原路返回）"，并且每步三重核对：
    ① 指令被接受（本机固件成功码 = 1，不是 0）
    ② 关节实测到位（按 8083 关节流连稳 4 拍判，不单信 motion_done 标志）
    ③ 位姿实测符合预期（位置/姿态都在容差内）
任何一条不对 → 立即停、打印现场、【不】自动下使能（项目惯例：要让人知道臂还绷着）。

三个动作：
  A 零位移：目标 = 当前位姿 + 当前关节，期望"接受但一动不动"（漂移 >0.15mm 就算异常）
  B J6 ±3°：绕法兰 z 轴原地转（法兰原点在 J6 轴线上 → 位置理论上 0mm）。
            转前 / +3°后 / 转回 0 各抓一帧，用 ORB 估两帧之间的转角：
            相机若真在转子上，±3° 应能被画面以 ≈3° 的旋转认出来（照片说了不算，画面说了算）
  C +5mm 上/回落：竖直平移。关节增量来自 fr3_math.solve_delta（纯正解数值雅可比
            + 局部最小二乘），全程不碰控制器逆解（禁区，四次崩溃实锤）；做完立刻回落。

运行（建议在自己的终端跑，方便手放急停）：
    python3 -I src/first_motion.py            # 真跑（要输 GO）
    python3 -I src/first_motion.py --dry      # 只读预演：算好每个目标打给你看，不发指令

前置：① 先切【自动模式】（本脚本检测到手动模式会拒绝发指令）；
      ② 手放急停；③ 关掉浏览器取流（相机独占，打不开只影响旁证、不挡运动）。
证据：data/logs/first_motion_<时间>.txt（全过程文字）、_cam[A|B|C].jpg（三帧对比）、
      .rpc.log（黑匣子：每一次 RPC 调用的原文记录）。
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I 模式不会自动带脚本目录

import config                                                    # noqa: E402
import fr3_math as M                                             # noqa: E402
from fr_arm import Arm                                           # noqa: E402
from fr_cam import Cam, CamError, rot_between                    # noqa: E402

SPEED, ACC = 10.0, 20.0        # 百分比（本机旧版 MoveL 只收百分比）；首次真动给最保守的
J6_DEG = 3.0                   # J6 原地转的角度
LIFT_MM = 5.0                  # 竖直平移
TOL_POS = 0.5                  # 运动后 实测 vs 目标 位置容差 mm
TOL_ANG = 0.5                  # 姿态容差 度
TOL_JOINT_SANITY = 1.0         # 关节残差只做合理性检查（到位判据用位姿——见 arrival 注释）
TOL_ZERO = 0.15                # 零位移步 / J6 原地转 允许的漂移 mm
IK_MAX_ERR_POS = 0.25          # 数值解位置残差上限（不过就不发指令）
IK_MAX_ERR_ANG = 0.25          # 数值解姿态残差上限
IK_MAX_DQ = 8.0                # 单个关节增量上限（5mm 位移不该超过它）
LIMIT_MARGIN = 3.0             # 关节软限位余量（度）


class Tee:
    """同时打屏 + 落盘（跑完我可以直接读文件，不靠人肉复制）。"""

    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def arrival(arm, p_tgt, pos_tol=0.3, ang_tol=0.3, stable_mm=0.02, stable_deg=0.02,
            stable_n=3, timeout=15.0):
    """判"到位且停稳"——用【位姿】判，不用关节差判。

    为什么（2026-10-16 首次真动实测教的一课）：控制器完成 MoveL 后，实际位姿可以精确到
    0.001mm，但关节上会留下 ~0.2° 的残差——残差向量落在雅可比的近零空间方向
    （本机停车位 σ_min≈0.107：0.2° 的关节偏差只反映成 0.001mm 的位姿偏差）。
    这不是"没到位"，是"位姿到位、关节自由"。对采集来说关节值无所谓（每帧记的是
    【实测】法兰位姿），但"什么时候可以抓帧"必须等位姿真的停稳——所以判据 =
    位姿接近目标 + 连续几帧位姿变化小于噪声。数据源用 8083 帧的 tcp 字段
    （实测与 RPC tcp_pose / fk(joints) 三者互差 <0.0005mm，还不用花 RPC 调用）。
    """
    t0 = time.time()
    prev = None
    hit = 0
    dp = da = float("inf")
    while time.time() - t0 < timeout:
        p = [float(v) for v in arm.st.tcp]
        dp = float(np.linalg.norm(np.asarray(p[:3]) - np.asarray(p_tgt[:3])))
        da = float(np.linalg.norm(M.rotvec_deg(
            M.pose6_to_T(p)[:3, :3] @ M.pose6_to_T(p_tgt)[:3, :3].T)))
        if prev is not None:
            mv = float(np.linalg.norm(np.asarray(p[:3]) - np.asarray(prev[:3])))
            mr = float(np.linalg.norm(M.rotvec_deg(
                M.pose6_to_T(p)[:3, :3] @ M.pose6_to_T(prev)[:3, :3].T)))
            hit = hit + 1 if (mv < stable_mm and mr < stable_deg) else 0
        prev = p
        if hit >= stable_n and dp <= pos_tol and da <= ang_tol:
            return True, dp, da, time.time() - t0
        # 停稳了但离目标远 = 指令多半被拒（例：2026-10-16 奇异位姿拒绝，6ms 就"返回成功"）。
        # 给 2s 启动宽限（"已接受、运动还没开始"的窗口），之后停稳即早退，不白等满 timeout。
        if hit >= stable_n and (dp > pos_tol or da > ang_tol) and time.time() - t0 > 2.0:
            return False, dp, da, time.time() - t0
        time.sleep(0.1)
    return False, dp, da, time.time() - t0


def do_move(arm, q_tgt, tag, say, tol_pos=TOL_POS, tol_ang=TOL_ANG):
    """发一条 MoveL（关节由我方给定，控制器不逆解）+ 三重核对。返回 (通过?, 现场数据)。"""
    p_tgt = arm.fk(q_tgt)                    # 只读正解，作为"目标位姿"的权威
    say(f"\n  ▶ {tag}")
    say(f"    目标关节 {[round(v, 3) for v in q_tgt]}")
    say(f"    目标位姿 {[round(v, 3) for v in p_tgt]}")
    t0 = time.time()
    arm.move_l(p_tgt, speed=SPEED, acc=ACC, joints=q_tgt)     # ① 指令（被拒会在这里抛）
    tsend = time.time() - t0
    ok_arr, dp_a, da_a, t_arr = arrival(arm, p_tgt)           # ② 位姿到位且停稳
    q_now = arm.joint_deg()                                   # ③ 实测（RPC 即时值）
    p_now = arm.tcp_pose()
    dj = max(abs(a - b) for a, b in zip(q_now, q_tgt))
    dp = float(np.linalg.norm(np.asarray(p_now[:3]) - np.asarray(p_tgt[:3])))
    da = float(np.linalg.norm(M.rotvec_deg(
        M.pose6_to_T(p_now)[:3, :3] @ M.pose6_to_T(p_tgt)[:3, :3].T)))
    say(f"    ① 指令接受（{tsend:.2f}s 内返回）")
    say(f"    ② 位姿到位 {'✓' if ok_arr else '✗'}：距目标 {dp_a:.4f}mm/{da_a:.4f}°、等 {t_arr:.2f}s")
    say(f"    ③ 实测关节 {[round(v, 3) for v in q_now]}")
    say(f"       实测位姿 {[round(v, 3) for v in p_now]}")
    good = ok_arr and dj <= TOL_JOINT_SANITY and dp <= tol_pos and da <= tol_ang
    say(f"    核对：位置差 {dp:.4f}mm（容差 {tol_pos}）、姿态差 {da:.4f}°（容差 {tol_ang}）、"
        f"关节残差 {dj:.4f}°（只看合理性，上限 {TOL_JOINT_SANITY}°，近零空间方向允许 ~0.2°）"
        f" → {'通过 ✓' if good else '不通过 ✗'}")
    return good, dict(q_now=q_now, p_now=p_now, dj=dj, dp=dp, da=da)


def main():
    ap = argparse.ArgumentParser(description="首次真动仪式（B3）")
    ap.add_argument("--dry", action="store_true", help="只读预演：算好目标、不发任何指令")
    ap.add_argument("--j6", type=float, default=J6_DEG, help=f"J6 原地转角（默认 {J6_DEG}°）")
    ap.add_argument("--lift", type=float, default=LIFT_MM, help=f"竖直位移（默认 {LIFT_MM}mm）")
    ap.add_argument("--descend", type=float, default=None, metavar="MM",
                    help="恢复模式：从当前位姿竖直下沉 MM mm（把臂从小偏移带回原位用）")
    ap.add_argument("--no-cam", action="store_true", help="不碰相机（相机被占用时用）")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = config.DATA_DIR / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    say = Tee(logdir / f"first_motion_{ts}.txt")
    os.environ.setdefault("FR_RPC_LOG", str(logdir / f"first_motion_{ts}.rpc.log"))

    say("=" * 70)
    say(f"B3 首次真动仪式 · {ts}")
    say("【DRY 预演：全程只读，不发任何指令】" if args.dry else
        "【真跑：会发运动指令——手放急停、人站好】")
    say("=" * 70)

    arm = Arm()
    arm.connect()
    st = arm.st

    # ---------- 前置检查 ----------
    say("\n[0] 现场状态")
    q0 = arm.joint_deg()
    p0 = arm.fk(q0)
    lim = arm.r.joint_soft_limits()
    say(f"  当前关节 {[round(v, 3) for v in q0]}")
    say(f"  当前位姿 {[round(v, 3) for v in p0]}  ← 法兰，工具号 {config.TOOL_ID}")
    say(f"  软限位   {[(round(a, 1), round(b, 1)) for a, b in lim]}")
    pre = [("急停未按下", st.estop == 0, f"estop={st.estop}"),
           ("无错误码", st.error_code == 0, f"error_code={st.error_code}"),
           ("当前不在运动中", st.motion_done == 1, f"motion_done={st.motion_done}"),
           ("自动模式", st.robot_mode == 0, f"robot_mode={st.robot_mode}")]
    bad = [c for c in pre if not c[1]]
    for name, ok, detail in pre:
        say(f"  {'✓' if ok else '✗'} {name}（{detail}）")
    if args.dry and bad and all(c[0] == "自动模式" for c in bad):
        say("  （DRY 预演只读、不发指令，模式这项不拦；真跑时会在这一步停下。）")
        bad = []
    if bad:
        say(f"\n✗ 前置不满足：{'、'.join(c[0] for c in bad)}")
        if st.robot_mode != 0:
            say("  手动模式下不往下走（约定：首次真动在【自动模式】做）。")
            say("  请先在 WebApp/示教器上切成自动模式，再重跑本脚本。")
        say("  未发任何指令。")
        arm.close()
        return 2

    # ---------- 恢复模式：一次性竖直下沉 ----------
    if args.descend is not None:
        mm = abs(float(args.descend))
        say(f"\n[恢复模式] 竖直下沉 {mm}mm（从当前位置带回；解完先核验残差，再发一条 MoveL）")
        fk6 = M.fk_throttled(arm.fk, 0.05)
        r = M.solve_delta(fk6, q0, dp=[0.0, 0.0, -mm], steps=1)
        qD = [float(v) for v in r["q"]]
        dqD = [float(v) for v in r["dq"]]
        pD = M.T_to_pose6(r["Tt"])
        say(f"  数值解：位置残差 {r['err_pos']:.4f}mm、姿态残差 {r['err_ang']:.4f}°、{r['iters']} 轮")
        say(f"  关节增量 {[round(v, 3) for v in dqD]}（max {max(abs(v) for v in dqD):.3f}°）")
        say(f"  目标位姿 {[round(v, 3) for v in pD]}")
        ok_plan = (r["err_pos"] <= IK_MAX_ERR_POS and r["err_ang"] <= IK_MAX_ERR_ANG
                   and max(abs(v) for v in dqD) <= IK_MAX_DQ)
        ok_lim = all(lo + LIMIT_MARGIN <= q <= hi - LIMIT_MARGIN for q, (lo, hi) in zip(qD, lim))
        say(f"  残差闸 {'通过 ✓' if ok_plan else '不通过 ✗'}；软限位裕量闸 {'通过 ✓' if ok_lim else '不通过 ✗'}")
        if args.dry:
            say("  [DRY] 只算不发。")
            arm.close()
            return 0
        if not (ok_plan and ok_lim):
            say("  ✗ 预演没过——不发指令。")
            arm.close()
            return 3
        say("\n  即将下沉一次。急停握好。")
        if input("  输 GO 回车 = 开跑；其他 = 取消：").strip().upper() != "GO":
            say("  已取消，未发任何指令。")
            arm.close()
            return 1
        ret = arm.enable()
        say(f"  使能返回 {ret}")
        if int(ret) != 1:
            arm.close()
            return 4
        time.sleep(0.5)
        okD, infoD = do_move(arm, qD, f"下沉 {mm}mm", say)
        say(f"\n  结果：{'✓ 完成' if okD else '✗ 不通过'}")
        say(f"  现在位姿 {[round(v, 3) for v in arm.tcp_pose()]}")
        say("  臂保持使能。")
        arm.close()
        return 0 if okD else 5

    # ---------- 预演：全部算好、全部核验 ----------
    say("\n[1] 只读预演：三个动作的目标全部先算出来（只用正解，不碰控制器逆解）")
    fk6 = M.fk_throttled(arm.fk, 0.05)                 # 正解是网络调用，节流

    say("  ── A 零位移")
    say("     目标 = 当前关节和位姿；接受里唯一合法的事就是【不动】")

    say(f"  ── B J6 原地 {args.j6:+.1f}° ×（+ 回 0，− 回 0）")
    qBp = list(q0)
    qBp[5] += args.j6
    qBn = list(q0)
    qBn[5] -= args.j6
    dBp = float(np.linalg.norm(np.asarray(arm.fk(qBp)[:3]) - np.asarray(p0[:3])))
    dBn = float(np.linalg.norm(np.asarray(arm.fk(qBn)[:3]) - np.asarray(p0[:3])))
    say(f"     +{args.j6}° 目标位置相对当前只挪 {dBp:.3f}mm（法兰原点在 J6 轴线上，理论 0）")
    say(f"     −{args.j6}° 目标位置相对当前只挪 {dBn:.3f}mm")

    say(f"  ── C 竖直 +{args.lift}mm 上（关节增量来自纯正解数值解）")
    rC = M.solve_delta(fk6, q0, dp=[0.0, 0.0, args.lift], steps=1)
    qC = [float(v) for v in rC["q"]]
    dqC = [float(v) for v in rC["dq"]]
    say(f"     数值解：位置残差 {rC['err_pos']:.4f}mm、姿态残差 {rC['err_ang']:.4f}°、"
        f"{rC['iters']} 轮")
    say(f"     关节增量 {[round(v, 3) for v in dqC]}（max {max(abs(v) for v in dqC):.3f}°）")
    okC_plan = (rC["err_pos"] <= IK_MAX_ERR_POS and rC["err_ang"] <= IK_MAX_ERR_ANG
                and max(abs(v) for v in dqC) <= IK_MAX_DQ)
    okC_lim = all(lo + LIMIT_MARGIN <= q <= hi - LIMIT_MARGIN for q, (lo, hi) in zip(qC, lim))
    say(f"     残差闸 {'通过 ✓' if okC_plan else '不通过 ✗'}；软限位裕量闸 "
        f"{'通过 ✓' if okC_lim else '不通过 ✗'}")
    if not (okC_plan and okC_lim):
        say("     ✗ 预演没过 —— 不会发任何指令。把上面的数报给 Claude 看。")
        arm.close()
        return 3

    # 顺带报一下当前位形的雅可比条件数（越接近奇异越差，>1000 就别玩笔画）
    J = M.num_jac6(fk6, q0)
    sv = np.linalg.svd(J, compute_uv=False)
    say(f"  ── 现场情报：雅可比奇异值 {[round(float(v), 4) for v in sv]}")
    say(f"     条件数 {sv[0] / max(sv[-1], 1e-12):.1f}（大 = 接近奇异，笔画的可用方向就少）")

    if args.dry:
        say("\n[DRY] 预演结束：以上目标都已算好并通过核验，但【一个字都没发】。")
        say("       真跑：python3 -I src/first_motion.py（手放急停，输 GO）")
        arm.close()
        return 0

    # ---------- GO ----------
    say("\n" + "─" * 70)
    say("即将真动（全程速度 10%、总位移 ≤5mm/≤3°）。请确认：")
    say("  ① 急停握在手里；② 臂周围没人、没障碍；③ 浏览器取流已关。")
    ans = input("  输 GO 回车 = 开跑；其他任何输入 = 取消：").strip().upper()
    if ans != "GO":
        say("已取消，未发任何指令。")
        arm.close()
        return 1

    results = []
    cam = None
    try:
        # ---------- 使能 ----------
        say("\n[2] 使能")
        ret = arm.enable()
        say(f"  RobotEnable(1) 返回 {ret}（本机固件成功码 = 1）")
        if int(ret) != 1:
            say("  ✗ 使能没成功 —— 停下，报数。")
            arm.close()
            return 4
        time.sleep(0.6)
        say(f"  使能后状态：estop={st.estop} error={st.error_code} motion_done={st.motion_done}")

        # ---------- 相机（可有可无，只影响 B 的旁证） ----------
        if not args.no_cam:
            try:
                cam = Cam()
                say(f"  相机已开：/dev/video{cam.dev} {cam.w}×{cam.h}")
            except CamError as e:
                say(f"  ⚠ 相机打不开（{e}）——B 步的画面旁证跳过，不影响运动本身。")

        # ---------- A 零位移 ----------
        say("\n[3] A 零位移（期望：接受但一动不动）")
        okA, _ = do_move(arm, q0, "A 零位移 MoveL", say, tol_pos=TOL_ZERO, tol_ang=0.1)
        results.append(("A 零位移", okA))
        if not okA:
            raise RuntimeError("零位移步就不对——停下（可能坐标系/关节方向理解有误）")

        # ---------- B J6 原地转 ----------
        say(f"\n[4] B J6 原地 {args.j6:+.1f}° → 回 0 → {args.j6:+.1f}° → 回 0")
        say("    （法兰原点在 J6 轴线上：位置应纹丝不动；画面应跟着转）")
        grayA = None
        if cam is not None:
            frA = cam.raw()
            grayA = cv2.cvtColor(frA, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(logdir / f"first_motion_{ts}_camA.jpg"), frA)
            say("    已存 camA.jpg（转前）")

        okB1, _ = do_move(arm, qBp, f"B1 J6 {args.j6:+.1f}°", say, tol_pos=TOL_ZERO, tol_ang=1.0)
        results.append((f"B1 J6 +{args.j6}°", okB1))
        if cam is not None:
            frB = cam.fresh(say)
            cv2.imwrite(str(logdir / f"first_motion_{ts}_camB.jpg"), frB)
            r = rot_between(grayA, cv2.cvtColor(frB, cv2.COLOR_BGR2GRAY))
            if r["ok"]:
                say(f"    画面 A→B：估到旋转 {r['ang_deg']:+.2f}°、平移 ({r['dx']:+.1f},{r['dy']:+.1f})px、"
                    f"内点 {r['n_in']}")
                say(f"    {'✓ 画面跟着 J6 转了（相机确在转子上）' if abs(r['ang_deg']) > 1.2 else '⚠ 画面几乎没动——相机真在转子上吗？（这一步先不动，报数）'}")
            else:
                say(f"    ⚠ 两帧对不上（{r['why']}）——看图说话，先往下走")
        if not okB1:
            raise RuntimeError("J6 转动步不对——停下")

        okB2, _ = do_move(arm, q0, "B2 回 0", say, tol_pos=TOL_ZERO, tol_ang=0.1)
        results.append(("B2 回 0", okB2))
        okB3, _ = do_move(arm, qBn, f"B3 J6 {args.j6:+.1f}°", say, tol_pos=TOL_ZERO, tol_ang=1.0)
        results.append((f"B3 J6 −{args.j6}°", okB3))
        okB4, _ = do_move(arm, q0, "B4 回 0", say, tol_pos=TOL_ZERO, tol_ang=0.1)
        results.append(("B4 回 0", okB4))
        if cam is not None:
            frC = cam.fresh(say)
            cv2.imwrite(str(logdir / f"first_motion_{ts}_camC.jpg"), frC)
            r = rot_between(grayA, cv2.cvtColor(frC, cv2.COLOR_BGR2GRAY))
            if r["ok"]:
                say(f"    画面 A→C（应≈0，证明转回原位画面也回来了）：旋转 {r['ang_deg']:+.2f}°、"
                    f"平移 ({r['dx']:+.1f},{r['dy']:+.1f})px、内点 {r['n_in']}")
        if not (okB2 and okB3 and okB4):
            raise RuntimeError("J6 回位步不对——停下")

        # ---------- C 竖直 +5mm ----------
        say(f"\n[5] C 竖直 +{args.lift}mm 上 → 回落")
        okC1, _ = do_move(arm, qC, f"C1 +{args.lift}mm 上", say)
        results.append((f"C1 +{args.lift}mm", okC1))
        if okC1:
            p_up = arm.tcp_pose()
            dp_up = np.asarray(p_up[:3]) - np.asarray(p0[:3])
            say(f"    实测位移向量 [{dp_up[0]:+.3f}, {dp_up[1]:+.3f}, {dp_up[2]:+.3f}]mm"
                f"（期望 [0, 0, +{args.lift}]）")
            if abs(dp_up[0]) > 0.5 or abs(dp_up[1]) > 0.5 or abs(dp_up[2] - args.lift) > 0.5:
                say("    ⚠ 位移方向和预期不符——先停下看数据，别继续")
                raise RuntimeError("+5mm 位移方向不对")
        else:
            say("    ⚠ C1 核对没过。把臂留在半空比带回原位更糟——只要现场健康"
                "（无报错、无急停、不在运动中）就自动带回原位；有任何异常就停手等你。")
            if st.error_code == 0 and st.estop == 0 and int(st.motion_done) == 1:
                okRec, _ = do_move(arm, q0, "C2′ 恢复：带回原位", say)
                results.append(("C2′ 恢复带回", okRec))
            else:
                say(f"    ✗ 现场不健康（error={st.error_code} estop={st.estop} "
                    f"motion_done={st.motion_done}）——不发任何指令，人来看现场。")
            raise RuntimeError("+5mm 步核对未过（已按健康情况处理）")
        okC2, _ = do_move(arm, q0, "C2 回落原位", say)
        results.append(("C2 回落", okC2))

        # ---------- 收尾 ----------
        say("\n[6] 收尾：与出发状态比对")
        q_end = arm.joint_deg()
        p_end = arm.tcp_pose()
        dj = max(abs(a - b) for a, b in zip(q_end, q0))
        dp = float(np.linalg.norm(np.asarray(p_end[:3]) - np.asarray(p0[:3])))
        say(f"  关节最大回差 {dj:.4f}°；位置回差 {dp:.4f}mm")
        say(f"  estop={st.estop} error={st.error_code} motion_done={st.motion_done} "
            f"robot_mode={st.robot_mode}")

    except Exception as e:                                  # noqa: BLE001
        say(f"\n✗✗ 中断：{type(e).__name__}: {e}")
        try:
            say(f"  现场：关节 {[round(v, 3) for v in arm.joint_deg()]} "
                f"位姿 {[round(v, 3) for v in arm.tcp_pose()]}")
        except Exception:                                   # noqa: BLE001
            say("  现场读数也拿不到了（通道断了？）")
        say("  未做任何自动下使能/复位——人来看现场决定。")
        results.append(("中断", False))

    finally:
        if cam is not None:
            cam.close()
        say("\n【总结】")
        for name, ok in results:
            say(f"  {'✓' if ok else '✗'} {name}")
        say(f"  黑匣子 {logdir}/first_motion_{ts}.rpc.log")
        say(f"  文字实录 {logdir}/first_motion_{ts}.txt")
        say("  臂【保持使能】（项目惯例：不自动下使能；要松臂在 WebApp 下使能）。")
        arm.close()
    return 0 if results and all(ok for _, ok in results) else 5


if __name__ == "__main__":
    sys.exit(main())
