# -*- coding: utf-8 -*-
"""视野扫描：小步转一个关节，把相机"此刻看到什么"存成一组照片（纯侦察）。

为什么需要：相机搬到了六轴转子上，旧的一整套"相机朝向"先验（板立哪、瞄哪个方向、
CAM_OFFSET）全部作废。与其猜，不如让腕子小步转一圈、每步存一帧——我看图之后告诉你
板该立在哪、高度多少、朝向怎么摆。

关节运动用 MoveJ（PTP），不碰控制器逆解（禁区）。
2026-10-16 首跑教训：第一版用 MoveL，J5 从 −5.6° 去 +2.4° 那步被控制器拒绝——
「下一指令关节配置发生变化（下一指令中存在奇异位姿，请使用 PTP 指令…）」：
q5=0° 正是腕部奇异（SVD 实测 σ_min≈1e-6、cond≈1.7e7），MoveL 要按笛卡尔路径规划、
过不去；MoveJ 是关节空间插值，可自由穿越奇异点，而且一步只动一个关节（线缆最友好）。
目标位姿仍只用 FK 预览/核验（arrival 判据要用），依然【零逆解】。
默认扫 J5（腕部俯仰）±40°、步长 8°：起始帧 + 10 张 + 归位核验帧。
也可以 --joint 1 扫底座（左右环视）——先看 J5 扫完的照片再决定下一步扫哪。

运行（自己的终端，手放急停；跑前关掉浏览器取流——相机独占）：
    python3 -I src/view_sweep.py                        # J5 ±40° 步长 8°
    python3 -I src/view_sweep.py --joint 1 --deg 20 --step 5
    python3 -I src/view_sweep.py --dry                  # 只算目标，不发指令
证据：data/view_sweep/<时间>/j5_p032.jpg 等 + poses.csv + sweep.txt（文字实录）
     + .rpc.log（黑匣子）
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrate as C                                          # noqa: E402
import config                                                  # noqa: E402
import fr3_math as M                                           # noqa: E402
from fr_arm import Arm                                         # noqa: E402
from fr_cam import Cam, CamError                               # noqa: E402

SPEED, ACC = 8.0, 20.0        # 侦察用小步，速度再压一点
MAX_STEP_MM = 300.0           # 单步相对起始位姿的最大位移（扫描本身不该挪太远）
MIN_Z = 200.0                 # 目标位姿的最低 z（别往下扎）
LIMIT_MARGIN = 3.0            # 关节软限位余量（度）


class Tee:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def arrival(arm, p_tgt, pos_tol=0.3, ang_tol=0.3, stable_mm=0.02, stable_deg=0.02,
            stable_n=3, timeout=15.0):
    """判"到位且停稳"——用【位姿】判，不用关节差判（与 first_motion.arrival 同一套判据）。

    2026-10-16 首次真动实测：控制器完成 MoveL 后位姿可精确到 0.001mm，但关节上会留
    ~0.2° 残差（残差在雅可比近零空间方向，σ_min≈0.107）。"位姿到位、关节自由"，
    所以等停稳只看位姿：数据源 = 8083 帧 tcp 字段（不花 RPC 调用）。
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


def main():
    ap = argparse.ArgumentParser(description="视野扫描（只侦察，不改任何标定）")
    ap.add_argument("--joint", type=int, default=5, choices=range(1, 7),
                    help="扫哪个关节（1~6，默认 5 = 腕部俯仰）")
    ap.add_argument("--deg", type=float, default=40.0, help="单侧最大角度（默认 40）")
    ap.add_argument("--step", type=float, default=8.0, help="步长（默认 8）")
    ap.add_argument("--dry", action="store_true", help="只算目标，不发指令")
    ap.add_argument("--no-cam", action="store_true", help="不开相机（相机被占用时）")
    args = ap.parse_args()
    jidx = args.joint - 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = config.DATA_DIR / "view_sweep" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    say = Tee(outdir / "sweep.txt")
    os.environ.setdefault("FR_RPC_LOG", str(outdir / "sweep.rpc.log"))

    say("=" * 70)
    say(f"视野扫描 J{args.joint} ±{args.deg:g}° 步长 {args.step:g}° · {ts}")
    say("【DRY：只算目标不发指令】" if args.dry else "【真跑：会小步转动关节】")
    say("=" * 70)

    arm = Arm()
    arm.connect()
    st = arm.st
    q0 = [float(v) for v in arm.joint_deg()]
    p0 = [float(v) for v in arm.fk(q0)]
    lim = arm.r.joint_soft_limits()
    say(f"\n起始关节 {[round(v, 3) for v in q0]}")
    say(f"起始位姿 {[round(v, 3) for v in p0]}")
    pre = [("急停未按下", st.estop == 0, f"estop={st.estop}"),
           ("无错误码", st.error_code == 0, f"error_code={st.error_code}"),
           ("当前不在运动中", st.motion_done == 1, f"motion_done={st.motion_done}"),
           ("自动模式", st.robot_mode == 0, f"robot_mode={st.robot_mode}")]
    bad = [c for c in pre if not c[1]]
    for name, ok, detail in pre:
        say(f"  {'✓' if ok else '✗'} {name}（{detail}）")
    if args.dry and bad and all(c[0] == "自动模式" for c in bad):
        say("  （DRY 只读，模式不拦；真跑时会在这一步停下。）")
        bad = []
    if bad:
        say(f"\n✗ 前置不满足：{'、'.join(c[0] for c in bad)}")
        if st.robot_mode != 0:
            say("  手动模式下不往下走（约定：真动在【自动模式】做）。")
        if st.error_code != 0:
            say("  error_code≠0：用 python3 -I src/clear_error.py 看故障原文（--clear 清除）。")
        say("  未发任何指令。")
        arm.close()
        return 2

    # ---------- 排程 ----------
    pos = [k * args.step for k in range(1, int(round(args.deg / args.step)) + 1)]
    plan = []                                     # [(角度, 要不要拍照)]
    for a in pos:                                 # 先负半圈：板在 −J5 侧（首跑 +8° 角点 14→10 掉了），
        plan.append((-a, True))                   # 且 +侧近似对称的落点贴着奇异带——先收好数据再冒险
    plan.append((0.0, False))                     # 回中，不拍
    for a in pos:
        plan.append((a, True))
    plan.append((0.0, False))                     # 归位，不拍

    say(f"\n排程：0°(起始拍) → " + " → ".join(
        f"{a:+.0f}°{'📷' if cap else ''}" for a, cap in plan))
    # 预算并核验每个目标
    checked = []
    for ang, cap in plan:
        q = list(q0)
        q[jidx] += ang
        p = arm.fk(q)
        d = float(np.linalg.norm(np.asarray(p[:3]) - np.asarray(p0[:3])))
        ok_lim = lim[jidx][0] + LIMIT_MARGIN <= q[jidx] <= lim[jidx][1] - LIMIT_MARGIN
        ok_box = d <= MAX_STEP_MM and p[2] >= MIN_Z
        checked.append((ang, cap, q, p, d, ok_lim and ok_box, ok_lim, ok_box))
    worst = max(checked, key=lambda r: r[4])
    say(f"  最大位移 {worst[4]:.1f}mm（在 {worst[0]:+.0f}°，上限 {MAX_STEP_MM:g}）")
    if not all(r[5] for r in checked):
        for ang, cap, q, p, d, ok, ok_lim, ok_box in checked:
            if not ok:
                say(f"  ✗ {ang:+.0f}°：限位{'✓' if ok_lim else '✗'}、范围{'✓' if ok_box else '✗'}"
                    f"（位移 {d:.1f}mm、z={p[2]:.1f}）")
        say("  ✗ 排程里有不安全的目标——不发任何指令。把上面的数给我看。")
        arm.close()
        return 3

    if args.dry:
        say("\n[DRY] 排程全部核验通过，没发任何指令。")
        arm.close()
        return 0

    say("\n" + "─" * 70)
    say(f"即将小步转动 J{args.joint}（速度 {SPEED:g}%、单步 {args.step:g}°、"
        f"总行程 ±{args.deg:g}°）。急停握好、人站好、浏览器取流关掉。")
    if input("  输 GO 回车 = 开跑；其他 = 取消：").strip().upper() != "GO":
        say("已取消，未发任何指令。")
        arm.close()
        return 1

    results = []
    cam = None
    rows = []
    det = C.BoardDetector()
    prev_gray = None
    try:
        ret = arm.enable()
        say(f"\n使能返回 {ret}（成功码 1）")
        if int(ret) != 1:
            raise RuntimeError("使能失败")
        time.sleep(0.5)

        # 签名自检：一条"原地 MoveJ"（目标=当前关节，机械上零位移）。签名/参数类型不对
        # 会在这里 Fault 掉——而不是死在第一个真步上（2026-10-16 实测：ovl 传 int 被拒，
        # 老固件要 float；见 fr_rpc.move_j 注释）。
        q_now = [float(v) for v in arm.joint_deg()]
        t_ss = time.time()
        arm.move_j(q_now, speed=SPEED)
        say(f"\n[签名自检] 原地 MoveJ 通过（接受 {time.time() - t_ss:.2f}s，零位移）")

        if not args.no_cam:
            try:
                cam = Cam()
                say(f"相机已开 /dev/video{cam.dev} {cam.w}×{cam.h}")
            except CamError as e:
                say(f"⚠ 相机打不开（{e}）——那这次扫描就白跑了，停下。")
                raise

        def shoot(tag, ang, q, p, fresh):
            """拍一张，存盘并报检测结果。"""
            fr = cam.fresh(say) if fresh else cam.raw()
            # 起/末帧单独命名（都发生在 0°——首跑的 j5_+000.jpg 被末帧盖过）
            path = outdir / (f"j{args.joint}_{tag}.jpg" if tag in ("start", "end")
                             else f"j{args.joint}_{ang:+04.0f}.jpg")
            cv2.imwrite(str(path), fr)
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            corners, ids = det.detect(gray)
            n = 0 if corners is None else int(len(corners))
            side = blur = 0.0
            if n:
                _x, _y, side = C.frame_stats(corners, (cam.w, cam.h))
                blur = C.board_blur(gray, corners)
            nonlocal prev_gray
            dprev = float(cv2.absdiff(gray, prev_gray).mean()) if prev_gray is not None else 0.0
            prev_gray = gray
            say(f"    📷 {path.name}：板角点 {n}/24、板宽 {side:.0f}px、清晰度 {blur:.0f}、"
                f"与前一帧均差 {dprev:.1f}")
            rows.append((tag, ang, *q, *p, n, side, blur, dprev, path.name))
            results.append((tag, True))

        # 起始帧
        say("\n[起始] 0°（运动前，直接抓）")
        shoot("start", 0.0, q0, p0, fresh=False)

        for i, (ang, cap, q, p, d, _ok, _l, _b) in enumerate(checked, 1):
            p_tgt = [float(v) for v in p]
            q_tgt = [float(v) for v in q]
            say(f"\n[{i}/{len(checked)}] J{args.joint} {ang:+.0f}°（位移 {d:.1f}mm）")
            t_cmd = time.time()
            arm.move_j(q_tgt, speed=SPEED)
            t_cmd = time.time() - t_cmd
            ok_arr, dp_a, da_a, t_arr = arrival(arm, p_tgt, timeout=45.0)
            say(f"    指令 {t_cmd:.2f}s、到位 {'✓' if ok_arr else '✗'}"
                f"（距目标 {dp_a:.4f}mm/{da_a:.4f}°、{t_arr:.2f}s）")
            if not ok_arr:
                raise RuntimeError(
                    f"{ang:+.0f}° 步没到位——停下（控制器 error_code={st.error_code}）"
                    f"；若 error_code≠0，用 python3 -I src/clear_error.py 看故障原文")
            if cap:
                shoot(f"j{args.joint}{ang:+.0f}", ang, q_tgt, p_tgt, fresh=True)

        # 归位核验
        say("\n[归位] 回 0°")
        arm.move_j(q0, speed=SPEED)
        ok_arr, dp_a, da_a, _ = arrival(arm, p0, timeout=45.0)
        say(f"    归位到位 {'✓' if ok_arr else '✗'}（距目标 {dp_a:.4f}mm/{da_a:.4f}°）")
        q_end = [float(v) for v in arm.joint_deg()]
        p_end = [float(v) for v in arm.tcp_pose()]
        dj = max(abs(a - b) for a, b in zip(q_end, q0))
        dp = float(np.linalg.norm(np.asarray(p_end[:3]) - np.asarray(p0[:3])))
        say(f"    归位 {'✓' if ok_arr else '✗'}：关节回差 {dj:.4f}°、位置回差 {dp:.4f}mm")
        say("\n[归位帧] 0°（与起始帧比对，核验画面回没回来）")
        shoot("end", 0.0, q0, p0, fresh=True)

    except Exception as e:                                   # noqa: BLE001
        say(f"\n✗✗ 中断：{type(e).__name__}: {e}")
        # 运动指令超时 ≠ 臂没动（2026-10-16 实测：客户端 3s 超时后，控制器把 48° 回中照跑完）。
        # 先等它真的停稳（状态制），再让 finally 的"实测停位姿"报告真相。
        try:
            if int(getattr(st, "motion_done", 1)) != 1:
                say("   （控制器仍在动——等停稳，最多 60s）")
                arm.wait_done(timeout=60.0)
        except Exception:                                    # noqa: BLE001
            pass
        results.append(("中断", False))
    finally:
        if cam is not None:
            cam.close()
        # 存 poses.csv
        with open(outdir / "poses.csv", "w", encoding="utf-8") as fh:
            fh.write("tag,j1,j2,j3,j4,j5,j6,x,y,z,rx,ry,rz,n_corners,side_px,blur,dprev,jpg\n")
            for r in rows:
                fh.write(",".join(str(round(v, 4)) if isinstance(v, float) else str(v)
                                  for v in r) + "\n")
        say("\n【总结】")
        say(f"  共 {len([r for r in results if r[1]])} 步完成、{len(rows)} 帧存于 {outdir}/")
        say("  请把这段话和照片目录给我看（我逐帧读图）。")
        say(f"  黑匣子 {outdir}/sweep.rpc.log")
        say(f"  臂【保持使能】，实测停位姿 {[round(float(v), 3) for v in arm.tcp_pose()]}"
            f"（正常=起始位姿；中断=中断处）。")
        arm.close()
    return 0 if results and all(ok for _, ok in results) else 5


if __name__ == "__main__":
    sys.exit(main())
