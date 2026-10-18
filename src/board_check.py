# -*- coding: utf-8 -*-
"""瞄姿体检：抓一帧 → 认板 → 全板投影/切边/余量 → 是否过【采集门槛】。

门槛和 auto_handeye 完全同款：24/24 角点 + 全板离画面边 ≥60px + 板宽 ≥350px。
挪板子或找瞄姿时用这个做即时反馈：差多少、往哪挪（板面 mm）、当前臂姿都打印出来。

用法（自己的终端；相机独占——先关浏览器取流）：
    python3 -I src/board_check.py                 # 只抓帧体检，不动臂
    python3 -I src/board_check.py --live          # 实时窗口：边滑板边看红框进黄框（q 退出并存证）
    python3 -I src/board_check.py --j5 -16        # 先把 J5 从当前位转 -16°，再体检（有 GO 和四闸）
    python3 -I src/board_check.py --j5 -16 --j6 -90   # 同时滚 J6（相机在转子上：滚 J6=画面滚转）
    python3 -I src/board_check.py --frame x.jpg   # 离线体检一张旧照片（不连机器人/相机）

⚠ --j5/--j6 是【相对当前位】的偏移：每次从停机位开始最省心；想回原姿态就跑取反的角度。
约定：只动 J5/J6 两条关节、MoveJ（不碰逆解）；不换模式；跑完臂保持使能；|q5| 别贴 0（奇异带）。
存证：data/view_sweep/check_<时间>.jpg（画了全板投影框：过=绿、不过=红 + 60px 门槛框）
      data/view_sweep/checks.csv（每次体检一行，方便看挪板的收敛过程）
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                  # noqa: E402
import view_sweep as VS                                        # noqa: E402  （复用 arrival/常量）
from fr_arm import Arm                                         # noqa: E402
from fr_cam import Cam, CamError                               # noqa: E402
from probe import board_report as BR                           # noqa: E402

OUTDIR = config.DATA_DIR / "view_sweep"


def annotate(frame, r, ok):
    """把全板投影框、门槛框、角点画上去（图内文字用 ASCII，中文只走终端）。"""
    im = frame.copy()
    g = 60
    cv2.rectangle(im, (g, g), (BR.IW - 1 - g, BR.IH - 1 - g), (0, 200, 255), 2)
    if r and not r.get("bad"):
        o = np.round(r["outline"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(im, [o], True, (0, 255, 0) if ok else (0, 0, 255), 3)
        for p in np.round(r["pts"]).astype(int):
            cv2.circle(im, tuple(p), 6, (255, 0, 255), -1)
        x0, y0, x1, y1 = r["bbox"]
        txt = (f"corners {r['n']}/24  bbox x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}]  "
               f"cut L{r['cut']['左']:.0f} T{r['cut']['上']:.0f} "
               f"R{r['cut']['右']:.0f} B{r['cut']['下']:.0f}  "
               f"need duv ({r['du']:+.0f},{r['dv']:+.0f})  {'PASS' if ok else 'FAIL'}")
    else:
        txt = (f"corners {r['n'] if r else 0}/24  "
               f"{'(board NOT found)' if not r else r['bad']}  FAIL")
    cv2.putText(im, txt, (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(im, txt, (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return im


def save_csv(row):
    p = OUTDIR / "checks.csv"
    new = not p.exists()
    with open(p, "a", encoding="utf-8") as fh:
        if new:
            fh.write("time,j1,j2,j3,j4,j5,j6,x,y,z,corners,cut_l,cut_t,cut_r,cut_b,"
                     "side_px,du,dv,dist_mm,pass,jpg\n")
        fh.write(",".join(str(round(v, 3)) if isinstance(v, float) else str(v)
                          for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser(description="瞄姿体检（采集门槛 24/24 + 离边≥60px）")
    ap.add_argument("--j5", type=float, default=0.0, help="先把 J5 从当前位转这个角度再体检")
    ap.add_argument("--j6", type=float, default=0.0,
                    help="先把 J6 从当前位转这个角度再体检（相机在转子上：滚 J6=画面滚转）")
    ap.add_argument("--frame", help="离线体检这张照片（不连机器人/相机）")
    ap.add_argument("--live", action="store_true",
                    help="实时瞄准窗口：边滑板边看红框进黄框（不动臂；q/ESC 退出并存证）")
    args = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- 离线：只看一张照片 ----------
    if args.frame:
        img = cv2.imread(args.frame)
        if img is None:
            print(f"读不了 {args.frame}")
            return 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r = BR.analyze(gray)
        ok, bad = BR.verdict(r)
        print(BR.format_report(os.path.basename(args.frame), r))
        print(f"  → {'✓ 过' if ok else '✗ 不过'}")
        return 0 if ok else 5

    # ---------- 实时瞄准窗口（滑板反馈环；不动臂） ----------
    if args.live:
        if abs(args.j5) > 1e-9 or abs(args.j6) > 1e-9:
            print("（--live 不动臂：--j5/--j6 被忽略）")
        arm = Arm()
        arm.connect()
        det, K = BR.C.BoardDetector(), BR.load_K()
        cam = None
        fr, r = None, None
        ok, bad = False, ["还没抓到帧"]
        tgt = (int((60 + (BR.IW - 1 - 60)) / 2), int((60 + (BR.IH - 1 - 60)) / 2))
        try:
            cam = Cam()
            print(f"相机已开 /dev/video{cam.dev} {cam.w}×{cam.h}")
            print("【实时瞄准】把板沿桌面滑动（别转动）——让画面里红框（全板投影）整个落进黄框。")
            print("  黄十字 = 目标中心；黄箭头 = 板心该走的方向；红框变绿 = 过门槛。")
            print("  按 q 或 ESC 退出并存证。相机独占——先关掉浏览器取流。")
            win = "board_aim"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            w = min(int(cam.w), 1400)
            cv2.resizeWindow(win, w, int(w * cam.h / cam.w))
            t_last, n_frames = 0.0, 0
            while True:
                fr = cam.raw()
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                r = BR.analyze(gray, det, K)
                ok, bad = BR.verdict(r)
                vis = annotate(fr, r, ok)
                cv2.drawMarker(vis, tgt, (0, 255, 255), cv2.MARKER_CROSS, 44, 3)
                if r and not r.get("bad"):
                    c = (int(r["center"][0]), int(r["center"][1]))
                    cv2.arrowedLine(vis, c, tgt, (0, 255, 255), 3, tipLength=0.02)
                msg = ("slide the board: red outline into the yellow box   |   "
                       + ("PASS - press q" if ok else "keep sliding"))
                cv2.putText(vis, msg, (70, BR.IH - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
                cv2.putText(vis, msg, (70, BR.IH - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0) if ok else (0, 255, 255), 2)
                cv2.imshow(win, vis)
                n_frames += 1
                if time.time() - t_last > 1.0:
                    t_last = time.time()
                    if r and not r.get("bad"):
                        print(f"  [{n_frames}] 角点 {r['n']}/24、duv ({r['du']:+.0f},{r['dv']:+.0f})px、"
                              f"切 左{r['cut']['左']:.0f} 上{r['cut']['上']:.0f} "
                              f"右{r['cut']['右']:.0f} 下{r['cut']['下']:.0f}px  "
                              f"{'✓ 过' if ok else '✗'}")
                    else:
                        print(f"  [{n_frames}] 认不出板（角点 {r['n'] if r else 0}）——让它回画面里")
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
        except cv2.error as e:                                   # noqa: BLE001
            print(f"✗ 打不开实时窗口（{e}）——改用无窗口流程：python3 -I src/board_check.py")
            arm.close()
            return 7
        finally:
            cv2.destroyAllWindows()
            if cam is not None:
                cam.close()
        # 存证（与无窗口体检同一套）
        OUTDIR.mkdir(parents=True, exist_ok=True)
        jpg = OUTDIR / f"check_{ts}.jpg"
        cv2.imwrite(str(jpg), annotate(fr, r, ok))
        jp = [float(v) for v in arm.joint_deg()]
        pp = [float(v) for v in arm.tcp_pose()]
        cut = r["cut"] if r and not r.get("bad") else {k: 0.0 for k in "左上右下"}
        save_csv([ts, *jp, *pp, (r["n"] if r else 0), cut["左"], cut["上"], cut["右"], cut["下"],
                  (r["side"] if r else 0.0), (r["du"] if r else 0.0), (r["dv"] if r else 0.0),
                  (r["dist"] if r else 0.0), int(ok), jpg.name])
        print("\n" + BR.format_report("<退出时那帧>", r))
        print(f"【结论】{'✓ 过采集门槛（红框已进黄框）——把这段贴我' if ok else '✗ 还没过：' + '；'.join(bad)}")
        print(f"  存证 {jpg}（已记 checks.csv）")
        print("  臂【保持使能】，全程未动。")
        arm.close()
        return 0 if ok else 5

    # ---------- 在线 ----------
    arm = Arm()
    arm.connect()
    st = arm.st
    q0 = [float(v) for v in arm.joint_deg()]
    p0 = [float(v) for v in arm.fk(q0)]
    print(f"当前关节 {[round(v, 3) for v in q0]}")
    print(f"当前位姿 {[round(v, 3) for v in p0]}")
    if abs(q0[4]) < 12.0:
        print(f"  ⚠ 当前 q5={q0[4]:+.2f}° 贴近腕部奇异带（|q5| 最好 ≥12°；上次实测 0° 时 σ≈1e-6）")

    p_tgt = None
    if abs(args.j5) > 1e-9 or abs(args.j6) > 1e-9:
        pre = [("急停未按下", st.estop == 0, f"estop={st.estop}"),
               ("无错误码", st.error_code == 0, f"error_code={st.error_code}"),
               ("当前不在运动中", st.motion_done == 1, f"motion_done={st.motion_done}"),
               ("自动模式", st.robot_mode == 0, f"robot_mode={st.robot_mode}")]
        bad = [c for c in pre if not c[1]]
        for name, ok, detail in pre:
            print(f"  {'✓' if ok else '✗'} {name}（{detail}）")
        if bad:
            print(f"\n✗ 前置不满足：{'、'.join(c[0] for c in bad)}——未发任何指令。")
            if st.error_code != 0:
                print("  error_code≠0：python3 -I src/clear_error.py 看故障原文（--clear 清除）。")
            arm.close()
            return 2
        if abs(args.j5) > 40.0 or abs(args.j6) > 180.0:
            print(f"✗ 偏移超限（J5 允许 ±40°、J6 ±180°）——不发指令。")
            arm.close()
            return 3
        q_tgt = list(q0)
        q_tgt[4] += args.j5
        q_tgt[5] += args.j6
        lim = arm.r.joint_soft_limits()
        for k, jn in ((4, "J5"), (5, "J6")):
            if abs(q_tgt[k] - q0[k]) > 1e-9 and not (lim[k][0] + 3.0 <= q_tgt[k] <= lim[k][1] - 3.0):
                print(f"✗ 目标 {jn}={q_tgt[k]:.2f}° 顶到软限位 {lim[k]}——不发指令。")
                arm.close()
                return 3
        p_tgt = [float(v) for v in arm.fk(q_tgt)]
        d = float(np.linalg.norm(np.asarray(p_tgt[:3]) - np.asarray(p0[:3])))
        print(f"\n计划：J5 {q0[4]:+.2f}° → {q_tgt[4]:+.2f}°、J6 {q0[5]:+.2f}° → {q_tgt[5]:+.2f}°"
              f"（位移 {d:.1f}mm、速度 {VS.SPEED:g}%）")
        if abs(q_tgt[4]) < 12.0:
            print(f"  ⚠ 目标 q5={q_tgt[4]:+.2f}° 贴近腕部奇异带（|q5| 最好 ≥12°）")
        if input("  输 GO 回车 = 动；其他 = 取消：").strip().upper() != "GO":
            print("已取消，未发任何指令。")
            arm.close()
            return 1
        ret = arm.enable()
        print(f"使能返回 {ret}（成功码 1）")
        if int(ret) != 1:
            arm.close()
            return 4
        time.sleep(0.5)
        arm.move_j(q_tgt, speed=VS.SPEED)
        ok_arr, dp, da, t_arr = VS.arrival(arm, p_tgt, timeout=45.0)
        print(f"到位 {'✓' if ok_arr else '✗'}（距目标 {dp:.4f}mm/{da:.4f}°、{t_arr:.2f}s）")
        if not ok_arr:
            print(f"✗ 没到位（error_code={st.error_code}）——先 clear_error 看故障；不体检了。")
            arm.close()
            return 5

    # ---------- 抓帧 ----------
    cam = None
    try:
        cam = Cam()
        print(f"相机已开 /dev/video{cam.dev} {cam.w}×{cam.h}")
        fr = cam.fresh(print)
    except CamError as e:
        print(f"✗ 相机打不开：{e}")
        arm.close()
        return 6
    try:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        r = BR.analyze(gray)
        ok, bad = BR.verdict(r)
        print("\n" + BR.format_report("<刚才那帧>", r))
        OUTDIR.mkdir(parents=True, exist_ok=True)
        jpg = OUTDIR / f"check_{ts}.jpg"
        cv2.imwrite(str(jpg), annotate(fr, r, ok))
        jp = [float(v) for v in arm.joint_deg()]
        pp = [float(v) for v in arm.tcp_pose()]
        cut = r["cut"] if r and not r.get("bad") else {k: 0.0 for k in "左上右下"}
        save_csv([ts, *jp, *pp, (r["n"] if r else 0), cut["左"], cut["上"], cut["右"], cut["下"],
                  (r["side"] if r else 0.0), (r["du"] if r else 0.0), (r["dv"] if r else 0.0),
                  (r["dist"] if r else 0.0), int(ok), jpg.name])
        print(f"\n【结论】{'✓ 过采集门槛（可以开采）' if ok else '✗ 不过：' + '；'.join(bad)}")
        if not ok and r and not r.get("bad"):
            print("  挪板方向别猜：先小挪 ~50mm，再跑一次本工具看 du,dv 是不是变小了——"
                  "变小=方向对，继续；变大=反方向。（du≈画面左右、dv≈画面上下）")
        print(f"  存证 {jpg}（画红线=全板投影框，黄框=60px 门槛线）")
        if abs(args.j5) > 1e-9 or abs(args.j6) > 1e-9:
            print(f"  臂停在 J5={q_tgt[4]:+.2f}°、J6={q_tgt[5]:+.2f}° 位（保持使能）。"
                  f"要回挪之前：再跑 --j5 {-args.j5:+.1f} --j6 {-args.j6:+.1f}。")
        else:
            print("  臂【保持使能】，未动。")
        arm.close()
        return 0 if ok else 5
    finally:
        if cam is not None:
            cam.close()


if __name__ == "__main__":
    sys.exit(main())
