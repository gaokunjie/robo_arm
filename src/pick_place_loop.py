# -*- coding: utf-8 -*-
"""取放循环（一条命令跑到用户中断）。

每轮：
    起点=初始位姿(HOME)
      ① 降到观察位(法兰 z=200, 保持 HOME 姿态) → 绕 J1 点拍 → 现场解出杯轴
      ② 复核/刷新 CUP_XY → 重建环/杯模型与插入姿态
      ③ 五段接近插入（复用 approach_cup.staged：每段过净空/限位/万向锁闸）
      ④ 抬杯到「杯底离盒顶 50mm」→ 定格 5s
      ⑤ 放回原位（下降→微降脱唇→径向撤出）
      ⑥ 撤回初始位姿(HOME) → 定格 5s
    重复。

⚠ 每条运动段都经 approach_cup.staged 的闸门（万向锁 GYM_HARD_DEG、环/杯净空 MIN_CLEAR、
   单段位移、限位、到位残差）；任一不过 → 停在原地并退出，绝不硬推。
⚠ 拍帧前一律 Cam.fresh()（运动后相机先吐 ~10s 旧帧，用 raw() 会拿旧帧配新 T）。

用法：
    python3 -u -I src/pick_place_loop.py            # 无限循环
    python3 -u -I src/pick_place_loop.py --cycles 1 # 只跑 1 轮
    python3 -u -I src/pick_place_loop.py --dry      # 只规划不发指令（预检，1 轮）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                                          # noqa: E402
import cup_pose                                                       # noqa: E402
import fr3_math as M                                                  # noqa: E402
import approach_cup as AC                                             # noqa: E402
import verify_table_place as V                                       # noqa: E402
from approach_cup import wait_move                                   # noqa: E402
from detect_cup import Detector                                      # noqa: E402
from fr_arm import Arm                                                # noqa: E402
from fr_cam import Cam                                                # noqa: E402
from fr_fkcache import FKCache                                        # noqa: E402

# ---- 参数 ----
Z_OBS = 200.0                               # 观察位法兰 z（此时相机≈码平面高度）
Z_RAISE = 350.0                             # 高位安全区（approach_cup 同值）
LIFT_FLANGE_Z = config.TABLE_Z + 50.0 + config.WEDGE_PLANE_Z_ABOVE_TABLE   # 403：杯底离盒顶 50mm
Z_PLACE = config.TABLE_Z + config.INSERT_PLANE_Z_ABOVE_TABLE              # 279：插入/落回面
TAPS = [-14.0, -9.0, -4.0, 4.0, 9.0, 14.0]  # 绕 J1 的观察点拍（度）
SPEED = 8.0
HOLD_S = 5.0
HOME_POSE = np.asarray(config.HOME_POSE, float)
HOME_JOINTS = np.asarray(config.HOME_JOINTS, float)


def shoot(arm, cam, det, tag, patt):
    """等画面追上 → 连拍取检出最多的一帧 → 用拍摄时刻的真实 TCP 姿态配 4×4。"""
    patt.mkdir(parents=True, exist_ok=True)
    cam.fresh()
    best, bn = None, -1
    for _ in range(3):
        fr = cam.raw()
        dd = cup_pose.detect_ex(fr, det)
        n = len([x for x in dd if x["id"] in (1, 2)])
        if n > bn:
            bn, best = n, fr
        time.sleep(0.6)
    T = M.pose6_to_T([float(v) for v in arm.tcp_pose()])
    if best is not None:
        import cv2
        cv2.imwrite(str(patt / f"{tag}.jpg"), best)
        dets = cup_pose.detect_ex(best, det)
    else:
        dets = []
    ids = sorted(d["id"] for d in dets)
    print(f"  [{tag}] 杯码 {bn}  ids={ids}", flush=True)
    return {"T": T, "dets": dets, "tag": tag}


def observe_localize(arm, cam, det, K, D, X, q_obs, outdir):
    """在 q_obs 上绕 J1 点拍并解杯轴。返回 axis_xy（np.array）或 None。"""
    shots = [shoot(arm, cam, det, "home", outdir)]
    for dq in TAPS:
        qt = q_obs.copy()
        qt[0] += dq
        bad = AC.guard_pose6(arm.fk(qt.tolist()), f"观察 J1{dq:+.0f}")
        if bad:
            print("  跳过 " + bad)
            continue
        arm.move_j(qt.tolist(), speed=SPEED, blocking=False)
        if not wait_move(arm, 60):
            print("  观察点拍超时")
            break
        shots.append(shoot(arm, cam, det, f"J1{dq:+.0f}", outdir))
        arm.move_j(q_obs.tolist(), speed=SPEED, blocking=False)
        wait_move(arm, 60)
    res = cup_pose.solve_pnp(shots, K, D, X, say=lambda *a, **k: None)
    cup_pose.report_pnp(res)
    if not res.get("pass"):
        return None, res
    return np.asarray(res["axis_xy"], float), res


def near(a, b, mm=5.0, deg=3.0):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.linalg.norm(a[:3] - b[:3])) < mm and \
        float(np.linalg.norm(M.rotvec_deg(M.pose6_to_T(a)[:3, :3] @ M.pose6_to_T(b)[:3, :3].T))) < deg


def main():
    argv = sys.argv
    dry = "--dry" in argv
    cycles = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--cycles=")), None)
    if "--cycles" in argv:
        cycles = int(argv[argv.index("--cycles") + 1])

    arm = Arm()
    arm.connect()
    assert arm.st.error_code == 0 and not arm.estop_pressed(), arm.errors()
    if getattr(arm.st, "robot_mode", 0) != 0:
        arm.r.set_mode(0)
    fc = FKCache(arm.r.fk)
    M.attach_cache(fc)
    lim = np.asarray(V.set_limits(arm.r), float)
    fk6 = M.fk_throttled(lambda q: fc.pose(list(q)))
    K, D, X = cup_pose.load_calib()
    cam = None if dry else Cam()
    det = Detector()
    Rh_home = M.pose6_to_T(HOME_POSE)[:3, :3]
    tree, cup = AC.build_models()

    def run_seg(label, pt, Rt, q):
        return np.asarray(AC.staged(arm, fc, fk6, lim, q, np.asarray(pt, float),
                                    Rt, tree, cup, label, dry), float)

    n, done = 0, 0
    try:
        q = np.asarray(arm.joint_deg(), float)
        cur = list(arm.fk(q.tolist()))
        if not near(cur, HOME_POSE):
            print(f"前往初始位姿（当前 {[round(v,1) for v in cur[:3]]}）")
            q = run_seg("→ HOME", HOME_POSE[:3], Rh_home, q)

        while cycles is None or done < cycles:
            n += 1
            print(f"\n{'='*18} 第 {n} 轮 {'='*18}", flush=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            outdir = config.DATA_DIR / "cup_shots" / f"loop_{stamp}"

            q = run_seg("①→观察位下降", [HOME_POSE[0], HOME_POSE[1], Z_OBS], Rh_home, q)
            q_obs = q.copy()

            if dry:
                print("[dry] 跳过视觉，沿用 config.CUP_AXIS_XY")
                axis_xy = np.asarray(config.CUP_AXIS_XY, float)
            else:
                print("观察定位中…")
                axis_xy, res = observe_localize(arm, cam, det, K, D, X, q_obs, outdir)
                if axis_xy is None:
                    print("✗ 本轮定位未过闸，停在观察位（不再运动）")
                    break
                print(f"✔ 本轮杯轴 {np.round(axis_xy,1).tolist()} "
                      f"|xy|={np.linalg.norm(axis_xy):.1f}mm")
                (outdir / "result.json").write_text(
                    json.dumps({"axis_xy": [float(axis_xy[0]), float(axis_xy[1])],
                                "res": {k: str(v) for k, v in res.items()
                                        if not isinstance(v, (list, dict))}},
                               ensure_ascii=False), encoding="utf-8")

            AC.CUP_XY = axis_xy
            tree, cup = AC.build_models()
            u, Rh = AC.insert_axes()
            hxy = np.array([u[0], u[1], 0.0])
            p_ins = np.array([axis_xy[0], axis_xy[1], Z_PLACE]) - AC.RC * hxy
            p_hover = p_ins - AC.D_OUT * hxy
            p_high = np.array([p_hover[0], p_hover[1], Z_RAISE])
            p_raise = np.array([HOME_POSE[0], HOME_POSE[1], Z_RAISE])
            p_lift = np.array([p_ins[0], p_ins[1], LIFT_FLANGE_Z])
            p_peel = np.array([p_ins[0], p_ins[1], Z_PLACE - 4.0])
            p_out = p_peel - AC.D_OUT * hxy
            print(f"杯轴 {np.round(axis_xy,1).tolist()}  环心抬升 z={LIFT_FLANGE_Z:.0f}")

            segs = [
                (p_raise, Rh_home, "③① 竖直抬到高位", False),
                (p_raise, Rh, "③② 转工具轴→水平", False),
                (p_high, Rh, "③③ 平移到进场位上方", False),
                (p_hover, Rh, "③④ 下降到插入高度", False),
                (p_ins, Rh, "③⑤ 径向推入到杯心", False),
                (p_lift, Rh, "④ 抬杯(杯底离盒顶50mm)", True),
                (p_ins, Rh, "⑤① 放回下降", True),
                (p_peel, Rh, "⑤② 微降脱唇", False),
                (p_out, Rh, "⑤③ 径向撤出", False),
                (HOME_POSE[:3], Rh_home, "⑥ 撤回初始位姿", False),
            ]

            trace, rng, q = [], [], q_obs
            for pt, Rt, lab, contact in segs:
                a = len(trace)
                q = np.asarray(AC.staged(arm, fc, fk6, lim, q, np.asarray(pt, float),
                                         Rt, tree, cup, lab, True, trace), float)
                rng.append((lab, a, len(trace), contact))

            abort = False
            for lab, a, b, contact in rng:
                if contact or b <= a:
                    continue
                m = min(w["seg_clear"] for w in trace[a:b])
                if m < AC.MIN_CLEAR:
                    print(f"✗ {lab} 最小净空 {m:+.1f} < {AC.MIN_CLEAR} → 本轮中止")
                    abort = True
            if abort:
                break

            if dry:
                print("[dry] 规划完成（含抬杯接触段），未发任何运动指令")
                done += 1
                continue

            plan = {"waypoints": trace, "cup_axis": AC.CUP_XY.tolist(),
                    "table_z": config.TABLE_Z}
            idx_lift = next(b for lab, a, b, c in rng if "抬杯" in lab)
            AC.replay(arm, lim, plan, 0, idx_lift)
            print(f"  ★ 已抬杯，定格 {HOLD_S:.0f}s")
            time.sleep(HOLD_S)
            AC.replay(arm, lim, plan, idx_lift, None)
            print(f"  ★ 已回初始位姿，定格 {HOLD_S:.0f}s")
            time.sleep(HOLD_S)
            done += 1
    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        if cam is not None:
            cam.close()
        arm.close()
        print("已断开。")


if __name__ == "__main__":
    main()
