# -*- coding: utf-8 -*-
"""固定轨迹取放循环：一次定位/一次规划，之后零视觉、零规划回放。

计划文件默认 `src/fixed_pick_place_plan.json`：
    杯轴 (-37.41, -584.82) 来自 2026-10-20 10:49 真机视觉实测；
    TABLE_Z=239（config 物理锚点）；插入面 z=279；抬杯法兰 z=403。
    共 71 个关节航点，10 段，7 个停点组。

运行时：
    HOME → 抬升/转插入姿态/平移/下降到进场位（自由段连续）
         → 径向插入（慢速、停）
         → 抬杯（慢速、停）
         → 放回下降（慢速、停）→ 微降脱唇（停）→ 径向撤出（停）
         → 回 HOME（停）→ 下一轮。

用法：
    python3 -u -I src/pick_place_fixed.py --dry
    python3 -u -I src/pick_place_fixed.py --cycles 1            # 默认 2.0x
    python3 -u -I src/pick_place_fixed.py --cycles 1 --safe     # 1.0x / max 20%
    python3 -u -I src/pick_place_fixed.py --cycles 1 --speed-scale 1.5 --max-speed-pct 30
    python3 -u -I src/pick_place_fixed.py --cycles 1 --acc-scale 1.5
    python3 -u -I src/pick_place_fixed.py                       # 无限循环

默认 2.0x：计划基线 1.0x 的所有段速度乘以 2，最高 36%（原 18%）。
接触段因此为 10%（原 5%），自由段 36%/24%/20%。这超过
config.MAX_SPEED_PCT=20，但只在本固定轨迹脚本内生效，config 未改。
若想恢复旧的 1.0x 行为，用 --safe。

安全：
    * 回放前逐点检查万向锁、关节限位、非接触段净空。
    * 每个停点组结束后读实际关节/笛卡尔位姿/错误码；任一不过停在原地。
    * 不使用视觉、不做 FK 规划、不重发超时指令。
    * 首次跑 2.0x 务必手放急停、目视整臂；不稳就 --safe 或 --speed-scale 1.5。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import fr3_math as M
import approach_cup as AC
import verify_table_place as V
from fr_arm import Arm
from fr_fkcache import FKCache
from fr_rpc import FRRpcError

PLAN_DEFAULT = Path(__file__).resolve().parent / "fixed_pick_place_plan.json"
TOL_JOINT_DEG = 0.5
TOL_HOME_DEG = 1.0
TOL_POS_MM = 1.5
TOL_ROT_DEG = 0.5
NEAR_DEG = 1.0
STABLE_S = 0.2
SEND_GAP_S = 0.01
WAIT_TIMEOUT_S = 180.0
SPEED_SCALE_DEFAULT = 2.0
SPEED_CAP_DEFAULT = 36.0
ACC_SCALE_DEFAULT = 1.0
BLEND_SCALE_DEFAULT = 1.0


def load_plan(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"固定轨迹不存在：{p}")
    plan = json.loads(p.read_text(encoding="utf-8"))
    if not plan.get("waypoints"):
        raise RuntimeError(f"固定轨迹没有航点：{p}")
    return plan


def ang_diff(a, b):
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0)


def wrap_joint(q, lim):
    qw = np.asarray(q, float).copy()
    for i in range(6):
        while qw[i] > lim[i, 1]:
            qw[i] -= 360.0
        while qw[i] < lim[i, 0]:
            qw[i] += 360.0
    return qw


def gym_off(pose):
    _, off = AC.gym_lock(M.pose6_to_T([float(v) for v in pose])[:3, :3])
    return float(off)


def plan_groups(plan):
    groups, cur = [], []
    for i, w in enumerate(plan["waypoints"]):
        cur.append(i)
        if w.get("stop_here"):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def check_plan(plan, lim=None, verbose=True, speed_scale=1.0,
               max_speed_pct=20.0, acc_scale=1.0, blend_scale=1.0):
    errors = []
    wps = plan["waypoints"]
    min_gym = (999.0, "")
    for w in wps:
        off = gym_off(w["pose"])
        if off < min_gym[0]:
            min_gym = (off, w.get("label", "?"))
        if off < AC.GYM_HARD_DEG:
            errors.append(f"万向锁硬闸：{w.get('label')} 距 ±90° 仅 {off:.4f}°")
        if lim is not None:
            qw = wrap_joint(w["q"], lim)
            oob = [f"j{j+1}" for j in range(6)
                   if not (lim[j, 0] - 1e-6 <= qw[j] <= lim[j, 1] + 1e-6)]
            if oob:
                errors.append(f"超限位：{w.get('label')} {oob}")
        if (not w.get("contact")) and float(w.get("seg_clear", 999.0)) < AC.MIN_CLEAR:
            errors.append(f"净空不足：{w.get('label')} "
                          f"{float(w['seg_clear']):+.2f} < {AC.MIN_CLEAR:.1f}")

    source = plan.get("source", {})
    home_q = np.asarray(source.get("home_q", wps[-1]["q"]), float)
    dj = float(np.max(ang_diff(wps[-1]["q"], home_q)))
    if dj > TOL_HOME_DEG:
        errors.append(f"终点不在 HOME：关节差 {dj:.3f}° > {TOL_HOME_DEG}°")

    home_pose = np.asarray(source.get("home_pose", wps[-1]["pose"]), float)
    T_end = M.pose6_to_T(wps[-1]["pose"])
    T_home = M.pose6_to_T(home_pose)
    dp = float(np.linalg.norm(T_end[:3, 3] - T_home[:3, 3]))
    dr = float(np.linalg.norm(M.rotvec_deg(T_end[:3, :3] @ T_home[:3, :3].T)))
    if dp > TOL_POS_MM or dr > TOL_ROT_DEG:
        errors.append(f"终点位姿不在 HOME：位置差 {dp:.3f}mm，姿态差 {dr:.3f}°")

    seg4 = [w for w in wps if w.get("seg") == 4]
    if seg4:
        Tr = M.pose6_to_T(seg4[-1]["pose"]) @ AC.M_FR
        cup = np.asarray(source.get("cup_axis_xy", [0.0, 0.0]), float)
        dxy = float(np.linalg.norm(Tr[:2, 3] - cup))
        if dxy > 1.0:
            errors.append(f"插入位环心离杯轴 {dxy:.2f}mm > 1.0mm")

    if verbose:
        print(f"固定计划：{plan.get('created', '?')}")
        print(f"  杯轴 {np.round(np.asarray(source.get('cup_axis_xy', [0, 0]), float), 2).tolist()}"
              f"  来源 {source.get('cup_axis_source', '?')}")
        print(f"  TABLE_Z {source.get('table_z')}  插入面 z {source.get('z_place')}"
              f"  抬杯法兰 z {source.get('z_lift')}")
        print(f"  航点 {len(wps)}  组 {len(plan_groups(plan))}")
        acc_pct = min(max(float(config.MAX_ACC_PCT) * acc_scale, 1.0), 100.0)
        print(f"  速度缩放 {speed_scale:.2f}x  上限 {max_speed_pct:.0f}%  "
              f"加速 {acc_pct:.0f}%  blend {blend_scale:.2f}x")
        print(f"  {'组':>3} {'段':28s} {'航点':>4} {'实际%':>7} {'blend':>7} {'停':>2} {'最小净空':>8}")
        for gi, idxs in enumerate(plan_groups(plan)):
            last = wps[idxs[-1]]
            labels = []
            for k in idxs:
                lab = wps[k].get("seg_label", "?")
                if not labels or labels[-1] != lab:
                    labels.append(lab)
            ws = [wps[k] for k in idxs]
            contact = all(bool(w.get("contact")) for w in ws)
            m = min(float(w.get("seg_clear", 999.0)) for w in ws)
            sp = [min(max(float(w.get("speed", 0.0)) * speed_scale, 1.0),
                      float(max_speed_pct)) for w in ws]
            bl = [min(500, max(0, int(round(int(w.get("blend_ms_eff", 0)) * blend_scale))))
                  for w in ws]
            sp_s = f"{min(sp):.0f}~{max(sp):.0f}" if min(sp) != max(sp) else f"{min(sp):.0f}"
            bl_s = f"{min(bl)}~{max(bl)}" if min(bl) != max(bl) else f"{min(bl)}"
            print(f"  {gi:>3} {' → '.join(labels):28s} {len(idxs):>4} "
                  f"{sp_s:>7} {bl_s:>7} "
                  f"{'是' if last.get('stop_here') else '否':>2} "
                  f"{'接触' if contact else f'{m:+8.2f}'}")
        print(f"  万向锁最小余量 {min_gym[0]:.3f}° @ {min_gym[1]}")
    return errors


def validate_plan(plan, lim=None):
    errors = check_plan(plan, lim=lim, verbose=False)
    if errors:
        raise RuntimeError("计划校验失败：\n  - " + "\n  - ".join(errors))


def dry_run(plan, args):
    errors = check_plan(plan, lim=None, verbose=True,
                        speed_scale=args.speed_scale,
                        max_speed_pct=args.max_speed_pct,
                        acc_scale=args.acc_scale,
                        blend_scale=args.blend_scale)
    if errors:
        print("\n✗ 计划离线校验失败：")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\n✓ 计划离线校验通过（未连接机械臂、未运动）")
    if args.max_speed_pct > config.MAX_SPEED_PCT:
        print(f"  ⚠ 当前上限 {args.max_speed_pct:.0f}% > config.MAX_SPEED_PCT="
              f"{config.MAX_SPEED_PCT}%；仅本脚本生效，config 未改")
    if args.speed_scale >= 1.9:
        print("  真机首次建议：--cycles 1（2.0x）手放急停；不稳改用 --safe 或 1.5x")
    else:
        print(f"  当前 {args.speed_scale:.2f}x / 上限 {args.max_speed_pct:.0f}%；"
              f"手放急停，先跑 --cycles 1")
    return 0


def ensure_home(arm, plan, lim, no_home_check=False):
    home_q = np.asarray(plan["source"]["home_q"], float)
    q = np.asarray(arm.joint_deg(), float)
    dj = float(np.max(ang_diff(q, home_q)))
    if dj <= TOL_HOME_DEG:
        print(f"✔ 已在 HOME（最大关节差 {dj:.3f}°）")
        return
    if no_home_check:
        print(f"⚠ 当前不在 HOME（最大关节差 {dj:.3f}°），按 --no-home-check 继续")
        return

    print(f"当前不在 HOME（最大关节差 {dj:.3f}°），先用带闸门的 AC.staged 回 HOME…")
    cup = np.asarray(plan["source"]["cup_axis_xy"], float)
    AC.CUP_XY = cup
    tree, cup_pts = AC.build_models()
    fc = FKCache(arm.r.fk)
    M.attach_cache(fc)
    fk6 = M.fk_throttled(lambda qq: fc.pose(list(qq)))
    home_pose = np.asarray(plan["source"]["home_pose"], float)
    Rh_home = M.pose6_to_T(home_pose)[:3, :3]
    q0 = np.round(q, 2)
    try:
        AC.staged(arm, fc, fk6, lim, q0, home_pose[:3], Rh_home, tree, cup_pts,
                  "→ HOME", False)
    except SystemExit as e:
        raise RuntimeError(f"回 HOME 被安全闸拦下：{e}") from e
    q1 = np.asarray(arm.joint_deg(), float)
    dj = float(np.max(ang_diff(q1, home_q)))
    if dj > TOL_HOME_DEG:
        raise RuntimeError(f"回 HOME 后关节差 {dj:.3f}° 仍超限")
    print(f"✔ 已回 HOME（最大关节差 {dj:.3f}°）")


def preflight(arm, plan, no_home_check):
    arm.connect()
    err = arm.errors().get("error_code")
    if err not in (None, 0):
        raise RuntimeError(f"机械臂有错误码 {err}，先清错")
    if arm.estop_pressed():
        raise RuntimeError("急停被按下，先复位")
    if getattr(arm.st, "robot_mode", 0) != 0:
        print("robot_mode != 0，切换到自动模式…")
        arm.r.set_mode(0)
        time.sleep(0.3)
        if getattr(arm.st, "robot_mode", 0) != 0:
            raise RuntimeError("切换自动模式失败")
    arm.enable()
    time.sleep(0.3)
    lim = np.asarray(V.set_limits(arm.r), float)
    validate_plan(plan, lim=lim)
    ensure_home(arm, plan, lim, no_home_check)
    return lim


def send_waypoint(arm, w, speed, acc, blend_ms):
    try:
        arm.r.move_j([float(x) for x in w["q"]],
                     [float(x) for x in w["pose"]],
                     config.TOOL_ID, config.USER_ID,
                     float(speed), float(acc), 100.0,
                     float(blend_ms))
    except FRRpcError as e:
        if e.uncertain:
            raise RuntimeError(
                f"{w.get('label')} MoveJ 响应超时/连接异常；已停止发送。"
                f"机械臂可能仍在执行最后一条已接受的指令，请立即目视/急停") from e
        raise RuntimeError(f"{w.get('label')} MoveJ 被拒绝：{e}") from e


def wait_group(arm, target_q, q_before, timeout=WAIT_TIMEOUT_S):
    t0 = time.time()
    started = False
    stable = 0.0
    while time.time() - t0 < timeout:
        err = getattr(arm.st, "error_code", None)
        if err not in (None, 0):
            raise RuntimeError(f"运动中出现 error_code={err}")
        if getattr(arm.st, "estop", None) == 1:
            raise RuntimeError("急停被按下")

        q = np.asarray(arm.st.joints, float)
        md = int(getattr(arm.st, "motion_done", 1))
        if not started:
            if md == 0 or float(np.max(ang_diff(q, q_before))) > 0.05:
                started = True
            elif time.time() - t0 > 0.8 and md == 1:
                started = True

        near = float(np.max(ang_diff(q, target_q))) < NEAR_DEG
        if started and near and md == 1:
            stable += 0.1
            if stable >= STABLE_S:
                return time.time() - t0
        else:
            stable = 0.0
        time.sleep(0.1)
    raise RuntimeError(f"等待到位超时 {timeout:.0f}s"
                       f"（目标关节 {np.round(target_q, 2).tolist()}）")


def run_group(arm, plan, indices, gi, speed_scale, max_speed_pct,
              acc_scale, blend_scale, no_blend, hold, verbose):
    wps = plan["waypoints"]
    target = wps[indices[-1]]
    target_q = np.asarray(target["q"], float)
    q_before = np.asarray(arm.st.joints, float)
    acc = min(max(float(config.MAX_ACC_PCT) * acc_scale, 1.0), 100.0)
    labels = []
    for k in indices:
        lab = wps[k].get("seg_label", "?")
        if not labels or labels[-1] != lab:
            labels.append(lab)
    print(f"\n【组 {gi}】{' → '.join(labels)}  航点 {len(indices)}  终点 {target.get('label')}")
    t0 = time.time()
    for k, idx in enumerate(indices):
        w = wps[idx]
        speed = min(max(float(w.get("speed", 5.0)) * speed_scale, 1.0),
                    float(max_speed_pct))
        blend = 0
        if not no_blend and k != len(indices) - 1:
            blend = min(500, max(0, int(round(
                int(w.get("blend_ms_eff", 0)) * blend_scale))))
        off = gym_off(w["pose"])
        if off < AC.GYM_HARD_DEG:
            raise RuntimeError(f"{w.get('label')} 距万向锁仅 {off:.4f}°，拒绝下发")
        send_waypoint(arm, w, speed, acc, blend)
        if verbose:
            print(f"  → {w.get('label')}  speed={speed:.1f}%  "
                  f"acc={acc:.0f}%  blend={blend}ms")
        time.sleep(SEND_GAP_S)

    dt = wait_group(arm, target_q, q_before)
    err = arm.errors().get("error_code")
    if err not in (None, 0):
        raise RuntimeError(f"组 {gi} 结束 error_code={err}")
    if arm.estop_pressed():
        raise RuntimeError(f"组 {gi} 结束急停被按下")

    qa = np.asarray(arm.joint_deg(), float)
    dj = float(np.max(ang_diff(qa, target_q)))
    pose_a = arm.fk([float(x) for x in qa])
    T_a = M.pose6_to_T(pose_a)
    T_t = M.pose6_to_T(target["pose"])
    dp = float(np.linalg.norm(T_a[:3, 3] - T_t[:3, 3]))
    dr = float(np.linalg.norm(M.rotvec_deg(T_a[:3, :3] @ T_t[:3, :3].T)))
    if dj > TOL_JOINT_DEG or dp > TOL_POS_MM or dr > TOL_ROT_DEG:
        raise RuntimeError(f"组 {gi} 到位残差超限：关节 {dj:.3f}° / "
                           f"位置 {dp:.3f}mm / 姿态 {dr:.3f}°")
    print(f"  ✔ 到位 关节差 {dj:.3f}°  位置差 {dp:.3f}mm  姿态差 {dr:.3f}°  "
          f"用时 {dt:.1f}s", flush=True)

    last_label = target.get("seg_label", "")
    if hold > 0 and ("抬杯" in last_label or "撤回初始位姿" in last_label):
        print(f"  ★ 定格 {hold:.1f}s")
        time.sleep(hold)
    return time.time() - t0


def run_loop(arm, plan, args):
    groups = plan_groups(plan)
    upto = args.upto_group
    if upto is not None:
        if upto < 0:
            raise RuntimeError("--upto-group 不能为负")
        if upto > len(groups):
            print(f"⚠ --upto-group {upto} 超过总组数 {len(groups)}，按 {len(groups)} 处理")
            upto = len(groups)
        groups = groups[:upto]
    if not groups:
        print("没有要执行的组。")
        return

    log_fh = None
    if args.log:
        lp = Path(args.log)
        lp.parent.mkdir(parents=True, exist_ok=True)
        log_fh = lp.open("a", encoding="utf-8")

    n = 0
    try:
        while args.cycles is None or n < args.cycles:
            n += 1
            print(f"\n{'=' * 18} 第 {n} 轮 {'=' * 18}", flush=True)
            t0 = time.time()
            ok = True
            err_msg = ""
            try:
                for gi, idxs in enumerate(groups):
                    run_group(arm, plan, idxs, gi, args.speed_scale,
                              args.max_speed_pct, args.acc_scale,
                              args.blend_scale, args.no_blend,
                              args.hold, args.verbose)
            except RuntimeError as e:
                ok = False
                err_msg = str(e)
                print(f"✗ {err_msg} → 本轮中止（停在原地，不再发运动指令）", flush=True)
            dt = time.time() - t0
            if ok:
                print(f"\n✔ 第 {n} 轮完成  用时 {dt:.1f}s", flush=True)
            if log_fh:
                log_fh.write(json.dumps({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "cycle": n, "ok": ok, "elapsed_s": round(dt, 2),
                    "error": err_msg, "speed_scale": args.speed_scale,
                    "max_speed_pct": args.max_speed_pct,
                    "acc_scale": args.acc_scale,
                    "blend_scale": args.blend_scale,
                    "no_blend": bool(args.no_blend),
                    "plan_created": plan.get("created"),
                    "cup_axis_xy": plan.get("source", {}).get("cup_axis_xy"),
                }, ensure_ascii=False) + "\n")
                log_fh.flush()
            if not ok:
                break
            if upto is not None:
                print(f"已按 --upto-group={upto} 停止；机械臂不在 HOME 时请先回 HOME")
                break
    finally:
        if log_fh:
            log_fh.close()


def main():
    ap = argparse.ArgumentParser(description="固定轨迹取放循环（零视觉、零规划回放）")
    ap.add_argument("--plan", default=str(PLAN_DEFAULT), help="固定轨迹 JSON")
    ap.add_argument("--cycles", type=int, default=None, help="轮数；省略=无限循环")
    ap.add_argument("--dry", action="store_true", help="只做离线校验，不连臂、不动")
    ap.add_argument("--check", action="store_true", help="同 --dry")
    ap.add_argument("--speed-scale", type=float, default=SPEED_SCALE_DEFAULT,
                    help=f"速度缩放（默认 {SPEED_SCALE_DEFAULT:.1f}x；1.0=旧单轮基线）")
    ap.add_argument("--max-speed-pct", type=float, default=SPEED_CAP_DEFAULT,
                    help=f"速度上限百分比（默认 {SPEED_CAP_DEFAULT:.0f}%%，"
                         f"可超过 config.MAX_SPEED_PCT={config.MAX_SPEED_PCT}）")
    ap.add_argument("--acc-scale", type=float, default=ACC_SCALE_DEFAULT,
                    help="加速度缩放（默认 1.0x=30%%；想更接近 2x 周期可试 1.5）")
    ap.add_argument("--blend-scale", type=float, default=BLEND_SCALE_DEFAULT,
                    help="blend 时间缩放（2x 速度下想保持拐角可设 0.5）")
    ap.add_argument("--safe", action="store_true",
                    help="恢复安全档：1.0x、上限 20%%、acc 1.0x")
    ap.add_argument("--no-blend", action="store_true", help="关闭平滑，逐点停")
    ap.add_argument("--hold", type=float, default=0.0, help="抬杯后/回HOME后定格秒数")
    ap.add_argument("--upto-group", type=int, default=None, help="只执行前 N 组（调试用）")
    ap.add_argument("--no-home-check", action="store_true", help="启动时不强制回 HOME")
    ap.add_argument("--log", default=None, help="可选 JSONL 运行日志")
    ap.add_argument("--verbose", action="store_true", help="逐航点打印")
    args = ap.parse_args()

    if args.safe:
        args.speed_scale = 1.0
        args.max_speed_pct = float(config.MAX_SPEED_PCT)
        args.acc_scale = 1.0
    if args.speed_scale <= 0:
        ap.error("--speed-scale 必须 > 0")
    if args.max_speed_pct <= 0:
        ap.error("--max-speed-pct 必须 > 0")
    if args.max_speed_pct > 100:
        ap.error("--max-speed-pct 不能超过 100")
    if args.acc_scale <= 0:
        ap.error("--acc-scale 必须 > 0")
    if args.blend_scale < 0:
        ap.error("--blend-scale 不能为负")
    if args.cycles is not None and args.cycles < 0:
        ap.error("--cycles 不能为负")

    plan = load_plan(args.plan)
    if args.dry or args.check:
        sys.exit(dry_run(plan, args))

    validate_plan(plan)
    arm = Arm()
    ok = True
    try:
        lim = preflight(arm, plan, args.no_home_check)
        acc = min(max(float(config.MAX_ACC_PCT) * args.acc_scale, 1.0), 100.0)
        print(f"✔ 预检通过：计划 {len(plan['waypoints'])} 航点 / "
              f"{len(plan_groups(plan))} 组  速度缩放 {args.speed_scale:.2f}x  "
              f"上限 {args.max_speed_pct:.0f}%  acc {acc:.0f}%  "
              f"blend={'关' if args.no_blend else f'{args.blend_scale:.2f}x'}", flush=True)
        if args.max_speed_pct > config.MAX_SPEED_PCT:
            print(f"⚠ 速度上限 {args.max_speed_pct:.0f}% 超过 "
                  f"config.MAX_SPEED_PCT={config.MAX_SPEED_PCT}%；"
                  f"config 未改，仅本脚本生效。首次请手放急停。", flush=True)
        run_loop(arm, plan, args)
    except KeyboardInterrupt:
        print("\n用户中断。尝试暂停…")
        try:
            arm.pause()
        except Exception as e:                                  # noqa: BLE001
            print(f"  暂停失败：{e}（必要时拍急停）")
    except RuntimeError as e:
        ok = False
        print(f"✗ {e}")
    except Exception as e:                                      # noqa: BLE001
        ok = False
        print(f"✗ {type(e).__name__}: {e}")
    finally:
        try:
            arm.close()
        except Exception:                                       # noqa: BLE001
            pass
        print("已断开。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
