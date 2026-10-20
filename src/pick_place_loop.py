# -*- coding: utf-8 -*-
"""取放循环（一条命令跑到用户中断）。

每轮：
    起点=初始位姿(HOME)；2026-10-20 起 HOME 就定义在【观察高度 z=Z_OBS】，
       所以每轮开头不再有"先下降到观察位"那一段，直接绕 J1 点拍 → 现场解出杯轴
       （旧方案 HOME 在 z=248.3=示教位姿，每轮先降到 200；竖直总行程不变，
        省的只是那 4 个航点的段切换/整定/到位检查，约 5~7s/轮）。
    ① 绕 J1 点拍 → 现场解出杯轴
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

import contextlib
import json
import sys
import threading
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
Q_DEC = 2            # 规划起点关节量化到 2 位小数
#   ⚠ 为什么：arm.joint_deg() 有 ~1e-4° 编码器噪声，而 FKCache 键是 4 位小数
#     （fr_fkcache._key 用 :.4f）。若不量化，每次起点差一点点 ⇒ 整条轨迹错开 ⇒
#     每轮凭空多出 ~900 个新位形、规划多花 ~95s。量化后「目标不变」的重复循环
#     全部命中持久缓存，规划近乎瞬时（实测：同输入第二遍未命中 0）。
HOME_TAUGHT = np.asarray(config.HOME_POSE, float)   # 示教位姿（z=248.3，仅参考）
Z_HOME = Z_OBS                               # 2026-10-20：循环 HOME 改到观察高度
#   ⚠ 为什么：观察位就是示教 HOME 沿 z 下降 48mm、保持同一姿态；把这个姿态直接当
#     HOME，每轮开头省掉"①→观察位下降"那一段（4 航点的规划/整定/到位检查 ≈5~7s）。
#     x/y/姿态仍沿用示教 HOME；config.HOME_POSE 本身不动（其它脚本的语义不变）。
HOME_POSE = HOME_TAUGHT.copy()
HOME_POSE[2] = Z_HOME


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


def observe_localize(arm, cam, det, K, D, X, q_obs, outdir, gate=None):
    """在 q_obs 上绕 J1 点拍并解杯轴。返回 axis_xy（np.array）或 None。

    gate：定位期间③①预规划线程可能正在发 FK；拍点之间的 MoveJ 也要过运动优先闸。
    """
    def _move(qt):
        if gate is None:
            arm.move_j(qt.tolist(), speed=SPEED, blocking=False)
        else:
            with gate.motion():
                arm.move_j(qt.tolist(), speed=SPEED, blocking=False)

    shots = [shoot(arm, cam, det, "home", outdir)]
    for dq in TAPS:
        qt = q_obs.copy()
        qt[0] += dq
        bad = AC.guard_pose6(arm.fk(qt.tolist()), f"观察 J1{dq:+.0f}")
        if bad:
            print("  跳过 " + bad)
            continue
        _move(qt)
        if not wait_move(arm, 60):
            print("  观察点拍超时")
            break
        shots.append(shoot(arm, cam, det, f"J1{dq:+.0f}", outdir))
        _move(q_obs)
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


def report_timing(tag, rng, plan_t, mov_t, s0, s1, npts):
    """打印逐段「规划/运动」耗时 + 本轮 FK 缓存命中率（mov_t=None 表示还没运动）。"""
    print(f"\n── 计时 [{tag}] ── {npts} 航点 ──")
    print(f"{'段':30s}{'规划s':>8s}{'运动s':>9s}")
    for lab, a, b, contact in rng:
        if b <= a:
            continue
        flag = " (接触)" if contact else ""
        mv = f"{mov_t.get(lab, 0.0):9.1f}" if mov_t is not None else "        -"
        print(f"{lab + flag:30s}{plan_t.get(lab, 0.0):8.1f}{mv}")
    line = f"{'合计':30s}{sum(plan_t.values()):8.1f}"
    if mov_t is not None:
        line += f"{sum(mov_t.values()):9.1f}"
    print(line)
    dh, dm = s1["hits"] - s0["hits"], s1["misses"] - s0["misses"]
    tot = dh + dm
    rate = 100.0 * dh / tot if tot else 0.0
    print(f"FK 本轮 命中 {dh}/{tot}（{rate:.0f}%）未命中 {dm}  缓存 {s1['entries']} 条")


def audit_seg(lab, wps, contact):
    """非接触段：最小净空 < MIN_CLEAR → 抛 RuntimeError（调用方停在原地，不硬推）。"""
    if contact or not wps:
        return
    m = min(w["seg_clear"] for w in wps)
    if m < AC.MIN_CLEAR:
        raise RuntimeError(f"{lab} 最小净空 {m:+.1f} < {AC.MIN_CLEAR:.1f}")


def preplan_seg(*, fc, fk6, q_start, seg, tree, cup):
    """规划单段（供『定位期间预规划目标无关的 ③①』用）。返回 (wps, 耗时s)。"""
    pt, Rt, lab, _ = seg
    t0 = time.time()
    wps = list(AC.staged_gen(fc, fk6, np.asarray(q_start, float), pt, Rt, tree, cup, lab))
    return wps, time.time() - t0


def run_pipeline(arm, fc, fk6, lim, q_start, segs, tree, cup, hold_s, exec_fn=None,
                 pre0=None, gate=None):
    """流水线【后台规划线程】：规划线程逐段算航点，主线程「算好一段就执行一段」。

    规划线程与主线程共用一条 RPC 连接（_LockedRpc 串行化）；运动期间主线程只轮询 :8083
    ⇒ 锁空闲 ⇒ 规划线程持续下发 FK ⇒ 规划与运动真正重叠（墙钟 ≈ max(规划, 运动)）。
    gate：可选 RpcGate。主线程下发 MoveJ 时关闭闸门，规划线程暂停发 FK，避免在
    控制器正要收运动指令时被后台查询抢占（2026-10-20 的 408 故障窗口）。
    pre0：(wps, plan_s) —— 第 0 段若已在【定位期间】预规划好就传入，规划线程从第 1 段起。
    返回 dict(trace, rng, plan_t, mov_t, times, t_wall, t_mov)。
    接触段跳过净空闸；任一段规划期净空不过 → 抛 RuntimeError。
    exec_fn：可选注入执行器 (a, b, pump, times)，默认 AC.replay；--sim 用它演练。
    """
    n = len(segs)
    planned = [None] * n
    ready = [threading.Event() for _ in range(n)]
    errors, plan_t = [], {}
    if pre0 is not None:
        planned[0], plan_t[segs[0][2]] = pre0
        ready[0].set()

    def planner():
        if pre0 is not None:
            q = np.asarray(pre0[0][-1]["q"], float) if pre0[0] else np.asarray(q_start, float)
            start = 1
        else:
            q = np.asarray(q_start, float)
            start = 0
        for k in range(start, n):
            pt, Rt, lab, c = segs[k]
            t0 = time.time()
            try:
                wps = list(AC.staged_gen(fc, fk6, q, pt, Rt, tree, cup, lab))
            except Exception as e:                       # noqa: BLE001
                errors.append(e)
                ready[k].set()
                return
            planned[k] = wps
            plan_t[lab] = time.time() - t0
            if wps:
                q = np.asarray(wps[-1]["q"], float)
            ready[k].set()

    t_wall0 = time.time()
    trace, rng, times = [], [], []
    th = threading.Thread(target=planner, name="planner", daemon=True)
    th.start()

    def exec_seg(a, b):
        if exec_fn is None:
            AC.replay(arm, lim, {"waypoints": trace}, a, b, times=times, quiet=True,
                      gate=gate)
        else:
            exec_fn(a, b, None, times)

    for k in range(n):
        ready[k].wait()
        if errors:
            raise RuntimeError(str(errors[0]))
        lab, contact = segs[k][2], segs[k][3]
        wps = planned[k]
        audit_seg(lab, wps, contact)
        a = len(trace)
        trace += wps
        rng.append((lab, a, len(trace), contact))
        print(f"\n【流水线 {k+1}/{n}】执行「{lab}」({len(wps)}点)"
              f"（本段规划已就绪 {plan_t.get(lab, 0.0):.1f}s）", flush=True)
        exec_seg(a, len(trace))
        if "抬杯" in lab and hold_s > 0:
            print(f"  ★ 已抬杯，定格 {hold_s:.0f}s")
            time.sleep(hold_s)

    if hold_s > 0:
        print(f"  ★ 已回初始位姿，定格 {hold_s:.0f}s")
        time.sleep(hold_s)

    dt_by_idx = dict(times)
    mov_t = {lab: sum(dt_by_idx.get(i, 0.0) for i in range(a2, b2))
             for lab, a2, b2, c in rng}
    t_mov = sum(dt for _, dt in times)
    return dict(trace=trace, rng=rng, plan_t=plan_t, mov_t=mov_t, times=times,
                t_wall=time.time() - t_wall0, t_mov=t_mov)


def sim_exec(a, b, pump, times):
    """--sim 的假执行器：不真动臂，仅记录时间戳（用于演练流水线逻辑）。"""
    for idx in range(a, b):
        times.append((idx, 0.0))


class _LockedRpc:
    """把 FRRpc 的每次调用串行化。

    规划线程（FK, :20003）与主线程（move_j/joint_deg, :20003）共用一条连接 ⇒ 必须互斥。
    运动期间主线程只轮询 :8083（不走本锁）⇒ 锁基本空闲 ⇒ 规划线程能持续下发 FK。
    """

    def __init__(self, r, lock):
        self._r = r
        self._lock = lock

    def __getattr__(self, name):
        fn = getattr(self._r, name)
        if not callable(fn):
            return fn

        def wrapped(*a, **k):
            with self._lock:
                return fn(*a, **k)
        return wrapped


class RpcGate:
    """运动指令优先闸：MoveJ 下发的那一刻，规划线程不得发新的 FK。

    只锁「下发」这一瞬，不锁整个运动：等待到位期间规划线程照常跑，保留流水线的
    重叠收益。另留 quiet_s 的安静窗口，避免 MoveJ 的响应刚回来、FK 就立刻压上去。

    为什么需要：2026-10-20 一次真机 408 就发生在后台规划持续发 FK 的窗口里。
    _LockedRpc 只保证串行，不保证「发运动指令时控制器是安静的」。
    """

    def __init__(self, quiet_s=0.05):
        self._cv = threading.Condition()
        self._busy = False
        self._quiet_s = float(quiet_s)

    def wait_idle(self):
        with self._cv:
            while self._busy:
                self._cv.wait()

    @contextlib.contextmanager
    def motion(self):
        with self._cv:
            while self._busy:
                self._cv.wait()
            self._busy = True
        try:
            yield
        finally:
            if self._quiet_s > 0:
                time.sleep(self._quiet_s)
            with self._cv:
                self._busy = False
                self._cv.notify_all()


def parse_cycles(argv):
    """解析 --cycles N / --cycles=N / -cycles N / -cycles=N（单双横线都收）。

    2026-10-20 实踩：文档写 --cycles，实际敲成 -cycles 时旧解析返回 None，
    脚本进无限循环。这里统一剥掉前导横线再比。
    """
    for i, a in enumerate(argv):
        name = a.lstrip("-")
        if name.startswith("cycles="):
            return int(name.split("=", 1)[1])
        if name == "cycles":
            return int(argv[i + 1])
    return None


def main():
    argv = sys.argv
    dry = "--dry" in argv
    sim = "--sim" in argv            # 流水线逻辑演练：规划照跑、运动用假执行器
    planonly = dry or sim
    cycles = parse_cycles(argv)

    arm = Arm()
    arm.connect()
    assert arm.st.error_code == 0 and not arm.estop_pressed(), arm.errors()
    if getattr(arm.st, "robot_mode", 0) != 0:
        arm.r.set_mode(0)
    _rpc_lock = threading.Lock()
    arm.r = _LockedRpc(arm.r, _rpc_lock)      # 规划线程与主线程共用连接 ⇒ 调用串行化
    gate = RpcGate()                          # 运动指令优先：发 MoveJ 时暂停后台 FK
    def _fk_rpc(q):
        gate.wait_idle()
        return arm.r.fk(q)

    fc = FKCache(_fk_rpc)                     # 缓存命中不走网络，也就无需过闸
    M.attach_cache(fc)
    lim = np.asarray(V.set_limits(arm.r), float)
    fk6 = M.fk_throttled(lambda q: fc.pose(list(q)))
    K, D, X = cup_pose.load_calib()
    cam = None if planonly else Cam()
    det = Detector()
    Rh_home = M.pose6_to_T(HOME_POSE)[:3, :3]
    tree, cup = AC.build_models()

    def run_seg(label, pt, Rt, q):
        return np.asarray(AC.staged(arm, fc, fk6, lim, q, np.asarray(pt, float),
                                    Rt, tree, cup, label, planonly), float)

    n, done = 0, 0
    try:
        q = np.round(np.asarray(arm.joint_deg(), float), Q_DEC)
        cur = list(arm.fk(q.tolist()))
        if not near(cur, HOME_POSE):
            print(f"前往初始位姿（当前 {[round(v,1) for v in cur[:3]]}）")
            q = run_seg("→ HOME", HOME_POSE[:3], Rh_home, q)

        while cycles is None or done < cycles:
            n += 1
            print(f"\n{'='*18} 第 {n} 轮 {'='*18}", flush=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            outdir = config.DATA_DIR / "cup_shots" / f"loop_{stamp}"

            # HOME 已在观察高度（见 Z_HOME）⇒ 不再单开"下降"段，直接进入观察。
            q_obs = q.copy()

            # ③① 与杯子无关（在 HOME xy 竖直抬升）⇒ 可在【定位】期间就预规划，把它的规划藏进定位墙钟。
            q_start = np.round(np.asarray(q_obs, float), Q_DEC)   # 量化起点（优化一）
            p_raise = np.array([HOME_POSE[0], HOME_POSE[1], Z_RAISE])
            seg0 = (p_raise, Rh_home, "③① 竖直抬到高位", False)
            pre, pre_th = {}, None
            if not planonly:
                def _pre():
                    try:
                        pre["r"] = preplan_seg(fc=fc, fk6=fk6, q_start=q_start, seg=seg0,
                                               tree=tree, cup=cup)
                    except Exception as e:                   # noqa: BLE001
                        pre["e"] = e
                pre_th = threading.Thread(target=_pre, name="preplan0", daemon=True)
                pre_th.start()

            t_loc = 0.0
            if planonly:
                print("[dry/sim] 跳过视觉，沿用 config.CUP_AXIS_XY")
                axis_xy = np.asarray(config.CUP_AXIS_XY, float)
            else:
                print("观察定位中…")
                t_l0 = time.time()
                axis_xy, res = observe_localize(arm, cam, det, K, D, X, q_obs, outdir,
                                                gate=gate)
                t_loc = time.time() - t_l0
                if axis_xy is None:
                    print("✗ 本轮定位未过闸，停在观察位（不再运动）")
                    if pre_th is not None:
                        pre_th.join()
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
            p_lift = np.array([p_ins[0], p_ins[1], LIFT_FLANGE_Z])
            p_peel = np.array([p_ins[0], p_ins[1], Z_PLACE - 4.0])
            p_out = p_peel - AC.D_OUT * hxy
            print(f"杯轴 {np.round(axis_xy,1).tolist()}  环心抬升 z={LIFT_FLANGE_Z:.0f}")

            pre0 = None
            if pre_th is not None:
                pre_th.join()
                if pre.get("r"):
                    pre0 = pre["r"]
                    print(f"  ③① 已随定位预规划（{pre0[1]:.1f}s，与定位重叠）")
                else:
                    print(f"  ⚠ ③① 预规划失败（{pre.get('e')}），回退流水线内规划")

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

            s0 = fc.stats()
            t_p0 = time.time()

            if dry:
                trace, rng, q = [], [], q_start
                plan_t = {}
                for pt, Rt, lab, contact in segs:
                    a = len(trace)
                    t0 = time.time()
                    q = np.asarray(AC.staged(arm, fc, fk6, lim, q, np.asarray(pt, float),
                                             Rt, tree, cup, lab, True, trace), float)
                    plan_t[lab] = time.time() - t0
                    rng.append((lab, a, len(trace), contact))
                t_plan = time.time() - t_p0
                s1 = fc.stats()
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
                report_timing(f"dry 第{n}轮", rng, plan_t, None, s0, s1, len(trace))
                print(f"定位 {t_loc:.1f}s  规划 {t_plan:.1f}s（未运动）")
                print("[dry] 规划完成（含抬杯接触段），未发任何运动指令")
                done += 1
                continue

            # —— 真机：流水线（优化二）；--sim 用假执行器演练 ——
            try:
                R = run_pipeline(arm, fc, fk6, lim, q_start, segs, tree, cup,
                                 0.0 if sim else HOLD_S,
                                 exec_fn=sim_exec if sim else None, pre0=pre0,
                                 gate=gate)
            except RuntimeError as e:
                print(f"✗ {e} → 本轮中止（停在原地）")
                break
            s1 = fc.stats()
            report_timing(f"第{n}轮", R["rng"], R["plan_t"], R["mov_t"], s0, s1,
                          len(R["trace"]))
            print(f"定位(+③①预规划) {t_loc:.1f}s  运动段墙钟 {R['t_wall']:.1f}s"
                  f"（规划 {sum(R['plan_t'].values()):.1f}s / 运动 {R['t_mov']:.1f}s 已重叠）"
                  f"  全轮 ≈ {t_loc + R['t_wall'] + 2 * HOLD_S:.1f}s")
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
