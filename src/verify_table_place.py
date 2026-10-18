# -*- coding: utf-8 -*-
"""方案 C 落位前的【可行性验证】：杯子直接放实验台面时，圆环能不能到位。

## 几何（方案 C：撤掉箱子，杯底直接站实验台面）

    实验台面 z = −85mm（基座系）—— 臂底座安装面离台面 85mm，基座系 z=0 在安装面
    杯底 = 台面 = −85   杯缘 = +35   ArUco 码心 = −45
    环心(插入) = −85+40  = −45  ⇒ 法兰 z =  45   （台面净空 130mm）
    环心(卡缘) = −85+114 = +29  ⇒ 法兰 z = 119   （台面净空 204mm）
    环心(抬升) = +49              ⇒ 法兰 z = 139   （台面净空 224mm）

## 方法（三步，全部只读）

    ① **种子表**：沿 q2/q3 网格解腕部，得到一批"工具精确朝下"的位形（含关节角）。
       —— 这一步不用优化器，纯 2×2 牛顿，稳定；实测 tilt 可到 0.00°。
    ② **就近取种**：按 (r,z) 距离挑最近的朝下位形作初值。
    ③ **抛光**：只对 (q2,q3) 最小化落点误差，每次评估重跑腕部解 ⇒ 朝向始终是硬约束。

## 三个必须避开的坑（2026-10-18 实踩）

    a) 腕部解**同时存在"朝上/朝下"两个分支**：只解 R02=R12=0 会收敛到朝上（tilt=180°）。
       必须显式要求 R22 < 0。
    b) **关节折返不能折到 (−180,180]**：q2/q4 限位是 [−265,85]（跨 350°），
       折到 +95° 反而越界。必须折回**各自限位区间**。
    c) 目标离种子太远时带罚函数的优化器会陷局部解（残差 100mm+）⇒ 必须先建种子表。

## 判据

残差 < 0.5mm 且 tilt < 0.1° 且全部关节在软限位内 ⇒ 该位姿成立。
⚠ 全程只调 GetForwardKin（带缓存），不发任何运动指令。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import fr3_math                                            # noqa: E402
from fr_fkcache import Budget, FKCache, Timeout             # noqa: E402
from fr_rpc import FRRpc                                   # noqa: E402

TABLE_Z = -85.0
RC = float(config.RING_CENTER_IN_FLANGE[2])
AZ = -85.7
BUDGET = 250.0
TARGETS = (("插入", TABLE_Z + 40 + RC),
           ("卡缘", TABLE_Z + 114 + RC),
           ("抬升", TABLE_Z + 114 + RC + config.LIFT_H))


LIM = None
BB = None

# ---- 成本模型（2026-10-18 立，防止再犯"没算预算就开跑"）----
FK_COST_S = fr3_math.FK_MIN_THROTTLE_S     # 0.05s/次，≈20 次/秒（防崩溃硬地板）
WRIST_ITERS = 8                            # 腕部解的热启动迭代上限
NEWTON_ITERS = 8                           # 位置抛光迭代
N_SEEDS = 2                                # 每个目标取几个种子


def cost_estimate(n_seed_cells, n_targets):
    """预估【未命中缓存】的 FK 次数与耗时。"""
    per_wrist = WRIST_ITERS * 3 + 1
    per_newton = NEWTON_ITERS * (3 * per_wrist + 1)
    calls = n_seed_cells * per_wrist + n_targets * N_SEEDS * per_newton
    return calls, calls * FK_COST_S


def set_limits(rpc):
    """从控制器取软限位并注入模块全局（fold/wrist_down/polish 都依赖它）。"""
    global LIM, BB
    LIM = soft_limits(rpc)
    BB = [(LIM[i, 0], LIM[i, 1]) for i in range(6)]
    return LIM


def soft_limits(rpc):
    """关节软限位**直接从控制器读**（GetSoftJointLimit），不硬编码。

    ⚠⚠ 2026-10-18 两次栽在"弧度 vs 度"：一次是 URDF 的 rpy（喂进收度数的
    `fr3_math.rot_*`，FK 偏 802mm），一次是这里的 `np.degrees([[-175,175],...])`
    —— 本来就是度，又乘了 57.3，限位变成 ±10026°，**关节根本没被约束**
    （解出 q4=-1588° 还"通过"了限位检查）。
    ⇒ 凡是"角"的量，本项目一律从控制器取，或至少加断言，绝不手工转换。
    """
    lim = np.asarray(rpc.joint_soft_limits(), float).reshape(6, 2)   # GetJointSoftLimitDeg
    assert np.all(np.abs(lim) < 360.0), f"软限位超出 ±360°，单位可能搞错了：{lim}"
    assert lim[0, 0] < 0 < lim[0, 1], f"软限位未夹住 0°：{lim}"
    return lim


def fold(q):
    """把关节角折回**各自限位区间**，且**只允许折 360° 的整数倍**。

    ⚠⚠ 2026-10-18 实踩两次：
      ① 折成 `lo + (q-lo) % (hi-lo)` 是**错的** —— 对 J4（限位跨 350°）它会折 350°，
         不是 360° 的整数倍 ⇒ 物理位形真的变了（实测落点偏 185mm）。
      ② 对 J6，超限 +352.7° 其实是"该腕翻转"的正常解：+352.7 ≡ −7.3（mod 360），
         落在限位内且离当前只有 7.3°。不折就永远被判"跳 352.7° > 60°"而卡死。

    正确做法：反复 ±360° 直到落进区间。旋转关节 q 与 q+360k **严格等价**，
    所以这是无损的；不像 ① 那种"折到区间内但改变了位形"。
    """
    q = np.asarray(q, float)
    out = np.empty(6)
    for i in range(6):
        lo, hi = LIM[i]
        v = float(q[i])
        for _ in range(4):                     # 限位跨度都 <720°，4 次足够
            if v > hi + 1e-9:
                v -= 360.0
            elif v < lo - 1e-9:
                v += 360.0
            else:
                break
        out[i] = v
    return out


def wrist_down(fc, q, iters=WRIST_ITERS, tol=1e-8):
    """固定 q1,q2,q3,q6，解 (q4,q5) 使法兰 z 轴 = 基座 −z（**只取朝下分支**）。"""
    q = np.asarray(q, float).copy()
    for _ in range(iters):
        T = fc.T(q)
        e = np.array([T[0, 2], T[1, 2]])
        if np.linalg.norm(e) < tol:
            break
        J = np.zeros((2, 2))
        for j in (3, 4):
            qq = q.copy()
            qq[j] += 1.0
            T2 = fc.T(qq)
            J[:, j - 3] = [T2[0, 2], T2[1, 2]] - e
        try:
            dq = np.linalg.lstsq(J, -e, rcond=None)[0]
        except np.linalg.LinAlgError:
            return q, 999.0, False
        m = float(np.max(np.abs(dq)))
        if m > 25.0:
            dq *= 25.0 / m
        q[3] = float(np.clip(q[3] + dq[0], LIM[3, 0], LIM[3, 1]))
        q[4] = float(np.clip(q[4] + dq[1], LIM[4, 0], LIM[4, 1]))
    T = fc.T(q)
    down = T[0, 2] ** 2 + T[1, 2] ** 2
    tilt = float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))))
    ok = down < 1e-10 and T[2, 2] < -0.99999          # ← 显式排除朝上分支
    return q, tilt, bool(ok)


def build_seeds(fc, budget, q1_list, q2_list, q3_list):
    """① 种子表：网格解腕部，收集"精确朝下"的位形（q1 两个方位各来一遍，
    因为"法兰方位 = q1"不成立，近似取一半转角能覆盖到）。"""
    seeds, cur = [], None
    for q1 in q1_list:
      cur = np.array([q1, -90.7, 0.0, -120.0, -60.0, 0.0])
      for q2 in q2_list:
        for q3 in q3_list:
            m0 = fc.misses
            if budget.left() < 25 or fc.misses - m0 > budget.seconds / FK_COST_S * 0.45:
                print(f"   （种子表按预算收敛于 {len(seeds)} 个）", flush=True)
                return seeds
            q = cur.copy()
            q[0], q[1], q[2] = q1, q2, q3
            q, tilt, ok = wrist_down(fc, q)
            if ok:
                cur = q
                T = fc.T(q)
                seeds.append((float(np.hypot(T[0, 3], T[1, 3])), float(T[2, 3]),
                              q.copy(), tilt))
    return seeds


def _eval(fc, p, q45):
    """给定 (q1,q2,q3)，解腕部并返回法兰落点。"""
    q = np.array([p[0], p[1], p[2], q45[0], q45[1], 0.0])
    q, tilt, ok = wrist_down(fc, q)
    T = fc.T(q)
    return T[:3, 3].copy(), tilt, ok, q


def polish(fc, xyz_t, seeds):
    """③ 就近取种 + 三维牛顿抛光。

    目标 = 法兰落点 3 个坐标，未知量 = (q1,q2,q3) 三个 ⇒ **3×3 牛顿**。

    ⚠ 为什么不能只解 (q2,q3)、靠 q1 给方位：**"法兰方位 = q1"是错的**。
      控制器 FK 实测：q1=-85.7° 的解落点在方位 108.2°（差 180°）。
      项目里也没有这种映射公式——它一直是解 6 自由度完整位姿。
      ⇒ 直接把 3 维落点当目标，让 q1 一起被解出来，不去猜方位↔q1 的关系。
    """
    if not seeds:
        return None
    tgt = np.asarray(xyz_t, float)
    ranked = sorted(seeds,
                    key=lambda s: (s[0] * np.cos(np.radians(AZ)) - tgt[0]) ** 2
                                  + (s[0] * np.sin(np.radians(AZ)) - tgt[1]) ** 2
                                  + (s[1] - tgt[2]) ** 2)
    best = None
    for r0, z0, q0, _ in ranked[:N_SEEDS]:
        p = np.array([q0[0], q0[1], q0[2]], float)
        q45 = np.array([q0[3], q0[4]], float)
        d = 0.5
        pos, tilt, good, q = _eval(fc, p, q45)
        for _ in range(NEWTON_ITERS):
            if not good or tilt > 0.2:
                good = False
                break
            e = pos - tgt
            if np.linalg.norm(e) < 1e-3:
                break
            J = np.zeros((3, 3))
            for j in range(3):
                pp = p.copy()
                pp[j] += d
                pj, _, _, _ = _eval(fc, pp, q45)
                J[:, j] = (pj - pos) / d
            try:
                dp = np.linalg.solve(J, -e)
            except np.linalg.LinAlgError:
                good = False
                break
            m = float(np.max(np.abs(dp)))
            if m > 20.0:
                dp *= 20.0 / m
            p = np.array([np.clip(p[0] + dp[0], BB[0][0], BB[0][1]),
                          np.clip(p[1] + dp[1], BB[1][0], BB[1][1]),
                          np.clip(p[2] + dp[2], BB[2][0], BB[2][1])])
            pos, tilt, good, q = _eval(fc, p, q45)
        if not good:
            continue
        q = fold(q)                                   # 折回各自限位区间
        T = fc.T(q)                                   # ← 用折回后的关节独立复核
        err = float(np.linalg.norm(T[:3, 3] - tgt))
        tl = float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))))
        if tl > 0.2:
            continue
        if best is None or err < best[1]:
            best = (q, err, tl, T[:3, 3].copy())
    return best


def main():
    rpc = FRRpc(ip=config.ROBOT_IP)
    fc = FKCache(rpc.fk)
    lim = set_limits(rpc)
    print("关节软限位（控制器 GetJointSoftLimitDeg，度）：")
    for i, (lo, hi) in enumerate(lim, 1):
        print(f"  j{i}: [{lo:7.1f}, {hi:7.1f}]")
    N_SEED_CELLS = 2 * 4 * 13
    N_TARGETS = 3
    calls, secs = cost_estimate(N_SEED_CELLS, N_TARGETS)
    print(f"成本预估：{calls} 次 FK ≈ {secs:.0f}s（节流 {FK_COST_S}s/次；"
          f"缓存命中不计费）  预算 {BUDGET:.0f}s")
    if secs > BUDGET * 0.8:
        print(f"⚠ 预估已占预算 {secs / BUDGET * 100:.0f}%，先缩减规模再跑")
    b = Budget(BUDGET)
    print(f"缓存 {len(fc.data)} 条 FK｜预算 {b.seconds:.0f}s\n")
    print(f"方案 C：台面 z={TABLE_Z:.0f}｜" + "｜".join(
        f"{n}: 环心z={z - RC:+.0f} → 法兰z={z:+.0f}（台面净空 {z - TABLE_Z:+.0f}mm）"
        for n, z in TARGETS))

    q1_list = (AZ, AZ + 180.0)
    q2_list = (-91.0, -72.0, -54.0, -38.0)
    q3_list = np.arange(-150.0, 151.0, 25.0)
    print(f"\n① 建种子表：q1 {len(q1_list)} × q2 {len(q2_list)} × q3 {len(q3_list)} …", flush=True)
    seeds = build_seeds(fc, b, q1_list, q2_list, q3_list)
    if not seeds:
        print("✗ 种子表为空 —— 预算或数值有问题，别据此下结论")
        return
    S = np.array([[s[0], s[1]] for s in seeds])
    print(f"   得到 {len(seeds)} 个精确朝下位形：r [{S[:,0].min():.0f},{S[:,0].max():.0f}]"
          f"  法兰z [{S[:,1].min():.0f},{S[:,1].max():.0f}]  最大tilt {max(s[3] for s in seeds):.2f}°",
          flush=True)

    print(f"\n{'目标':>8s}{'r':>6s}{'法兰x':>8s}{'法兰y':>8s}{'法兰z':>8s}"
          f"{'残差mm':>8s}{'tilt':>8s}  关节解")
    print("-" * 104, flush=True)
    allok = True

    def xy(r):
        return np.array([r * np.cos(np.radians(AZ)), r * np.sin(np.radians(AZ))])

    jobs = [(f"r=425", 425.0, z) for _, z in TARGETS]
    if b.left() > 120:                       # 余量够才加做杯距扫描
        jobs += [(f"r={r:.0f}", r, TARGETS[1][1]) for r in (330, 380, 470, 515)]
    for tag, r_t, z_t in jobs:
        try:
            b.check(tag)
            got = polish(fc, np.append(xy(r_t), z_t), seeds)
        except Timeout as exc:
            print(f"\n⚠ 预算用尽（{exc}）—— 已有结果仍然有效", file=sys.stderr, flush=True)
            break
        if got is None:
            print(f"{tag:>8s}  ✗ 无朝下解"); allok = False; continue
        q, err, tilt, p = got
        inlim = bool(np.all(q >= LIM[:, 0] - 1e-6) and np.all(q <= LIM[:, 1] + 1e-6))
        ok = err < 0.5 and tilt < 0.1 and inlim
        allok &= ok
        note = "" if inlim else "  ⚠越限位"
        print(f"{tag:>8s}{r_t:6.0f}{p[0]:8.1f}{p[1]:8.1f}{p[2]:8.1f}{err:8.3f}{tilt:7.2f}°  "
              f"[{', '.join(f'{v:.2f}' for v in q)}]  {'✓ 成立' if ok else '✗ 不成立'}{note}",
              flush=True)

    fc.save(force=True)
    st = fc.stats()
    print(f"\n缓存 {st['entries']} 条，命中 {st['hits']}，新查 {st['misses']}"
          f"（命中率 {st['hit_rate']*100:.0f}%） 用时 {b.elapsed():.0f}s/{b.seconds:.0f}s")
    print("\n★ " + ("三个关键位姿全部成立 ⇒ 可以撤箱子、杯底直接放实验台面。"
                    if allok else "存在不成立的位姿 ⇒ 先别撤箱子。"))


if __name__ == "__main__":
    main()
