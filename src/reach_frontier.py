# -*- coding: utf-8 -*-
"""给定水平距 r，**工具竖直向下**时法兰 z 能到多高 → 杯子要放多近才够得着。

## 为什么要有这个脚本（它替代掉的那些做法）

原做法：在控制器 FK 上直接跑 scipy 数值优化，扫 9 个 r × 6 个起点。
代价 = 9×6×470 ≈ 25000 次 FK ÷ 20 次/秒 ≈ **21 分钟**，而且中途零输出
⇒ 2026-10-18 连续三次被当成卡死而中止。

现在：① `fr_fkcache.FKCache` 把每次查询落盘，重复实验零成本；
② `Budget` 墙钟到点就带着部分结果退出；③ 每行 flush。

## 物理判据（为什么必须是"工具朝下"）

环与法兰**同轴**（`RING_CENTER_IN_FLANGE=[0,0,90]`，沿工具 z）⇒ 环平面垂直于工具 z。
杯是竖直的，环要横着套进去 ⇒ **工具 z 必须竖直**。这不是姿态偏好，是硬约束。

## 用法
    python3 -I src/reach_frontier.py                 # 默认 r=425，预算 150s
    python3 -I src/reach_frontier.py --r 350 400 425 450 --budget 240
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
from fr_fkcache import Budget, FKCache, Timeout            # noqa: E402
from fr_rpc import FRRpc                                   # noqa: E402

# 控制器软限位（GetSoftJointLimit 实测，与 URDF 一致）
LIM = np.degrees(np.array([[-175.0, 175.0], [-265.0, 85.0], [-160.0, 160.0],
                           [-265.0, 85.0], [-175.0, 175.0], [-175.0, 175.0]]))
BOUND = [(LIM[i, 0], LIM[i, 1]) for i in range(6)]

W_R, W_T = 2e3, 2e4
"""惩罚权重：工具朝下 2e4 远重于 r 偏差 2e3。
判据是"工具必须朝下"（错了整个抓取方案作废），r 差一点只是够不够得着。"""


def make_obj(fc, r_target):
    def f(q):
        T = fc.T(q)
        r = float(np.hypot(T[0, 3], T[1, 3]))
        return (-T[2, 3]
                + W_T * (T[0, 2] ** 2 + T[1, 2] ** 2)
                + W_R * (r - r_target) ** 2)
    return f


def warm_starts(fc, r_target, prev):
    """从上一个 r 的解热启动 —— 沿前沿走一步只需几十次 FK，不用每次重头多起点。"""
    out = []
    if prev is not None:
        for dq in (0.0, 5.0, -5.0, 10.0, -10.0):
            q = prev.copy()
            q[1] = np.clip(q[1] + dq, LIM[1, 0], LIM[1, 1])
            out.append(q)
    if not out:
        out = [[13.5, -129.0, 82.8, -71.3, -21.6, 46.9],
               [-85.0, -100.0, -60.0, -80.0, -40.0, 0.0],
               [-85.0, -140.0, 20.0, -100.0, -90.0, 0.0],
               [-85.0, -60.0, -140.0, -60.0, -100.0, 0.0]]
    return out


def solve_at(fc, r_target, starts, budget, maxiter=400, tol_pend=2e-3):
    """在 r_target 处求"工具朝下的最高法兰 z"。返回 (z, r, q, 偏离角°) 或 None。"""
    from scipy.optimize import minimize
    best = None
    for s in starts:
        if budget.left() < 15:
            break
        try:
            res = minimize(make_obj(fc, r_target), s, method="L-BFGS-B",
                           bounds=BOUND, options=dict(maxiter=maxiter, eps=1.5,
                                                      ftol=1e-14, gtol=1e-14))
        except Timeout:
            raise
        except Exception as exc:                              # noqa: BLE001
            print(f"      起点失败：{exc}", flush=True)
            continue
        T = fc.T(res.x)
        pen = float(T[0, 2] ** 2 + T[1, 2] ** 2)
        if pen >= tol_pend:
            continue
        z = float(T[2, 3])
        r = float(np.hypot(T[0, 3], T[1, 3]))
        if best is None or z > best[0]:
            best = (z, r, np.asarray(res.x, float), np.degrees(np.sqrt(pen)))
    return best


def main():
    ap = argparse.ArgumentParser(description="工具朝下时的可达前沿")
    ap.add_argument("--r", type=float, nargs="*", default=[425.0],
                    help="要测的水平距（可多个，按给出顺序做，前一个给后一个热启动）")
    ap.add_argument("--budget", type=float, default=150.0, help="墙钟预算（秒）")
    ap.add_argument("--table-z", type=float, default=None,
                    help="承载面 z（默认取 config.TABLE_Z）")
    args = ap.parse_args()

    tz = args.table_z if args.table_z is not None else config.TABLE_Z
    rc = config.RING_CENTER_IN_FLANGE[2]
    z_ins = tz + config.INSERT_PLANE_Z_ABOVE_TABLE + rc
    z_wedge = tz + config.WEDGE_PLANE_Z_ABOVE_TABLE + rc
    z_lift = z_wedge + config.LIFT_H
    print("=== 目标位姿（法兰高度；环心在法兰下方 90mm，工具竖直向下）===")
    print(f"  承载面 TABLE_Z = {tz}   环心悬于法兰下 {rc:.0f}mm")
    print(f"  插入位 环面 z={tz + config.INSERT_PLANE_Z_ABOVE_TABLE:.0f} → 法兰 z={z_ins:.0f}")
    print(f"  卡缘位 环面 z={tz + config.WEDGE_PLANE_Z_ABOVE_TABLE:.0f} → 法兰 z={z_wedge:.0f}")
    print(f"  抬升位 +{config.LIFT_H:.0f}                → 法兰 z={z_lift:.0f}")
    print(f"  预算 {args.budget:.0f}s\n")

    rpc = FRRpc(ip=config.ROBOT_IP)
    fc = FKCache(rpc.fk)
    print(f"缓存：已有 {len(fc.data)} 条", flush=True)
    b = Budget(args.budget, "前沿扫描")

    print(f"{'目标r':>7s}{'实测r':>8s}{'法兰z':>9s}{'偏离角':>9s}{'环心z':>8s}  判读")
    print("-" * 62, flush=True)
    rows, prev = [], None
    for r_t in args.r:
        try:
            b.check(f"r={r_t}")
            got = solve_at(fc, r_t, warm_starts(fc, r_t, prev), b)
        except Timeout as exc:
            print(f"\n⚠ 预算用尽（{exc}），已有 {len(rows)} 个结果，继续用它们下结论。",
                  file=sys.stderr, flush=True)
            break
        if got is None:
            print(f"{r_t:7.0f}{'—':>8s}{'无解':>9s}", flush=True)
            continue
        z, r, q, deg = got
        prev = q
        rows.append((r_t, z, deg))
        j1 = ("✓够" if z >= z_ins else f"✗差{z_ins - z:.0f}") + "插入" + \
             ("／✓够" if z >= z_lift else f"／✗差{z_lift - z:.0f}") + "抬升"
        print(f"{r_t:7.0f}{r:8.1f}{z:9.1f}{deg:8.2f}°{z - rc:8.1f}  {j1}", flush=True)

    fc.save(force=True)
    st = fc.stats()
    print(f"\nFK 缓存：条目 {st['entries']}，命中 {st['hits']}，新查 {st['misses']} "
          f"（命中率 {st['hit_rate']*100:.0f}%）  用时 {b.elapsed():.0f}s/{b.seconds:.0f}s")
    if rows:
        ok_ins = [r for r, z, _ in rows if z >= z_ins]
        ok_lift = [r for r, z, _ in rows if z >= z_lift]
        print(f"\n★ 够得着【插入位 法兰z={z_ins:.0f}】的最大杯距 = "
              f"{max(ok_ins):.0f}mm" if ok_ins else
              f"\n★ 在测过的所有杯距上都够不着插入位（z={z_ins:.0f}）")
        print(f"★ 够得着【抬升位 法兰z={z_lift:.0f}】的最大杯距 = "
              f"{max(ok_lift):.0f}mm" if ok_lift else
              f"★ 在测过的所有杯距上都够不着抬升位（z={z_lift:.0f}）")


if __name__ == "__main__":
    main()
