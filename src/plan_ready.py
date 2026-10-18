# -*- coding: utf-8 -*-
"""A 方案第 1 步：规划"当前停放位 → 中性 ready 位"的安全路径（**只读，不发指令**）。

## 为什么需要这一步

机械臂当前停在 `[13.48,-129.04,82.81,-71.34,-21.6,46.87]`，法兰 z=545。
任何"相机在杯上方"的观察位都在另一个 IK 分支（`see_cup.py` 解出 J4 要翻 244°），
被 cap 闸（60°/关节）正确拦下。
先把臂挪到一个**各关节跳变小、且远离台面与杯子**的中性位，再从那里规划观察位。

## 用什么规划

`grasp.march_to`（项目自带）：每步只走 ≤35mm / ≤12°，用 `solve_delta(steps=3,iters=8)` 解，
**每步都核验** 残差 / 软限位 / |Δq|；任一步不过就停在那里并报告，绝不硬推。
FK 走全局缓存 ⇒ 未命中才付节流。

## ⚠ 本脚本能验什么、不能验什么（别误读"路径已核验"）

**能验**：落点轨迹、关节限位、每关节跳变、法兰相对【台面】与【杯轴】的距离。
**不能验**：小臂/肘部/腕壳与台面、铝型材支架的碰撞 —— 控制器 FK 只给法兰位姿，
本项目没有整臂连杆模型。
⇒ 因此输出里同时给出"法兰最小净空"，**执行时请用户全程目视盯臂**，手放急停。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import fr3_math as M                                       # noqa: E402
from fr_fkcache import Budget, FKCache                      # noqa: E402
from fr_rpc import FRRpc                                   # noqa: E402
import grasp                                               # noqa: E402
import verify_table_place as V                             # noqa: E402

# ---- 停放位的安全判据（2026-10-18 修正）----
# 原来错用"法兰水平离杯轴 >220mm"当过滤条件 ⇒ 把【正悬在杯子上方】的正确候选
# 全部误杀。正确判据是**环 vs 杯**的两种通过方式之一：
#   (a) 环整体高于杯缘（竖直净空足）—— 这正是停放位该有的样子；
#   (b) 环水平躲开杯身（距离 > 环外半径 + 杯半径 + 余量）。
MIN_FLANGE_ABOVE_TABLE = 200.0      # 法兰离台面的最小净空
MIN_RING_ABOVE_RIM = 80.0           # 环底高于杯缘的最小竖直净空（走 (a) 时）
RING_RO = config.RING_OD / 2.0
CUP_RIM_R = config.CUP_OD_RIM / 2.0
CUP_TOP_Z = config.TABLE_Z + config.CUP_HEIGHT       # 杯缘 z = +35
CUP_XY = np.asarray(config.CUP_AXIS_XY, float)


def ring_clear(ring_c, ring_n):
    """返回 (是否安全, 说明)。ring_n = 环平面法向（工具 z）。"""
    ring_bottom = ring_c[2] - config.RING_THICK / 2.0
    d = float(np.hypot(ring_c[0] - CUP_XY[0], ring_c[1] - CUP_XY[1]))
    if ring_bottom > CUP_TOP_Z + MIN_RING_ABOVE_RIM:
        return True, f"环底 {ring_bottom:.0f} 高于杯缘 {CUP_TOP_Z:.0f} 净空 {ring_bottom-CUP_TOP_Z:+.0f}mm"
    if d > RING_RO + CUP_RIM_R + 30.0:
        return True, f"环水平躲开 {d:.0f}mm（>{RING_RO+CUP_RIM_R+30:.0f}）"
    return False, f"环底 {ring_bottom:.0f} 杯缘 {CUP_TOP_Z:.0f} 距离 {d:.0f}mm 都不足"


def main():
    rpc = FRRpc(ip=config.ROBOT_IP)
    fc = FKCache(rpc.fk)
    M.attach_cache(fc)                                   # 节流只在未命中时发生
    lim = V.set_limits(rpc)
    q_now = np.asarray(rpc.joint_deg(), float)
    T_now = fc.T(q_now)
    print(f"当前关节 {np.round(q_now,2).tolist()}")
    print(f"当前法兰 {np.round(T_now[:3,3],1).tolist()}  "
          f"(台面净空 {T_now[2,3]-config.TABLE_Z:+.0f}mm)\n")

    # ---- ① 选 ready 位：tool 竖直、远离杯与台面、且各关节跳变小 ----
    seeds = V.build_seeds(fc, Budget(100), (V.AZ, V.AZ + 180.0),
                          (-91., -72., -54., -38.), np.arange(-150., 151., 25.))
    print(f"精确朝下种子 {len(seeds)} 个", flush=True)

    def xy(r, az=V.AZ):
        return np.array([r * np.cos(np.radians(az)), r * np.sin(np.radians(az))])

    cands = []
    for r in (420., 460., 500., 540.):
        for z in (260., 320., 380., 440.):
            try:
                g = V.polish(fc, np.append(xy(r), z), seeds)
            except Exception:                                # noqa: BLE001
                continue
            if g is None:
                continue
            q, err, tl, p = g
            dq = np.abs(q - q_now)
            ring = p + fc.T(q)[:3, :3] @ np.array([0.0, 0.0, V.RC])
            ok, why = ring_clear(ring, fc.T(q)[:3, 2])
            cands.append(dict(q=q, err=err, tilt=tl, p=p, r=r, z=z, ok=ok, why=why,
                              mx=float(dq.max()), dq=dq, ring=ring))
            print(f"  候选 r={r:.0f} 法兰z={z:.0f}: 残差{err:.3f}mm tilt{tl:.2f}° "
                  f"max|Δq|={dq.max():5.1f}°  {'✓' if ok else '✗'} {why}", flush=True)
    if not cands:
        print("✗ 没找到任何 ready 候选")
        return

    safe = [c for c in cands
            if c["err"] < 0.5 and c["tilt"] < 0.1 and c["ok"]
            and c["p"][2] - config.TABLE_Z > MIN_FLANGE_ABOVE_TABLE]
    safe.sort(key=lambda c: c["mx"])
    print(f"\n通过净空/精度筛选的候选 {len(safe)} 个", flush=True)
    if not safe:
        print("✗ 无候选同时满足精度与净空")
        return
    for c in safe[:5]:
        print(f"   max|Δq|={c['mx']:6.1f}°  Δq={np.round(c['dq'],0).astype(int).tolist()}"
              f"  r={c['r']:.0f} z={c['z']:.0f}")
    tgt = safe[0]

    # ---- ② march_to 规划路径（只读）----
    print(f"\n规划路径 → ready(r={tgt['r']:.0f}, 法兰z={tgt['z']:.0f}) "
          f"目标关节 {np.round(tgt['q'],2).tolist()}", flush=True)
    b = Budget(200)
    class _Arm:                                   # 只读壳：march_to 只用 joint_soft_limits
        r = rpc
        def joint_deg(self):
            return list(rpc.joint_deg())
    fk6 = M.fk_throttled(lambda q: fc.pose(list(q)))
    try:
        q_fin, info = grasp.march_to(_Arm(), fc.T(tgt["q"]), fk6, list(q_now),
                                     max_dp=35.0, max_drot=12.0, max_iter=40,
                                     tag="ready")
    except Exception as exc:                        # noqa: BLE001
        print(f"✗ march_to 异常：{type(exc).__name__}: {exc}")
        return
    fc.save(force=True)

    # ---- ③ 路径复核 ----
    q_fin = np.asarray(q_fin, float)
    T_fin = fc.T(q_fin)
    err = float(np.linalg.norm(T_fin[:3, 3] - tgt["p"]))
    print(f"\n=== 路径复核 ===")
    print(f"  终点法兰 {np.round(T_fin[:3,3],1).tolist()}  目标 {np.round(tgt['p'],1).tolist()}"
          f"  残差 {err:.2f}mm")
    print(f"  终点关节 {np.round(q_fin,2).tolist()}")
    print(f"  总关节变化 {np.round(np.abs(q_fin-q_now),1).tolist()}  "
          f"max {np.max(np.abs(q_fin-q_now)):.1f}°")
    print(f"  台面净空 {T_fin[2,3]-config.TABLE_Z:+.0f}mm")
    ring_f = T_fin[:3, 3] + T_fin[:3, :3] @ np.array([0.0, 0.0, V.RC])
    ok_f, why_f = ring_clear(ring_f, T_fin[:3, 2])
    print(f"  终点环心 {np.round(ring_f,1).tolist()}  {'✓' if ok_f else '✗'} {why_f}")
    print(f"  限位内: {bool(np.all(q_fin>=lim[:,0]-1e-6) and np.all(q_fin<=lim[:,1]+1e-6))}")
    st = fc.stats()
    print(f"\n缓存 {st['entries']} 条，新查 {st['misses']}，命中 {st['hits']}"
          f"（{st['hit_rate']*100:.0f}%）")
    if err > 2.0:
        print("\n★ 终点残差 >2mm —— 不可直接执行，需要再规划（不猜）")
    else:
        print("\n★ 路径可执行。本脚本【未发任何运动指令】。")


if __name__ == "__main__":
    main()
