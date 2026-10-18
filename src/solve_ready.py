# -*- coding: utf-8 -*-
"""求一个中性 ready 位：工具竖直向下、臂抬起、远离杯与台面。

用途：当前停放姿态（臂前伸、手腕已滚转）与任何"相机在杯上方"的观察位差得太远
——`see_cup.py --lookdown-plan` 解出 J4 要翻 244°，被 cap 闸（60°/关节）全拦。
先把臂挪到一个**各关节跳变都小**的中性位，再从那里规划观察位。

判据：与当前姿态的每关节跳变 ≤55°（闸门 60°，留余量），残差 <0.5mm、tilt <0.1°。

只读：全程只调带缓存的控制器正解，不发任何运动指令。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import fr3_math                                             # noqa: E402
from fr_fkcache import Budget, FKCache                      # noqa: E402
from fr_rpc import FRRpc                                    # noqa: E402
import verify_table_place as V                              # noqa: E402


def main():
    rpc = FRRpc(ip=config.ROBOT_IP)
    fc = FKCache(rpc.fk)
    fr3_math.attach_cache(fc)
    lim = V.set_limits(rpc)
    q_now = np.asarray(rpc.joint_deg(), float)
    print(f"当前关节 {np.round(q_now,2).tolist()}")
    print(f"当前法兰 {np.round(fc.T(q_now)[:3,3],1).tolist()}")
    seeds = V.build_seeds(fc, Budget(90), (V.AZ, V.AZ + 180.0),
                          (-91., -72., -54., -38.),
                          np.arange(-150., 151., 25.))
    print(f"精确朝下种子 {len(seeds)} 个（tilt 最大 "
          f"{max(s[3] for s in seeds):.2f}°）\n", flush=True)

    def xy(r, az=V.AZ):
        return np.array([r * np.cos(np.radians(az)), r * np.sin(np.radians(az))])

    print(f"{'r':>6s}{'法兰z':>8s}{'残差mm':>8s}{'tilt':>8s}{'max|dq|':>9s}"
          f"{'各关节Δq':>36s}  判读")
    print("-" * 90, flush=True)
    best = None
    for r in (360., 400., 440., 480.):
        for z in (300., 400., 500.):
            try:
                g = V.polish(fc, np.append(xy(r), z), seeds)
            except Exception as exc:                            # noqa: BLE001
                print(f"{r:6.0f}{z:8.0f}  ✗ {exc}", flush=True)
                continue
            if g is None:
                print(f"{r:6.0f}{z:8.0f}  ✗ 无朝下解", flush=True)
                continue
            q, err, tl, p = g
            dq = np.abs(q - q_now)
            mx = float(dq.max())
            ok = mx <= 55.0 and err < 0.5 and tl < 0.1
            print(f"{r:6.0f}{z:8.0f}{err:8.3f}{tl:7.2f}°{mx:9.1f}"
                  f"{np.round(dq,0).astype(int).tolist()!s:>36s}"
                  f"  {'✓ 一步可达' if ok else ''}", flush=True)
            if ok and (best is None or mx < best[0]):
                best = (mx, r, z, q.copy(), p.copy())
    fc.save(force=True)
    if best is None:
        print("\n✗ 没有一步到位的 ready 位 —— 需要分段（见报告）")
        return
    mx, r, z, q, p = best
    T = fc.T(q)
    ring = p + T[:3, :3] @ np.array([0.0, 0.0, V.RC])
    d_ring = float(np.hypot(ring[0] - config.CUP_AXIS_XY[0],
                            ring[1] - config.CUP_AXIS_XY[1]))
    print(f"\n★ ready 位：法兰 ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})  r={r:.0f}  max|dq|={mx:.1f}°")
    print(f"  关节 {np.round(q,2).tolist()}")
    print(f"  环心 z={ring[2]:.0f}（台面净空 {ring[2]-config.TABLE_Z:+.0f}mm）  "
          f"环心离杯轴 {d_ring:.0f}mm")


if __name__ == "__main__":
    main()
