# -*- coding: utf-8 -*-
"""进场位【只读】验证 —— 方案 C，真实杯轴。

## 为什么必须单独验（2026-10-18 几何裁决）
环是水平 C 形环（内径93/外径113/缺口85）；杯是有锥度的回转体，**杯缘外径 101**。
    杯缘 101 > 缺口 85 且 > 内径 93  ⇒ 环**不可能**从杯缘正上方套下去。
竖直下降时环心必须离杯轴 > 缘半径 50.8 + 环外半径 56.5 = **107.3mm**（数值扫描实测：d=107 仍侵入、d=110 起全程净空）。
⇒ **`config.D_OUT=25` 对"下降"是错的**（注释只算了环内壁 vs 杯身 9.4mm，漏了杯缘）。
⇒ 正确顺序：**大站位下降到插入高度 → 再径向推进**。推进段从 d=115 推到 0 实测全程无侵入。

## 本脚本解什么
    hover : 环心 = 杯轴 + D_APP·û（站位 115mm），z = 插入面；工具竖直、缺口朝杯轴
并把法兰落点/关节解出来复核可达（残差/tilt/限位）。全程只读，不发运动指令。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import fr3_math
from fr_fkcache import FKCache
from fr_rpc import FRRpc
import verify_table_place as V

CUP_AXIS = np.array([-31.8, -589.0])      # 视觉实测（|xy|=589.9，与卷尺 590 差 0.2mm）
CUP_BOTTOM_Z = -62.2                      # 视觉实测（与卡尺 −85 差 22.8mm，按 S14 用视觉值）
D_APP = 115.0                             # 站位（>107.3，留 ~8mm 余量）
RC = float(config.RING_CENTER_IN_FLANGE[2])
Z_INS = CUP_BOTTOM_Z + config.INSERT_PLANE_Z_ABOVE_TABLE     # 插入时【环心】z


def gap_dir_of(q):
    T = fr3_math.fk(q) if False else None
    return None


def main():
    rpc = FRRpc(ip=config.ROBOT_IP)
    fc = FKCache(rpc.fk); fr3_math.attach_cache(fc)
    lim = V.set_limits(rpc)
    az_cup = float(np.degrees(np.arctan2(CUP_AXIS[1], CUP_AXIS[0])))
    print(f"杯轴 {CUP_AXIS.tolist()}  |xy|={np.linalg.norm(CUP_AXIS):.1f}  杯底z={CUP_BOTTOM_Z}")
    print(f"插入时环心 z={Z_INS:.1f}  ⇒ 法兰 z = {Z_INS+RC:.1f}")
    print(f"站位 D_APP={D_APP:.0f}mm（下限 107.3）\n")

    results = {}
    # û_out = 杯轴→环心 的方向。测两种：向基座(径向内) / 背离基座(径向外) / +x 侧向
    candidates = {
        "径向内(朝基座)": az_cup + 180.0,
        "径向外(背离)":   az_cup,
        "+x 侧向":        0.0,
    }
    for name, u_az in candidates.items():
        u = np.array([np.cos(np.radians(u_az)), np.sin(np.radians(u_az))])
        ring_xy = CUP_AXIS + D_APP * u
        tgt = np.append(ring_xy, Z_INS + RC)                    # 法兰落点
        seeds = V.build_seeds(fc, V.Budget(80), (u_az, u_az + 180.0),
                              (-91., -72., -54., -38.), np.arange(-150., 151., 25.))
        g = V.polish(fc, tgt, seeds)
        if g is None:
            print(f"[{name}] û方位 {u_az:7.1f}°  环心{np.round(ring_xy,1).tolist()}  ✗ 无朝下解")
            continue
        q, err, tilt, p = g
        T = fc.T(q)
        rc = p + T[:3, :3] @ np.array([0., 0., RC])
        inl = bool(np.all(q >= lim[:, 0] - 1e-6) and np.all(q <= lim[:, 1] + 1e-6))
        # 缺口朝向：工具系 RING_GAP_DIR_DEG，转基座系
        gd = config.RING_GAP_DIR_DEG
        gv = T[:3, :3] @ np.array([np.cos(np.radians(gd)), np.sin(np.radians(gd)), 0.])
        gv = gv[:2] / np.linalg.norm(gv[:2])
        to_axis = CUP_AXIS - rc[:2]; to_axis /= np.linalg.norm(to_axis)
        ang = float(np.degrees(np.arccos(np.clip(gv @ to_axis, -1, 1))))
        # 缺口修正：滚转 q6 使 gv 转到 to_axis
        best = None
        for q6 in np.arange(-175, 175.1, 0.5):
            qq = q.copy(); qq[5] = q6
            T2 = fc.T(qq)
            gv2 = T2[:3, :3] @ np.array([np.cos(np.radians(gd)), np.sin(np.radians(gd)), 0.])
            gv2 = gv2[:2] / np.linalg.norm(gv2[:2])
            a = float(np.degrees(np.arccos(np.clip(gv2 @ to_axis, -1, 1))))
            if best is None or a < best[0]: best = (a, q6)
        a6, q6 = best
        qq = q.copy(); qq[5] = q6
        inl6 = bool(np.all(qq >= lim[:, 0] - 1e-6) and np.all(qq <= lim[:, 1] + 1e-6))
        print(f"[{name}] û方位 {u_az:7.1f}°  环心{np.round(ring_xy,1).tolist()}  法兰{np.round(p,1).tolist()}")
        print(f"      残差 {err:.3f}mm  tilt {tilt:.2f}°  限位{'✓' if inl else '✗'}  "
              f"缺口(未修){ang:.1f}°→(q6={q6:+.1f}°){a6:.2f}°  {'✓' if a6<5.8 and inl6 else '✗'}")
        print(f"      关节 [{', '.join(f'{v:.2f}' for v in qq)}]\n")
        results[name] = (qq, err, tilt, a6, inl6, p)
    fc.save(force=True)
    ok = [k for k, v in results.items() if v[1] < 0.5 and v[2] < 0.1 and v[4]]
    print("★ 可用进场方向：" + ("、".join(ok) if ok else "无"))


if __name__ == "__main__":
    main()
