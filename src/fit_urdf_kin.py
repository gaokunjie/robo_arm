# -*- coding: utf-8 -*-
"""用真机 FK 样本拟合 URDF 模型参数（基座系变换 + 关节零位偏置 + 符号）。

样本 = 控制器 GetForwardKin 的纯数学输出（无重力形变），全部只读采集：
  18 行 j2_sweep 表（2026-10-15 实验，0.1mm 精度）
  W1   which_rpy 实验（3 位小数）
  A1   对准位姿（2 位小数）
参数: β(基座绕Z) + tx,ty,tz + o2..o6（o1 与 β 简并，固定 0）。
用法: python3 -I fit_urdf_kin.py <urdf路径>
"""
import re
import sys
from itertools import product

import numpy as np
from scipy.optimize import least_squares

URDF = sys.argv[1] if len(sys.argv) > 1 else "dl_urdf/fairino3_v6.urdf"

txt = open(URDF, encoding="utf-8").read()
joints = []
for m in re.finditer(
    r'<joint\s+name="(j\d)"[^>]*>\s*<origin\s+xyz="([^"]*)"\s+rpy="([^"]*)"\s*/>\s*'
    r'<parent\s+link="[^"]*"\s*/>\s*<child\s+link="[^"]*"\s*/>\s*<axis\s+xyz="([^"]*)"\s*/>',
    txt,
):
    name, xyz, rpy, axis = m.groups()
    joints.append(dict(name=name,
                       xyz=np.array([float(v) for v in xyz.split()]) * 1000.0,
                       rpy=np.array([float(v) for v in rpy.split()]),
                       axis=np.array([float(v) for v in axis.split()])))
assert len(joints) == 6


def rot(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def axis_rot(a, ang):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def fk_pos(q_deg, signs):
    """URDF FK，返回法兰位置 (mm, 基座=base_link)。"""
    T = np.eye(4)
    for i, jnt in enumerate(joints):
        T_i = np.eye(4)
        T_i[:3, :3] = rot(*jnt["rpy"])
        T_i[:3, 3] = jnt["xyz"]
        A = np.eye(4)
        A[:3, :3] = axis_rot(jnt["axis"], np.radians(signs[i] * q_deg[i]))
        T = T @ T_i @ A
    return T[:3, 3]


SWEEP_J0 = [13.439, None, 82.863, -70.917, -13.581, -83.411]
SWEEP = [
    (-197.0, (388.2, -112.1, 380.8)), (-193.0, (371.0, -116.2, 404.8)),
    (-189.4, (354.2, -120.2, 425.2)), (-185.0, (332.0, -125.5, 448.6)),
    (-181.0, (310.4, -130.6, 468.4)), (-177.0, (287.4, -136.1, 486.5)),
    (-173.0, (263.3, -141.9, 503.0)), (-169.0, (238.2, -147.9, 517.7)),
    (-165.0, (212.1, -154.1, 530.5)), (-161.0, (185.2, -160.6, 541.4)),
    (-157.0, (157.6, -167.1, 550.4)), (-153.0, (129.5, -173.9, 557.4)),
    (-149.0, (101.0, -180.7, 562.4)), (-145.0, (72.2, -187.6, 565.3)),
    (-141.0, (43.3, -194.5, 566.1)),  (-137.0, (14.4, -201.4, 564.8)),
    (-133.0, (-14.4, -208.2, 561.5)), (-129.0, (-42.8, -215.0, 556.1)),
]
SAMPLES = [([SWEEP_J0[0], j2] + SWEEP_J0[2:], np.array(pos)) for j2, pos in SWEEP]
SAMPLES.append(([-12.64, -180.513, 35.86, -34.236, -1.627, -84.47],
                np.array([415.327, -300.115, 378.262])))
SAMPLES.append(([13.24, -128.69, -78.71, 18.48, -14.76, -83.90],
                np.array([414.19, -106.67, 352.82])))

J = np.array([s[0] for s in SAMPLES])          # N×6
Q = np.array([s[1] for s in SAMPLES])          # N×3 目标位置


def make_resid(signs):
    def resid(p):
        beta, tx, ty, tz = p[:4]
        off = np.array([0.0] + list(p[4:9]))    # o1=0, o2..o6
        out = []
        Rb = rot(0, 0, beta)
        for k in range(len(J)):
            q = signs * (J[k] + off)
            pred = Rb @ fk_pos(q, np.ones(6)) + np.array([tx, ty, tz])
            out.append(pred - Q[k])
        return np.concatenate(out)
    return resid


results = []
for signs in product([1, -1], repeat=6):
    s = np.array(signs, float)
    best = None
    for beta0 in (0, 90, 180, -90):
        p0 = np.array([beta0, 400.0, -100.0, 400.0, 0, 0, 0, 0, 0], float)
        try:
            r = least_squares(make_resid(s), p0, method="lm", max_nfev=6000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    rms = np.sqrt(2 * best.cost / len(best.fun))
    results.append((rms, tuple(int(v) for v in signs), best))

results.sort(key=lambda r: r[0])
print("=== 前 4 名（符号组合 + 拟合参数）===")
for rms, s, r in results[:4]:
    p = r.x
    print(f"\n符号 {s}   RMS = {rms:.3f} mm")
    print(f"  β={p[0]:8.3f}°  t=({p[1]:8.2f},{p[2]:8.2f},{p[3]:8.2f}) mm")
    print(f"  偏置 o2..o6 = {np.round(p[4:9], 3).tolist()}°")

rms, s, r = results[0]
print("\n=== 最优组合逐样本残差 ===")
resid = make_resid(np.array(s, float))(r.x).reshape(-1, 3)
for (j, q), d in zip(SAMPLES, resid):
    print(f"  J2={j[1]:7.1f}  残差 {np.round(d, 3).tolist()}  |{np.linalg.norm(d):.3f}mm")
