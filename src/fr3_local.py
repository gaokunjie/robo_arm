# -*- coding: utf-8 -*-
"""FR3 本地运动学模型（URDF + 真机拟合）——离线 FK/IK，不发任何指令。

## 为什么需要它（2026-10-18，第 5 次控制器崩溃之后立的纪律）

`fr3_math.solve_delta` 用控制器的 `GetForwardKin` 当 FK 引擎。一次雅可比 = 6 发，
一次规划 = 45s/步 × 40 步 = 30 分钟；把节流降到 0.002s 做可达域扫描时，
**几十万发把控制器应用层打死**（FK_MIN_THROTTLE_S=0.05 是那次事故留下的疤）。

⇒ 批量离线计算（可达图、IK 扫描、多起点搜索）**禁止**用实时控制器当 FK 引擎。
本模块就是那个替代品：纯 numpy，微秒级，无网络。

## 模型形式

    p_flange = Rz(β) · FK_urdf(q + [0, o2..o6]) + t        （β 绕基座 Z）

`o1` 与 `β` 简并，固定为 0。`signs` 是零位方向的候选（老固件与 v6 URDF 可能反号）。
真机位姿约定与 calibrate/fr3_math 一致：R = Rz(rz)·Ry(ry)·Rx(rx)（ZYX 右乘）。

## 校准纪律

⚠ **本模型在校准样本上 <1mm 之前，不得单独用来下"够得着/够不着"的结论**——
必须 (a) 在留出样本上验证过，且 (b) 与控制器 FK 在**目标位形上**逐点比对过。
理由：`vendor/fr3_urdf/README.md` 记录过未校准时直接 FK 与真机差 60~300mm。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from fr3_math import T_to_pose6, pose6_to_T, rot_x, rot_y, rot_z

ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = ROOT / "vendor" / "fr3_urdf" / "fairino3_v6.urdf"
PARAMS_PATH = ROOT / "data" / "fr3_local_kin.json"


# ---------------------------------------------------------------- URDF

def parse_urdf(path=URDF_PATH):
    """抽 6 个 revolute 关节的 origin/axis。返回 list[dict]。"""
    txt = Path(path).read_text(encoding="utf-8")
    joints = []
    for m in re.finditer(
        r'<joint\s+name="(j\d)"[^>]*type="revolute"[^>]*>(.*?)</joint>', txt, re.S
    ):
        name, body = m.groups()
        o = re.search(r'<origin\s+xyz="([^"]*)"\s+rpy="([^"]*)"', body)
        a = re.search(r'<axis\s+xyz="([^"]*)"', body)
        if not o or not a:
            continue
        joints.append(dict(
            name=name,
            xyz=np.array([float(v) for v in o.group(1).split()]) * 1000.0,  # m → mm
            rpy=np.array([float(v) for v in o.group(2).split()]),
            axis=np.array([float(v) for v in a.group(1).split()]),
        ))
    if len(joints) != 6:
        raise RuntimeError(f"URDF 解析出 {len(joints)} 个关节，应为 6：{path}")
    return joints


JOINTS = parse_urdf()


def selftest(verbose=True):
    """自检：**独立实现**的标量 FK 与批量 FK 必须逐点一致。

    为什么要有这个：批量 FK 是拟合的热路径（每次函数调用算全部样本），
    而"批量写错了但内部自洽"这类 bug 拟合 RMS 抓不到 —— 表现为"怎么拟都 200mm+"
    这种含糊现象。留一个与批量实现**不共享代码**的标量参考，逐点比对才能当场定位。

    2026-10-18 实踩：`_rpy_to_R` 把 URDF 的弧度 rpy 当成度喂给 fr3_math.rot_*，
    FK 整体偏 802mm，但逐步推演完全自洽。**自洽 ≠ 正确**。
    """
    def axis_rot_ref(a, ang):
        a = np.asarray(a, float)
        a = a / np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

    def fk_ref(q_deg):
        """独立参考实现：直接按 URDF 弧度语义构造旋转，不复用本模块任何辅助函数。"""
        T = np.eye(4)
        for i, jnt in enumerate(JOINTS):
            rx, ry, rz = (float(v) for v in jnt["rpy"])
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            Ti = np.eye(4)
            Ti[:3, :3] = Rz @ Ry @ Rx
            Ti[:3, 3] = jnt["xyz"]
            A = np.eye(4)
            A[:3, :3] = axis_rot_ref(jnt["axis"], np.radians(float(q_deg[i])))
            T = T @ Ti @ A
        return T

    rng = np.random.default_rng(0)
    Q = rng.uniform(-180.0, 180.0, (300, 6))
    mine = urdf_fk_batch(Q)
    ref = np.array([fk_ref(q) for q in Q])
    err = float(np.abs(mine - ref).max())

    # 链长自洽：全零位下 J2 在 z=140、臂展 = |J6−J2| 应为 520+204 量级
    T0 = fk_ref([0, 0, 0, 0, 0, 0])
    d = float(np.linalg.norm(T0[:3, 3] - np.array([0.0, 0.0, JOINTS[1]["xyz"][2]])))

    if verbose:
        print(f"  批量 FK vs 独立参考实现（300 个随机位形）：最大差 {err:.3e} mm")
        print(f"  全零位法兰 {np.round(T0[:3, 3], 2).tolist()}  "
              f"离 J2 轴 {d:.1f}mm（理论最大 724）")
    if err > 1e-6:
        raise AssertionError(f"批量 FK 与参考实现不一致（{err:.3e} mm）—— 别拿去拟合")
    if not (500.0 < d < 760.0):
        raise AssertionError(f"链长自检失败：J2→法兰 {d:.1f}mm 不在合理区间")
    if verbose:
        print("  自检通过")
    return err


def axis_rot(a, ang):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _rpy_to_R(rpy_rad):
    """URDF 的 <origin rpy> 单位是**弧度**（URDF 规范如此）。

    ⚠⚠ 绝不能直接喂给 `fr3_math.rot_x/rot_y/rot_z` —— 那三个入参是【度】，内部
    会再 `math.radians` 一次。本文件 j2/j5/j6 的 rpy 都是 1.5708（弧度 = 90°），
    错当成度就变成 1.57°，**FK 整体偏 802mm，而逐步自洽完全看不出来**（2026-10-18 实踩）。
    这里显式转弧度→度再交给 fr3_math，保证与全项目其它代码同一套旋转实现。
    """
    rx, ry, rz = (float(v) for v in rpy_rad)
    return rot_z(np.degrees(rz)) @ rot_y(np.degrees(ry)) @ rot_x(np.degrees(rx))


LINK_KEYS = ("sh", "a1", "a2", "w1", "w2")
"""可拟合的 5 个链长（mm），各自对应 URDF 里哪个 origin：

    sh = j2 的 z   —— 肩枢高度
    a1 = |j3 的 x| —— 大臂
    a2 = |j4 的 x| —— 小臂
    w1 = j5 的 z   —— 腕第一段
    w2 = j6 的 z   —— 腕第二段

⚠ 为什么必须把它们当自由参数：只用【零位偏置 + 基座变换】去拟合（老脚本的做法），
在 j2∈[−197,−129] 那条窄缝上能做到 2mm，但一外推到整个关节域就崩到 140mm。
链长错是**比例性**误差，零位偏置（角度量）根本补不上。
"""

DEFAULT_LINKS = dict(sh=140.0, a1=280.0, a2=240.01, w1=102.0, w2=102.0)


def apply_links(links=None):
    """按链长参数生成一份关节表（不修改全局 JOINTS）。"""
    L = dict(DEFAULT_LINKS)
    if links:
        L.update({k: float(v) for k, v in links.items() if k in L})
    js = [dict(j) for j in JOINTS]
    js[1]["xyz"] = np.array([0.0, 0.0, L["sh"]])
    js[2]["xyz"] = np.array([-L["a1"], 0.0, 0.0])
    js[3]["xyz"] = np.array([-L["a2"], 0.0, 0.0])
    js[4]["xyz"] = np.array([0.0, 0.0, L["w1"]])
    js[5]["xyz"] = np.array([0.0, 0.0, L["w2"]])
    return js, L


def urdf_fk_batch(Q, joints=None):
    """批量 URDF FK。Q: N×6 → (N,4,4)。

    ⚠ **必须批量**：拟合时 `residuals` 每个函数调用都要算全部样本。
    逐样本循环 = 每次调用 N 次 FK，配合大 max_nfev 就是十亿量级 —— 2026-10-18
    的符号扫描就是这样把 20 分钟耗光、看着像卡死的。
    """
    Q = np.atleast_2d(np.asarray(Q, float))
    js = JOINTS if joints is None else joints
    n = len(Q)
    T = np.tile(np.eye(4), (n, 1, 1))
    for i, jnt in enumerate(js):
        Ti = np.tile(np.eye(4), (n, 1, 1))
        Ti[:, :3, :3] = _rpy_to_R(jnt["rpy"])
        Ti[:, :3, 3] = jnt["xyz"]
        ang = np.radians(Q[:, i])
        a = jnt["axis"] / np.linalg.norm(jnt["axis"])
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        s, c = np.sin(ang), np.cos(ang)
        A = np.tile(np.eye(3), (n, 1, 1)) + s[:, None, None] * K + (1.0 - c)[:, None, None] * (K @ K)
        R4 = np.tile(np.eye(4), (n, 1, 1))
        R4[:, :3, :3] = A
        T = T @ Ti @ R4
    return T


def urdf_fk(q_deg, joints=None):
    """URDF 原始 FK（不含 β/t/零位偏置）。返回 4×4。"""
    return urdf_fk_batch(np.asarray(q_deg, float).reshape(1, 6), joints)[0]


# ---------------------------------------------------------------- 拟合参数

class Model:
    """p_flange = Rz(β)·FK(q + off; links) + t。"""

    def __init__(self, beta=0.0, t=(0.0, 0.0, 0.0), off=(0.0,) * 6, signs=(1,) * 6,
                 links=None):
        self.beta = float(beta)
        self.t = np.asarray(t, float)
        self.off = np.asarray(off, float)
        self.signs = np.asarray(signs, float)
        self.links = dict(DEFAULT_LINKS)
        if links:
            self.links.update({k: float(v) for k, v in links.items() if k in DEFAULT_LINKS})
        self._joints = apply_links(self.links)[0]

    # ---- 正解 ----
    def fk_T_batch(self, Q):
        """批量正解。Q: N×6 → (N,4,4)（与 fk_T 同模型，纯性能路径）。"""
        Q = np.atleast_2d(np.asarray(Q, float))
        T = urdf_fk_batch(self.signs * (Q + self.off), self._joints)
        Tb = np.tile(np.eye(4), (len(Q), 1, 1))
        Tb[:, :3, :3] = rot_z(self.beta)
        T = Tb @ T
        T[:, :3, 3] += self.t
        return T

    def fk_T(self, q_deg):
        return self.fk_T_batch(q_deg)[0]

    def fk(self, q_deg):
        """法兰位姿 [x,y,z,rx,ry,rz]（与控制器 / calibrate 同约定）。"""
        return T_to_pose6(self.fk_T(q_deg))

    # ---- 序列化 ----
    def to_dict(self):
        return dict(beta=self.beta, t=self.t.tolist(), off=self.off.tolist(),
                    signs=[int(v) for v in self.signs], links=self.links)

    @classmethod
    def from_dict(cls, d):
        return cls(d["beta"], d["t"], d["off"], d["signs"], d.get("links"))

    def save(self, path=PARAMS_PATH):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path=PARAMS_PATH):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"本地运动学模型不存在：{p}\n"
                "先跑 src/fit_local_kin.py（用真机 FK 样本拟合 + 留出验证）")
        return Model.from_dict(json.loads(p.read_text(encoding="utf-8")))

    # ---- 拟合 ----
    def residuals(self, Q, P):
        """Q: N×6 真机关节(deg)；P: N×3 真机法兰位置(mm)。返回 N×3 残差。"""
        return self.fk_T_batch(Q)[:, :3, 3] - np.asarray(P, float)


def fit(Q, P, signs=(1,) * 6, free_links=False, verbose=False):
    """拟合 β + t + 零位偏置 o2..o6（o1 与 β 简并，固定 0），可选再放 5 个链长。

    Q: N×6 真机关节角(deg)   P: N×3 真机法兰位置(mm)

    `free_links=True` 时参数从 9 个增到 14 个（+sh,a1,a2,w1,w2）。
    **新采的跨域样本必须放链长**：只用角度类参数拟合，在窄缝数据上能做到 2mm，
    但外推到整个关节域就崩到 140mm —— 链长是比例性误差，角度偏置补不上。

    ⚠ 用 `method='trf'` + 显式界，而不是无界 `lm`：无界 LM 在错模型上不失败，
    只会一路跑到 β=3247° 的退化解而 RMS 照样"收敛"（2026-10-18 实测 349mm）。
    """
    from scipy.optimize import least_squares

    Q = np.asarray(Q, float)
    P = np.asarray(P, float)
    n_par = 14 if free_links else 9
    if len(Q) < n_par * 2:
        raise ValueError(f"只有 {len(Q)} 个样本，不足以定 {n_par} 个参数（需要 >= {n_par * 2}）")
    s = np.asarray(signs, float)
    LK = list(LINK_KEYS)

    def unpack(p):
        beta, t, off2 = p[0], p[1:4], p[4:9]
        links = None
        if free_links:
            links = {k: p[9 + i] for i, k in enumerate(LK)}
        return Model(beta, t, [0.0] + list(off2), s, links)

    def resid(p):
        return unpack(p).residuals(Q, P).ravel()

    lo = [-720.0, -3000.0, -3000.0, -3000.0, -180.0, -180.0, -180.0, -180.0, -180.0]
    hi = [720.0, 3000.0, 3000.0, 3000.0, 180.0, 180.0, 180.0, 180.0, 180.0]
    if free_links:
        for k in LK:
            lo.append(DEFAULT_LINKS[k] * 0.80)
            hi.append(DEFAULT_LINKS[k] * 1.20)
    lo, hi = np.array(lo), np.array(hi)

    t0 = P.mean(axis=0)
    best = None
    for b0 in (0.0, 90.0, 180.0, -90.0):
        p0 = [b0, t0[0], t0[1], t0[2], 0.0, 0.0, 0.0, 0.0, 0.0]
        if free_links:
            p0 += [DEFAULT_LINKS[k] for k in LK]
        p0 = np.clip(np.array(p0, float), lo, hi)
        try:
            r = least_squares(resid, p0, method="trf", bounds=(lo, hi), max_nfev=4000)
        except Exception as exc:                                  # noqa: BLE001
            if verbose:
                print(f"    β0={b0} 失败：{exc}")
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("所有初值都没收敛")
    m = unpack(best.x)
    rms = float(np.sqrt(np.mean(np.sum(m.residuals(Q, P) ** 2, axis=1))))
    return m, rms


# ---------------------------------------------------------------- 平面闭式可达性

def planar_reach_limit(reach=622.0, shoulder_z=140.0):
    """法兰"正对某点、工具竖直向下"时的闭式可达边界。

    推导（可复核，不依赖任何拟合参数）：
      J2 在基座 z=zs；J2→J4 是 280+240 的平面 2 链；腕 J4→J5→J6 各 102。
      工具 z 轴竖直向下 ⇒ θ5+θ6 = 180°，于是
          p_J4 = p_flange + 102·(sinθ5, 1−cosθ5)
      右端矢量长度 = 204·|sin(θ5/2)| ≤ 204，**方向遍历整圈** ⇒
          min_θ5 |p_flange + 102·u − p_J2| = | |p_flange − p_J2| − 102 |
      2 链够得到 ⟺ | |p_flange−p_J2| − 102 | ≤ 520，且 ≥ 40。
      ⇒ 充要条件（对距离而言）：
          **|p_flange − p_J2| ≤ 520 + 102 = 622 = 臂展**
      即   d² + (z_flange − 140)² ≤ 622²

    返回 (reach, shoulder_z)。这两个数来自 URDF，不是猜的。
    """
    return float(reach), float(shoulder_z)


def max_d_for_z(z_flange, reach=622.0, shoulder_z=140.0):
    """给定法兰高度 z，允许的最大水平距离 d（mm）。够不着返回 0。"""
    dz = float(z_flange) - float(shoulder_z)
    v = reach ** 2 - dz ** 2
    return float(np.sqrt(v)) if v > 0 else 0.0


def down_axis_reachable(d, z_flange, reach=622.0, shoulder_z=140.0):
    """距离层面的闭式判定（不含关节限位）。"""
    return d ** 2 + (z_flange - shoulder_z) ** 2 <= reach ** 2


def grasp_flange_z(ring_plane_z, ring_cen_in_flange=90.0):
    """环中面高度 → 法兰高度（工具 z 竖直向下时，法兰在环上方）。"""
    return float(ring_plane_z) + float(ring_cen_in_flange)
