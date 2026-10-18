# -*- coding: utf-8 -*-
"""FR3 数学小件：位姿约定 + 【只用正解】的数值运动学（离线性，不发任何指令）。

为什么要它：控制器逆解（GetInverseKin / GetInverseKinRef）是禁区——四次应用层崩溃
的唯一共同因子（HANDOFF §3，全发生在工作空间边界附近的位姿，纯规划就崩、臂没动过）。
但 arm.move_l(pose, joints=[...]) 允许调用方【自己带关节】原样下发，于是"想要一个小位移"
这件事可以整体绕开控制器：

    给目标位移 → 数值雅可比 J(q)（6 次正解） → 阻尼最小二乘解 Δq → 再用正解核验达成度
    → arm.move_l(正解出的位姿, joints=q0+Δq)。

全程只有 GetForwardKin（本项目实测可信：实测关节→正解 vs 实测法兰位姿 差 0.0007mm）。

姿态约定与 calibrate.py 完全一致：**R = Rz(rz)·Ry(ry)·Rx(rx)**（ZYX，依次右乘；
2026-10-15 真机上用 J6+30° 钉死的——换约定前先重跑那个实验）。

⚠ 正解是网络调用（一次约 2~5ms）。雅可比一次要 6 发；调用方用 fk_throttled 节流，
  别在密循环里裸调（2026-10-16 两次 ~600 次调用把控制器打掉线的教训）。
"""

import math
import time

import numpy as np


# ---------------- 基本旋转 ----------------

def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rotvec_to_R(v_deg):
    """旋转向量（度，方向=轴、模=角）→ 旋转矩阵（Rodrigues）。"""
    v = np.radians(np.asarray(v_deg, float))
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def rotvec_deg(R):
    """旋转矩阵 → 旋转向量（度）。用标准 log 公式，小角度稳定、无特征向量符号歧义。"""
    R = np.asarray(R, float)
    tr = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    ang = math.degrees(math.acos(tr))
    if ang < 1e-7:
        return np.zeros(3)
    if ang > 179.99:                       # 贴近 π：用反对称部分会退化，取对称部分
        A = (R + np.eye(3)) / 2.0
        ax = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(ax))
        v = A[:, i] / (ax[i] if ax[i] > 1e-12 else 1.0)
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        return v * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float)
    return v / (2.0 * math.sin(math.radians(ang))) * ang


# ---------------- 位姿 ↔ 齐次矩阵（与 calibrate 同约定） ----------------

def pose6_to_T(pose):
    """[x,y,z,rx,ry,rz]（mm/度）→ 4×4。姿态 R = Rz(rz)·Ry(ry)·Rx(rx)。"""
    x, y, z, rx, ry, rz = (float(v) for v in pose)
    T = np.eye(4)
    T[:3, :3] = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    T[:3, 3] = (x, y, z)
    return T


def T_to_pose6(T):
    """4×4 → [x,y,z,rx,ry,rz]。R = Rz·Ry·Rx 的逆：ry=asin(−R20)、rz=atan2(R10,R00)、
    rx=atan2(R21,R22)；ry=±90° 的万向节位形单独处理（本项目位姿最接近的一次差 3°）。"""
    T = np.asarray(T, float)
    R = T[:3, :3]
    sy = -R[2, 0]
    if abs(sy) < 0.999999:
        ry = math.asin(sy)
        rz = math.atan2(R[1, 0], R[0, 0])
        rx = math.atan2(R[2, 1], R[2, 2])
    else:                                   # cy=0：把 rz 归零，角度全记在 rx 上
        ry = math.copysign(math.pi / 2, sy)
        rz = 0.0
        rx = math.atan2(math.copysign(R[0, 1], sy), math.copysign(R[0, 2], sy))
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
            math.degrees(rx), math.degrees(ry), math.degrees(rz)]


# ---------------- 数值运动学（只用正解） ----------------

FK_MIN_THROTTLE_S = 0.05
"""⛔ 正解调用节流的**硬下限**（2026-10-17 加，因打崩过一次控制器）。

事故记录：把 fk_throttled 的 sleep 降到 0.002s 做离线可达性扫描，几分钟内打出
**几十万次 GetForwardKin**，控制器应用层当场死掉（:20003/:8083 全关、只有 :80 活着）——
与 10-16 那四次崩溃完全同一个 signature，只能彻底断电 ≥1 分钟恢复。

为什么以前没出事：文档里"FK 可信"依据的是"每行 11 发、累计数小时"，那是**零星**调用。
**批量离线扫描（可达图、IK 扫描、多起点搜索）绝对不能用实时控制器当 FK 引擎。**
正确做法：① 用 URDF/自研本地模型；② 或先一次性缓存一批 FK 样本再离线算。

⇒ 下限 0.05s：即使调用方传 0.002 也会被抬到 0.05（≈20 次/秒上限）。
   这仍只够"交互级"调用；真要做扫描请改用本地模型。
"""


_FK_CACHE = None
"""可选注入的 FK 缓存（见 `attach_cache`）。注入后 `fk_throttled` 只在**未命中**时付节流代价。"""


def attach_cache(cache):
    """把 `fr_fkcache.FKCache` 注入本模块，之后所有 `fk_throttled` 自动走缓存。

    ## 为什么必须挂在【这里】而不是挂在调用方的 fk 上（2026-10-18 踩过）

        fk6 = M.fk_throttled(arm.fk, 0.01)      ← see_cup.py / grasp.py 都是这么写的

    节流的 sleep 在**被包装函数的外面**，所以如果把缓存挂在 `arm.fk` 上：
        fk6(q) → sleep(0.05) → cached_fk(q) → 命中也要睡 50ms ⇒ **命中率再高也没用**。
    挂进 `fk_throttled` 内部才是对的：
        fk6(q) → cached_fk(q) → 命中直接返回（0ms）；未命中才 sleep + 发控制器。

    这条纪律的意义：控制器 FK 是 ~20 次/秒的**硬地板**（第 5 次崩溃后立的），
    任何"每次调用都先睡 50ms"的设计都会把一次规划拖成几十分钟。
    """
    global _FK_CACHE
    _FK_CACHE = cache
    return cache


def detach_cache():
    global _FK_CACHE
    _FK_CACHE = None


def fk_throttled(fk, sleep_s=0.05):
    """给正解加节流（网络调用）。返回包装函数，接口不变。

    ⚠ sleep_s 会被硬抬到 FK_MIN_THROTTLE_S（见上面的事故记录）。
    ⚠ 若已 `attach_cache`，**节流只在缓存未命中时发生**（命中 0ms）。
    """
    if _FK_CACHE is not None:
        return _FK_CACHE.pose
    s = max(float(sleep_s), FK_MIN_THROTTLE_S)
    cache = {}

    def f(q):
        key = tuple(round(float(v), 6) for v in q)
        hit = cache.get(key)
        if hit is not None:
            return hit
        time.sleep(s)
        v = fk(q)
        if len(cache) > 30000:
            cache.clear()
        cache[key] = v
        return v
    return f


def num_jac6(fk6, q, d=0.5):
    """6×6 数值雅可比（前向差分，每个关节 1 次正解）。
    行 0~2 = d位置(mm) / d关节(deg)；行 3~5 = d【基座系】旋转(deg) / d关节(deg)。"""
    q = np.asarray(q, float)
    T0 = pose6_to_T(fk6(q))
    J = np.zeros((6, 6))
    for i in range(6):
        qq = q.copy()
        qq[i] += d
        T1 = pose6_to_T(fk6(qq))
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / d
        J[3:, i] = rotvec_deg(T1[:3, :3] @ T0[:3, :3].T) / d
    return J


def _solve_once(fk6, q0, dp, drot, iters, tol_pos, tol_ang, max_step, damp0, damp_max):
    """单段牛顿迭代（solve_delta 的内核）。"""
    q0 = np.asarray(q0, float)
    T0 = pose6_to_T(fk6(q0))
    Tt = T0.copy()
    Tt[:3, 3] = T0[:3, 3] + np.asarray(dp, float)
    Tt[:3, :3] = rotvec_to_R(drot) @ T0[:3, :3]
    q = q0.copy()
    hist = []
    damp = damp0
    prev = None
    it = -1
    for it in range(iters):
        Tc = pose6_to_T(fk6(q))
        e = np.zeros(6)
        e[:3] = Tt[:3, 3] - Tc[:3, 3]
        e[3:] = rotvec_deg(Tt[:3, :3] @ Tc[:3, :3].T)
        hist.append((float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))))
        if hist[-1][0] <= tol_pos and hist[-1][1] <= tol_ang:
            break
        norm = hist[-1][0] + hist[-1][1]
        if prev is not None:                      # 自适应阻尼
            if norm > prev * 0.9:
                damp = min(damp * 4.0, damp_max)
            elif norm < prev * 0.5:
                damp = max(damp * 0.5, 1e-4)
        prev = norm
        J = num_jac6(fk6, q)
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
        m = float(np.max(np.abs(dq)))
        if m > max_step:
            dq *= max_step / m
        q = q + dq
    Tc = pose6_to_T(fk6(q))
    return dict(q=q,
                err_pos=float(np.linalg.norm(Tt[:3, 3] - Tc[:3, 3])),
                err_ang=float(np.linalg.norm(rotvec_deg(Tt[:3, :3] @ Tc[:3, :3].T))),
                iters=it + 1, hist=hist)


def solve_delta(fk6, q0, dp=(0.0, 0.0, 0.0), drot=(0.0, 0.0, 0.0), steps=1,
                iters=15, tol_pos=0.05, tol_ang=0.05, max_step=3.0,
                damp0=0.01, damp_max=10.0):
    """局部小位移解：求 q ≈ q0，使 FK(q) 的位置 = FK(q0)+dp、姿态 = 基座系转过 drot。

    做法：牛顿迭代 + 阻尼最小二乘（J^T(JJ^T+λI)^{-1}），每步夹在 max_step 度内。
    阻尼 λ 自适应（LM 风格）：残差掉得快就放松（快收敛），变差就收紧（近奇异时保命）。
    真机标定是这个量级定下的——固定 λ=0.3 时是慢吞吞的线性收敛（8 轮才到 0.08mm），
    λ=0.01 时 5 轮就 0.001mm；纯粹为了近奇异位形才需要往回收。

    steps>1 时把 dp/drot 均分成 steps 段【逐段解】（每段拿上一段结果当起点）。
    为什么：单次解大位移会掉进局部最优（假臂实测 −60mm 单次解停在 40mm 残差，
    +30mm 停在 2.4mm；拆成 6~12 段后全部收敛到微米级）。本项目将来"定尺度对"
    的 60mm 笔画就走 steps≈12。**最终残差统一对【完整目标】核验**，分段的中间结果不算数。

    ⚠ 方向性（假臂两处位形全扫描实测，真机同理）：朝【更深进入工作空间】的方向
      （−X/−Y/+Z 一类）60mm 全部收敛到 0.004mm；朝【往外伸】的方向（+X/+Y/−Z）
      10~40mm 就 FAIL（半路贴上奇异流形，σ_min 崩到 1e-3，求解器如实停下）。
      所以笔画方向不能拍脑袋：**先解、看残差、残差不过不许动**。

    返回 dict：
        q          求到的关节
        err_pos    对完整目标核验的位置残差 mm
        err_ang    对完整目标核验的姿态残差 deg
        dq         q − q0
        iters      总迭代数
        worst_step 各分段内部的最大 (位置残差+姿态残差)（趋势监控用）
        Tt         完整目标齐次矩阵
    ⚠ 调用方必须自己核验 err_pos/err_ang 与 |dq| 的上限，核验不过就别发这条运动。
    """
    q0 = np.asarray(q0, float)
    dp = np.asarray(dp, float)
    drot = np.asarray(drot, float)
    q = q0.copy()
    worst = 0.0
    tot = 0
    for i in range(max(1, int(steps))):
        r = _solve_once(fk6, q, dp / steps, drot / steps,
                        iters, tol_pos, tol_ang, max_step, damp0, damp_max)
        q = r["q"]
        worst = max(worst, r["err_pos"] + r["err_ang"])
        tot += r["iters"]
    T0 = pose6_to_T(fk6(q0))
    Tt = T0.copy()
    Tt[:3, 3] = T0[:3, 3] + dp
    Tt[:3, :3] = rotvec_to_R(drot) @ T0[:3, :3]
    Tc = pose6_to_T(fk6(q))
    return dict(q=q,
                err_pos=float(np.linalg.norm(Tt[:3, 3] - Tc[:3, 3])),
                err_ang=float(np.linalg.norm(rotvec_deg(Tt[:3, :3] @ Tc[:3, :3].T))),
                dq=q - q0, iters=tot, worst_step=worst, Tt=Tt)


# ---------------- 离线自测 ----------------

def _selftest():
    rng = np.random.default_rng(7)
    # 1) 约定往返：随机姿态 → T → pose6 → T
    worst = 0.0
    for _ in range(200):
        p = [rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(0, 800),
             rng.uniform(-180, 180), rng.uniform(-89, 89), rng.uniform(-180, 180)]
        back = T_to_pose6(pose6_to_T(p))
        d = np.linalg.norm(np.asarray(back[:3]) - np.asarray(p[:3]))
        dR = np.linalg.norm(rotvec_deg(pose6_to_T(back)[:3, :3] @ pose6_to_T(p)[:3, :3].T))
        worst = max(worst, d, dR)
    print(f"  位姿往返（含 ±89° 全域）：最大误差 {worst:.2e}（位置 mm / 旋转 deg 同量级）")

    # 2) 合成 6R（标准 DH 链，α 交替 ±90°，离奇异性远的起始位形），解小位移
    #    注：别拿低自由度玩具 FK 当测试台——"5mm 且姿态不动"在自由度不够的模型上
    #    数学上无解，求解器只能给最小二乘折中，看着像"没收敛"，其实是对的（真踩过）。
    DH = [  # (a mm, alpha deg, d mm)
        (30.0, 90.0, 152.0), (400.0, 0.0, 0.0), (30.0, 90.0, 0.0),
        (0.0, -90.0, 380.0), (0.0, 90.0, 0.0), (0.0, 0.0, 90.0),
    ]

    def fake_fk(q):
        T = np.eye(4)
        for qi, (a, al, d) in zip(np.asarray(q, float), DH):
            Rz, Tz, Tx, Rx = (np.eye(4) for _ in range(4))
            Rz[:3, :3] = rot_z(qi)
            Tz[2, 3] = d
            Tx[0, 3] = a
            Rx[:3, :3] = rot_x(al)
            T = T @ Rz @ Tz @ Tx @ Rx
        return T_to_pose6(T)

    q0 = [45.0, -55.0, 70.0, -25.0, 35.0, 40.0]      # 良态起始位形（σ_min≈0.35，cond≈49）
    cases = (([0, 0, 5.0], 1, "+5mm Z"), ([-20.0, 0, 0], 1, "−20mm X"),
             ([0, 0, 60.0], 12, "+60mm Z"), ([-60.0, 0, 0], 12, "−60mm X"),
             ([0, -60.0, 0], 12, "−60mm Y"))
    for dp, steps, name in cases:
        r = solve_delta(fake_fk, q0, dp=dp, steps=steps)
        print(f"  合成 6R 解 {name:>9}（{steps} 段）：位置残差 {r['err_pos']:.5f}mm、姿态残差 "
              f"{r['err_ang']:.5f}°、{r['iters']} 轮、max|Δq|={np.max(np.abs(r['dq'])):.2f}°")
        assert r["err_pos"] < 0.1 and r["err_ang"] < 0.1, f"solve_delta 自测没过: {name}"

    # 3) 已知【不可行】方向：+30mm Y 且姿态严格冻结——本玩具位形上做不到。
    #    独立复核：scipy.optimize.least_squares（LM，30 次随机重启）最好解 2.384mm/1.330°，
    #    本函数 2.373mm/1.351°（略优且不同解落在同一条 Pareto 前沿）→ 是问题无解，不是 bug。
    #    这条测试的用处：提醒【残差核验必须发生在发运动指令之前】——解不出来就不许动。
    r = solve_delta(fake_fk, q0, dp=[0, 30.0, 0], steps=6)
    print(f"  已知不可行例（+30mm Y）：残差 {r['err_pos']:.3f}mm/{r['err_ang']:.3f}°"
          f"（scipy 独立复核同量级）→ 这种解【不许发指令】，由调用方的残差闸拦住")
    assert r["err_pos"] > 0.5, "不可行例应当保持大残差"
    print("  自测通过（纯离线，未连机器人）")


if __name__ == "__main__":
    _selftest()
