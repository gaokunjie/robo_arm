# -*- coding: utf-8 -*-
"""只读采集真机 FK 样本（本地运动学模型的地基）。

⚠ **只读**：全程只调 `GetForwardKin` / `GetRobotState`，不发任何运动指令。
⚠ **节流**：`fr3_math.FK_MIN_THROTTLE_S = 0.05`（约 20 次/秒）是第 5 次控制器崩溃后
   立的硬上限，**不许在本脚本里绕过它**。默认 60 个样本 ≈ 3 秒。

为什么要重新采：上一版拟合数据（`fit_urdf_kin.py`）里 18/20 个样本在同一条 J2 扫描弧上，
约束退化 ⇒ 拟合 RMS 2.9mm 好看，但两个独立位形的残差是 20.5mm / 4.65mm。
**自洽 ≠ 准**：跨位形随机采样才有定参数的约束力。

用法：
    python3 -I src/collect_fk_samples.py                # 默认 60 个
    python3 -I src/collect_fk_samples.py --n 120
    python3 -I src/collect_fk_samples.py --check-only   # 只验连通性（2 发）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import fr3_math                                            # noqa: E402
from fr3_math import pose6_to_T                            # noqa: E402
from fr_rpc import FRRpc                                    # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "fk_samples.json"

# 采样盒（度）——刻意覆盖工作区附近的大范围，而不是只在一条弧上。
# 依据：URDF 关节限位 j2/j4 [-265,85]、j3 ±162、j5/j6 ±175；FR3 手册工作区约 ±170。
JOINT_BOX = np.array([
    [-170.0, 170.0],     # j1 底座回转
    [-120.0, 120.0],     # j2 大臂
    [-150.0, 150.0],     # j3 小臂
    [-170.0, 170.0],     # j4 腕滚转
    [-120.0, 120.0],     # j5 腕俯仰
    [-170.0, 170.0],     # j6 法兰滚转
])


def sample_box(rng, n):
    """Latin-hypercube 采样：每个维分层，比纯随机更均匀（更好的约束）。"""
    u = (np.arange(n)[:, None] + rng.random((n, 6))) / n
    for i in range(6):
        rng.shuffle(u[:, i])
    lo, hi = JOINT_BOX[:, 0], JOINT_BOX[:, 1]
    return lo + u * (hi - lo)


# 抓取工作区（基座系）：法兰 z 在 400~620、水平距 350~600 —— 见 config 注释与
# fr3_local.planar_reach_limit 的闭式。训练样本要覆盖这里，否则拟合在目标位形外推。
WORK_POS = dict(x=(-650.0, 650.0), y=(-650.0, 650.0), z=(150.0, 750.0))
WORK_R_MIN, WORK_R_MAX = 300.0, 660.0


def usable(pose):
    """这个 FK 结果是不是落在可信工作区里。不是 ⇒ 返回 False（丢弃该样本）。

    为什么必须筛：GetForwardKin 对不可达/奇异位形可能返回垃圾数，
    把它当训练数据会把模型往错的方向拽（而拟合 RMS 照样好看 —— 这类错误 RMS 抓不到）。
    """
    x, y, z = pose[:3]
    if not all(np.isfinite(v) for v in pose[:3]):
        return False
    r = float(np.hypot(x, y))
    return (WORK_POS["x"][0] <= x <= WORK_POS["x"][1]
            and WORK_POS["y"][0] <= y <= WORK_POS["y"][1]
            and WORK_POS["z"][0] <= z <= WORK_POS["z"][1]
            and WORK_R_MIN <= r <= WORK_R_MAX)


def tool_down(pose):
    """法兰 z 轴是否竖直向下（T[2,2]≈−1）——套取位/抬升位真正用的位形。"""
    return float(pose6_to_T(pose)[2, 2]) < -0.985


def sample_tool_down(rng, n):
    """定向生成【工具竖直向下】的位形。

    为什么要定向：随机采样 375 个里朝下的只有 **0** 个（工具朝下是一个 5 维流形上的
    一个点集，随机撞中的概率极低）—— 而套取位和抬升位**全部**在朝下流形上。
    只用随机样本拟合，等于在目标位形上外推。

    构造（依据 URDF，见 fr3_local.py 推导）：
      工具 z 轴 = Rot(q2+q3+q4+q5+q6)·(0,0,1)；要它指向基座 −z ⇒ 累积角 = 180°（mod 360°）。
      ⇒ 自由挑 q1,q2,q3,q4,q5，再令 q6 = 180 − (q2+q3+q4+q5)。
    q6 超出 ±175° 的直接丢弃（那不是合法位形）。
    """
    lo, hi = JOINT_BOX[:, 0], JOINT_BOX[:, 1]
    out = []
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        q = lo + rng.random(6) * (hi - lo)
        q6 = 180.0 - (q[1] + q[2] + q[3] + q[4])
        q6 = (q6 + 180.0) % 360.0 - 180.0          # 折回 (−180,180]
        if abs(q6) > 175.0:
            continue
        out.append(np.array([q[0], q[1], q[2], q[3], q[4], q6]))
    return np.asarray(out, float) if out else np.zeros((0, 6))


def main():
    ap = argparse.ArgumentParser(description="只读采集真机 FK 样本")
    ap.add_argument("--n", type=int, default=150, help="目标可用样本数")
    ap.add_argument("--pool", type=int, default=0, help="候选池大小（默认 n 的 2.5 倍）")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check-only", action="store_true", help="只验连通性（2 发 FK）")
    args = ap.parse_args()

    print(f"连接 {config.ROBOT_IP}（只读模式）…")
    rpc = FRRpc(ip=config.ROBOT_IP)
    joints = rpc.joint_deg()
    tcp = rpc.tcp_pose()
    print(f"  当前关节 {np.round(joints, 2).tolist()}")
    print(f"  当前 TCP  {np.round(tcp, 2).tolist()}")

    rng = np.random.default_rng(args.seed)
    if args.check_only:
        Q = sample_box(rng, 2)
    else:
        n_pool = args.pool or int(args.n * 2.5)
        Q_free = sample_box(rng, int(n_pool * 0.4))            # 自由位形：定标全局
        Q_down = sample_tool_down(rng, n_pool - len(Q_free))   # 朝下位形：定标工作区
        Q = np.vstack([Q_free, Q_down])
        order = rng.permutation(len(Q))
        Q = Q[order]
        print(f"  池：自由 {len(Q_free)} + 朝下 {len(Q_down)}（构造命中 {len(Q_down)}/{n_pool - len(Q_free)}）")

    print(f"\n采 {len(Q)} 个候选（节流 {fr3_math.FK_MIN_THROTTLE_S}s，约 "
          f"{len(Q) * fr3_math.FK_MIN_THROTTLE_S:.0f}s）…")
    # ⚠⚠ 必须走 fk_throttled —— 直接裸调 rpc.fk() 就是第 5 次崩溃的成因
    #    （几分钟几十万发 GetForwardKin → 应用层死机）。节流是硬下限，抬不高也绕不过。
    fk = fr3_math.fk_throttled(rpc.fk)

    rows, n_bad, n_down = [], 0, 0
    t0 = time.time()
    for i, q in enumerate(Q):
        try:
            pose = np.asarray(fk(q), float)
        except Exception as exc:                                # noqa: BLE001
            n_bad += 1
            print(f"  ✗ 第 {i} 个样本失败：{exc}")
            continue
        if not usable(pose):
            n_bad += 1
            continue
        down = tool_down(pose)
        n_down += int(down)
        rows.append((q, pose[:3], pose[3:], down))
        if (i + 1) % 25 == 0 or i == len(Q) - 1:
            print(f"  {i + 1}/{len(Q)}  可用 {len(rows)}（朝下 {n_down}）  {time.time() - t0:.1f}s")

    P = np.asarray([r[1] for r in rows], float)
    print(f"\n可用 {len(rows)}/{len(Q)}（丢弃 {n_bad}，其中朝下 {n_down}）用时 {time.time() - t0:.1f}s")
    print(f"  位置范围 x[{P[:,0].min():.0f},{P[:,0].max():.0f}] "
          f"y[{P[:,1].min():.0f},{P[:,1].max():.0f}] z[{P[:,2].min():.0f},{P[:,2].max():.0f}]")
    R = np.hypot(P[:, 0], P[:, 1])
    print(f"  水平距 r[{R.min():.0f},{R.max():.0f}]  —— 套取位 r≈425、抬升位法兰 z≈549 在此范围内")

    payload = dict(
        note="只读 GetForwardKin 采样，无运动指令。喂 src/fit_local_kin.py。"
             "含 flange_rpy_deg 与 tool_down 标记（朝下位形=套取/抬升真正用的位形）。",
        ip=config.ROBOT_IP, seed=args.seed, pool=len(Q), n=len(rows),
        n_tool_down=n_down, throttle_s=fr3_math.FK_MIN_THROTTLE_S,
        joints_deg=[r[0].tolist() for r in rows],
        flange_xyz_mm=P.tolist(),
        flange_rpy_deg=[r[2].tolist() for r in rows],
        tool_down=[bool(r[3]) for r in rows],
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    print(f"→ {args.out}")
    if len(rows) < 40:
        print("⚠ 可用样本 <40，拟合会欠定（9 个参数至少要 12 且位形多样）——加大 --pool 重采")


if __name__ == "__main__":
    main()
