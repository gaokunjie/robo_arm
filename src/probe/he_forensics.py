# -*- coding: utf-8 -*-
"""D 批采集数据的事后取证（纯离线：只读 jpg + flange_poses.csv + collect_log.csv，不连机器人/相机）。

--handeye-solve 首跑：板面误差/轴多样性/留一都健康，但 ①轴对应残差中位 1.36°（仿真对照
0.12°）、②分区解【旋转】差 3.6~4.2°（判据 0.5°）、③k=1.0755（板尺寸自检）。①②③ 是三种病：
    A 几张坏帧（局部）        → 删了重解就行
    B 时间相关漂移（蠕变）    → 板/相机在采集期间动过 → 分段或重采
    C 视角相关系统误差（板弯/畸变/内参）→ 换平整板/查内参
本工具用三路证据把它们分开：
  ③ 轴对应残差用【同一个 R】拟一遍，再把每对残差按
     · 帧间隔（漂移 ⇒ 隔得越远越差）· 时间间隔（分钟）· 4×4 时间分块矩阵；
     另列"谁在牵连"（逐帧相关对的中位残差）和最差的 10 对
  ④ 时间切分解（前/后半、三等分）：漂移 ⇒ 前后两半解出的 X 差得远
  ⑤ 位置切分解（左右/上下/远近，与 solve 同款）：视角系统误差在这里露头
另附 ①逐帧板面误差+留一表（点名坏帧）、②轴残差全量、⑥候选剔除集诊断（定位凶手，不是凑判据）。
还有两条量尺（⑨/⑩ 都量板侧尺度因子 = 板尺寸 s_a/s_true 或机器人位移尺度）：
  ⑨ 直尺式：板链位移 ÷ 法兰位移——板侧尺度错 ⇒ 读出该因子；焦距错 ⇒ 读 ≈1.00
  ⑩ 画面平移 ÷ 模型预测平移——与 ⑨ 独立的二次核验，同签名
     ⑩a 合成真值表（S0 全对 / S1 板小 7.5% / S2 焦距短 7.5% / S2b 焦距短 2% × 5 视角）已钉死方向：
       板小 ⇒ ⑨≈⑩≈板因子、R 误差 0.00°；焦距错 ⇒ ⑨≈⑩≈1.00 而 R 误差 0.60°、|t| 也膨胀
     ⚠ 2026-10-16 旧版曾把 ⑨ 写成「板因子×焦距因子」——被自检证伪（焦距错不在这两个量里出现）。

用法：
    python3 -I src/probe/he_forensics.py
    python3 -I src/probe/he_forensics.py --dir data/handeye_shots
"""

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calibrate as C                                          # noqa: E402

METHOD = cv2.CALIB_HAND_EYE_PARK
DROPSETS = [(), ("007",), ("008",), ("022",), ("026",),
            ("007", "008"), ("022", "026"), ("007", "008", "022", "026")]


# ---------------------------------------------------------------- 工具
def solve_one(idx, Rg, tg, Rc, tc):
    """Park 解（与 solve 同款调用），失败返回 None。"""
    try:
        return C._he_solve_once([Rg[i] for i in idx], [tg[i] for i in idx],
                                [Rc[i] for i in idx], [tc[i] for i in idx], METHOD)
    except Exception as e:                                     # noqa: BLE001
        print(f"    （子集解失败：{e}）")
        return None


def diffX(Ta, Tb):
    dt = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
    dr = float(C._rot_axis_angle(Ta[:3, :3].T @ Tb[:3, :3])[0])
    return dt, dr


def pair_axes(poses, R_t2c):
    """与 _he_axis_match 完全同款地建对：(i, j, a 基座系轴, b 相机→板系轴)，只取 θ>15°。"""
    Ts = [C._pose6_to_T(p) for p in poses]
    R_bc = [R.T for R in R_t2c]                                # 相机→板（方向陷阱见 docstring）
    out = []
    for i in range(len(Ts)):
        for j in range(i + 1, len(Ts)):
            Aij = Ts[j][:3, :3] @ Ts[i][:3, :3].T
            ang, ax_a = C._rot_axis_angle(Aij)
            if ang < 15.0:
                continue
            _ang2, ax_b = C._rot_axis_angle(R_bc[j] @ R_bc[i].T)
            out.append((i, j, ax_a, ax_b))
    return out


def fit_R(pairs):
    """把 b 轴对到 a 轴的最佳旋转 R（逐行复刻 _he_axis_match 的迭代+SVD，同种子）。"""
    a = np.array([p[2] for p in pairs])
    b = np.array([p[3] for p in pairs])
    rng = np.random.default_rng(0)
    starts = [np.eye(3)] + [np.array(cv2.Rodrigues(rng.normal(size=3))[0]) for _ in range(48)]
    best = (None, -2.0)
    for R0 in starts:
        R = np.array(R0)
        for _ in range(60):
            dots = np.einsum("ij,ij->i", a, b @ R.T)
            s = np.sign(dots); s[s == 0.0] = 1.0
            H = (b * s[:, None]).T @ a
            U, _S, Vt = np.linalg.svd(H)
            Rn = Vt.T @ U.T
            if np.linalg.det(Rn) < 0:
                Vt = Vt.copy(); Vt[-1] *= -1.0
                Rn = Vt.T @ U.T
            if np.abs(Rn - R).max() < 1e-13:
                R = Rn
                break
            R = Rn
        o = float(np.mean(np.abs(np.einsum("ij,ij->i", a, b @ R.T))))
        if o > best[1]:
            best = (R, o)
    return best[0]


def pair_errs(pairs, R):
    a = np.array([p[2] for p in pairs])
    b = np.array([p[3] for p in pairs])
    return np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", a, b @ R.T)), -1.0, 1.0)))


def shift_test_pair(cA, idA, cB, idB, Ra, ta, Rb, tb, TfA, TfB, K, D, objp):
    """⑩ 单对：|实测画面平移| ÷ |模型预测平移|。

    实测 = 两帧共有角点的位移均值（向量）——旋转残差绕质心转、均值里一阶相消；
    预测 = 模型世界里把板 3D 点（用假设边长 objp、PnP 位姿）按
           「长度=法兰位移、方向=PnP 位移方向」刚移一下，再投影的位移均值。
    姿态冻结 + 同一刚体 ⇒ 真值 |实测|/|预测| = 板侧尺度因子（s_a/s_true，或机器人位移尺度）；
    焦距/内参错在这个比值里≈相消（⑩a 真值表：焦距错 7.5% 只读 0.9982，板小 7.5% 读 1.0760）。
    返回 (比值, 方向差°, 实测px, 预测px, Δ法兰mm, 共用角点数) 或 None。
    """
    mB = {int(v): m for m, v in enumerate(idB)}
    idx = [(m, mB[int(v)]) for m, v in enumerate(idA) if int(v) in mB]
    if len(idx) < 12:
        return None
    ma = np.array([m for m, _ in idx])
    mb = np.array([p for _, p in idx])
    ids = np.array([int(idA[m]) for m in ma])
    meas = (cB[mb] - cA[ma]).mean(axis=0)
    if C._rot_axis_angle(TfA[:3, :3].T @ TfB[:3, :3])[0] > 1.2:      # 冻结姿态守卫
        return None
    df = float(np.linalg.norm(TfB[:3, 3] - TfA[:3, 3]))
    if df < 8.0:
        return None
    d = np.asarray(tb).ravel() - np.asarray(ta).ravel()
    nd = float(np.linalg.norm(d))
    if nd < 3.0:
        return None
    uh = d / nd
    P1 = np.asarray(ta).ravel()[None, :] + objp[ids] @ Ra.T
    z3 = np.zeros(3)
    p1, _ = cv2.projectPoints(P1, z3, z3, K, D)
    p2, _ = cv2.projectPoints(P1 + uh[None, :] * df, z3, z3, K, D)
    pred = (p2 - p1).reshape(-1, 2).mean(axis=0)
    npred = float(np.linalg.norm(pred))
    if npred < 4.0:
        return None
    nmeas = float(np.linalg.norm(meas))
    ang = float(np.degrees(np.arctan2(pred[0] * meas[1] - pred[1] * meas[0],
                                      float(np.dot(pred, meas)))))
    return nmeas / npred, ang, nmeas, npred, df, len(idx)


def sim_selfcheck():
    """⑩a 合成世界自检/真值表：四种「谁错了」× 5 个视角，看 ⑨式/⑩/平面残差/R 误差。

    干净针孔世界（f_a=1000、20 格 25mm 板、板距 ~400mm），两帧只差一次纯平移
    （相机系 [58,8,3]mm）。故事：
      S0 全对                    → ⑨=⑩≈1.000，残差 0，R 误差 0（钉「干净世界读干净数」）
      S1 板真小 1/1.0755（f 对）  → 板尺寸错的签名
      S2 焦距真短 1/1.0755（板对）→ 焦距错的签名（残差/R 会不会露马脚？）
      S2b 焦距短 1.02            → 小焦距错的签名
    ⚠ 2026-10-16 首版曾断言「⑨ 在焦距故事也该读 1.0755」——被本自检当场证伪，故扩成真值表。
    """
    K_a = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    D0 = np.zeros(5)
    gx, gy = np.meshgrid(np.arange(1, 7) * 25.0, np.arange(1, 5) * 25.0)
    Xk_a = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)   # 假设几何（25mm）
    poses = [(cv2.Rodrigues(np.array([0.06, 0.10, 0.02]))[0], np.array([-20.0, 15.0, 400.0])),
             (cv2.Rodrigues(np.array([0.13, 0.22, 0.05]))[0], np.array([45.0, -25.0, 420.0])),
             (cv2.Rodrigues(np.array([0.24, -0.15, 0.10]))[0], np.array([-60.0, 35.0, 390.0])),
             (cv2.Rodrigues(np.array([0.18, 0.05, -0.12]))[0], np.array([10.0, 60.0, 450.0])),
             (cv2.Rodrigues(np.array([0.30, 0.30, -0.04]))[0], np.array([55.0, -55.0, 430.0]))]
    dcam = np.array([58.0, 8.0, 3.0])
    dframe = float(np.linalg.norm(dcam))
    TfA = np.eye(4)
    TfB = np.eye(4)
    TfB[:3, 3] = [dframe, 0.0, 0.0]                              # 假法兰：冻结姿态、位移=真实
    idd = np.arange(len(Xk_a))
    ok_all = True
    for name, s_board, s_f in (("S0 全对              ", 1.0, 1.0),
                               ("S1 板真小 1/1.0755   ", 1.0 / 1.0755, 1.0),
                               ("S2 焦距真短 1/1.0755 ", 1.0, 1.0 / 1.0755),
                               ("S2b 焦距短 1/1.02    ", 1.0, 1.0 / 1.02)):
        Xk_true = Xk_a * s_board
        Kt = K_a.copy()
        Kt[0, 0] = Kt[1, 1] = 1000.0 * s_f
        res_l, tr_l, rr_l, r9_l, r10_l = [], [], [], [], []
        for R_true, t_true1 in poses:
            P1 = t_true1 + Xk_true @ R_true.T
            px1, _ = cv2.projectPoints(P1, np.zeros(3), np.zeros(3), Kt, D0)
            px2, _ = cv2.projectPoints(P1 - dcam, np.zeros(3), np.zeros(3), Kt, D0)
            px1 = px1.reshape(-1, 2).astype(np.float64)
            px2 = px2.reshape(-1, 2).astype(np.float64)
            _ok1, rv1, tv1 = cv2.solvePnP(Xk_a, px1, K_a, D0, flags=cv2.SOLVEPNP_ITERATIVE)
            _ok2, rv2, tv2 = cv2.solvePnP(Xk_a, px2, K_a, D0, flags=cv2.SOLVEPNP_ITERATIVE)
            R1, R2 = cv2.Rodrigues(rv1)[0], cv2.Rodrigues(rv2)[0]
            PJ, _ = cv2.projectPoints(Xk_a, rv1, tv1, K_a, D0)
            res_l.append(float(np.linalg.norm(px1 - PJ.reshape(-1, 2), axis=1).mean()))
            tr_l.append(float(np.linalg.norm(tv1.ravel()) / np.linalg.norm(t_true1)))
            rr_l.append(float(C._rot_axis_angle(R1.T @ R_true)[0]))
            r9_l.append(float(np.linalg.norm(tv2.ravel() - tv1.ravel()) / dframe))
            out = shift_test_pair(px1, idd, px2, idd, R1, tv1.ravel(), R2, tv2.ravel(),
                                  TfA, TfB, K_a, D0, Xk_a)
            r10_l.append(out[0] if out else float("nan"))
        m = lambda a: float(np.median(a))                           # noqa: E731
        print(f"    {name} 平面残差 {m(res_l):5.2f}px  |t|/真距 {m(tr_l):.3f}  "
              f"R误差 {m(rr_l):5.2f}°  ⑨式 {m(r9_l):.4f}  ⑩ {m(r10_l):.4f}")
        if s_board == 1.0 and s_f == 1.0:
            ok_all = ok_all and abs(m(r9_l) - 1.0) < 0.005 and abs(m(r10_l) - 1.0) < 0.005 \
                and m(res_l) < 0.01 and m(rr_l) < 0.01
        if s_board != 1.0:
            ok_all = ok_all and abs(m(r10_l) - 1.0 / s_board) < 0.01
    print("    （S0 必须全零全一；S1 必须读出板因子——这两条是方向护栏。"
          "S2/S2b 读数本身就是「焦距错长什么样」的答案）")
    return ok_all


def tsec(hms):
    h, m, s = (int(v) for v in hms.split(":"))
    return h * 3600 + m * 60 + s


def load_times(shots, names):
    """collect_log.csv 的非 CHECK accepted 行（按序）↔ 帧 001..N；000 用运行起始时刻。

    返回 (时刻, 标签, 板距 mm, 倾角°)，元素数 = len(names)；对不上整体返回 None。
    """
    p = shots / "collect_log.csv"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) > 23 and parts[23] == "1" and not parts[1].startswith("CHECK"):
            rows.append((parts[0].strip(), parts[1].strip(), parts[21], parts[22],
                         parts[6], parts[7], parts[8]))            # 距/倾/q4/q5/q6
    run_start = names[0][len("he_"):-len("_000.jpg")]           # he_YYYYMMDD_HHMMSS_NNN
    hms = f"{run_start[9:11]}:{run_start[11:13]}:{run_start[13:15]}"
    if len(rows) == len(names) - 1:
        rows = [(hms, "(基线帧)", "", "", "", "", "")] + rows
    elif len(rows) != len(names):
        return None
    return ([r[0] for r in rows], [r[1] for r in rows],
            [float(r[2]) if r[2] else None for r in rows],
            [float(r[3]) if r[3] else None for r in rows],
            [(float(r[4]), float(r[5]), float(r[6])) if r[4] else None for r in rows])


def load_checks(shots):
    """run_*.txt 里的核验行 + collect_log.csv 的 CHECK 行时刻，按序配对。"""
    rp = sorted(shots.glob("run_*.txt"))
    cp = shots / "collect_log.csv"
    if not rp or not cp.exists():
        return []
    vals = []
    for line in rp[-1].read_text(encoding="utf-8").splitlines():
        if "板位姿漂移" in line:
            seg = line.split("板位姿漂移")[1].split("mm")[0].strip()
            deg = line.split("mm/")[1].split("°")[0].strip()
            vals.append((float(seg), float(deg)))
    times = []
    for line in cp.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) > 24 and parts[1].startswith("CHECK"):
            times.append((parts[0].strip(), parts[1].strip()))
    out = []
    for k in range(min(len(vals), len(times))):
        out.append((times[k][0], times[k][1], vals[k][0], vals[k][1]))
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="D 批采集数据事后取证（纯离线）")
    ap.add_argument("--dir", default=str(C.HE_SHOTS_DIR))
    ap.add_argument("--k", type=float, default=1.0,
                    help="把板距按真尺度换算 t/k（k=假设边长/实际边长）再算平移类分析；"
                         "旋转类判据不受影响——用来分离 k 的伪影")
    args = ap.parse_args()
    shots = Path(args.dir)

    rows = C._he_read_csv(shots / C.HE_POSES_CSV)
    if len(rows) < 6:
        sys.exit(f"{shots} 里 csv 帧太少")
    K, D = C._try_load_K(C.OUT_YAML)
    if K is None:
        sys.exit("读不到内参")
    det = C.BoardDetector()

    names, poses, R_t2c, t_t2c, per_px, cxy = [], [], [], [], [], []
    corners_all, ids_all = [], []                              # ⑩ 用：逐帧角点 + id
    for name, pose in rows:
        img = cv2.imread(str(shots / name))
        if img is None:
            print(f"  ⚠ {name} 读不出（跳）")
            continue
        corners, ids = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if corners is None or len(corners) < 12:
            print(f"  ⚠ {name} 这次认不出（跳）")
            continue
        R, t = C._solve_board(det, corners, ids, K, D)
        if R is None:
            continue
        obj_i = det.objp[ids.ravel()].astype(np.float64)
        proj, _ = cv2.projectPoints(obj_i, cv2.Rodrigues(R)[0], t.reshape(3, 1), K, D)
        per_px.append(float(cv2.norm(corners.astype(np.float64),
                                     proj.astype(np.float64), cv2.NORM_L2) / math.sqrt(len(proj))))
        cp = corners.reshape(-1, 2).mean(axis=0)
        cxy.append([float(cp[0]), float(cp[1]), float(np.linalg.norm(t))])
        corners_all.append(corners.reshape(-1, 2).astype(np.float64))
        ids_all.append(ids.ravel().astype(int))
        names.append(name)
        poses.append(pose)
        R_t2c.append(R)
        t_t2c.append(t.ravel() if hasattr(t, "ravel") else t)
    n = len(names)
    print(f"有效帧 {n} 张（时序即采集顺序）\n")
    t_raw = [np.asarray(t).ravel() for t in t_t2c]             # ⑨ 直尺 k 用原始值

    tinfo = load_times(shots, names)
    if tinfo:
        times, tags, dists, tilts, wrists = tinfo
        tsecs = [tsec(t) for t in times]
    else:
        print("⚠ 读不到 collect_log.csv 时间列——时间分析退回帧序")
        times = [f"#{i:02d}" for i in range(n)]
        tags = [""] * n
        dists = [None] * n
        tilts = [None] * n
        wrists = [None] * n
        tsecs = [float(i) * 24.0 for i in range(n)]            # 假时间轴：1 帧 ≈ 24s
    if args.k != 1.0:
        print(f"（--k {args.k:g}：板距已按真尺度换算 t/k；只影响平移类分析，旋转类不变）")
        t_t2c = [np.asarray(t).ravel() / args.k for t in t_t2c]

    # 腕部偏离（相对冻结瞄姿的 J4/J5/J6 最大差）——腕动越大，相机支架/线缆受扰越大
    aim_wrist = None
    _rp = sorted(shots.glob("run_*.txt"))
    if _rp:
        for line in _rp[-1].read_text(encoding="utf-8").splitlines():
            if "当前关节" in line:
                try:
                    aim_wrist = [float(v) for v in
                                 line.split("[")[1].split("]")[0].split(",")][3:6]
                except (IndexError, ValueError):
                    aim_wrist = None
                break
    wr = [None] * n
    if aim_wrist is not None:
        for i in range(n):
            if wrists[i] is not None:
                wr[i] = max(abs(wrists[i][k] - aim_wrist[k]) for k in range(3))

    checks = load_checks(shots)
    if checks:
        print("检查点漂移读数（回瞄姿时 vs 基线，采集器自己量的；板「没动」= <2.0mm）：")
        for t, tag, mm, deg in checks:
            print(f"    {t}  {tag:<14} {mm:5.2f}mm / {deg:.2f}°")
        print()

    # R 与 t 的两种形状（与 calibrate 的调法一致：Rg 3×3、tg (3,1)）
    Rg = [C._pose6_to_T(p)[:3, :3].copy() for p in poses]
    tg = [C._pose6_to_T(p)[:3, 3].reshape(3, 1).copy() for p in poses]

    # 轴对应残差 + 逐帧牵连（先算出来，① 的表里一起用；口径与 solve 完全一致）
    pairs = pair_axes(poses, R_t2c)
    R_axis = fit_R(pairs)
    err = pair_errs(pairs, R_axis)
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    blame = np.full(n, np.nan)
    for f in range(n):
        m = (ii == f) | (jj == f)
        if m.sum() >= 5:
            blame[f] = float(np.median(err[m]))

    # ---------- ① 逐帧板面误差 + 留一（按时序；留一与 solve 同款公式）----------
    loo = []
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        Tk = solve_one(idx, Rg, tg, R_t2c, t_t2c)
        if Tk is None:
            loo.append(float("nan"))
            continue
        pk = np.array([(C._pose6_to_T(poses[i]) @ Tk @ np.block(
            [[R_t2c[i], np.asarray(t_t2c[i]).reshape(3, 1)], [np.zeros(3), 1.0]]))[:3, 3]
            for i in idx]).mean(axis=0)
        pred = (np.linalg.inv(Tk) @ (np.linalg.inv(C._pose6_to_T(poses[k])) @ np.append(pk, 1.0)))[:3]
        loo.append(float(np.linalg.norm(pred - np.asarray(t_t2c[k]).ravel())))
    loo = np.array(loo)
    print("① 逐帧（按采集顺序）：板面误差 px · 留一 mm · 板距/倾角 · 腕偏 · 轴牵连（°）")
    for i in range(n):
        flag = "  ⚠" if loo[i] > 3.0 else ""
        dd = f"{dists[i]:4.0f}" if dists[i] is not None else "   —"
        tt = f"{tilts[i]:4.1f}" if tilts[i] is not None else "  —"
        ww = f"{wr[i]:4.1f}" if wr[i] is not None else "   —"
        bb = f"{blame[i]:5.2f}" if np.isfinite(blame[i]) else "    —"
        print(f"    {i:02d} {times[i]} {tags[i]:<10} {per_px[i]:6.3f}px  loo {loo[i]:5.2f}mm"
              f"  {dd}mm {tt}°  腕偏 {ww}°  牵连 {bb}{flag}")
    print(f"    留一中位 {np.nanmedian(loo):.2f}mm  最大 {np.nanmax(loo):.2f}mm")

    # ---------- ② 轴对应残差：全量（复现 solve 的报告值）----------
    print(f"\n② 轴对应残差（θ>15°）：中位 {np.median(err):.2f}°  最大 {err.max():.2f}°  {len(pairs)} 对"
          f"（solve 报的是同一口径）")

    # ---------- ③ 同一 R 下逐对残差，切三刀 ----------
    print("\n③a 按【帧间隔】：漂移 ⇒ 隔得越远越差；板弯/内参等视角误差 ⇒ 各档平。")
    for lo, hi in ((0, 4), (5, 12), (13, 99)):
        m = (np.abs(jj - ii) >= lo) & (np.abs(jj - ii) <= hi)
        if m.sum() < 6:
            print(f"    间隔 {lo:>2}-{hi if hi < 99 else '∞'}: 对太少（{m.sum()}）")
            continue
        print(f"    间隔 {lo:>2}-{hi if hi < 99 else '∞'}：{m.sum():>3} 对，中位 {np.median(err[m]):.2f}°  "
              f"75分位 {np.percentile(err[m], 75):.2f}°  最大 {err[m].max():.2f}°")

    print("\n③b 按【时间间隔】（分钟）——更直接：")
    dts = np.array([abs(tsecs[p[1]] - tsecs[p[0]]) / 60.0 for p in pairs])
    for lo, hi in ((0, 0.5), (0.5, 2.0), (2.0, 1e9)):
        m = (dts >= lo) & (dts < hi)
        hilab = f"{hi:g}min" if hi < 1e8 else "∞min"
        if m.sum() < 6:
            print(f"    {lo:g}-{hilab}: 对太少（{m.sum()}）")
            continue
        print(f"    {lo:>4g}-{hilab:<6}：{m.sum():>3} 对，中位 {np.median(err[m]):.2f}°  "
              f"75分位 {np.percentile(err[m], 75):.2f}°  最大 {err[m].max():.2f}°")

    nb = 4
    edges = [int(round(k * n / nb)) for k in range(nb + 1)]
    blocks = [(edges[b], edges[b + 1]) for b in range(nb)]
    print("\n③c 4×4 时间分块矩阵（每格=两时间块之间所有对的中位残差°，括号=对数）")
    blab = [f"{blocks[b][0]:02d}-{blocks[b][1] - 1:02d}" for b in range(nb)]
    print("         " + "".join(f"{b:>12}" for b in blab))
    for b1 in range(nb):
        row = f"    {blab[b1]:>5}"
        for b2 in range(nb):
            m = 0
            if b2 == b1:
                m = (ii >= blocks[b1][0]) & (jj < blocks[b1][1]) & (ii < blocks[b1][1])
            elif b2 > b1:
                m = (ii >= blocks[b1][0]) & (ii < blocks[b1][1]) & \
                    (jj >= blocks[b2][0]) & (jj < blocks[b2][1])
            cnt = int(np.sum(m)) if not isinstance(m, int) else 0
            if cnt >= 6:
                row += f"{np.median(err[m]):>7.2f}({cnt:>3})"
            else:
                row += f"{'·':>7}({cnt:>3})"
        print(row)
    print("    （看形状：远块比近块全面更大 = 单调漂移；某一行/列整片热 = 那几张坏；"
          "都平 = 不随时间是视角/几何病）")

    print("\n③d 最差的 10 对（点名）：")
    order = np.argsort(-err)
    for k in order[:10]:
        b1 = int(ii[k]); b2 = int(jj[k])
        print(f"    {names[b1][-7:-4]}({times[b1]}) × {names[b2][-7:-4]}({times[b2]})"
              f"  Δ帧 {b2 - b1:>2}  Δt {abs(tsecs[b2] - tsecs[b1]) / 60:.1f}min  {err[k]:.2f}°")

    bl_sorted = sorted([(float(blame[f]), f) for f in range(n) if np.isfinite(blame[f])],
                       reverse=True)
    print("\n③e 谁在牵连（与该帧相关的对的中位残差，前 8）：")
    for med, f in bl_sorted[:8]:
        cnt = int(((ii == f) | (jj == f)).sum())
        print(f"    {names[f][-7:-4]}（{times[f]} {tags[f]}） 中位 {med:.2f}°（{cnt} 对）")

    # ---------- ④ 时间切分解 ----------
    def tsplit(a, b, label):
        Ta = solve_one(a, Rg, tg, R_t2c, t_t2c)
        Tb = solve_one(b, Rg, tg, R_t2c, t_t2c)
        if Ta is None or Tb is None:
            return
        dt, dr = diffX(Ta, Tb)
        span = abs(np.mean([tsecs[k] for k in b]) - np.mean([tsecs[k] for k in a])) / 60.0
        print(f"    {label}：{len(a)} vs {len(b)}（时间中心隔 {span:.1f}min）→ "
              f"Δt {dt:6.2f}mm  Δr {dr:6.3f}°")

    print("\n④ 时间切分解（Park）：漂移 ⇒ 前后两半解出的 X 差得远；板没动 ⇒ 应≈0")
    h = n // 2
    tsplit(list(range(0, h)), list(range(h, n)), f"前半 {h} vs 后半 {n - h}")
    t3 = n // 3
    i1, i2 = list(range(0, t3)), list(range(t3, 2 * t3))
    i3 = list(range(2 * t3, n))
    tsplit(i1, i2, "第1/3 vs 第2/3")
    tsplit(i2, i3, "第2/3 vs 第3/3")
    tsplit(i1, i3, "第1/3 vs 第3/3（最远时间跨度）")

    # ---------- ⑤ 位置切分解（与 solve 同款判据）----------
    print("\n⑤ 位置切分解（与 --handeye-solve 同款；判据 Δr < 0.5°、Δt 看噪声带）：")
    for axis_name, key in (("左右", 0), ("上下", 1), ("远近", 2)):
        vals = np.array([cxy[i][key] for i in range(n)])
        lo = [i for i in range(n) if vals[i] < np.median(vals)]
        hi = [i for i in range(n) if vals[i] >= np.median(vals)]
        Ta = solve_one(lo, Rg, tg, R_t2c, t_t2c)
        Tb = solve_one(hi, Rg, tg, R_t2c, t_t2c)
        if Ta is None or Tb is None:
            continue
        dt, dr = diffX(Ta, Tb)
        flag = "  ← 对不上" if dr >= 0.5 else ""
        print(f"    {axis_name}：{len(lo)} vs {len(hi)} → Δt {dt:6.2f}mm  Δr {dr:6.3f}°{flag}")
        print(f"      lo: {' '.join(tags[i][:6] for i in lo)}")
        print(f"      hi: {' '.join(tags[i][:6] for i in hi)}")

    # ---------- ⑥ 候选剔除集诊断 ----------
    print("\n⑥ 候选剔除集（纯诊断；哪种剔法让 ②③④⑤ 同时收敛——凶手长什么样）：")
    print(f"    {'剔除':<24} {'轴中位°':>7} {'留一中位mm':>9} {'时间Δr°':>8} {'位置Δr°(左右/上下/远近)':>18}")
    for drops in DROPSETS:
        didx = [i for i in range(n) if names[i][-7:-4] not in drops]
        poses_k = [poses[i] for i in didx]
        R_k = [R_t2c[i] for i in didx]
        tk = [t_t2c[i] for i in didx]
        Rg_k = [Rg[i] for i in didx]
        tg_k = [tg[i] for i in didx]
        pk_pairs = pair_axes(poses_k, R_k)
        amk = None
        if len(pk_pairs) >= 6:
            amk = float(np.median(pair_errs(pk_pairs, fit_R(pk_pairs))))
        hh = len(didx) // 2
        T1 = solve_one(list(range(0, hh)), Rg_k, tg_k, R_k, tk)
        T2 = solve_one(list(range(hh, len(didx))), Rg_k, tg_k, R_k, tk)
        dtr = diffX(T1, T2)[1] if (T1 is not None and T2 is not None) else float("nan")
        vv = []
        for key in (0, 1, 2):
            vals = np.array([cxy[i][key] for i in didx])
            lo = [j for j in range(len(didx)) if vals[j] < np.median(vals)]
            hi2 = [j for j in range(len(didx)) if vals[j] >= np.median(vals)]
            Ta = solve_one(lo, Rg_k, tg_k, R_k, tk)
            Tb = solve_one(hi2, Rg_k, tg_k, R_k, tk)
            vv.append(diffX(Ta, Tb)[1] if (Ta is not None and Tb is not None) else float("nan"))
        loo2 = []
        for kk in range(len(didx)):
            sub = [j for j in range(len(didx)) if j != kk]
            Tk = solve_one(sub, Rg_k, tg_k, R_k, tk)
            if Tk is None:
                continue
            pk = np.array([(C._pose6_to_T(poses_k[j]) @ Tk @ np.block(
                [[R_k[j], np.asarray(tk[j]).reshape(3, 1)], [np.zeros(3), 1.0]]))[:3, 3]
                for j in sub]).mean(axis=0)
            pred = (np.linalg.inv(Tk) @ (np.linalg.inv(C._pose6_to_T(poses_k[kk])) @ np.append(pk, 1.0)))[:3]
            loo2.append(float(np.linalg.norm(pred - np.asarray(tk[kk]).ravel())))
        lmed = float(np.median(loo2)) if loo2 else float("nan")
        dlab = "+".join(drops) if drops else "（不剔）"
        amtxt = f"{amk:7.2f}" if amk is not None else "   —   "
        vtxt = "/".join(f"{v:.2f}" if np.isfinite(v) else "—" for v in vv)
        print(f"    {dlab:<24} {amtxt} {lmed:9.2f} {dtr:8.3f} {vtxt:>18}")

    # ---------- ⑦ 帧抽样稳定性：⑤ 的 Δr 里有多少是"Park 抽 16 帧自己就抖"？ ----------
    print("\n⑦ 帧抽样稳定性（Park）：随机抽 16/23 帧各 200 次，相对全量解的散布")
    Tfull = solve_one(list(range(n)), Rg, tg, R_t2c, t_t2c)
    rng2 = np.random.default_rng(0)
    for m in (16, 23):
        if m > n:
            continue                                           # 小批次（≤16 帧）跳过抽不出的档
        dts2, drs2, crash = [], [], 0
        for _ in range(200):
            idx = sorted(int(v) for v in rng2.choice(n, m, replace=False))
            T = solve_one(idx, Rg, tg, R_t2c, t_t2c)
            if T is None:
                crash += 1
                continue
            dt, dr = diffX(Tfull, T)
            if dr > 10.0 or dt > 100.0:                        # Park 在小子集上会整段崩
                crash += 1
                continue
            dts2.append(dt); drs2.append(dr)
        if dts2:
            print(f"    n={m}：Δt 中位 {np.median(dts2):5.2f}mm / 95分位 {np.percentile(dts2, 95):5.2f}mm   "
                  f"Δr 中位 {np.median(drs2):6.3f}° / 95分位 {np.percentile(drs2, 95):6.3f}°   "
                  f"数值崩 {crash}/200")
        else:
            print(f"    n={m}：全部数值崩（{crash}/200）")
    print("    （⑤ 各分区 Δr 若小于这里的 95 分位，就不是「两边真不一样」，只是子集解不稳）")

    # ---------- ⑧ 全量解下每帧板原点在基座系里的散布（×时间线，与检查点读数并排）----------
    print("\n⑧ 每帧板原点投回基座系（全量 Park 解）：板没动 ⇒ 聚一点；检查点读数插在时间线原位")
    Tbb = [C._pose6_to_T(poses[i]) @ Tfull @ np.block(
        [[R_t2c[i], np.asarray(t_t2c[i]).reshape(3, 1)], [np.zeros(3), 1.0]]) for i in range(n)]
    pb = np.array([T[:3, 3] for T in Tbb])
    pmean = pb.mean(axis=0)
    dev = np.linalg.norm(pb - pmean, axis=1)
    print(f"    位置散 RMS {float(np.sqrt((dev ** 2).mean())):.2f}mm   最大 {dev.max():.2f}mm")
    wv = np.array([wr[i] if wr[i] is not None else np.nan for i in range(n)])
    okm = np.isfinite(wv)
    if okm.sum() >= 8:
        c1 = float(np.corrcoef(wv[okm], dev[okm])[0, 1])
        okb = okm & np.isfinite(blame)
        c2 = float(np.corrcoef(wv[okb], blame[okb])[0, 1]) if okb.sum() >= 8 else float("nan")
        print(f"    相关性：腕偏 vs 板原点散 r = {c1:+.2f}（{int(okm.sum())} 帧）；"
              f"腕偏 vs 轴牵连 r = {c2:+.2f}")
        print("    （腕动越大⇔误差越大 ⇒ 相机支架/线缆被腕部动作扰动的实锤；r≈0 ⇒ 另有其因）")
    ev = [(tsecs[i], 0, i) for i in range(n)]
    if tinfo and checks:
        ev += [(tsec(ck[0]), 1, ck) for ck in checks]
        ev.sort()
        for t0, kind, obj in ev:
            hms = f"{int(t0 // 3600):02d}:{int(t0 % 3600 // 60):02d}:{int(t0 % 60):02d}"
            if kind == 0:
                i = obj
                d = pb[i] - pmean
                print(f"    {hms} 帧{i:02d} {tags[i]:<10} Δ[{d[0]:+6.2f} {d[1]:+6.2f} {d[2]:+6.2f}]mm"
                      f"  |Δ| {dev[i]:5.2f}")
            else:
                _t, ctag, mm, deg = obj
                print(f"    {hms} 【检查点 {ctag:<9}】板漂移读数 {mm:.2f}mm / {deg:.2f}°")
    else:
        for i in range(n):
            d = pb[i] - pmean
            print(f"    帧{i:02d} {tags[i]:<10} Δ[{d[0]:+6.2f} {d[1]:+6.2f} {d[2]:+6.2f}]mm  |Δ| {dev[i]:5.2f}")

    # ---------- ⑨ 直尺式 k：纯平移对逐对比模长（姿态差<2°、法兰位移>40mm）----------
    # 不用回归、不用 R_X：姿态冻结 ⇒ 相机位移 = 法兰位移（刚体）；板链测的 |u| 若是
    # k 倍真 mm，则 |u|/|Δ法兰| 就是 k。与 solve 的回归 k 独立，用来互相印证。
    # ⚠ 用原始 t（--k 换算过就不用它，避免自证）。
    print("\n⑨ 直尺式 k（纯平移对逐对比模长；与回归 k 独立）")
    Tf = [C._pose6_to_T(p) for p in poses]
    ks_d = []
    for i in range(n):
        for j in range(i + 1, n):
            if C._rot_axis_angle(Tf[j][:3, :3].T @ Tf[i][:3, :3])[0] > 2.0:
                continue
            df = float(np.linalg.norm(Tf[j][:3, 3] - Tf[i][:3, 3]))
            if df < 40.0:
                continue
            Ti = np.eye(4); Ti[:3, :3], Ti[:3, 3] = R_t2c[i], np.asarray(t_raw[i]).ravel()
            Tj = np.eye(4); Tj[:3, :3], Tj[:3, 3] = R_t2c[j], np.asarray(t_raw[j]).ravel()
            u = (Tj @ np.linalg.inv(Ti))[:3, 3]
            du = float(np.linalg.norm(u))
            if du < 20.0:
                continue
            ks_d.append((du / df, i, j, df, du))
    if len(ks_d) >= 4:
        kv = np.array([x[0] for x in ks_d])
        print(f"    {len(ks_d)} 对：k 中位 {np.median(kv):.4f}  均值 {kv.mean():.4f}  "
              f"σ {kv.std(ddof=1):.4f}  范围 [{kv.min():.4f}, {kv.max():.4f}]")
        for kk2, i, j, df, du in sorted(ks_d, reverse=True)[:8]:
            print(f"    {names[i][-7:-4]}→{names[j][-7:-4]}  Δ法兰 {df:5.1f}mm  Δ板链 {du:5.1f}mm  "
                  f"比值 {kk2:.4f}")
        print("    （⑨/⑩ 读的都是板侧尺度因子：板尺寸错 ⇒ 读出该因子；焦距错 ⇒ ≈1.00（⑩a 真值表））")
    else:
        print(f"    纯平移对大少（{len(ks_d)} 对）")

    # ---------- ⑩ 画面平移 ÷ 模型预测：单独量板因子（与焦距解耦）----------
    print("\n⑩ 画面平移 ÷ 模型预测（姿态冻结对；与 ⑨ 独立二次核验板侧尺度因子）")
    print("    ⑩a 合成世界自检（钉方向）：")
    if not sim_selfcheck():
        print("    ⚠⚠ 自检不过——下面的实数不能信，先修这段数学")
    objp = det.objp.astype(np.float64)
    rows10 = []
    for i in range(n):
        for j in range(i + 1, n):
            if C._rot_axis_angle(Tf[j][:3, :3].T @ Tf[i][:3, :3])[0] > 0.8:
                continue
            out = shift_test_pair(corners_all[i], ids_all[i], corners_all[j], ids_all[j],
                                  R_t2c[i], t_raw[i], R_t2c[j], t_raw[j],
                                  Tf[i], Tf[j], K, D, objp)
            if out is not None:
                rows10.append((i, j, out))
    if rows10:
        rows10.sort(key=lambda r: r[2][4], reverse=True)
        for i, j, (ratio, ang, nmeas, npred, df, nn) in rows10:
            print(f"    {names[i][-7:-4]}→{names[j][-7:-4]}  Δ法兰 {df:5.1f}mm  "
                  f"实测 {nmeas:6.1f}px  预测 {npred:6.1f}px  比值 {ratio:.4f}  "
                  f"方向差 {ang:+5.1f}°  共用 {nn} 点")
        rr = np.array([r[2][0] for r in rows10])
        big = np.array([r[2][0] for r in rows10 if r[2][4] >= 25.0])
        line = f"    全 {len(rr)} 对：中位 {np.median(rr):.4f}"
        if len(big):
            line += f"    Δ≥25mm 的 {len(big)} 对：中位 {np.median(big):.4f}"
        print(line)
        print("    （读 ~1.00 ⇒ 板侧无尺度错，k≈1.07 另有他因；读 ~1.075 ⇒ 板真小 7.5%（≈23.2mm）；"
              "读 ~1.04 ⇒ 板≈24mm。焦距错在这两个量里都≈1.00——去 ⑧ 的 R 误差里找它）")
    else:
        print("    姿态冻结对太少，量不出")


if __name__ == "__main__":
    main()
