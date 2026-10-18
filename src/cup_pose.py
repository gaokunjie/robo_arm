# -*- coding: utf-8 -*-
"""cup_pose：运行时杯子定位（像素 → 基座系杯轴 xy + 码平面 z）。P4 视觉定位唯一入口。

⚠ 杯上 6 枚码【全是 ID 1】（60° 均布）——跨帧按 ID 配对三角测量不成立。

主方法（2026-10-19 起）= 码法线【径向直线 LS 交点】：
    每帧每条检测 → solvePnP(柱面模型) → 码心 Pb + 码外法线 n̂（基座系）
    → 过 Pb 沿 n̂ 的"径向线"必过杯轴 ⇒ 所有径向线的最小二乘交点 = 杯轴 xy。
    · 对 R（码高处半径）与码心径向偏差 δ 双免疫：直线方程里根本没有 R，沿 n̂ 平移不改直线。
    · 只需【方向】⇒ 用比位置更松的方向闸（PNP_RMS_GATE_PX_DIR），把位置不达标但方向可用的
      码也纳入交点（这类码常独供一个方位角，剔掉会让交点病态、漂几 mm）。
交叉核对（仅参考，不采信）：法线点法（Pb − R·n̂，对 R/δ 敏感）+ 圆心拟合（放大 28×）。
    旧"射线∩z 平面 + 圆拟合"路径已降级为 --legacy 诊断对照。

口径（冻结，勿动）：
    X   ← data/hand_eye.yaml（2026-10-16 冻结；p_flange = R·p_cam + t）
    K,D ← data/camera_intrinsics.yaml（锁焦版 fx≈1462；AF 开启期的数据全作废）
    r0  = (R40+码厚)·cos(半包角) —— 吸收"四角均值=弦中点"效应；R40 由 CUP_PROFILE 插值。
    ⚠ 相机图像流相对世界是【180° 旋转】的（GP_Flip_Mirror 硬件链）；一切像素运算都在
      实际像素空间里做，投影/叠加照原始坐标画即可——只是"人眼看图"要转正再读。

用法（离线重算；真机采样走 src/see_cup.py）：
    python3 -I src/cup_pose.py data/cup_shots/<stamp>     # 目录内 *.json（含 image 字段）
    python3 -I src/cup_pose.py shotA.json shotB.json
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                   # noqa: E402
import geometry                                                 # noqa: E402
import fr3_math as M                                            # noqa: E402
from detect_cup import Detector                                 # noqa: E402

def marker_plane_z():
    """码中心平面的基座系 z —— 单一出处 = config.TABLE_Z + MARKER_Z_ABOVE_TABLE。

    ⚠ 2026-10-17：本模块原先有一份私有副本 NOMINAL_TABLE_Z = 442.0，与 config.TABLE_Z(None)、
      see_cup.LD_AIM.z、_ld_path_check 的硬编码 z=442 合计四份 z 锚，互相差 80~135mm。
      现在只认 config；config 未实测 = 拒绝解算（见 config.require_scene）。
    """
    return config.marker_plane_z()


# ---------------------------------------------------------------- 标定载入
def load_calib():
    """→ (K, D, X)。X = 法兰←相机 4×4（p_flange = R·p_cam + t）。"""
    p = config.DATA_DIR / config.CALIB_CAMERA
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        sys.exit(f"读不到内参：{p}")
    K = np.asarray(fs.getNode("camera_matrix").mat(), float)
    D = np.asarray(fs.getNode("dist_coeffs").mat(), float).ravel()
    fs.release()
    p = config.DATA_DIR / "hand_eye.yaml"
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        sys.exit(f"读不到手眼：{p}")
    R = np.asarray(fs.getNode("R_flange_cam").mat(), float)
    t = np.asarray(fs.getNode("t_flange_cam_mm").mat(), float).reshape(3)
    fs.release()
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = t
    return K, D, X


def marker_circle_r():
    """r0：码中心（四角均值点）到杯轴的水平距离。

    四角均值点 = 弦中点效应：落在表面往里 (1−cos 半包角)·R；码厚再往外 +t。
    净 = (R40+t)·cos(半包角)（与逐角精确公式差 <0.1mm）。
    """
    r40 = geometry.cup_od_at(config.MARKER_Z_ABOVE_TABLE)[0] / 2.0
    half = np.radians(geometry.marker_wrap_deg(config.MARKER_Z_ABOVE_TABLE,
                                               config.ARUCO_MARKER_MM) / 2.0)
    return float((r40 + config.MARKER_THICK) * np.cos(half))


# ---------------------------------------------------------------- 检测/射线
def refine_corners(gray, corners):
    """cornerSubPix 亚像素精化（失败回退原值，不炸）。"""
    c = np.asarray(corners, np.float32).reshape(-1, 1, 2).copy()
    try:
        cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    except cv2.error:
        pass
    return c.reshape(4, 2).astype(float)


def detect_ex(img, det=None):
    """检测 + 精化 → [{id, corners(4×2), center(2,), span_px}]。"""
    det = det or Detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for d in det.detect(img):
        c = refine_corners(gray, d["corners"])
        span = float(np.mean([np.linalg.norm(c[(k + 1) % 4] - c[k]) for k in range(4)]))
        out.append({"id": int(d["id"]), "corners": c, "center": c.mean(axis=0), "span_px": span})
    return out


def _ray_cam(K, D, uv):
    p = cv2.undistortPoints(np.asarray(uv, float).reshape(1, 1, 2), K, D).reshape(2)
    v = np.array([p[0], p[1], 1.0])
    return v / np.linalg.norm(v)


def base_ray(T_f, X, K, D, uv):
    """基座系射线 (origin, dir)，dir 单位化。"""
    TB = T_f @ X
    return TB[:3, 3].copy(), TB[:3, :3] @ _ray_cam(K, D, uv)


def _ray_plane_xy(o, d, z):
    """射线 ∩ 平面 z=z 的 xy；掠射/朝后 → None。"""
    if abs(d[2]) < 0.15:
        return None
    t = (z - o[2]) / d[2]
    if t <= 0:
        return None
    return (o + t * d)[:2]


# ============================================================================
# PnP 主方法（2026-10-17）
# ----------------------------------------------------------------------------
# 为什么是主方法（存量帧实测，2026-10-17）：
#   · 码自身姿态直接给杯轴方向：码局部 X 轴与竖直夹角 ±2.8°（15/15 检测）
#     ⇒ 单枚码即可定方向，**不需要圆拟合 / 多视角 / orbit**
#   · 深度由 solvePnP 直接给出 ⇒ **完全不吃 TABLE_Z**
#     实测：所有点 z 整体偏 20mm → 拟合出的杯轴 xy 变化 0.00mm（完全解耦）
#   · 径向残差 RMS 1.69mm（好批次），在容差 5.12mm 之内
#   对照旧法（射线 ∩ z=TABLE_Z+40 平面 + 圆拟合）：残差 5.2~12.3mm，且 xy 对
#   TABLE_Z 的灵敏度 2.6mm/mm —— 那个 z 平面 detour 是历史 42mm 偏差的产地。
#
# ⚠ 柱面建模（关键，修正 detect_cup.py 文档里记的「弦长 ≠ 标称边长，偏 7~9%」）：
#   码贴在半径 R 的柱面上，角点不在切平面上：
#       u = 切向弧长坐标，ψ = u/R
#       x_obj = R·sin(ψ)        （切向：弦 < 弧）
#       z_obj = −R·(1 − cos ψ)  （径向：角点比码心"陷进去"）
#   用平面方块当模型 → 尺寸高估 ~2%、深度系统性偏大、xy 被拽偏。
# ============================================================================

MARKER_UP_LOCAL_AXIS = 1        # 实测（2026-10-17 存量帧）：码局部【Y】轴 ≈ 竖直 = 杯轴方向
                                  # 局部 Y 的 z 分量 ≈ −0.997；符号随贴码朝向，取反成向上
# ⚠ 曾误判为局部 X（因 arcsin(≈0) 误读成"≈竖直"，实为"水平"），已修正
PNP_RMS_GATE_PX = 1.0          # solvePnP 重投影 RMS 闸（实测真码 0.10~0.49px）
PNP_RMS_GATE_PX_DIR = 3.0      # 直线交点法的【仅方向】闸：位置/深度精度要求可放宽，方向仍稳
                               #   实测：2.91px 的码位置偏 ~3.8mm，但方向可用，且独供一个方位角
                               #   ⇒ 若按 1.0px 剔掉它，剩码方位重合、交点病态漂 2.5mm
PNP_TILT_GATE_DEG = 15.0        # 杯轴与竖直夹角闸：超了说明杯倒了/位姿不对
PNP_TRIM_SIGMA = 3.0           # 迭代剔野：残差 > TRIM_SIGMA×中位绝对残差 就剔
PNP_RMS_PASS_MM = 3.0          # 验收闸①：径向残差 RMS 超过它 = 不采信（容差 5.12 的 60%）
PNP_LINE_DIST_PASS_MM = 3.0    # 验收闸①（主）：直线交点法中，各码径向线到交点的垂距（中位）
                               #   直线交点法对 R 与码心径向偏差 δ 双免疫，是【主结果】依据。
                               #   现场 4 线实测垂距 1.5/3.2/4.7/1.5mm（中位 2.35）⇒ 3.0 是合理闸。
PNP_LINE_SPREAD_MIN_DEG = 30.0 # 验收闸②（主·条件数）：各码径向【方向】的方位张角（mod 180）下限。
                               #   垂距小 ≠ 交点稳：两线近乎平行时交点也"看似自洽"却病态。
                               #   实测教训：3 枚码里两枚方位差 0.3°（几乎平行），交点漂 2.5mm；
                               #   补上第 3 个方位（158°）后张角 ~120°、交点稳定。取 30° 为下限。
PNP_NORMAL_SPREAD_PASS_MM = 8.0  # 信息项（交叉核对，不拦）：点法各码独立算出的杯轴散布
                                  #   ⚠ 为什么不能用它在线标定 R（试过了，不行，别再试）：
                                  #     单次采集里可读的码全挤在 ~30mm 内（方位几乎相同），
                                  #     散布对 R 几乎不敏感（233456 批：R 36.55→46.00，散布 5.17→5.06）；
                                  #     但轴【位置】对 R 是 1:1（同一批 ΔR=9.45 → 轴动 ~10mm ✓）。
                                  #     ⇒ R 只能卡尺实测（config.CUP_PROFILE 的 40mm 处），
                                  #       或者恢复 orbit 让码方位铺开再标 —— 现选前者。



def marker_cyl_obj(half_mm, R, n=4):
    """码在柱面上的 4 个角点（局部系：x=切向、y=竖直、z=外法线）。"""
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))[:n]:
        u = sx * half_mm
        psi = u / R
        pts.append([R * math.sin(psi), sy * half_mm, -R * (1.0 - math.cos(psi))])
    return np.asarray(pts, np.float64)


def marker_radius():
    """码中心所在柱面的半径 R = 杯半径(码高处) + 码厚。"""
    return geometry.cup_od_at(config.MARKER_Z_ABOVE_TABLE)[0] / 2.0 + config.MARKER_THICK


def marker_pose_cam(corners, K, D, R=None):
    """单枚码 → 相机系位姿 (tvec, Rmat, 重投影RMSpx, 深度)。柱面模型。"""
    R = marker_radius() if R is None else R
    obj = marker_cyl_obj(config.ARUCO_MARKER_MM / 2.0, R)
    img = np.asarray(corners, np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    err, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(((err.reshape(4, 2) - img) ** 2).sum(axis=1).mean()))
    return tvec.reshape(3), cv2.Rodrigues(rvec)[0], rms, float(np.linalg.norm(tvec))


def axis_from_marker_normals(recs, R, camera_xy=None):
    """【主估计器】用码自身法线直接定杯轴 —— 不拟合圆心，灵敏度 1:1。

    码贴在半径 R 的柱面外侧 ⇒  码心_xy = 杯轴_xy + R·径向外法线_水平
    故  杯轴_xy = 码心_xy − R·n̂_h。

    为什么必须这样而不是拟合圆心（2026-10-17 实测）：
      可读的码全在杯的【近侧】（方位跨度约 90° 弧），弧拟合的径向方向条件数极差——
      半径 R 差 2.15mm → 拟合出的轴动 60mm，放大约 28×。
      而 R 来自 CUP_PROFILE 两点内插，本身就有 ±1mm 量级不确定度 ⇒ 圆心法根本不可用。
      法线法：R 错 δ → 轴错 δ（1:1，无放大）；法线方位角误差 ~0.34° → 0.22mm。
    """
    out = []
    for c in recs:
        n = c["Rb"][:, 2]                              # 柱面模型下局部 z 指向轴内侧
        nh = np.array([n[0], n[1]])                    # 水平分量（径向外法线就在 xy 平面内）
        nn = float(np.linalg.norm(nh))
        if nn < 1e-9:
            out.append(None)
            continue
        nh = nh / nn
        cam = np.asarray(c["cam_xy"] if camera_xy is None else camera_xy, float)
        to_marker = c["Pb"][:2] - cam                  # 相机 → 码
        if np.linalg.norm(to_marker) < 1e-9:
            out.append(None)
            continue
        to_marker = to_marker / np.linalg.norm(to_marker)
        # 码能被解码 ⇒ 它在杯的【近侧】、正面朝相机 ⇒ 外法线由【轴指向码】，
        # 也就是由相机一侧指回来。所以要 nh 与 (相机→码) 反向。
        # ⚠ 2026-10-17 首版把这里写成"同向"，结果杯轴被放到码的靠相机一侧 ——
        #   标注图上十字落在码的下方（更近），而正确位置应在码的【上方/更远】。
        #   判据：杯轴 |xy| 必须【大于】码的 |xy|（相机在 +y 侧时）。
        if np.dot(nh, to_marker) > 0:
            nh = -nh

        out.append({"tag": c["tag"], "id": c["id"], "n_h": nh,
                    "axis": c["Pb"][:2] - R * nh,
                    "radial_deg": float(np.degrees(np.arctan2(nh[1], nh[0]))),
                    "norm_tilt_deg": float(np.degrees(np.arccos(
                        np.clip(-c["Rb"][2, 2], -1, 1))))})
    return [o for o in out if o is not None]


def axis_from_marker_lines(recs):
    """【主估计器·首选】码法线径向直线的 LS 交点 —— 对 R 与码心径向偏差 δ 双免疫。

    每枚码给出码心 Pb（基座系）与该处外法线的水平分量 n̂_h（径向）。
    过 Pb 沿 n̂_h 的直线即该码的"径向线"，杯轴必在这条线上：
      · 不管 R（码高处杯半径 + 码厚）标得准不准 —— 直线方程里根本没有 R；
      · 不管曲面 PnP 把码心沿径向偏了多少 δ —— 沿 n̂ 平移不改变"过 Pb 沿 n̂ 的直线"。
    ⇒ 杯轴 = 所有径向线的最小二乘交点（各线垂距平方和最小，闭式线性解）。
    返回 (per_line, axis_xy)；per_line 每项含该码到交点的垂距 line_dist（一致性度量）。

    为什么不用点法（Pb − R·n̂）：点法把每码投影成一个点再取中位，对 δ 与 R 敏感；
    config.CUP_AXIS_XY 现场记录用的就是这个直线交点法（4 条线到交点距 1.5/3.2/4.7/1.5mm）。
    """
    rows, rhs, used = [], [], []
    for c in recs:
        nh = np.array([c["Rb"][0, 2], c["Rb"][1, 2]], float)
        nn = float(np.linalg.norm(nh))
        if nn < 1e-9:
            continue
        nh = nh / nn
        nperp = np.array([-nh[1], nh[0]])          # 直线法向（垂距 = |a·n̂⊥ − p·n̂⊥|）
        p = c["Pb"][:2]
        rows.append(nperp)
        rhs.append(float(nperp @ p))
        used.append(c)
    if len(rows) < 2:
        return [], None
    A = np.array(rows)
    b = np.array(rhs)
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    per = [{"tag": c["tag"], "id": c["id"], "line_dist": float(abs(row @ a - r)),
            "az_deg": float(np.degrees(np.arctan2(c["Rb"][1, 2], c["Rb"][0, 2])) % 180.0)}
           for c, row, r in zip(used, rows, rhs)]
    return per, (float(a[0]), float(a[1]))


def _fit_vertical_axis(Pxy, r0, start):
    """竖直轴等半径圆柱的轴心 xy：min Σ(|P−a| − r0)²。"""
    from scipy.optimize import least_squares
    a = np.asarray(start, float)

    def f(p):
        return np.hypot(Pxy[:, 0] - p[0], Pxy[:, 1] - p[1]) - r0
    return least_squares(f, a, method="lm", xtol=1e-13, ftol=1e-13).x


def solve_pnp(shots, K=None, D=None, X=None, say=print, r0=None):
    """shots: [{"T": 4×4(法兰), "dets": [...], "tag": str}]

    返回 dict（字段与旧 solve() 对齐，便于并排比较）：
      axis_xy        杯轴 xy（主结果）
      z_markers      码平面 z（中位数，PnP 深度给的，**不是 TABLE_Z 推的**）
      rms_mm         径向残差 RMS（半径固定 ⇒ 无自由参数的真实误差量）
      implied_table_z  z_markers − MARKER_Z_ABOVE_TABLE（仅诊断，不写 config）
      tilt_deg       码局部 X 轴给出的杯轴与竖直夹角（杯是否直立）
      per_det / n_det / n_reject ...
    """
    if K is None:
        K, D, X = load_calib()
    R = marker_radius()
    # ⚠ 半径口径：柱面模型下解出的码心【就是真实柱面上的点】，半径就是 R；
    #   marker_circle_r()（34.40）是给【平面模型】用的弦长修正值，用在这里等于修正两次。
    r0 = R if r0 is None else r0

    cand = []
    for sh in shots:
        Tb = sh["T"] @ X
        for det in sh["dets"]:
            if det["id"] not in _cup_ids():
                continue
            mp = marker_pose_cam(det["corners"], K, D, R)
            if mp is None:
                cand.append({"tag": sh["tag"], "id": det["id"],
                             "why": "solvePnP 失败", "why_dir": "solvePnP 失败"})
                continue
            tvec, Rcam, rms, depth = mp
            Rb = Tb[:3, :3] @ Rcam
            uy = Rb[:, MARKER_UP_LOCAL_AXIS]              # 码局部 Y ≈ 杯轴
            up = uy if uy[2] >= 0 else -uy               # 统一取向上
            tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
            rec = {"tag": sh["tag"], "id": det["id"], "tvec": tvec, "Rcam": Rcam,
                   "depth": depth, "pnp_rms_px": rms, "tilt_deg": tilt,
                   "Pb": (Tb @ np.append(tvec, 1.0))[:3],
                   "Rb": Rb, "cam_xy": Tb[:3, 3][:2],
                   "span_px": det.get("span_px"), "center": det.get("center"),
                   "why": "", "why_dir": ""}
            if rms > PNP_RMS_GATE_PX:
                rec["why"] = f"PnP 重投影 {rms:.2f}px > {PNP_RMS_GATE_PX}px"
            elif tilt > PNP_TILT_GATE_DEG:
                rec["why"] = f"杯轴与竖直 {tilt:.1f}° > {PNP_TILT_GATE_DEG}°（杯倒了?）"
            # 方向可用性：直线交点法只需方向 n̂（位置/深度精度另论），故有独立更松的闸。
            # 实测：PnP 2.91px 的码位置差 ~3.8mm，但方向仍可用，且它独供第 3 个方位角
            #（0.3° 重合的那两枚使 3 枚交点病态）⇒ 不含它时交点漂 2.5mm。
            if rms > PNP_RMS_GATE_PX_DIR:
                rec["why_dir"] = f"PnP 重投影 {rms:.2f}px > {PNP_RMS_GATE_PX_DIR}px（方向也不可用）"
            elif tilt > PNP_TILT_GATE_DEG:
                rec["why_dir"] = f"杯轴与竖直 {tilt:.1f}° > {PNP_TILT_GATE_DEG}°"
            cand.append(rec)

    good = [c for c in cand if not c["why"]]          # 精度集（点法/圆心法用）
    good_dir = [c for c in cand if not c["why_dir"]]  # 方向集（直线交点法用，⊇ good）
    res = {"r0_mm": r0, "n_all": len(cand), "n_det": len(good),
           "n_dir": len(good_dir),
           "n_reject": len(cand) - len(good), "method": "solvePnP + 柱面模型",
           "per_det": []}
    if good_dir:
        res["tilt_deg"] = float(np.median([c["tilt_deg"] for c in good_dir]))
        res["tilt_max_deg"] = float(max(c["tilt_deg"] for c in good_dir))
    if len(good) < 2:
        res["error"] = (f"精度可用码 {len(good)} 枚 <2（共检出 {len(cand)}，"
                        f"方向可用 {len(good_dir)}）——解不出杯轴（挪观察位/确认码在视野里）")
        for c in cand:
            res["per_det"].append({"tag": c["tag"], "id": c["id"],
                                   "why": c["why"] or "样本不足"})
        return res

    P = np.array([c["Pb"] for c in good])
    # ---- 主估计器 A【首选】：码法线径向直线的 LS 交点（对 R/δ 双免疫） ----
    cam_xy = float(np.median([c["Pb"][0] for c in good_dir])), float(
        np.median([c["Pb"][1] for c in good_dir]))
    lines, a_line = axis_from_marker_lines(good_dir)
    res["n_lines"] = len(lines)
    if a_line is not None:
        dl = np.array([l["line_dist"] for l in lines])
        res["axis_xy_line"] = a_line
        res["line_dist_med_mm"] = float(np.median(dl))
        res["line_dist_max_mm"] = float(dl.max())
        az = [l["az_deg"] for l in lines]
        res["line_spread_deg"] = float(max(
            (min(abs(az[i] - az[j]), 180.0 - abs(az[i] - az[j])))
            for i in range(len(az)) for j in range(i + 1, len(az)))) if len(az) > 1 else 0.0
    # ---- 交叉核对 B：法线点法（Pb − R·n̂，对 R/δ 敏感，仅参考） ----
    nrm = axis_from_marker_normals(good, R, None)
    res["n_normals"] = len(nrm)
    if nrm:
        A = np.array([n["axis"] for n in nrm])
        a_norm = np.median(A, axis=0)
        spread = np.linalg.norm(A - a_norm, axis=1)
        res["axis_xy_normal"] = (float(a_norm[0]), float(a_norm[1]))
        res["normal_spread_mm"] = float(np.median(spread))
        res["normal_spread_max_mm"] = float(spread.max())
        res["radial_spread_deg"] = float(max(
            abs((n["radial_deg"] - a + 180) % 360 - 180)
            for n in nrm for a in (float(np.degrees(np.arctan2(a_norm[1], a_norm[0]))),)))
    # ---- 交叉核对：圆心拟合（放大 28×，只当质量指标，不当结果） ----
    keep = list(range(len(good)))
    start = np.array([float(np.median(P[:, 0])), float(np.median(P[:, 1]))])
    a = _fit_vertical_axis(P[keep, :2], r0, start)
    for _round in range(3):                                 # 迭代剔野
        rad = np.hypot(P[keep, 0] - a[0], P[keep, 1] - a[1]) - r0
        rms = float(np.sqrt((rad ** 2).mean()))
        mad = float(np.median(np.abs(rad)))
        if mad <= 1e-6:
            break
        worst = int(np.argmax(np.abs(rad)))
        if abs(rad[worst]) <= PNP_TRIM_SIGMA * mad:
            break
        keep.pop(worst)
        if len(keep) < 2:
            keep.append(worst)
            break
        a = _fit_vertical_axis(P[keep, :2], r0, a)

    rad = np.hypot(P[keep, 0] - a[0], P[keep, 1] - a[1]) - r0
    rms = float(np.sqrt((rad ** 2).mean()))
    zm = float(np.median(P[keep, 2]))
    dropped = [i for i in range(len(good)) if i not in keep]
    for i, c in enumerate(good):
        r = float(np.hypot(c["Pb"][0] - a[0], c["Pb"][1] - a[1]) - r0)
        res["per_det"].append({
            "tag": c["tag"], "id": c["id"], "depth_mm": round(c["depth"], 1),
            "pnp_rms_px": round(c["pnp_rms_px"], 2),
            "tilt_deg": round(c["tilt_deg"], 1),
            "resid_mm": round(r, 2),
            "Pb": [round(float(v), 1) for v in c["Pb"]],
            "used": i in keep,
            "why": "" if i in keep else f"剔野（径向残差 {r:+.2f}mm）"})
    for c in cand:
        if c["why"]:
            why = c["why"]
            if not c.get("why_dir"):        # 位置不达标但方向可用 → 直线交点法仍采用
                why += "（方向仍用于直线交点）"
            res["per_det"].append({"tag": c["tag"], "id": c["id"], "why": why})
    res["axis_xy_circle"] = (float(a[0]), float(a[1]))
    final = (res.get("axis_xy_line")
             or res.get("axis_xy_normal")
             or (float(a[0]), float(a[1])))
    res.update(axis_xy=final, z_markers=zm, rms_mm=rms,
               max_resid_mm=float(np.abs(rad).max()), n_trim=len(dropped),
               implied_table_z=zm - config.MARKER_Z_ABOVE_TABLE,
               cand_spread_mm=float(max(
                   np.linalg.norm(P[keep][i] - P[keep][j])
                   for i in range(len(keep)) for j in range(len(keep)))) if len(keep) > 1 else 0.0)
    # 验收（2026-10-19 重定，与主估计器=直线交点法一致）：
    #   ① 直线交点垂距（几何一致性） ② 各码方向张角（条件数，防"平行线假自洽"） ③ 杯轴倾角
    #   径向残差 RMS / 点法散布降为【信息项】：它们量的是圆心拟合轴（本题几何下放大 28×、
    #   已退化），用退化轴算出的残差无判别力，不再拦。
    res["pass"] = bool(
        res.get("line_dist_med_mm", 9e9) <= PNP_LINE_DIST_PASS_MM
        and res.get("line_spread_deg", 0.0) >= PNP_LINE_SPREAD_MIN_DEG
        and res.get("tilt_max_deg", 90.0) <= PNP_TILT_GATE_DEG)
    return res


def _cup_ids():
    return {config.ARUCO_ID_CUP_A, config.ARUCO_ID_CUP_B}


def report_pnp(res, say=print):
    if res.get("error"):
        say(f"✗ {res['error']}")
        for p in res["per_det"]:
            say(f"    [{p['tag']}] id{p['id']}  {p['why']}")
        return False
    say(f"方法 {res['method']}   精度可用 {res['n_det']}/{res['n_all']} 枚"
        f"（预处理剔 {res['n_reject']}，迭代剔野 {res['n_trim']}）"
        f"   方向可用 {res.get('n_dir', res['n_det'])} 枚（直线交点法用）")
    say(f"半径固定 r0 = {res['r0_mm']:.2f} mm（卡尺 + 杯轮廓算出 ⇒ 残差无自由参数）")
    say(f"{'帧':<14}{'ID':>3}{'视距':>7}{'PnPpx':>7}{'倾角':>7}"
        f"{'基座系xyz':>26}{'径向残差':>9}  用")
    for p in res["per_det"]:
        if "why" in p and not p.get("Pb"):
            say(f"{p['tag']:<14}{p['id']:>3}  —— 剔除：{p['why']}")
            continue
        say(f"{p['tag']:<14}{p['id']:>3}{p['depth_mm']:>7.0f}{p['pnp_rms_px']:>7.2f}"
            f"{p['tilt_deg']:>7.1f}  ({p['Pb'][0]:>7.1f},{p['Pb'][1]:>8.1f},{p['Pb'][2]:>7.1f})"
            f"{p['resid_mm']:>+9.2f}  {'✓' if p['used'] else '✗ ' + p['why']}")
    say(f"\n★ 杯轴 xy（直线交点法·主结果）= ({res['axis_xy'][0]:.1f}, {res['axis_xy'][1]:.1f}) mm"
        f"   |xy| = {np.hypot(*res['axis_xy']):.1f} mm")
    if "axis_xy_line" in res:
        say(f"  直线交点一致性：各码径向线到交点垂距 中位 {res['line_dist_med_mm']:.2f} / "
            f"最大 {res['line_dist_max_mm']:.2f} mm；方向张角 "
            f"{res.get('line_spread_deg', float('nan')):.0f}°（对 R/δ 免疫）")
    if "axis_xy_normal" in res:
        axn = res["axis_xy_normal"]
        say(f"  交叉核对·点法（Pb−R·n̂，对 R/δ 敏感）= ({axn[0]:.1f}, {axn[1]:.1f})"
            f"   与主结果差 {np.hypot(res['axis_xy'][0]-axn[0], res['axis_xy'][1]-axn[1]):.1f} mm"
            f"   点法散布 中位 {res.get('normal_spread_mm', float('nan')):.2f} / 最大 "
            f"{res.get('normal_spread_max_mm', float('nan')):.2f} mm")
    if "axis_xy_circle" in res:
        say(f"  交叉核对·圆心法（放大 28×，仅参考）= ({res['axis_xy_circle'][0]:.1f},"
            f" {res['axis_xy_circle'][1]:.1f})   与主结果差 "
            f"{np.hypot(res['axis_xy'][0]-res['axis_xy_circle'][0], res['axis_xy'][1]-res['axis_xy_circle'][1]):.1f} mm")
    say(f"  码平面 z = {res['z_markers']:.1f} mm（PnP 深度给的）"
        f"   [信息] 径向残差 RMS {res['rms_mm']:.2f} / 最大 {res['max_resid_mm']:.2f} mm")
    say(f"  杯轴倾角 中位 {res['tilt_deg']:.1f}° / 最大 {res['tilt_max_deg']:.1f}°（杯应直立）")
    say(f"  诊断：若以此推 TABLE_Z = {res['implied_table_z']:.1f} mm"
        f"（仅对照用，不写 config —— PnP 的 xy 与 z 完全解耦，不依赖它）")
    say(f"  验收闸：直线交点垂距 ≤ {PNP_LINE_DIST_PASS_MM}mm 且 方向张角 ≥ "
        f"{PNP_LINE_SPREAD_MIN_DEG:.0f}° 且 倾角 ≤ {PNP_TILT_GATE_DEG:.0f}° → "
        f"{'✓ 通过' if res['pass'] else '✗ 不过'}"
        f"   （径向 RMS ≤{PNP_RMS_PASS_MM}mm / 点法散布 ≤{PNP_NORMAL_SPREAD_PASS_MM}mm 为信息项）")
    return res["pass"]


# ---------------------------------------------------------------- 圆拟合（诊断对照）

def _roi_filter(shots, K, D, X, aim, r_px_gate, z_try, say=print):
    """剔除【不属于这只杯子】的检测。

    2026-10-16 23:5x 实锤的坑：orbit 拍到桌面上另一张打印卡片（上面也有
    DICT_4X4_50 的码），检出的 id=38/39 span只有 33~41px（真杯码 85~105px），
    圆心均值直接被拽偏 300mm，zm 自由解乱跳（473/405/424 三次三个数）。

    两道物理上成立的闸（不是拍脑袋阈值）：
      ① 归属闸：杯码中心必在【杯轴的投影】r_px_gate 像素内——码到杯轴只有 r0≈34.4mm，
         在 330mm 视距/fx=1462 下约 150px。卡片上的码离杯轴几百毫米，必被剔。
      ② 尺寸离群闸：同一批里 span 偏离中位数太多的剔掉（杯子不动、视距受控，
         真码的 span 应彼此接近；卡片/远处打印物会明显偏小）。
    返回被剔除的明细，供报告留痕。
    """
    dropped = []
    for sh in shots:
        T = sh["T"]
        inv = np.linalg.inv(T @ X)
        Pb = inv @ np.array([aim[0], aim[1], z_try, 1.0])
        if Pb[2] <= 0:
            continue
        uv_aim, _ = cv2.projectPoints(Pb[:3], np.zeros(3), np.zeros(3), K, D)
        uv_aim = uv_aim.ravel()
        keep = []
        for det in sh["dets"]:
            dv = float(np.linalg.norm(np.asarray(det["center"], float) - uv_aim))
            if dv > r_px_gate:
                dropped.append({"tag": sh["tag"], "id": det["id"],
                                "center_px": list(np.round(det["center"], 1)),
                                "dist_to_cup_px": round(dv, 1), "why": "不在杯轴附近"})
                continue
            keep.append(det)
        sh["dets"] = keep
    spans = [d["span_px"] for sh in shots for d in sh["dets"]]
    if len(spans) >= 3:
        med = float(np.median(spans))
        for sh in shots:
            keep = []
            for det in sh["dets"]:
                if det["span_px"] < 0.60 * med:
                    dropped.append({"tag": sh["tag"], "id": det["id"],
                                    "center_px": list(np.round(det["center"], 1)),
                                    "span_px": det["span_px"], "why": f"尺寸离群(<{0.6*med:.0f}px)"})
                    continue
                keep.append(det)
            sh["dets"] = keep
    if dropped:
        say(f"  · ROI/尺寸闸剔除 {len(dropped)} 个非杯码："
            + " ".join(f"[{d['tag']}:id{d['id']}:{d['why']}]" for d in dropped))
    return dropped


def solve(shots, K=None, D=None, X=None, say=print, free_zm=False):
    """shots: [{"T": 4×4, "dets": [...], "tag": str}]

    返回 dict：
      axis_xy        杯轴基座系 (x, y)（主结果）
      z_markers      码平面高度 z 拟合值（free_zm 时）/ 标称值（退化时）
      rms_mm         拟合残差 RMS（对 r0 的径向偏差）
      per_det        [{tag, id, center_px, resid_mm, cand_xy}] 逐检测明细
      r0_mm / zm_free / rms_free / rms_fixed / zm_free_shift  诊断字段
      n_det / n_shot
    """
    if K is None:
        K, D, X = load_calib()
    r0 = marker_circle_r()

    # 归属/尺寸闸先跑——不过闸的检测根本不该进拟合（见 _roi_filter 文档）
    z_m = marker_plane_z()
    dropped = _roi_filter(shots, K, D, X, np.asarray(config.CUP_AXIS_XY, float),
                          r_px_gate=320.0, z_try=z_m, say=say)

    rays = []                                        # (o, d, shot_idx, det)
    for si, sh in enumerate(shots):
        for det in sh["dets"]:
            o, d = base_ray(sh["T"], X, K, D, det["center"])
            rays.append((o, d, si, det))
    n_all = len(rays)
    rays = [ry for ry in rays if abs(ry[1][2]) >= 0.15]   # 掠射射线【拟合前去重】
    # 2026-10-16 教训：瞄准位相机 z≈480 ≈ 码面 482 → 射线近水平，拟合病态甚至全灭。
    # 掠射帧（|d_z|<0.15）对平面求交只贡献噪声——直接算丢，不参与残差（原先罚 50 会被拉平）。
    n_det = len(rays)
    res = {"r0_mm": r0, "n_det": n_det, "n_all": n_all, "n_grazing": n_all - n_det,
           "n_dropped_roi": len(dropped),
           "n_shot": sum(1 for sh in shots if sh["dets"]), "per_det": []}
    if n_det < 3:
        res["error"] = (f"可用射线 {n_det} 条 <3（共检测 {n_all}，掠射丢 {n_all - n_det}）"
                        f"，拟合不了（挪俯视位/确认码在视野里）")
        return res

    def resid_fn(params, z):
        ax, ay = params
        out = []
        for o, d, _si, _det in rays:
            q = _ray_plane_xy(o, d, z)
            if q is None:
                out.append(50.0)                     # 掠射的踢出去（不该发生）
                continue
            out.append(float(np.hypot(q[0] - ax, q[1] - ay) - r0))
        return np.array(out)

    # 初值：每条检测各自回推 r0 的方向（水平指离相机）→ 候选均值
    cand0 = []
    for o, d, _si, _det in rays:
        q = _ray_plane_xy(o, d, z_m)
        if q is None:
            continue
        u = q - o[:2]
        nu = np.linalg.norm(u)
        cand0.append(q + r0 * u / nu if nu > 1e-6 else q)
    if not cand0:
        res["error"] = "全部射线掠射——位姿不对？"
        return res
    a0 = np.mean(cand0, axis=0)

    from scipy.optimize import least_squares
    sol_fix = least_squares(lambda p: resid_fn(p, z_m), a0, method="trf")
    rms_fix = float(np.sqrt((sol_fix.fun ** 2).mean()))

    if free_zm:
        # ---- 全局 z 扫描 + 局部精修（2026-10-16 23:5x 修）----
        # 旧实现把初值 a0 和 bounds 全锚在 config 的实测码平面 z 上：
        # 一旦真值离标称 >50mm（杯垫高/台架改过），a0 就偏出收敛域，3 参最小二乘
        # 掉进局部极小 → 报 zm=405/RMS 11.4 这种"看着有解其实是坑"的结果。
        # 改成：对 z 做一维粗扫（每点内层只拟合 a,b，2 参便宜），取全局最优 z
        #     再从该点起 3 参精修。粗扫给的是全局 basin，不依赖标称初值。
        def fit_ab(z, starts):
            pts = []
            for o, d, _si, _det in rays:
                q = _ray_plane_xy(o, d, z)
                if q is not None:
                    pts.append(q)
            if len(pts) < 3:
                return None
            P = np.asarray(pts)

            def f2(p):
                return np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) - r0
            best = None
            for st in starts:
                try:
                    s = least_squares(f2, st, method="trf")
                except Exception:                                # noqa: BLE001
                    continue
                r = float(np.sqrt((s.fun ** 2).mean()))
                if best is None or r < best[0]:
                    best = (r, float(s.x[0]), float(s.x[1]))
            return best

        starts = [a0, np.asarray(config.CUP_AXIS_XY, float)]
        z_lo = z_m - 200.0
        z_hi = z_m + 120.0
        bestz = None
        for z in np.arange(z_lo, z_hi + 1e-9, 4.0):
            b = fit_ab(float(z), starts)
            if b and (bestz is None or b[0] < bestz[1][0]):
                bestz = (float(z), b)
        # 用上一轮解更新 starts（沿扫描推进），让 3 参精修从真 basin 起
        starts = [np.array([bestz[1][1], bestz[1][2]]), a0]

        def r3(p):
            return resid_fn(p[:2], p[2])
        z0 = bestz[0]
        sol3 = least_squares(r3, [bestz[1][1], bestz[1][2], z0], method="trf",
                             bounds=([a0[0] - 300, a0[1] - 300, z0 - 12.0],
                                     [a0[0] + 300, a0[1] + 300, z0 + 12.0]))
        rms_free = float(np.sqrt((sol3.fun ** 2).mean()))
        zm_free = float(sol3.x[2])
        shift = abs(zm_free - z_m)
        # 判据：自由拟合多一个参数本来就会降残差——降幅要明显（>0.10mm）、
        # 位移合理（<200mm）、样本够多（≥6 条）才采信；否则 zm 不可辨，用标称值。
        use_free = (rms_free < rms_fix - 0.10) and shift < 200.0 and n_det >= 6
        res.update(zm_free=zm_free, rms_free=rms_free, rms_fixed=rms_fix,
                   zm_free_shift=shift, using_free=bool(use_free))
        if use_free:
            ax, ay, zm = float(sol3.x[0]), float(sol3.x[1]), zm_free
            rms = rms_free
        else:
            ax, ay, zm = float(sol_fix.x[0]), float(sol_fix.x[1]), z_m
            rms = rms_fix
    else:
        ax, ay, zm = float(sol_fix.x[0]), float(sol_fix.x[1]), z_m
        rms = rms_fix

    for o, d, si, det in rays:
        q = _ray_plane_xy(o, d, zm)
        if q is None:
            rd = float("nan")
        else:
            rd = float(np.hypot(q[0] - ax, q[1] - ay) - r0)
        u = (q - o[:2]) if q is not None else np.zeros(2)
        nu = float(np.linalg.norm(u))
        res["per_det"].append({
            "tag": shots[si]["tag"], "id": det["id"],
            "center_px": [round(float(det["center"][0]), 1), round(float(det["center"][1]), 1)],
            "span_px": round(float(det["span_px"]), 1),
            "resid_mm": round(rd, 2),
            "cand_xy": [round(float(q[0] + r0 * u[0] / nu), 1),
                        round(float(q[1] + r0 * u[1] / nu), 1)] if nu > 1e-6 else None})
    cands = np.array([p["cand_xy"] for p in res["per_det"] if p["cand_xy"]])
    res.update(axis_xy=(ax, ay), z_markers=zm, rms_mm=rms,
               cand_spread_mm=float(max(np.linalg.norm(a - b) for a in cands for b in cands))
               if len(cands) > 1 else 0.0)
    # 交叉核对：能跨帧配对时（每帧内同 ID 不重复且跨帧重复）另跑三角测量
    counts = {}
    for sh in shots:
        for det in sh["dets"]:
            counts[det["id"]] = counts.get(det["id"], 0) + 1
    res["tri_check"] = _tri_crosscheck(shots, K, D, X, say)
    return res


def _tri_crosscheck(shots, K, D, X, say):
    """仅在"同 ID 跨帧、且每帧内同 ID 至多一条"时才可跑（本杯 6×ID1 时不适用）。"""
    per = {}
    for si, sh in enumerate(shots):
        seen = set()
        for det in sh["dets"]:
            if det["id"] in seen:
                return None                        # 同帧同 ID 重复 → 不可配对
            seen.add(det["id"])
            per.setdefault(det["id"], []).append(base_ray(sh["T"], X, K, D, det["center"]))
    out = {}
    for i, rs in per.items():
        if len(rs) < 2:
            continue
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for o, d in rs:
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ o
        P3, *_ = np.linalg.lstsq(A, b, rcond=None)
        out[i] = [round(float(v), 2) for v in P3]
    return out or None


# ---------------------------------------------------------------- 报告
def report(res, say=print):
    if res.get("error"):
        say(f"✗ {res['error']}")
        return False
    say(f"检测 {res['n_det']} 条（{res['n_shot']} 帧有码）  r0 = {res['r0_mm']:.2f} mm"
        f"  码平面 z = {res['z_markers']:.1f}（= TABLE_Z {config.TABLE_Z} + {config.MARKER_Z_ABOVE_TABLE:g}）")
    say(f"{'帧':<12} {'ID':>3} {'中心px':>16} {'边长px':>7} {'残差mm':>7}  单条候选xy")
    for p in res["per_det"]:
        cand = f"({p['cand_xy'][0]:7.1f},{p['cand_xy'][1]:7.1f})" if p["cand_xy"] else "—"
        say(f"{p['tag']:<12} {p['id']:>3} ({p['center_px'][0]:6.0f},{p['center_px'][1]:5.0f})"
            f" {p['span_px']:>7.1f} {p['resid_mm']:>7.2f}  {cand}")
    if res.get("n_grazing"):
        say(f"（掠射丢弃 {res['n_grazing']}/{res['n_all']} 条——瞄准位近水平射线的锅，俯视位应趋零）")
    if "zm_free" in res:
        say(f"拟合：zm 自由解 {res['zm_free']:.1f}mm（残差 {res['rms_free']:.2f}） vs "
            f"zm={config.TABLE_Z}+{config.MARKER_Z_ABOVE_TABLE:g} 固定（残差 {res['rms_fixed']:.2f}） → "
            f"{'采信自由解(仅诊断,不可作测量值)' if res['using_free'] else '采信实测 z 锚'}")
    say(f"\n★ 杯轴 xy = ({res['axis_xy'][0]:.1f}, {res['axis_xy'][1]:.1f}) mm"
        f"   码平面 z = {res['z_markers']:.1f} mm   拟合 RMS = {res['rms_mm']:.2f} mm")
    say(f"  逐条候选最大两两差 {res['cand_spread_mm']:.2f} mm（含单帧方位散布，粗指标）")
    if res.get("tri_check"):
        say(f"  三角测量交叉核对（ID 可分时）：{res['tri_check']}")
    return res["rms_mm"] < 1.5


# ---------------------------------------------------------------- CLI
def _load_shot(jsonp, det=None):
    meta = json.loads(Path(jsonp).read_text(encoding="utf-8"))
    imgp = Path(jsonp).parent / meta["image"]
    img = cv2.imread(str(imgp))
    if img is None:
        sys.exit(f"读不到图：{imgp}")
    T = M.pose6_to_T([float(v) for v in meta["pose6"]])
    return {"T": T, "dets": detect_ex(img, det), "tag": meta.get("tag", Path(jsonp).stem),
            "image": str(imgp)}


def main():
    args = sys.argv[1:]
    legacy = "--legacy" in args
    args = [a for a in args if a != "--legacy"]
    if not args:
        sys.exit(__doc__)
    jsons = []
    for a in args:
        p = Path(a)
        jsons.extend([q for q in sorted(p.glob("*.json"))
                      if not q.name.startswith("result")]   # result*.json 是输出，不是 shot
                     if p.is_dir() else [p])
    if not jsons:
        sys.exit("没找到 *.json（见文件头用法）")
    det = Detector()
    K, D, X = load_calib()
    shots = []
    for jp in jsons:
        sh = _load_shot(jp, det)
        shots.append(sh)
        ids = [d["id"] for d in sh["dets"]]
        print(f"[{sh['tag']:<12}] 检出 {len(ids)} 码 {sorted(ids)}")
    if legacy:
        print("\n———— 旧法：射线 ∩ z=TABLE_Z+40 平面 + 圆拟合（诊断对照，已降级）————")
        res = solve(shots, K, D, X)
        ok = report(res)
    else:
        print("\n———— PnP 主方法（solvePnP + 柱面模型；不吃 TABLE_Z）————")
        res = solve_pnp(shots, K, D, X)
        ok = report_pnp(res)
    out = jsons[0].parent if jsons[0].suffix == ".json" else jsons[0]
    try:
        (Path(out) / "result_pnp.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
