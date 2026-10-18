# -*- coding: utf-8 -*-
"""单臂取放 · 几何自检（纯计算，不碰机械臂、不碰相机）

把 config 里的实测尺寸代进"开口环套杯"的几何约束，回答四个问题：

  1. 谁在卡？—— 是环内径(93) 还是 缺口宽度(85)？
  2. 环平面最低能插多高？（插入窗口的硬上限）
  3. 每个候选插入高度还剩多少精度预算？（= 允许多大的定位误差）
  4. 码贴多高、贴多大？

随时可跑：  python3 -I src/geometry.py
改完 config 里的实测值再跑一遍，所有下游数字自动跟着变。
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

MM = "mm"


# ============================================================================
# 杯身轮廓
# ============================================================================

def cup_od_at(z):
    """杯身外径 d(z)：按 CUP_PROFILE 线性内插。

    z 超出轮廓范围时按端点值截断（不外推——外推出来的数是编的）。
    返回 (外径, 是否在实测范围内)。
    """
    prof = sorted(C.CUP_PROFILE)
    if z <= prof[0][0]:
        return prof[0][1], z == prof[0][0]
    if z >= prof[-1][0]:
        return prof[-1][1], z == prof[-1][0]
    for (z0, d0), (z1, d1) in zip(prof, prof[1:]):
        if z0 <= z <= z1:
            t = (z - z0) / (z1 - z0)
            return d0 + t * (d1 - d0), True
    return prof[-1][1], False


def height_where_od_equals(target_od):
    """反解：杯身外径等于 target_od 的高度。轮廓外返回 None。"""
    prof = sorted(C.CUP_PROFILE)
    lo, hi = prof[0], prof[-1]
    if not (lo[1] <= target_od <= hi[1]):
        return None
    for (z0, d0), (z1, d1) in zip(prof, prof[1:]):
        if min(d0, d1) <= target_od <= max(d0, d1) and d1 != d0:
            return z0 + (target_od - d0) * (z1 - z0) / (d1 - d0)
    return None


# ============================================================================
# 核心约束
# ============================================================================

def ring_band(h):
    """环平面中面在高度 h 时，环在竖直方向占的区间 [h-t/2, h+t/2]。"""
    half = C.RING_THICK / 2.0
    return h - half, h + half


def insert_budget(h):
    """环平面中面高度 h 处插入时，**单边**的净余量（mm）。

    约束链：
      · 插入瞬间杯轴与环心重合，杯身要穿过缺口 → 缺口内杯身最大外径必须 ≤ 缺口宽度
      · 环有厚度，杯身在环的整个厚度带内都占位 → 取带内最粗处（= 环顶边处）
      · 码有厚度，杯身 + 码才是真实占位 → 再扣掉码厚
    余量 ≤ 0 表示这个高度根本插不进去。
    """
    _, z_top = ring_band(h)
    od, _ok = cup_od_at(z_top)
    return (C.RING_GAP - od - 2.0 * C.MARKER_THICK) / 2.0


def insert_headroom(h):
    """同上，但不扣码厚——用来比较"纯几何"余量。"""
    _, z_top = ring_band(h)
    od, _ok = cup_od_at(z_top)
    return (C.RING_GAP - od) / 2.0


def wedge_plane_height():
    """环升到多高会被杯身挡住（= 卡缘，杯子从这里开始被抬走）。

    环内孔是 93：当环顶边所在高度的杯身外径长到 93，环就再也升不上去，
    杯子被顶起来。返回环**中面**的高度。
    """
    z = height_where_od_equals(C.RING_ID)
    return None if z is None else z - C.RING_THICK / 2.0


def wedge_plane_height_upper():
    """卡缘高度的悲观上界：假设杯缘是外翻唇边、杯身到杯口才长到 93。"""
    z = height_where_od_equals(C.RING_ID)
    if z is None:
        return None
    top = max(p[0] for p in C.CUP_PROFILE)
    z_worst = top if C.CUP_PROFILE[-1][1] >= C.RING_ID else z
    return z_worst - C.RING_THICK / 2.0


def marker_wrap_deg(z, w):
    """宽 w 的码贴在直径 d(z) 处，在杯身上包住的圆心角（度）。

    用**弧长**算，不是弦长：纸不会拉伸，码两条竖边之间的弧长恒等于码宽，
    所以包角 = w / 半径（弧度制）。
    （按弦长 2·asin(w/D) 会高估：极端例子 D = w 时弦长公式给 180°，
      实际是 360×50/(π×50) = 114.6°。早期版本用的就是弦长，已改。）
    """
    od, _ok = cup_od_at(z)
    if od <= 0:
        return 0.0
    return math.degrees(w / (od / 2.0))


def marker_size_rule(od):
    """按"包角 ≤ 90°"选码边长。

    包角 = w/(od/2) ≤ π/2  →  od ≥ (2/π)·w ≈ 1.273·w
    """
    for size in (50.0, 45.0, 40.0):
        if od >= (2.0 / math.pi) * size:
            return size
    return None


# ============================================================================
# 报告
# ============================================================================

def main():
    print("=" * 74)
    print("单臂取放 · 几何自检")
    print("=" * 74)
    print(f"环：内径 {C.RING_ID} / 外径 {C.RING_OD} / 缺口 {C.RING_GAP} / 厚 {C.RING_THICK}")
    print(f"杯：高 {C.CUP_HEIGHT} / 底 {C.CUP_OD_BOTTOM} / 中 {C.CUP_OD_MID} "
          f"/ 口 {C.CUP_OD_RIM} / 码厚 {C.MARKER_THICK}")
    print(f"轮廓点：{C.CUP_PROFILE}")
    print()

    # ---- 1. 谁在卡 ----
    print("─" * 74)
    print("[1] 谁在卡？")
    print("─" * 74)
    if C.RING_GAP < C.RING_ID:
        print(f"  缺口 {C.RING_GAP} < 环内径 {C.RING_ID}"
              f"  →  插入受【缺口】卡，抬升受【环内径】卡，两个数不是一回事：")
        print(f"      · 插入：杯身要挤过缺口 → 插入处杯身外径必须 < {C.RING_GAP}")
        print(f"      · 抬升：杯身进到环内孔后 → 长到 {C.RING_ID} 就被挡住，这就是【卡缘】")
    else:
        print(f"  环内径 {C.RING_ID} ≤ 缺口 {C.RING_GAP} → 内径是唯一卡点，"
              f"剧本公式 (内径-杯径)/2 成立")
    print()

    # ---- 2. 插入窗口 ----
    print("─" * 74)
    print("[2] 插入高度窗口（环中面离杯底的高度）")
    print("─" * 74)
    h_hard = None
    half = C.RING_THICK / 2.0
    z_hard = height_where_od_equals(C.RING_GAP)
    if z_hard is not None:
        h_hard = z_hard - half
        print(f"  硬上限：杯身长到 {C.RING_GAP}（= 缺口宽）是在杯底往上 "
              f"{z_hard:.1f}mm 处")
        print(f"          → 环中面必须低于 {h_hard:.1f}mm，再高一度都插不进去")
    ok_lip = C.CUP_OD_RIM > C.RING_GAP
    print(f"  杯口外径 {C.CUP_OD_RIM} vs 缺口 {C.RING_GAP}："
          f"{'杯缘出不了缺口 ✓（杯子被兜住，不会从侧面漏出去）' if ok_lip else '✗ 杯子会从缺口滑出去！'}")
    print(f"  杯口外径 {C.CUP_OD_RIM} vs 环内径 {C.RING_ID}："
          f"{'✓ 杯缘进不了内孔，抬升时必被挡住' if C.CUP_OD_RIM > C.RING_ID else '✗ 杯子会整个穿过环，卡不住'}")
    print()

    prof = sorted(C.CUP_PROFILE)
    print("  杯身轮廓（实测点之间线性内插）：")
    for z, d in prof:
        print(f"    z = {z:6.1f}mm  →  外径 {d:6.1f}mm")
    print()

    # ---- 3. 精度预算表 ----
    print("─" * 74)
    print("[3] 各候选插入高度的精度预算  =  (缺口 − 插入处杯径 − 2×码厚) ÷ 2")
    print("─" * 74)
    print(f"  {'环中面高':>9} {'环顶边处杯径':>13} {'单边余量':>10} {'扣码厚后':>10}"
          f" {'50mm包角':>10} {'45mm包角':>10} {'按规则选码':>10}")
    print(f"  {'─'*9:>9} {'─'*13:>13} {'─'*10:>10} {'─'*10:>10}"
          f" {'─'*10:>10} {'─'*10:>10} {'─'*10:>10}")
    for h in (90, 80, 70, 60, 50, 45, 40, 35, 30, 25):
        zb, zt = ring_band(h)
        od_top, inrange = cup_od_at(zt)
        pure = insert_headroom(h)
        net = insert_budget(h)
        w50 = marker_wrap_deg(h, 50.0)
        w45 = marker_wrap_deg(h, 45.0)
        pick = marker_size_rule(cup_od_at(h)[0])
        mark = "*" if net <= 0 else " "
        note = ""
        if net <= 0:
            note = "插不进"
        print(f"  {h:>7.0f}mm {od_top:>11.1f}mm {pure:>9.2f}mm {net:>9.2f}mm "
              f"{w50:>9.1f}° {w45:>9.1f}° {str(pick)+'mm' if pick else '换杯子':>10}{mark}{'  '+note if note else ''}")
    print()
    print("  读法：『扣码厚后』那一列 = 允许的定位误差（单边）。认「≥ 4mm 舒服 /")
    print("        3~4mm 能试 / < 3mm 别碰」这条线，插入高度就定下来了。")
    print()

    # ---- 4. 卡缘与抬升行程 ----
    print("─" * 74)
    print("[4] 卡缘高度与抬升行程")
    print("─" * 74)
    wp = wedge_plane_height()
    wu = wedge_plane_height_upper()
    if wp is not None:
        print(f"  杯身长到 {C.RING_ID}（= 环内径）在杯底往上 "
              f"{height_where_od_equals(C.RING_ID):.1f}mm 处")
        print(f"  → 环中面升到 {wp:.1f}mm 就被挡住，杯子从这里开始被顶起来")
        if wu is not None and abs(wu - wp) > 1:
            print(f"  ⚠ 这只是**按实测点线性内插**的结果。杯缘那段（{prof[-2][0]:.0f}→"
                  f"{prof[-1][0]:.0f}mm）的真实形状没量过：")
            print(f"      越早变粗 → 越早卡住（{wp:.1f}）；越晚变粗 → 越晚卡住，理论极值 {wu:.1f}")
            print(f"      → P1 手工套一次，眼睛看着环停在杯身哪里，量出来填 "
                  f"WEDGE_PLANE_Z_ABOVE_TABLE")
        print(f"  → 自由行程 = 卡缘高 − 插入高；再加 20mm 让杯底离桌")
        for h in (50, 45, 40, 35, 30):
            lift = (wp - h) + 20.0
            lift_hi = (wu - h) + 20.0 if wu else lift
            warn = "  ⚠ 超单条 100mm 上限，得拆两段" if lift_hi > C.MAX_MOVE_MM else ""
            print(f"      插入 {h}mm → 抬升 {lift:.0f}~{lift_hi:.0f}mm{warn}")
    print()

    # ---- 5. 码 ----
    print("─" * 74)
    print("[5] 码贴多高、多大")
    print("─" * 74)
    for h in (45, 40, 35):
        od, _ = cup_od_at(h)
        pick = marker_size_rule(od)
        if pick is None:
            print(f"  插入 {h}mm（杯径 {od:.1f}）→ 约束①判死：杯太细，换杯子")
            continue
        w = marker_wrap_deg(h, C.ARUCO_MARKER_MM)
        print(f"  插入 {h}mm（杯径 {od:.1f}）→ 约束①（包角 ≤90°）上限 {od/(2/math.pi):.1f}mm；"
              f"当前用 {C.ARUCO_MARKER_MM:.0f}mm，包角 {w:.1f}° ✓")
    print()
    print(f"  约束②（2026-10-15 A1 实测后加的硬约束）：**单枚码的可读窗口 ≥ 相邻码间隔**。")
    print(f"    实测：50mm 码（包角 79°）只读到 ±25°；25mm 码（包角 39.5°）读到 ±45°。")
    print(f"    当前 {C.ARUCO_MARKERS_PER_CUP:.0f} 枚均布 ⇒ 间隔 {360/C.ARUCO_MARKERS_PER_CUP:.0f}°，"
          f"窗口 ±45° = 90° ≥ 间隔 ✓（留了一倍余量）")
    # 裁条模板：按**码所在高度**的周长算，不是按杯身中部——用户是照这个条子贴码的
    for h in (45, 40, 35):
        od, _ = cup_od_at(h)
        circ = math.pi * od
        print(f"  贴码高度 {h}mm：周长 π×{od:.1f} = {circ:.1f}mm → 每枚管 "
              f"{circ/C.ARUCO_MARKERS_PER_CUP:.1f}mm（{360/C.ARUCO_MARKERS_PER_CUP:.0f}° 弧长）；"
              f"裁条模板 = 半周长 {circ/2:.1f}mm")
    print(f"  （杯身中部 {C.CUP_OD_MID}mm 处的周长是 {math.pi*C.CUP_OD_MID:.1f}mm，"
          f"只在码贴中部时才用这个数）")
    print()
    print("=" * 74)


if __name__ == "__main__":
    main()
