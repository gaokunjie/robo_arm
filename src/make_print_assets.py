# -*- coding: utf-8 -*-
"""生成打印素材：杯子 ArUco 码（枚数/边长取 config，2026-10-15 起 = 6 枚 25mm 60° 均布）
+ 相机标定 ChArUco 板。

为什么用 254 DPI：254/25.4 = 10.00 像素/毫米，是个整数。
这样码边长就是整百像素，A4 就是整 2100×2970 像素——
打印出来"标称尺寸 = 真实尺寸"，不用在纸上再换算一次误差。

生成后脚本会自己把图重新检测一遍并量尺寸，量错了直接报错，
所以能进打印机的图一定是尺寸正确的。

杯子的码边长取决于【码所在高度】处的杯身外径——注意不是杯身中部外径，
套取高度在杯身下部，比中部细。默认值直接按 config 里的杯身轮廓在套取高度处内插。

运行：
    python3 -I src/make_print_assets.py                              # 用 config 的默认值
    python3 -I src/make_print_assets.py --charuco-only               # 只出标定板
    python3 -I src/make_print_assets.py --cup-od 72.5                # 手动给贴码处杯径
    python3 -I src/make_print_assets.py --cup-od 72.5 --marker-mm 45 # 手动指定码尺寸
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config                                              # noqa: E402
import geometry                                            # noqa: E402

# ---------------- 版面常数 ----------------
DPI = 254.0                       # 正好 10 像素/毫米
PXMM = DPI / 25.4                 # = 10.0
A4_W, A4_H = 2100, 2970           # 210×297mm @ 10px/mm
BLACK, WHITE = 0, 255

# Noto Sans CJK 的 .ttc 里打包了 JP/KR/SC/TC/HK 五套，index=2 是简体。
# （别用 DroidSansFallback：它缺 ASCII 字形，数字会印成一排方块。）
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_INDEX = 2


def mm(px):
    """像素 → 毫米。"""
    return px / PXMM


def font(size_px):
    return ImageFont.truetype(FONT_PATH, size_px, index=FONT_INDEX)


def pick_marker_mm(od_at_marker):
    """按【码所在高度】的杯身外径选码边长。

    规则：码宽在杯身上包住的圆心角不超过 90°。按**弧长**算（纸不会拉伸，码两条
    竖边之间的弧长恒等于码宽）：包角 = 码宽 ÷ 半径 ≤ π/2
        → 周长 π·D ≥ 4 × 码宽  →  D ≥ (4/π) × 码宽 ≈ 1.273 × 码宽。
    （旧版按弦长 2·asin(w/D) 推出 1.41，是错的：极端例子 D = w 时弦长公式给 180°，
      实际只有 114.6°。）

    注意传的是【码所在高度】的杯径，不是杯身中部杯径——套取高度不在中部，两者差很多。
    """
    for size in (50.0, 45.0, 40.0):
        if od_at_marker >= (4.0 / math.pi) * size:
            return size
    return None


# ---------------- 打印素材 1：杯子 ArUco 码 ----------------
def make_aruco_page(marker_id, marker_mm, od_at_marker, print_scale=1.0):
    """A4 一页：多枚同 ID 的码（ARUCO_MARKERS_PER_CUP 枚均布用）
    + 一条【一格弧长】的定位条（重复贴 n 次绕一圈画 n 条线）。

    print_scale 只影响印在纸上的说明文字（告诉用户这页有没有做缩放补偿），
    真正的缩放由 apply_print_scale 在出图前统一做。
    """
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    side_px = int(round(marker_mm * PXMM))
    marker = cv2.aruco.drawMarker(d, marker_id, side_px)      # 黑框边缘即"码边长"

    canvas = np.full((A4_H, A4_W), WHITE, np.uint8)

    # 三枚码横排
    n = config.ARUCO_MARKERS_PER_CUP
    gap = 140
    total = n * side_px + (n - 1) * gap
    x0 = (A4_W - total) // 2
    my = 620
    for k in range(n):
        xx = x0 + k * (side_px + gap)
        canvas[my:my + side_px, xx:xx + side_px] = marker

    img = Image.fromarray(canvas).convert("RGB")
    dr = ImageDraw.Draw(img)

    # 标题
    dr.text((120, 90), f"杯子 ArUco 码  ·  {config.ARUCO_DICT}  ·  ID {marker_id}  ·  共 {n} 枚",
            font=font(56), fill=(0, 0, 0))
    dr.text((120, 176), f"码边长（黑框外沿）= {marker_mm:.1f} mm —— 打印后请用尺子复核",
            font=font(38), fill=(180, 0, 0))
    dr.text((120, 232), f"（按【码所在高度】处的杯身外径 {od_at_marker:.1f} mm 选定；"
                        f"此时码在杯身上包住 {math.degrees(marker_mm / (od_at_marker / 2)):.0f}°）",
            font=font(34), fill=(90, 90, 90))

    _dim_line(dr, x0, my - 90, x0 + side_px, my - 90, f"{marker_mm:.1f} mm")

    # 120° 弧长定位条：印成一条可以整条剪下来的实体，比拿尺子量好使
    arc = math.pi * od_at_marker / n                            # 相邻两枚码之间的弧长
    sy = my + side_px + 250
    dr.text((120, my + side_px + 170),
            f"定位条：长度 {arc:.1f} mm（= 贴码高度处周长 {math.pi * od_at_marker:.1f} ÷ {n}）",
            font=font(36), fill=(0, 0, 0))
    _cut_strip(dr, 120, sy, arc)
    dr.text((120, sy + 150),
            f"用法：沿虚线把这条【整条剪下来】，绕在杯身上、对准贴码高度，在两头各画一条竖线（这就是两枚码的中心位置）；",
            font=font(32), fill=(60, 60, 60))
    dr.text((120, sy + 195),
            f"然后以其中一条线为起点再贴一次、再画线；绕完一圈共 {n} 条线，最后一条应正好落回起点附近。",
            font=font(32), fill=(60, 60, 60))

    # 供核对打印比例用的 100mm 尺（和上面那条是两回事，别搞混）
    dr.text((120, sy + 290), "下面这根是【打印比例校验尺】，全长必须是 100.0 mm，跟定位条无关：",
            font=font(32), fill=(0, 0, 0))
    _scale_bar(dr, 120, sy + 350, 1000, 42)
    # 竖向的那根：贴在右边空白走廊（x≥1855 从 y≈1150 到 2230 全空，已逐像素验过）。
    # 起点挂在 markers 下沿下方 80px，这样改码尺寸时跟着走、不会撞上三枚码。
    # cap_dy=-88：这页尺子上方有一整段空白（y 1106~1155px 已逐像素验空），
    # 把「纵向」抬到独立一行，免得和最上面的刻度「0」连读成「纵向0」。
    bar_v_mm = _scale_bar_v(dr, 1855, my + side_px + 80, 1000, cap_dy=-88)

    # 说明
    _notes(dr, [
        "打印设置",
        "  纸张 A4 ｜ 缩放 100%（选『实际大小』，千万不要选『适应可打印区域』）｜ 黑白即可",
        "  建议用【不干胶标签纸】打印，直接撕下来贴，比涂胶水服帖。",
        "",
    ] + print_scale_note(print_scale, bar_v_mm) + [
        "",
        "贴到杯子上",
        f"  高度：{n} 枚码的中心必须在【同一个高度】，且 = 套取高度"
        f"（现定在离杯底 {config.INSERT_PLANE_Z_ABOVE_TABLE:.0f} mm，改高度要重打本页）。",
        "  周向：用上面的定位条把三枚码按 120° 贴均匀。",
        "  朝向：码的【上边缘朝杯口】，三枚要保持一致（软件靠码的上下判断杯轴方向）。",
        "  贴完覆一层薄透明胶带，防止环滑过时把边角带翘。",
        f"  贴好后量『码中心离杯底多高』，填进 config 的 MARKER_Z_ABOVE_TABLE。",
        "",
        f"本页是 ID {marker_id}（{'左杯 A' if marker_id == config.ARUCO_ID_CUP_A else '右杯 B'}），"
        f"另一只杯子请打印 ID {config.ARUCO_ID_CUP_B if marker_id == config.ARUCO_ID_CUP_A else config.ARUCO_ID_CUP_A}。",
    ], 120, sy + 460)

    return img


# ---------------- 打印素材 1b：窄码版 ----------------
# 2026-10-15 定：50mm 码在 40mm 高处的包角是 79°，A1 整圈实测覆盖率只有 44%——
# 三枚 120° 根本轮不到一圈。合成平面码在同样的像素尺寸下能撑到 80°+，
# 所以瓶颈既不是斜角、也不是分辨率，而是【码被杯身弯掉】；射线-圆柱仿真复现了这一点。
# 修法只有一个方向：把码贴窄（包角小）、贴多（每枚要管的范围小）。包角 = 码宽 ÷ 杯半径：
#     25mm → 40°（仿真能读到 ±30°）      20mm → 32°（仿真 ±35°）
# 6 枚 60° 均布每枚只需管 ±30°；8 枚 45° 均布每枚只需管 ±22.5°（余量更大，做备用）。
NARROW_PRIMARY_MM = 25.0
NARROW_PRIMARY_N = 6
NARROW_BACKUP_MM = 20.0
NARROW_BACKUP_N = 8


def make_narrow_aruco_page(marker_id, od_at_marker, primary_mm=NARROW_PRIMARY_MM,
                           n_primary=NARROW_PRIMARY_N, backup_mm=NARROW_BACKUP_MM,
                           n_backup=NARROW_BACKUP_N):
    """A4 一页：主用 N 枚窄码（一行）+ 备用 M 枚更窄码（两行）+ 两条定位条 + 40mm 高度尺。

    版面用的是"标称坐标"（10px/mm）；要不要按打印机反向补偿由 apply_print_scale 统一做，
    本函数不管。留白按 12mm 设计（外面还有 assert_blank_margins 兜底）。
    """
    assert n_backup % 2 == 0, "备用码要排两行，枚数必须是偶数"
    X0 = 150          # 正文/条/尺统一的左边距：120 会让居中刻度字和粗刻度线越界
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    canvas = np.full((A4_H, A4_W), WHITE, np.uint8)

    s1 = int(round(primary_mm * PXMM))
    s2 = int(round(backup_mm * PXMM))
    m1 = cv2.aruco.drawMarker(d, marker_id, s1)
    m2 = cv2.aruco.drawMarker(d, marker_id, s2)

    # 主用：一整行。缝 55px = 5.5mm ≥ 1 个模块（25/6=4.17mm），安静区够、下剪刀也够。
    gap1 = 55
    w1 = n_primary * s1 + (n_primary - 1) * gap1
    x1, y1 = (A4_W - w1) // 2, 430
    for k in range(n_primary):
        xx = x1 + k * (s1 + gap1)
        canvas[y1:y1 + s1, xx:xx + s1] = m1

    # 备用：两行，每行一半
    per = n_backup // 2
    gap2 = 70
    w2 = per * s2 + (per - 1) * gap2
    x2, y2 = (A4_W - w2) // 2, 810
    for r in range(2):
        yy = y2 + r * (s2 + 70)
        for k in range(per):
            xx = x2 + k * (s2 + gap2)
            canvas[yy:yy + s2, xx:xx + s2] = m2

    img = Image.fromarray(canvas).convert("RGB")
    dr = ImageDraw.Draw(img)

    wrap1 = math.degrees(primary_mm / (od_at_marker / 2))
    wrap2 = math.degrees(backup_mm / (od_at_marker / 2))
    dr.text((X0, 120), f"杯子 ArUco 码 · 窄码版 · {config.ARUCO_DICT} · ID {marker_id}",
            font=font(50), fill=(0, 0, 0))
    dr.text((X0, 200), f"主用 {n_primary} 枚 × {primary_mm:.1f} mm　＋　"
                        f"备用 {n_backup} 枚 × {backup_mm:.1f} mm　（黑框边长就是这个数）",
            font=font(34), fill=(180, 0, 0))
    dr.text((X0, 258), f"为什么改窄：50mm 码在贴码高度绕杯身 79°，实测只能读到 ±25°，"
                        f"三枚 120° 轮不到一圈（整圈覆盖率 44%）。",
            font=font(29), fill=(90, 90, 90))
    dr.text((X0, 296), f"换成窄码，包角降到 {wrap1:.0f}° / {wrap2:.0f}°：平面码同样的像素数能撑到 80°+，"
                        f"弯得越少读得越斜。",
            font=font(29), fill=(90, 90, 90))
    dr.text((X0, 365), f"【主用】{n_primary} 枚 × {primary_mm:.1f} mm："
                        f"每 {360 // n_primary}° 一枚，每枚只需管 ±{180 // n_primary}°——先贴这套",
            font=font(31), fill=(0, 0, 0))
    dr.text((X0, 745), f"【备用】{n_backup} 枚 × {backup_mm:.1f} mm："
                        f"每 {360 // n_backup}° 一枚，每枚只需管 ±{180 // n_backup}°——主用不够才用",
            font=font(31), fill=(0, 0, 0))

    # 定位条：长度 = 贴码高度处的【半个周长】，绕两轮就把一圈均分完（比"每次只量一格"少累积误差）
    circ = math.pi * od_at_marker
    half_mm = circ / 2.0
    half_px = int(round(half_mm * PXMM))
    dr.text((X0, 1330), f"定位条（剪下来用）：全长 {half_mm:.1f} mm = 周长 {circ:.1f} 的 ½；"
                         f"中心线对准离杯底 {config.MARKER_Z_ABOVE_TABLE:.1f} mm",
            font=font(31), fill=(0, 0, 0))
    ticks_a = [int(round(k * half_px / (n_primary / 2.0))) for k in range(n_primary // 2 + 1)]
    ticks_b = [int(round(k * half_px / (n_backup / 2.0))) for k in range(n_backup // 2 + 1)]
    _tick_strip(dr, X0, 1375, half_px, ticks_a)
    dr.text((X0, 1483), f"① {n_primary} 枚用：每 {half_mm / (n_primary / 2.0):.1f} mm 一条刻度"
                         f"（绕两轮画出 {n_primary} 条线）", font=font(27), fill=(60, 60, 60))
    _tick_strip(dr, X0, 1530, half_px, ticks_b)
    dr.text((X0, 1638), f"② {n_backup} 枚用：每 {half_mm / (n_backup / 2.0):.1f} mm 一条刻度"
                         f"（绕两轮画出 {n_backup} 条线）", font=font(27), fill=(60, 60, 60))

    # 40mm 高度尺：贴在定位条右边，用来在杯身上找"码中心高度"
    dr.text((1450, 1325), "40mm 高度尺", font=font(30), fill=(180, 0, 0))
    _height_gauge(dr, 1360, 1400, config.MARKER_Z_ABOVE_TABLE, w=120)

    # 打印比例校验尺（和定位条是两回事，别搞混）
    dr.text((X0, 1710), "打印比例校验尺：下面这根全长必须是 100.0 mm（下面 ① 要量它）",
            font=font(31), fill=(0, 0, 0))
    _scale_bar(dr, X0, 1750, 1000, 42)
    bar_v_mm = _scale_bar_v(dr, 1800, 1400, 1000, cap_dy=-42)

    _notes(dr, [
        "打印设置",
        "  纸张 A4 ｜ 缩放选【实际大小 / 100%】（别选『适应可打印区域』）｜ 黑白即可 ｜ 建议不干胶标签纸",
        "  本页【没有】做缩放补偿：按 100% 打出来就是标称尺寸。",
        "  （上一版画大 1/0.96 是因为那条打印链会缩 4%；本页直接按 100% 打，靠 ① 校验。）",
        "",
    ] + print_scale_note(1.0, bar_v_mm, extra_flags="--narrow ") + [
        "",
        "贴到杯子上",
        f"  高度：全部码的中心必须在【同一个高度】= 离杯底 {config.MARKER_Z_ABOVE_TABLE:.1f} mm"
        f"（用高度尺量，别目测）。",
        "  周向：定位条中心线对准那个高度，按刻度在杯身上画竖线，每枚码的中心压在一条线上。",
        "  朝向：每枚码都让【纸上这个方向】朝杯口，几枚保持一致（软件靠码的上下判杯轴方向）。",
        "  贴完覆一层薄透明胶带，防止环滑过时把边角带翘。",
        "  贴好后量『码中心离杯底多高』和一枚码的实际边长，告诉我，我更新 config。",
        "",
        f"本页是 ID {marker_id}（左杯 A）。等这轮尺寸定下来再出 ID "
        f"{config.ARUCO_ID_CUP_B}（右杯 B），免得白打一张。",
    ], X0, 1900, dy=40, size=31, max_px=1620)   # 120+1620=1740 < 竖尺 1800

    assert_blank_margins(img)
    return img


def _tick_strip(dr, x, y, length_px, tick_px, h=100):
    """可剪定位条：上下虚线（剪切线）+ 中点线（对准码中心高度）+ 若干竖刻度。"""
    for xx in range(x, x + length_px, 26):
        seg = min(16, x + length_px - xx)
        dr.line([(xx, y), (xx + seg, y)], fill=(120, 120, 120), width=2)
        dr.line([(xx, y + h), (xx + seg, y + h)], fill=(120, 120, 120), width=2)
    dr.rectangle([x, y + 6, x + length_px - 1, y + h - 6], outline=(0, 0, 0), width=3)
    ym = y + h // 2
    dr.line([(x, ym), (x + length_px, ym)], fill=(150, 150, 150), width=2)
    for t in tick_px:
        dr.line([(x + t, y + 8), (x + t, y + h - 8)], fill=(0, 0, 0), width=5)


def _height_gauge(dr, x, y, length_mm, w=60):
    """高度尺：顶边 = length_mm（红），底边 = 0（虚线，对盒顶/杯底）。"""
    L = int(round(length_mm * PXMM))
    dr.rectangle([x, y, x + w, y + L], outline=(0, 0, 0), width=2)
    dr.line([(x - 14, y), (x + w + 14, y)], fill=(180, 0, 0), width=5)
    for xx in range(x - 14, x + w + 14, 20):
        dr.line([(xx, y + L), (xx + min(12, x + w + 14 - xx), y + L)],
                fill=(120, 120, 120), width=2)
    f = font(26)
    dr.text((x - 8, y + L - 42), "0", font=f, fill=(60, 60, 60))
    dr.text((x + 10, y + 12), f"{length_mm:.1f} mm", font=f, fill=(180, 0, 0))
    return L


def _cut_strip(dr, x, y, length_mm, h=90):
    """一条正好 length_mm 长的可剪定位条：上下画虚线（剪切线），两头带刻度。"""
    L = int(round(length_mm * PXMM))
    # 上下剪切虚线
    for xx in range(x, x + L, 26):
        seg = min(16, x + L - xx)
        dr.line([(xx, y), (xx + seg, y)], fill=(120, 120, 120), width=2)
        dr.line([(xx, y + h), (xx + seg, y + h)], fill=(120, 120, 120), width=2)
    # 实体条文 + 两端标记
    dr.rectangle([x, y + 6, x + L - 1, y + h - 6], outline=(0, 0, 0), width=3)
    for xx in (x, x + L):
        dr.line([(xx, y + 6), (xx, y + h - 6)], fill=(180, 0, 0), width=4)
    # 端点标注
    f = font(32)
    dr.text((x - 6, y + h + 8), "0", font=f, fill=(180, 0, 0))
    lab = f"{length_mm:.1f}"
    dr.text((x + L - dr.textlength(lab, font=f) + 6, y + h + 8), lab, font=f, fill=(180, 0, 0))
    # 中点刻度（10mm 一格），方便顺手核对
    step = int(round(10 * PXMM))
    for k in range(1, L // step):
        xx = x + k * step
        dr.line([(xx, y + h - 26), (xx, y + h - 6)], fill=(0, 0, 0), width=2)


def _dim_line(dr, x0, y0, x1, y1, label):
    """带箭头的尺寸标注线。"""
    dr.line([(x0, y0), (x1, y1)], fill=(0, 0, 0), width=3)
    for x, s in ((x0, 1), (x1, -1)):
        dr.line([(x, y0), (x + 22 * s, y0 - 12)], fill=(0, 0, 0), width=3)
        dr.line([(x, y0), (x + 22 * s, y0 + 12)], fill=(0, 0, 0), width=3)
    f = font(40)
    w = dr.textlength(label, font=f)
    dr.text(((x0 + x1) / 2 - w / 2, y0 - 62), label, font=f, fill=(0, 0, 0))


def _scale_bar(dr, x, y, length_px, h):
    """黑白相间的比例尺，每 10mm 一格。（length_px 取 10mm 的整数倍）"""
    step = int(round(10 * PXMM))
    for k, xx in enumerate(range(x, x + length_px, step)):
        dr.rectangle([xx, y, xx + step - 1, y + h],
                     fill=(0, 0, 0) if k % 2 == 0 else (255, 255, 255),
                     outline=(0, 0, 0), width=2)
    f = font(34)
    nb = length_px // step
    for k in range(nb + 1):
        xx = x + k * step
        dr.line([(xx, y + h), (xx, y + h + 18)], fill=(0, 0, 0), width=3)
        if k % 5 == 0:
            lab = f"{k * 10}"
            w = dr.textlength(lab, font=f)
            dr.text((xx - w / 2, y + h + 24), lab, font=f, fill=(0, 0, 0))
    dr.text((x + length_px + 60, y + h + 24), f"mm（全长 {nb * 10}）", font=f, fill=(0, 0, 0))


def _scale_bar_v(dr, x, y, length_px, w=40, cap_dy=-42):
    """竖版比例尺，从上往下，每 10mm 一格。用来验证打印机【纵向】的缩放。

    为什么非要一根竖的：横尺只能量出 x 方向的缩放。如果打印机/驱动只缩横向
    （或横纵缩得不一样），ArUco 码会印成非正方形、标定板方格会变矩形——前者让
    位姿解算歪掉，后者直接把错误比例灌进手眼标定的平移量。一根横尺看不出来。

    x, y = 左上角；w = 尺身宽度（px）；length_px 取 10mm 的整数倍（不足一格的部分
    会被丢掉，返回值是【实际画出来的】总长，调用方要拿这个数去写说明文字）。
    刻度线和数字都在尺身【右侧】，所以左侧要留够空间、也别贴着正文。

    cap_dy：标题相对 y 的偏移。默认 -42 时标题会和最上面那个「0」挤在同一行
    （读起来像「纵向0」）；上面有空间就传更负的值把它抬到独立一行。
    """
    length_px = (length_px // int(round(10 * PXMM))) * int(round(10 * PXMM))
    step = int(round(10 * PXMM))
    nb = length_px // step
    for k in range(nb):
        yy = y + k * step
        dr.rectangle([x, yy, x + w - 1, yy + step - 1],
                     fill=(0, 0, 0) if k % 2 == 0 else (255, 255, 255),
                     outline=(0, 0, 0), width=2)
    f = font(28)
    for k in range(nb + 1):
        yy = y + k * step
        dr.line([(x + w, yy), (x + w + 16, yy)], fill=(0, 0, 0), width=3)
        dr.text((x + w + 26, yy - 17), f"{k * 10}", font=f, fill=(0, 0, 0))
    # 标题放在尺子【上方】——放下面会和正文/页边撞
    dr.text((x, y + cap_dy), "纵向", font=font(30), fill=(0, 0, 0))
    return nb * 10          # 返回标称总长（mm）


def _notes(dr, lines, x, y, dy=44, size=34, max_px=1700):
    """正文块。max_px 是单行宽度上限——右边有竖校验尺（老页在 x=1855、窄码版在 x=1800），
    排版写超了会撞上去，撞了不会报错、只会静默印歪，所以这里直接拦。"""
    f = font(size)
    for i, ln in enumerate(lines):
        w = dr.textlength(ln, font=f)
        assert x + w <= x + max_px, (
            f"正文第 {i + 1} 行太宽（{w:.0f}px > {max_px}），会撞上右边的竖尺：{ln[:24]}…")
        dr.text((x, y + i * dy), ln, font=f,
                fill=(0, 0, 0) if not ln.startswith("  ") else (60, 60, 60))


def print_scale_note(scale, bar_v_mm=100, extra_flags=""):
    """生成『打印后必做』那几行——按当前是否在做缩放补偿给出不同的话。

    篇幅是紧的：ArUco 页正文区到页面底只剩约 5 行余量，加行前先算版面。
    extra_flags 会插进「重新生成」那条命令里（窄码版要带 --narrow，不然会印出旧版面）。
    """
    head = ["打印后必做的一步（决定后面算得准不准，别跳）",
            f"  ① 量两根校验尺：横的那根 = 100.0 mm，右边竖的那根 = {bar_v_mm:.0f}.0 mm。"]
    if abs(scale - 1.0) > 1e-6:
        body = [
            f"  ② 本页已按 PRINT_SCALE={scale:.4f} 反向补偿：画的是标称的 {1/scale:.4f} 倍，",
            f"     你的打印机缩到 {scale*100:.1f}% 后落到纸上正好是标称长度。",
            "  ③ 两根都对 → 补偿生效。**换了打印机或改了打印设置必须重来**。",
            "     横对、竖不对（或两根偏差不一样）→ 打印机【非等比】缩放，别硬凑：",
            "       把横、竖两个实测长度都告诉我，补偿要改成横纵分开做。",
        ]
    else:
        body = [
            "  ② 不是标称长度 → 查看器/驱动把整页缩了。最常见的原因是选了",
            "     『适应可打印区域 / Fit to printable area』。改成【实际大小 / 100%】重打。",
            "  ③ 没有『实际大小』这个选项的话：记下横尺实测长度 L（mm），然后",
            f"     python3 -I src/make_print_assets.py {extra_flags}--print-scale <L÷100>   重新生成",
            "     例：量到 96.0mm → --print-scale 0.96",
        ]
    tail = ["  ④ 最后量 ArUco 黑框边长，应 = 标称值 ±0.3mm；不是就回 ② 重来。",
            "     差得不多也别将就——这个数直接决定后面算位置准不准。"]
    return head + body + tail


# ---------------- 打印素材 2：ChArUco 标定板 ----------------
def make_charuco_page(squares=None, square_mm=None, marker_mm=None):
    """A4 一页 ChArUco 板，用于标相机内参（参数默认值单一出处 = config.CHARUCO_*）。

    方格 25mm × 7列5行 = 175×125mm，正好放进 A4 还剩白边。内角点 6×4 = 24 个。
    """
    squares = squares or config.CHARUCO_SQUARES
    square_mm = config.CHARUCO_SQUARE_MM if square_mm is None else square_mm
    marker_mm = config.CHARUCO_MARKER_MM if marker_mm is None else marker_mm
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.CHARUCO_DICT))
    board = cv2.aruco.CharucoBoard_create(squares[0], squares[1], square_mm, marker_mm, d)

    bw = int(round(squares[0] * square_mm * PXMM))        # 1750
    bh = int(round(squares[1] * square_mm * PXMM))        # 1250
    bimg = np.full((bh, bw), WHITE, np.uint8)
    board.draw((bw, bh), bimg, marginSize=0, borderBits=1)   # 10px/mm，方格正好 250px

    canvas = np.full((A4_H, A4_W), WHITE, np.uint8)
    bx, by = (A4_W - bw) // 2, 780
    canvas[by:by + bh, bx:bx + bw] = bimg

    img = Image.fromarray(canvas).convert("RGB")
    dr = ImageDraw.Draw(img)

    dr.text((120, 90), "相机标定板 ChArUco  ·  用于标内参 + 手眼",
            font=font(56), fill=(0, 0, 0))
    dr.text((120, 176), f"方格 {square_mm:.1f} mm × {squares[0]}列 × {squares[1]}行"
                        f"　小码 {marker_mm:.1f} mm　字典 {config.CHARUCO_DICT}",
            font=font(38), fill=(180, 0, 0))

    _dim_line(dr, bx, by - 90, bx + bw, by - 90, f"整板宽 {squares[0] * square_mm:.1f} mm")

    bar_y = by + bh + 200
    dr.text((120, bar_y - 90), "比例尺：下面这根尺子全长必须是 100.0 mm",
            font=font(38), fill=(0, 0, 0))
    _scale_bar(dr, 120, bar_y, 1000, 46)
    # 竖向尺：右边空白区（x≥1855、y 2053~2908 全空，已逐像素验过；板底边在 2052px，
    # 所以标题从 2063 起）。长度取 75mm 而不是 80：最后那个刻度的数字以 yy-17 为顶，
    # 会往下挂约 18px，80mm 的话末位数字正好越过 apply_print_scale 的裁剪线。
    bar_v_mm = _scale_bar_v(dr, 1855, 2105, 750)

    _notes(dr, [
        "打印设置",
        "  纸张 A4 ｜ 缩放 100%（选『实际大小』）｜ 【务必用哑光纸】，亮面纸反光会让标定点飘。",
        "  有硬卡纸/雪弗板就先裱上去，板面越平整标定越准。",
        "",
        "打印后必做的一步（决定标定准不准）",
        f"  量两根比例尺：横的 100.0 mm、右边竖的 {bar_v_mm:.0f}.0 mm，不是就重打。",
        "  ⚠ 横竖【分别】量：只缩一个方向 = 非等比缩放，方格变矩形，标定比例就是错的。",
        "    别硬凑，把两个实测值告诉我。板不能有折痕，卷了就用双面胶贴到硬板上。",
        "",
        "怎么用（P3 会给你详细步骤）",
        "  举着这块板在相机前慢慢晃，覆盖画面各个角落和各个倾角，采 30~40 张。",
    ], 120, bar_y + 140)

    return img, bw, bh, square_mm


# ---------------- 自校验 ----------------
def apply_print_scale(img, scale):
    """把整页按 1/scale 放大、再居中裁回 A4——抵消打印机自己的缩放。

    scale = 打印机实测的缩放比（量打印页上的 100mm 校验尺得到 L，scale = L/100）。
    页面画大一点，打印缩回去正好落到标称尺寸。

    放大后四周多出来的边被裁掉——**但被裁的地方必须是空白**，否则说明版面真出界了，
    那就有内容会丢，必须报错而不是闷声印出去。
    """
    if abs(scale - 1.0) < 1e-6:
        return img
    k = 1.0 / scale
    w, h = img.size
    big = img.resize((int(round(w * k)), int(round(h * k))), Image.LANCZOS)
    left, top = (big.width - w) // 2, (big.height - h) // 2

    arr = np.asarray(big.convert("L"))
    strips = []
    if top > 0:
        strips += [arr[:top, :], arr[top + h:, :]]
    if left > 0:
        strips += [arr[:, :left], arr[:, left + w:]]
    if strips:
        darkest = min(int(s.min()) for s in strips)
        assert darkest > 200, (
            f"放大 {k:.4f} 倍后有内容被裁掉（残边最暗像素 {darkest}/255）——"
            f"PRINT_SCALE={scale} 让版面出界了。请先把打印设置改成『实际大小 / 100%』"
            f"再重打，别靠补偿硬凑。")

    return big.crop((left, top, left + w, top + h))


def verify_aruco(img, marker_id, marker_mm):
    """把生成的页面重新检测一遍，量尺寸。"""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, d)
    assert ids is not None, "自校验失败：生成的页面上检测不到 ArUco 码"
    ids = np.asarray(ids).ravel()
    n = int((ids == marker_id).sum())
    assert n == config.ARUCO_MARKERS_PER_CUP, \
        f"自校验失败：期望 {config.ARUCO_MARKERS_PER_CUP} 枚 ID {marker_id}，检测到 {n} 枚"
    idx = int(np.where(ids == marker_id)[0][0])
    pts = corners[idx].reshape(-1, 2)
    sides_mm = [mm(float(np.linalg.norm(pts[k] - pts[(k + 1) % 4]))) for k in range(4)]
    mean_mm = sum(sides_mm) / 4
    err = abs(mean_mm - marker_mm)
    print(f"   自校验 ID {marker_id}: 检出 {n} 枚, 首枚四边 "
          f"{[round(s, 2) for s in sides_mm]} mm, 均值 {mean_mm:.2f} mm, 与标称差 {err:.2f} mm")
    assert err < 0.6, f"自校验失败：码边长 {mean_mm:.2f}mm 与标称 {marker_mm}mm 差太多"
    return mean_mm


def assert_blank_margins(img, margin_px=120):
    """四边各留 margin_px（默认 12mm）纯白——内容出界要在印之前拦住。

    打印机打不到边：A4 四周有 4~5mm 不可打印区。排版一旦压线，纸面上就是切边，
    而且不报错。这里直接按像素验，比"看版面坐标"可靠。
    """
    a = np.asarray(img.convert("L"))
    for name, strip in (("上", a[:margin_px, :]), ("下", a[-margin_px:, :]),
                        ("左", a[:, :margin_px]), ("右", a[:, -margin_px:])):
        d = int(strip.min())
        assert d > 200, (
            f"版面出界：{name}边 {mm(margin_px):.0f}mm 以内有内容（最暗 {d}/255）。"
            f"打印机会切边，先把版面收进来。")


def verify_narrow_aruco(img, marker_id, primary_mm, n_primary, backup_mm, n_backup):
    """窄码版自校验：同一 ID 有两组尺寸，按实测边长分组点数、分别核尺寸。"""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, d)
    assert ids is not None, "自校验失败：生成的页面上检测不到 ArUco 码"
    ids = np.asarray(ids).ravel()
    sel = [c for c, i in zip(corners, ids) if int(i) == marker_id]
    assert len(sel) == n_primary + n_backup, \
        f"自校验失败：期望 {n_primary + n_backup} 枚 ID {marker_id}，检测到 {len(sel)} 枚"
    mid = (primary_mm + backup_mm) / 2.0
    sizes = []
    for c in sel:
        p = c.reshape(-1, 2)
        sides = [float(np.linalg.norm(p[k] - p[(k + 1) % 4])) for k in range(4)]
        sizes.append(mm(sum(sides) / 4))
    for name, group, want, cnt in (("主用", [s for s in sizes if s > mid], primary_mm, n_primary),
                                   ("备用", [s for s in sizes if s <= mid], backup_mm, n_backup)):
        m = sum(group) / len(group) if group else 0.0
        print(f"   自校验 {name}: 检出 {len(group)}/{cnt} 枚，平均边长 {m:.2f} mm "
              f"(标称 {want:.1f} mm，差 {abs(m - want):.2f} mm)")
        assert len(group) == cnt, f"自校验失败：{name} 应为 {cnt} 枚，实测 {len(group)} 枚"
        assert abs(m - want) < 0.6, f"自校验失败：{name} 边长 {m:.2f}mm ≠ {want:.1f}mm"
    mn = min(sizes)
    assert mn > 0.9 * backup_mm, f"自校验失败：最小边长只有 {mn:.2f}mm"


def _cluster_1d(vals, tol_px):
    """把一串数值按 tol_px 的容差聚成簇（单链接），返回每簇均值，升序。

    用来把"同一列/同一行的角点"归并成一个坐标。容差取 2px 是因为方格间距在
    250px 量级，而需要合并的抖动 <0.01px、需要分开的间距 >200px，中间空得很。
    """
    v = np.sort(np.asarray(vals, dtype=float))
    out, cur = [], [v[0]]
    for x in v[1:]:
        if x - cur[-1] <= tol_px:
            cur.append(x)
        else:
            out.append(float(np.mean(cur)))
            cur = [x]
    out.append(float(np.mean(cur)))
    return np.array(out)


def verify_charuco(img, squares, square_mm):
    """检测板上的小码 + 插值角点，量方格间距。"""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.CHARUCO_DICT))
    board = cv2.aruco.CharucoBoard_create(squares[0], squares[1], square_mm,
                                          config.CHARUCO_MARKER_MM, d)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, d)
    assert ids is not None, "自校验失败：标定板上检测不到小码"
    n_exp = squares[0] * squares[1] // 2
    ret, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    assert ci is not None and len(ci) > 0, "自校验失败：插值不出 ChArUco 角点"
    cc = cc.reshape(-1, 2)
    # 同一列的角点 x 不会严格相等：整页放大 1/PRINT_SCALE 倍时经过 LANCZOS 重采样，
    # 同列四个角点的 x 会差 0.0001~0.003px。所以必须按【容差聚类】，不能 round 后
    # unique —— round 合并不掉这种亚像素差，会得出一串 0mm 的假间距。
    xs = _cluster_1d(cc[:, 0], tol_px=2.0)
    gaps = [mm(float(g)) for g in np.diff(xs)][:5]
    print(f"   自校验 标定板: 检测到 {len(ids)}/{n_exp} 个小码, 插值出 {len(ci)} 个角点")
    print(f"   自校验 角点排成 {len(xs)} 列")
    assert gaps, f"自校验失败：角点没聚成多列（只有 {len(xs)} 列），量不出方格间距"
    print(f"   自校验 方格间距: {[round(g, 2) for g in gaps]} mm  (标称 {square_mm} mm)")
    assert abs(gaps[0] - square_mm) < 0.6, f"自校验失败：方格 {gaps[0]:.2f}mm ≠ {square_mm}mm"
    return len(ci)


# ---------------- 主流程 ----------------
def save(img, stem):
    png = config.PRINT_DIR / f"{stem}.png"
    pdf = config.PRINT_DIR / f"{stem}.pdf"
    img.save(png, dpi=(DPI, DPI))
    img.convert("RGB").save(pdf, "PDF", resolution=DPI)
    print(f"   写出 {png.name}  ({img.width}×{img.height}px = {mm(img.width):.0f}×{mm(img.height):.0f}mm)")
    print(f"   写出 {pdf.name}")
    return png


def main():
    ap = argparse.ArgumentParser(description="生成打印素材")
    ap.add_argument("--cup-od", type=float, default=None,
                    help="【贴码高度】处的杯身外径（mm）。不给就按 config.CUP_PROFILE "
                         "在 config.INSERT_PLANE_Z_ABOVE_TABLE 处内插")
    ap.add_argument("--marker-mm", type=float, default=config.ARUCO_MARKER_MM,
                    help="码边长（mm）。不给就按 --cup-od 自动选")
    ap.add_argument("--charuco-only", action="store_true", help="只出标定板")
    ap.add_argument("--narrow", action="store_true",
                    help="出【窄码版】整页（2026-10-15 起的主线）：主用窄码 + 备用窄码 + "
                         "两条定位条 + 40mm 高度尺，只出这一页")
    ap.add_argument("--primary-mm", type=float, default=NARROW_PRIMARY_MM,
                    help=f"窄码版主用码边长（默认 {NARROW_PRIMARY_MM:.0f}）")
    ap.add_argument("--n-primary", type=int, default=NARROW_PRIMARY_N,
                    help=f"窄码版主用枚数（默认 {NARROW_PRIMARY_N}）")
    ap.add_argument("--backup-mm", type=float, default=NARROW_BACKUP_MM,
                    help=f"窄码版备用码边长（默认 {NARROW_BACKUP_MM:.0f}）")
    ap.add_argument("--n-backup", type=int, default=NARROW_BACKUP_N,
                    help=f"窄码版备用枚数，偶数（默认 {NARROW_BACKUP_N}）")
    ap.add_argument("--print-scale", type=float, default=config.PRINT_SCALE,
                    help="打印机实测的缩放比（量打印页上 100mm 校验尺得到 L，填 L/100）。"
                         "默认取 config.PRINT_SCALE。例：量到 96.0mm → 0.96")
    args = ap.parse_args()
    ps = args.print_scale
    if not (0.8 <= ps <= 1.25):
        print(f"  ✗ --print-scale={ps} 超出合理范围（0.8~1.25）。"
              f"先量准那根 100mm 校验尺再来。")
        return 1

    config.PRINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"打印素材 · {DPI:.0f} DPI = {PXMM:.2f} 像素/毫米, A4 = {A4_W}×{A4_H}px "
          f"({mm(A4_W):.0f}×{mm(A4_H):.0f}mm)")
    if abs(ps - 1.0) > 1e-6:
        print(f"  ⚠ PRINT_SCALE = {ps:.4f}：整页会放大 {1/ps:.4f} 倍输出，"
              f"用来抵消你打印机的 {ps*100:.1f}% 缩放")
        print(f"    换打印机/改打印设置后必须调回 1.0 并重打，否则会静默印出偏大的图")

    # ---- 窄码版（A1 整圈实测覆盖率 44% 之后的修正版）----
    if args.narrow:
        print("\n[窄码版] 主用窄码（一行）+ 备用窄码（两行）+ 双定位条 + 40mm 高度尺")
        cup_od = args.cup_od
        if cup_od is None and config.MARKER_Z_ABOVE_TABLE is not None:
            h = config.MARKER_Z_ABOVE_TABLE
            cup_od, inrange = geometry.cup_od_at(h)
            print(f"  贴码高度 {h:.0f}mm → 该处杯径 {cup_od:.1f} mm（按 config.CUP_PROFILE 内插）")
            if not inrange:
                print("  ⚠ 这个高度落在实测轮廓点之外，内插结果不可信，先补测杯径")
        if cup_od is None:
            print("  ✗ 缺【贴码高度】处的杯身外径，用 --cup-od <直径> 给。")
            return 1
        n_p, n_b = args.n_primary, args.n_backup
        if args.narrow and (n_p < 3 or n_b < 4 or n_b % 2):
            print(f"  ✗ 枚数不合理：主用 {n_p}（≥3）、备用 {n_b}（≥4 且为偶数）")
            return 1
        for label, mm_, n in (("主用", args.primary_mm, n_p), ("备用", args.backup_mm, n_b)):
            wrap = math.degrees(mm_ / (cup_od / 2))
            need = 180.0 / n
            print(f"  {label} {n} 枚 × {mm_:.1f}mm：包角 {wrap:.1f}°，"
                  f"每枚要管 ±{need:.1f}° —— 紧不紧看 A1 整圈实测")
        total_mm = n_p * args.primary_mm
        print(f"  贴码总占用弧长 {total_mm:.0f}mm / 周长 {math.pi * cup_od:.1f}mm"
              f"（剩 {(math.pi * cup_od - total_mm) / n_p:.1f}mm/缝）")
        mid = config.ARUCO_ID_CUP_A
        img = make_narrow_aruco_page(mid, cup_od, args.primary_mm, n_p, args.backup_mm, n_b)
        img = apply_print_scale(img, ps)
        if abs(ps - 1.0) > 1e-6:
            print(f"   已按 PRINT_SCALE={ps:.4f} 补偿：本页画出 {args.primary_mm / ps:.2f}/"
                  f"{args.backup_mm / ps:.2f}mm，落到纸上正好 {args.primary_mm:.2f}/"
                  f"{args.backup_mm:.2f}mm")
        verify_narrow_aruco(img, mid, args.primary_mm / ps, n_p, args.backup_mm / ps, n_b)
        save(img, f"aruco_{config.ARUCO_DICT}_id{mid}_narrow_"
                  f"{args.primary_mm:.0f}mm_x{n_p}_{args.backup_mm:.0f}mm_x{n_b}")
        print("\n完成（只出了窄码版这一页）。")
        return 0

    # ---- 杯子 ArUco 码 ----
    if not args.charuco_only:
        print("\n[1/2] 杯子 ArUco 码")
        cup_od, marker_mm = args.cup_od, args.marker_mm
        if cup_od is None and config.INSERT_PLANE_Z_ABOVE_TABLE is not None:
            h = config.INSERT_PLANE_Z_ABOVE_TABLE
            cup_od, inrange = geometry.cup_od_at(h)
            print(f"  贴码高度 {h:.0f}mm → 该处杯径 {cup_od:.1f} mm"
                  f"（按 config.CUP_PROFILE 内插）")
            if not inrange:
                print("  ⚠ 这个高度落在实测轮廓点之外，内插结果不可信，先补测杯径")
        if marker_mm is None and cup_od is not None:
            marker_mm = pick_marker_mm(cup_od)
            if marker_mm is None:
                print(f"  ✗ 贴码处杯径 {cup_od:.1f} mm 太细：就算用最小的 40mm 码，"
                      f"码也会绕杯身超过 90°。换个粗点的杯子，或把码往上贴。")
                return 1
            print(f"  按杯径 {cup_od:.1f} mm 自动选定码边长 {marker_mm:.0f} mm"
                  f"（包角 {geometry.marker_wrap_deg(config.INSERT_PLANE_Z_ABOVE_TABLE, marker_mm):.1f}°）")
        if cup_od is None or marker_mm is None:
            print("  ✗ 缺参数，跳过（不生成，免得打错重打）。")
            print("    需要【贴码高度】处的杯身外径——不是杯身中部外径，两者差很多。")
            print("    二选一：")
            print("      python3 -I src/make_print_assets.py --cup-od <贴码处直径>")
            print("      python3 -I src/make_print_assets.py --cup-od <直径> --marker-mm <边长>")
            print("    选码尺寸的规则（包角 ≤ 90°，按弧长算）：")
            for s in (50, 45, 40):
                print(f"      杯径 ≥ {(4 / math.pi) * s:5.1f} mm  →  {s} mm 码")
            return 1
        for mid in (config.ARUCO_ID_CUP_A, config.ARUCO_ID_CUP_B):
            print(f"  ID {mid}:")
            img = make_aruco_page(mid, marker_mm, cup_od, ps)
            img = apply_print_scale(img, ps)
            if abs(ps - 1.0) > 1e-6:
                print(f"   已按 PRINT_SCALE={ps:.4f} 补偿：本页画出 {marker_mm/ps:.2f}mm，"
                      f"打印机缩到 {ps*100:.1f}% 后落到纸上正好 {marker_mm:.2f}mm")
            verify_aruco(img, mid, marker_mm / ps)
            save(img, f"aruco_{config.ARUCO_DICT}_id{mid}_{marker_mm:.0f}mm_x{config.ARUCO_MARKERS_PER_CUP}")

    # ---- 标定板 ----
    print("\n[2/2] ChArUco 标定板")
    img, _bw, _bh, sq = make_charuco_page()
    img = apply_print_scale(img, ps)
    if abs(ps - 1.0) > 1e-6:
        print(f"   已按 PRINT_SCALE={ps:.4f} 补偿：本页方格画出 {sq/ps:.2f}mm，"
              f"落到纸上正好 {sq:.2f}mm")
    verify_charuco(img, config.CHARUCO_SQUARES, sq / ps)
    save(img, f"charuco_{config.CHARUCO_SQUARES[0]}x{config.CHARUCO_SQUARES[1]}_"
              f"{sq:.0f}mm_A4")

    print("\n完成。打印时记得：纸张 A4、缩放 100%、标定板用哑光纸。")
    print("打完后**先用尺子量那根 100mm 比例尺**，确认长度对了再往下走。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
