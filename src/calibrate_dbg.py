# -*- coding: utf-8 -*-
"""批次 A · A2 相机内参标定（ChArUco 板，手举着拍）+ 批次 D · 手眼标定

A2 要回答的问题：**这台相机的 fx/fy/cx/cy 和畸变系数是多少？**
手眼标定（批次 D）、像素↔毫米换算（批次 E）全部建立在这组数上，
所以判据只有一条：重投影误差 **< 0.5px**（规划 §0.5）。

D 要回答的问题：**相机装歪了多少？**（法兰→相机的固定变换 X）。
判据：留一法交叉验证 **中位 < 2mm**（规划 D 批）。手眼要拿 A2 的内参当输入。

板 = src/make_print_assets.py 打印的那张：7 列 × 5 行、方格 25mm、小码 18mm、DICT_6X6_250。
⚠ 板必须**哑光、平整、按 1.0 打**；打印后量两根比例尺（横的 100.0mm、右边竖的 75.0mm），
   横竖要分别量——只缩一个方向 = 非等比缩放，方格变矩形，标定比例就是错的。

用法：
    python3 -I src/calibrate.py --capture        # 举着板在相机前晃，回车存一帧、q 结束
    python3 -I src/calibrate.py --solve          # 算内参 → 打印重投影误差 → 存 yaml
    python3 -I src/calibrate.py --handeye        # 手眼采集：拖臂摆位姿，回车存（不发运动指令）
    python3 -I src/calibrate.py --handeye-solve  # 解 AX=XB → 交叉验证 → 存 data/hand_eye.yaml

⚠ **必须带 `-I` 跑**（本项目所有脚本的惯例，见下面 _make_board() 的注释）：
   带 -I 走系统 OpenCV 4.6 的 aruco 老 API；不带 -I 会加载用户目录的 4.10（新 API）。
   两条路都写了，但别混着用——打印板子的 make_print_assets.py 只会在 4.6 下跑通。

拍摄协议（30~40 张）：
    · 板要**贴在硬底上、握着硬底拍**（雪弗板/亚克力/木板/硬卡纸，四角胶带贴牢）。
      ⚠ 别直接手举裁切垫或裸 A4：软垫被手一捏就弯，倾角一大弯得更狠——第四批（2026-10-15
      16:15）就是这么废的：整批斜着拍，垫子在照片里肉眼可见地拱起来，每视角误差
      0.380→0.566px、cy 从 572 漂到 722。合成实测弓 1.5mm → RMS 1.7~2.2px、畸变错 200%
    · 板要占画面够大：正对时板宽 ≥600px（离相机 ~40cm 上下）。
      合成实测：板宽 274px 的平板只能标到 0.5px 上下（勉强），415px 就 0.05px
    · 板要走到画面【四角】**和上下两半**——畸变靠边角约束，主点（尤其 cy）靠画面上下都有板。
      实测两批板心全在画面下半部（y 0.43~0.87），cy 一个 572 一个 722；举高/举远补上半部
    · 远近各来几档；**绕横竖轴各 ±30~40° 的斜视 ≥8 张**（看状态行的倾角读数）。这条最要紧：
      板一正对，焦距和径向畸变在投影里就几乎等价、互相顶替，fx 能飘 8~10%
      （第二批 50 张倾角中位 8° → 奇偶两半 fx 差 3.1%、换畸变模型跳到 1713）
    · 不反光、不糊。终端实时给角点数/板心位置/板的大小/模糊度**/倾角/距离**，认不出角点的帧按回车会被拒
    · 每张之间板动一动，别在同一个位姿连拍

    ⚠ 倾角/距离是拿 data/camera_intrinsics.yaml 里**已有的内参**当"瞄准镜"估的（solvePnP）：
      K 差 ±1.5% 只让倾角偏 ±0.3°（2026-10-15 实测），指导"转到 30°"够用；
      但它只是瞄准镜，判据以 --solve 的结果为准。头一次拍（还没有 yaml）就不显示倾角。

⚠ 相机是独占的（config.CAMERA_DEV）：跑之前先关掉 camera_stream.py 等占相机的进程。
"""

import argparse
import math
import os
import select
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                     # noqa: E402

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("需要 opencv-python 和 numpy")

SHOTS_DIR = config.DATA_DIR / "calib_shots"      # 存帧
OUT_YAML = config.DATA_DIR / "camera_intrinsics.yaml"

MIN_CORNERS = 12          # 一帧至少认出这么多内角点才算数（板上一共 24 个）
MIN_VIEWS_HARD = 10       # 有效视图少于这个数，直接拒绝标定（出来的是垃圾）
MIN_VIEWS_SOFT = 20       # 少于这个数照算，但提醒回去补拍（规划要 30~40）
BLUR_WARN = 150.0         # 板面区域的拉普拉斯方差低于它 → 提示"偏糊"（只提示，不拦）
OUTLIER_FACTOR = 3.0      # 单视图误差 > 中位数 × 这个数 → 报为离群视图
# 以下三条是 2026-10-15 用"已知真值"的合成板标出来的经验线（见 memory: calib-bow-size-limits）：
BOARD_GOOD_PX = 600.0     # 想要的正对板宽。合成：274px 平板只到 RMS 0.5，415px → 0.05
BOARD_MIN_PX = 450.0      # 板宽中位低于它 → 焦距/畸变定不准，实时提示和 --solve 都按它判
SMOOTH_WARN_PX = 0.5      # 每视角重投影误差中位超它 → 板不平/弯了（跟内参无关的硬指标）。
                          # 合成：平板 0.03~0.11px，弓 1.5mm → 1.0~1.6px；废批那 104 张 = 1.03px
STAB_WARN_PCT = 1.0       # 奇偶帧各标一半，两半的 fx 差超过它 → 焦距没被钉住（倾斜不够）。
                          # 合成实测：倾角 ±38° 的两半差 0.05%；第二批（倾角中位 8°）差 3.1%
TILT_GOAL_DEG = 25.0      # "大倾斜"的线（--solve 也按它数）
TILT_GOAL_N = 8           # 一批里想要这么多张 >25° 的斜视——焦距全靠它们钉住。
                          # 实测：全正对的一批 fx 在 1404~1713 间飘（8~10%），倾角够的合成批只差 0.05%

HE_SHOTS_DIR = config.DATA_DIR / "handeye_shots"      # 手眼存帧 + flange_poses.csv
HE_OUT_YAML = config.DATA_DIR / "hand_eye.yaml"
HE_POSES_CSV = "flange_poses.csv"     # 存帧目录里：帧文件名 + 当帧法兰位姿（一行一张）

HE_HARD_SHOTS = 10        # 少于这么多张直接拒绝解（AX=XB 的底线，规划写 10~15）
HE_MIN_SHOTS = 12         # 采集界面的进度分母
HE_BOARD_MIN_PX = 350.0   # 板宽低于它不收：手眼是拿每张的板姿拼出来的，板小 = 板姿不准
HE_BOARD_GOOD_PX = 500.0  # 建议值（约 350~550mm 距离）
HE_STILL_MM = 0.3         # 存入前连读两次法兰位姿，位置差超过它 = 臂还在晃，拒收
HE_STILL_DEG = 0.15       # 同上，姿态差（度）
HE_SIMILAR_DEG = 8.0      # 与已有位姿的转角小于它 → 提示"太像了，白占一张"
HE_CROSS_MM = 2.0         # 交叉验证判据（规划 D 批：< 2mm），中位数按它判；这是【地板】不是全部判据
HE_SPLIT_BOOT = 25        # 分区解的噪声带重采样次数。
                          # 2026-10-15 实测：真机角点残差 ~0.45px 是【白噪声】（留一半拟合光滑
                          # 曲面预测另一半，留出 RMS ≥ 总 RMS，四批数据 + 仿真对照全部如此），
                          # 而 7v7 分区解对这种噪声的放大是 ~15 倍：0.45px → 分区差中位 5~7mm。
                          # 所以固定 2mm 阈值只有"角点干净到 0.1px"时才可达（合成数据的水平），
                          # 真机永远过不了 → 必须按数据自己的噪声定带宽（见 _he_split_boot）。
HE_LOO_METHOD = "Park"    # 写进 yaml / 算交叉验证用的解法（合成真值比过五种，见离线测试）
AXIS_MIN_EIG = 0.02       # 转动轴散布最小特征值，低于它 = 轴基本共面（AX=XB 平移不可观测）
AXIS_WEAK_EIG = 0.08      # 偏弱提醒线（还不至于解不了；补哪一类看缺口方向，见 _he_gap_advice）
HE_NEED_CORNERS = 24      # 手眼帧必须"板拍全"。缺角的板姿朝向会被带偏：2026-10-15 那批 16 张里
                          # 有 10 张只认出 12~18 个角点，轴对应残差 1.9~3.5°（合成 0.00°），整批报废
HE_EDGE_MARGIN_PX = 60    # 板角点包围盒离画面边至少这么多像素（贴边的角点定位不准）
HE_DUP_MM = 0.3           # 与上一张法兰位置差 < 它、姿态差 < HE_DUP_DEG → 判为重复帧、拒存
HE_DUP_DEG = 0.2          # （多按一下回车就会发生；上批 006/007 就是这么来的）
HE_AXIS_GOOD_DEG = 0.7    # 轴对应残差中位（数据自洽性，X-free）：合成 0.00°、仿真 0.12°，缺角那批 2.96°
HE_AXIS_BAD_DEG = 1.5     # 高于它 → 判"数据本身不自洽"：先修数据，其它判据都还轮不到谈
HE_SCALE_MAX_SIG = 0.02   # 板尺寸自检的误差棒宽过它就判"这次看不出"（不挡，转由卡尺兜底）：
                          # 全瞄准的位姿集实测 ±2.5%（4% 的错都在噪声里），补了不转腕平移后 ±0.3%。


# ---------------------------------------------------------------- 板与检测
# ⚠ cv2 有两套 aruco API，而本机两个版本都装着：
#     `python3 -I`（本项目所有脚本的跑法）→ 系统 OpenCV 4.6.0：老 API
#          CharucoBoard_create / interpolateCornersCharuco
#     不加 -I  → 用户目录里的 4.10.0：新 API CharucoDetector，
#          且已经删掉 CharucoBoard_create（拿不到的属性会 AttributeError）
#   两条路都写在这儿，但**实际跑的是 4.6 那条**。2026-10-15 实测：两个版本渲染/识别的
#   板子图样逐像素相同（见 memory: opencv-dual-install-aruco），所以混着也不算错；
#   但打印板子的 make_print_assets.py 只认老 API，统一用 -I 省得给自己挖坑。


def _make_board():
    """→ (board, 角点物体坐标 (N,3) float32, 字典)。"""
    nx, ny = config.CHARUCO_SQUARES
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.CHARUCO_DICT))
    if hasattr(cv2.aruco, "CharucoBoard_create"):                 # OpenCV ≤ 4.6
        board = cv2.aruco.CharucoBoard_create(nx, ny, config.CHARUCO_SQUARE_MM,
                                              config.CHARUCO_MARKER_MM, dic)
        objp = board.chessboardCorners                            # 老 API 是属性
    else:                                                         # OpenCV ≥ 4.7
        board = cv2.aruco.CharucoBoard((nx, ny), config.CHARUCO_SQUARE_MM,
                                       config.CHARUCO_MARKER_MM, dic)
        if hasattr(board, "setLegacyPattern"):   # 打印出来的那张板是 4.6 的老图样；
            board.setLegacyPattern(True)         # 本机 4.10 实测两种图样逐像素相同，
        objp = board.getChessboardCorners()      # 设一下只是把话说明白
    # 角点 id 就是 objp 的下标 → 检测出来的 id 直接拿来索引物体坐标，天然对齐
    return board, np.asarray(objp, dtype=np.float32).reshape(-1, 3), dic


class BoardDetector:
    """认板 → 亚像素角点。两种 API 背后是同一件事：detect(gray) → (角点, id)。"""

    def __init__(self):
        self.board, self.objp, self.dic = _make_board()
        self.n_corners = len(self.objp)
        self.api = "4.6" if hasattr(cv2.aruco, "CharucoBoard_create") else "4.7+"
        params = (cv2.aruco.DetectorParameters_create() if self.api == "4.6"
                  else cv2.aruco.DetectorParameters())
        # 亚像素细化——<0.5px 的判据靠它
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if self.api == "4.7+":
            self._det = cv2.aruco.CharucoDetector(self.board, cv2.aruco.CharucoParameters(),
                                                  params)
        self.params = params

    def detect(self, gray):
        """→ (角点 N×1×2 float32, id N×1 int32)；认不出就 (None, None)。"""
        if self.api == "4.7+":
            corners, ids, _dc, _di = self._det.detectBoard(gray)
        else:
            mc, mi, _rej = cv2.aruco.detectMarkers(gray, self.dic, parameters=self.params)
            if mi is None or len(mi) == 0:
                return None, None
            _n, corners, ids = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, self.board)
        if ids is None or len(ids) == 0:
            return None, None
        return (np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
                np.asarray(ids, dtype=np.int32).reshape(-1, 1))


def frame_stats(corners, size):
    """板在画面里的归一化位置 + 板有多大（角点包围盒的最长边，px）。

    ⚠ 这里**故意不给"倾斜角"**：内参还没标出来时，倾角和焦距在投影里是互相抵消的，
    随便猜个 f 去解位姿能把 31° 说成 1°（本机合成板实测，见 memory 里的记录）。
    倾斜覆盖率等 --solve 拿到真 K 之后再判——那时才算得准。
    """
    pts = corners.reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    side = max(pts[:, 0].ptp(), pts[:, 1].ptp())
    return float(cx / size[0]), float(cy / size[1]), float(side)


def board_blur(gray, corners):
    """糊不糊：只算【板所占那块】的拉普拉斯方差。

    全幅方差在这儿没用——板占画面越大，全幅值越被白背景摊薄：实测同一张板上
    离得远（板小）全幅 86、板区 3261；离得近（板大）全幅 193、板区 1249。
    板区值才跟"糊不糊"对应（本机合成板实测：σ=0.6px → 1250，σ=1.5px → 110，
    σ=3px → 22）。认不出角点就退回全幅——那种时候状态行提示的是"认不全"。
    """
    if corners is None or len(corners) < 4:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    pts = corners.reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    dx, dy = 0.10 * (x1 - x0), 0.10 * (y1 - y0)
    h, w = gray.shape[:2]
    sub = gray[max(int(y0 - dy), 0):min(int(y1 + dy) + 1, h),
               max(int(x0 - dx), 0):min(int(x1 + dx) + 1, w)]
    return float(cv2.Laplacian(sub, cv2.CV_64F).var()) if sub.size else 0.0


# ---------------------------------------------------------------- 采集

def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_DEV, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"/dev/video{config.CAMERA_DEV} 打不开——是不是别的进程占着相机？")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_H)
    for _ in range(8):                       # 预热：前几帧曝光/白平衡不稳
        cap.read()
    return cap


def _read_command(timeout=0.35):
    """非阻塞读一行终端输入；超时返回 None。

    不用 cv2.imshow 开窗：本机 Wayland 下 Qt 窗口是黑的（碰杯-法奥指令映射.md 记过），
    所以这里跟 a1_static_test.py --watch 一个路子——**只用终端读数**。
    不阻塞才能让上面那行状态一直刷新。

    EOF（stdin 是管道/重定向且已读空）也返回 None：否则 readline() 回空串，
    会被上层当成"按了回车"→ 一路存帧停不下来。
    """
    r, _w, _x = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    line = sys.stdin.readline()
    return line.strip() if line != "" else None


def _try_load_K(path):
    """读已有内参 yaml，给采集时的实时倾角/距离当"瞄准镜"。读不到 → (None, None)。

    倾角只有 K 已知时才算得准。K 本身带 ±1.5% 误差时倾角只偏 ±0.3°
    （2026-10-15 拿真帧把 fx 扰动 ±1.5% 实测），指导"转到 30°"绰绰有余。
    """
    if not Path(path).exists():
        return None, None
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None, None
    K, D = fs.getNode("camera_matrix").mat(), fs.getNode("dist_coeffs").mat()
    fs.release()
    if K is None or D is None:
        return None, None
    return np.asarray(K, np.float64), np.asarray(D, np.float64).ravel()


def _solve_board(det, corners, ids, K, D):
    """solvePnP → (R 相机←板, t 相机←板 mm)。点太少或解不出 → (None, None)。

    手眼要的正是这对比倾角/距离更原始的 (R,t)，所以从 frame_pose 里拆出来共用。
    """
    if K is None or corners is None or ids is None or len(corners) < 6:
        return None, None
    try:
        ok, rvec, tvec = cv2.solvePnP(det.objp[ids.ravel()].astype(np.float64),
                                      corners.astype(np.float64), K, D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None
        return cv2.Rodrigues(rvec)[0], np.asarray(tvec).ravel()
    except cv2.error:
        return None, None


def frame_pose(det, corners, ids, K, D):
    """→ (板相对光轴的倾角度, 板心距 mm)；没 K 或算不出 → (None, None)。

    角度定义跟 --solve 完全一样（板法线 vs 光轴），数字可以互相对照。
    """
    R, tvec = _solve_board(det, corners, ids, K, D)
    if R is None:
        return None, None
    tilt = math.degrees(math.acos(min(1.0, abs(float(R[2, 2])))))
    c_board = np.asarray(det.objp, np.float64).mean(axis=0)
    return float(tilt), float(np.linalg.norm(tvec + R @ c_board))


def _status_line(n_saved, n_corner, n_all, blur, x, y, side, tilt=None, dist_mm=None,
                 missing_tilt=0):
    """一行原地刷新的状态（\\r 覆盖）。"""
    msg = f"已存 {n_saved:>2} 张 | 角点 {n_corner:>2}/{n_all} | 模糊度 {blur:>5.0f}"
    if n_corner:
        msg += f" | 板心 {x * 100:>3.0f}%,{y * 100:>3.0f}% | 板宽 {side:>4.0f}px"
        if tilt is not None:
            msg += f" | 倾角 {tilt:>3.0f}° 距离 {dist_mm:>4.0f}mm"
    if n_corner < MIN_CORNERS:
        msg += "  ⇒ 认不全，把板拿正/拿近"
    else:
        # 提示拼在一起说，别让"板偏小"把"缺大倾斜"顶掉——后者才是这批的瓶颈
        tips = []
        if missing_tilt > 0 and tilt is not None:
            if tilt >= TILT_GOAL_DEG:
                tips.append(f"这张够斜，存！还缺 {missing_tilt - 1} 张")
            else:
                tips.append(f"还缺 {missing_tilt} 张大倾斜（板像开门样转开 30~40°）")
        if blur < BLUR_WARN:
            tips.append("偏糊，稳一下")
        if side < BOARD_MIN_PX:
            tips.append(f"板偏小，再拿近些（目标 ≥{BOARD_GOOD_PX:.0f}px）")
        msg += "  ⇒ " + "；".join(tips) if tips else "  ⇒ 回车=存这张"
    sys.stdout.write("\r" + msg.ljust(150))
    sys.stdout.flush()


def cmd_capture(args):
    det = BoardDetector()
    if len(args.shots) > 1:
        sys.exit("--capture 只能给一个存帧目录（多目录是 --solve 才用的）")
    shots_dir = args.shots[0]
    shots_dir.mkdir(parents=True, exist_ok=True)
    before = len(list(shots_dir.glob("*.jpg")))
    if before:
        print(f"⚠ {shots_dir} 里已有 {before} 张旧图。这次存的会跟它们混在一起被 --solve 一起用；")
        print("  不想混就先把旧图挪走（比如 mkdir old && mv *.jpg old/）。")
    cap = open_camera()
    K, D = _try_load_K(args.out)             # 上次 --solve 的结果，给实时倾角当瞄准镜

    saved = []                               # [(文件名, x, y, 板宽px, 倾角, 距离mm)]
    cells = {}                               # 3×3 覆盖计数
    print(f"存帧目录：{shots_dir}")
    if K is not None:
        print(f"（瞄准镜：复用 {Path(args.out).name} 的内参，状态行会实时显示倾角/距离）")
    else:
        print("（没读到已有内参：这次状态行不显示倾角。先 --solve 跑一次，下一批就有瞄准镜）")
    print("【回车】= 存这一帧    【q】= 结束（相机单流：别同时开 camera_stream.py 等）")
    print("举着板：先正对多拍几张 → 再让板走到画面【四角】"
          f"→ 最后把板转开拍 ≥{TILT_GOAL_N} 张【大倾斜】（看倾角数字 ≥{TILT_GOAL_DEG:.0f}°）。")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                sys.exit("取流中断")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = det.detect(gray)
            n = 0 if corners is None else len(corners)
            blur = board_blur(gray, corners)
            x = y = side = 0.0
            tilt = dist_mm = None
            if n:
                x, y, side = frame_stats(corners, (gray.shape[1], gray.shape[0]))
                tilt, dist_mm = frame_pose(det, corners, ids, K, D)
            missing = 0
            if K is not None:
                n_tilted = sum(1 for s in saved if s[4] is not None and s[4] >= TILT_GOAL_DEG)
                missing = max(0, TILT_GOAL_N - n_tilted)
            _status_line(len(saved), n, det.n_corners, blur, x, y, side,
                         tilt, dist_mm, missing)

            cmd = _read_command()
            if cmd is None:
                continue
            if cmd.lower() in ("q", "quit"):
                break
            if cmd not in ("", " "):
                print("\n  （只认【回车】=存、【q】=结束）")
                continue
            if n < MIN_CORNERS:
                print(f"\n  ✗ 只认出 {n} 个角点（要 ≥{MIN_CORNERS}）——把板拿正/拿近点再来，"
                      f"这张没存")
                continue
            name = time.strftime("calib_%Y%m%d_%H%M%S_") + f"{len(saved):03d}.jpg"
            cv2.imwrite(str(shots_dir / name), frame)
            saved.append((name, x, y, side, tilt, dist_mm))
            cell = min(int(y * 3), 2) * 3 + min(int(x * 3), 2)
            cells[cell] = cells.get(cell, 0) + 1
            warn = "  ⚠ 偏糊" if blur < BLUR_WARN else ""
            extra = f"  倾角 {tilt:.0f}° 距离 {dist_mm:.0f}mm" if tilt is not None else ""
            print(f"\n  ✓ {name}  角点 {n}  板心({x * 100:3.0f}%,{y * 100:3.0f}%)"
                  f"  板宽 {side:.0f}px{extra}  模糊度 {blur:.0f}{warn}")
    except KeyboardInterrupt:
        print("\n（Ctrl-C 结束）")
    finally:
        cap.release()

    sys.stdout.write("\r" + " " * 150 + "\r")
    _capture_summary(saved, cells)


def _capture_summary(saved, cells):
    """收尾把"采全了没有"摊开讲——比"我拍了 35 张"有用得多。"""
    print()
    print("─" * 66)
    if not saved:
        print("一张都没存。再来一次：python3 -I src/calibrate.py --capture")
        return
    n = len(saved)
    print(f"共存 {n} 张（规划要 30~40 张）")
    print("板心覆盖（画面 3×3 格，每格几张）：")
    for row in range(3):
        print("   " + "  ".join(f"{cells.get(row * 3 + col, 0):3d}" for col in range(3)))
    empty = [k for k in range(9) if cells.get(k, 0) == 0]
    if empty:
        pos = "、".join({0: "左上", 1: "上中", 2: "右上", 3: "左中", 4: "正中", 5: "右中",
                         6: "左下", 7: "下中", 8: "右下"}[k] for k in empty)
        print(f"  ⚠ 这些格子还是 0：{pos} —— 畸变靠边角约束，四角尤其要拍到")
    if cells.get(0, 0) + cells.get(1, 0) + cells.get(2, 0) == 0:
        print("  ⚠ 画面【上半部】一张都没有：cy（主点纵坐标）要画面上下都有板才定得住，"
              "实测两批上半部全空时 cy 差 150px —— 把板举高些/举远些补几张")
    sides = sorted(t[3] for t in saved)
    lo, hi, mid = sides[0], sides[-1], sides[len(sides) // 2]
    print(f"板宽 {lo:.0f}~{hi:.0f}px（中位 {mid:.0f}px，远近拉开的比例 {hi / lo:.2f}）")
    if hi / lo < 1.6:
        print(f"  ⚠ 远近没拉开（{hi / lo:.2f}）：远、中、近各来几档，焦距才定得准")
    if mid < BOARD_MIN_PX:
        print(f"  ⚠ 整体偏小（中位 {mid:.0f}px）：板越小焦距越定不准——合成实测 274px 的"
              f"【完全平板】也只标到 0.5px 上下，415px 才 0.05px。")
        print(f"    下一批拿到离相机 ~40cm，让正对时板宽 ≥{BOARD_GOOD_PX:.0f}px。")
    tilts = sorted(t[4] for t in saved if t[4] is not None)
    if not tilts:
        print(f"  倾斜读不了（采集时没内参可当瞄准镜）")
    else:
        n_tilt = sum(1 for t in tilts if t >= TILT_GOAL_DEG)
        print(f"倾角（瞄准镜估的，跟 --solve 同定义）中位 {tilts[len(tilts) // 2]:.0f}°，"
              f"最大 {tilts[-1]:.0f}°；"
              f"≥{TILT_GOAL_DEG:.0f}° 的有 {n_tilt} 张（要 ≥{TILT_GOAL_N}）")
        if n_tilt < TILT_GOAL_N:
            print(f"  ⚠ 大倾斜不够：焦距和径向畸变在投影里几乎等价，全靠斜视才分得开——"
                  f"第二批 0 张 >25° 时 fx 在 1404~1713 间飘。")
    print("  板贴没贴平，--solve 会判（采集时算不准，所以不在这儿编）")
    print("─" * 66)
    print(f"下一步：python3 -I src/calibrate.py --solve")


# ---------------------------------------------------------------- 解算

def _load_views(det, objp_all, shots_dir):
    """读一个或多个存帧目录里的所有 jpg。多目录 = 把几批并起来标一遍。

    什么时候要并：一批里位姿条件不全（比如某批板小、某批站位偏），单批标不稳，
    但几批合起来条件就齐了 —— 相机没动过，内参就该是同一组。合成与真批都验证过：
    并集的 fx 明显比任何单批稳。
    """
    dirs = list(shots_dir) if isinstance(shots_dir, (list, tuple)) else [shots_dir]
    objpoints, imgpoints, names, sizes = [], [], [], []
    for d in dirs:
        shots = sorted(d.glob("*.jpg"))
        if not shots:
            sys.exit(f"{d} 里没有图——先跑 --capture")
        print(f"读 {d}（{len(shots)} 张）")
        for fp in shots:
            img = cv2.imread(str(fp))
            if img is None:
                print(f"  ！读不出 {fp.name}，跳过")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids = det.detect(gray)
            n = 0 if corners is None else len(corners)
            if n < MIN_CORNERS:
                print(f"  − {fp.name}: 角点 {n} < {MIN_CORNERS}，弃用")
                continue
            objpoints.append(objp_all[ids.ravel()])      # 索引就是角点 id，天然对齐
            imgpoints.append(corners)
            names.append(fp.name)
            sizes.append((gray.shape[1], gray.shape[0]))
    return objpoints, imgpoints, names, sizes


def cmd_solve(args):
    det = BoardDetector()
    objpoints, imgpoints, names, sizes = _load_views(det, det.objp, args.shots)
    n = len(objpoints)
    if n < MIN_VIEWS_HARD:
        sys.exit(f"只有 {n} 张有效视图（至少要 {MIN_VIEWS_HARD}）——回去补拍")
    if len(set(sizes)) > 1:
        sys.exit(f"图片尺寸不一致 {sorted(set(sizes))}——目录里混了别的图？")
    w, h = sizes[0]
    if n < MIN_VIEWS_SOFT:
        print(f"⚠ 只有 {n} 张有效视图（规划要 30~40）——先算一版看误差，多半还得补拍")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    per = []
    for i in range(n):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / math.sqrt(len(proj))
        per.append(float(err))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    med = sorted(per)[n // 2]
    order = sorted(range(n), key=lambda i: -per[i])

    print()
    print("=" * 78)
    print(f"内参标定结果 · {n} 张有效视图 · 图像 {w}×{h}")
    print("=" * 78)
    print(f"  fx = {fx:9.2f} px    fy = {fy:9.2f} px    （fx/fy 差 {abs(fx - fy) / fy * 100:.2f}%）")
    print(f"  cx = {cx:9.2f} px    cy = {cy:9.2f} px    （图像中心 {(w - 1) / 2:.1f}, {(h - 1) / 2:.1f}）")
    print(f"  畸变 [k1 k2 p1 p2 k3] = {np.array(dist).ravel().round(5).tolist()}")
    print(f"  整体重投影 RMS = {rms:.3f} px    （判据 < 0.5px）  中位单视图 {med:.3f} px")
    print()
    print("  误差最大的 6 张（看这些图是不是糊了/反光/板动了）：")
    for i in order[:6]:
        flag = "  ← 离群" if per[i] > med * OUTLIER_FACTOR else ""
        print(f"    {names[i]}  {per[i]:.3f} px{flag}")
    bad = [i for i in range(n) if per[i] > med * OUTLIER_FACTOR]
    print()

    # ---- 两条跟内参无关的硬指标：板有多大、板面自不自洽 ----
    # 板弯不弯能被这里抓住，是因为弯板只在畸变/焦距层面"几乎"装得下：拟合会挪 D 去凑，
    # 结果 k1/p1 离谱。去畸变后再做单应性反而看不出弯（8 自由度能把 1.5mm 弓完全吃掉），
    # 所以判据用"标完之后每视角还剩多少误差"最稳。
    sides = sorted(max(pts[:, 0].ptp(), pts[:, 1].ptp())
                   for pts in (np.asarray(p).reshape(-1, 2) for p in imgpoints))
    med_side = sides[n // 2]
    print(f"  板在画面里的宽度：中位 {med_side:.0f}px（{sides[0]:.0f}~{sides[-1]:.0f}）"
          f"   目标 ≥{BOARD_GOOD_PX:.0f}px")
    if med_side < BOARD_MIN_PX:
        print(f"    ⚠ 整体偏小：合成实测同尺寸(≈274px)的【完全平板】也只能标到 0.5px 上下，"
              f"拿大到 415px 就掉到 0.05px。这批就算 RMS 好看，余量也小 —— 下一批拿近些。")
    if med > SMOOTH_WARN_PX:
        print(f"  ⚠ 每视角误差中位 {med:.3f}px 偏大：这是跟内参无关的硬指标，"
              f"合成平板只有 0.03~0.11px，把板弓 1.5mm 就变成 1.0~1.6px ——")
        print(f"    说明板在这批照片里【没贴平/弯了】。最要紧的是别【手举着一块软垫】拍：")
        print(f"    裁切垫/纸板被手一捏就弯，大倾角时肉眼都看得出来（第四批就是这样，")
        print(f"    每视角 0.380→0.566、cy 从 572 漂到 722）。")
        print(f"    把打印纸四角用胶带贴到一块【硬板】（雪弗板/亚克力/木板/硬卡纸）上，握着硬板拍。")
    else:
        print(f"  ✓ 板面自洽（每视角中位 {med:.3f}px；合成平板 0.03~0.11px）")
    print()

    # ---- 位姿多样性：现在 K 是真值了，倾角/距离才算得准 ----
    c_board = np.asarray(det.objp, np.float64).mean(axis=0)
    tilts, dists = [], []
    for i in range(n):
        R = cv2.Rodrigues(np.asarray(rvecs[i], np.float64))[0]
        nv = R[:, 2]
        tilts.append(math.degrees(math.acos(min(1.0, abs(float(nv[2]))))))
        dists.append(float(np.linalg.norm(np.asarray(tvecs[i]).ravel() + R @ c_board)))
    n_tilt = sum(1 for t in tilts if t > 25.0)
    ts, ds = sorted(tilts), sorted(dists)
    print(f"  位姿多样性（用标出来的 K 反算）：")
    print(f"    倾角 中位 {ts[n // 2]:.0f}°  最大 {ts[-1]:.0f}°；超过 25° 的 {n_tilt} 张")
    print(f"    板心距相机 {ds[0]:.0f}~{ds[-1]:.0f}mm  中位 {ds[n // 2]:.0f}mm")

    # ---- 取景覆盖：板心落在画面的哪些格子里。主点（尤其 cy）靠画面上下都拍到才定得住。
    # 实测：两批都是画面下半部 100%，cy 一个 572、一个 722 —— 差 150px 的根子在这里。
    grid = np.zeros((3, 3), int)
    for pts in imgpoints:
        p = np.asarray(pts).reshape(-1, 2)
        gx = min(int(p[:, 0].mean() / max(w - 1, 1) * 3), 2)
        gy = min(int(p[:, 1].mean() / max(h - 1, 1) * 3), 2)
        grid[gy, gx] += 1
    print("  取景覆盖（板心落在画面 3×3 的哪一格）：")
    for r in range(3):
        print("     " + "  ".join(f"{grid[r, c]:3d}" for c in range(3)))
    if int(grid[0].sum()) == 0:
        print(f"    ⚠ 画面【上半部】一张都没有：cy（主点纵坐标）要靠画面上下都有板才定得住 ——")
        print(f"      实测两批上半部全空，cy 一个 572 一个 722（差 150px）。举高些/举远些补几张。")
    elif int((grid == 0).sum()):
        print(f"    ⚠ 还有 {int((grid == 0).sum())} 格是 0：畸变靠边角约束，四角尤其要拍到。")
    print()

    # ---- 子集稳定性：同一批帧劈两半各标一遍。正对/小倾角时焦距和径向畸变几乎不可分，
    # 换个帧子集 fx 就飘；只有倾斜超过 25° 的视角才把两者分开（见 STAB_WARN_PCT 注释）。
    unstable, fx_spread = False, None
    if n >= 16:
        Ks = []
        for idx in (list(range(0, n, 2)), list(range(1, n, 2))):
            _r, K2, _d, _r2, _t2 = cv2.calibrateCamera(
                [objpoints[i] for i in idx], [imgpoints[i] for i in idx], (w, h), None, None)
            Ks.append(K2)
        fx1, fx2 = float(Ks[0][0, 0]), float(Ks[1][0, 0])
        cy1, cy2 = float(Ks[0][1, 2]), float(Ks[1][1, 2])
        fx_spread = abs(fx1 - fx2) / ((fx1 + fx2) / 2) * 100
        print(f"  子集稳定性（奇偶帧各半再标一遍）：fx {fx1:.1f} vs {fx2:.1f} → 差 {fx_spread:.2f}%；"
              f"cy 差 {abs(cy1 - cy2):.0f}px")
        if fx_spread > STAB_WARN_PCT:
            unstable = True
            print(f"    ⚠ 焦距没被钉住（两半差 {fx_spread:.2f}% > {STAB_WARN_PCT}%）：缺的是【倾斜】——")
            print(f"      正对/小倾角再多也补不上，绕横竖轴各 ±30~40° 拍 ≥{TILT_GOAL_N} 张，"
                  f"畸变和焦距才分得开。")
        print()

    ok = True
    if rms >= 0.5:
        ok = False
        print("  ✗ RMS ≥ 0.5px：没到判据。按证据排的排查顺序：")
        print("    ① 标定板没贴平（手持纸板、被捏着一个角）→ 合成实测弓 1.5mm 就 RMS 1.7~2.2px、")
        print("       畸变系数错 200% —— 贴到硬底上重拍，这条最常见；")
        print("    ② 板在画面里太小（上面板宽中位 <450px 时）→ 先拿近再谈别的；")
        print("    ③ 采集时板动了、或个别帧糊 → 按上面的表把离群帧删了重算；")
        print("    ④ 板的比例尺打错了（横竖尺量出来不是 100.0/75.0）→ 重打板。")
    if n < MIN_VIEWS_SOFT:
        ok = False
        print(f"  ! 视图数 {n} < {MIN_VIEWS_SOFT}：就算 RMS 好看也别信，补到 30 张以上再说。")
    if unstable:
        ok = False
        print(f"  ! 焦距在不同帧子集间飘 {fx_spread:.1f}%（倾斜不够）：这版参数先别接进 config，"
              f"下次补 ±30~40° 的斜视再标。")
    if abs(fx - fy) / fy > 0.01:
        ok = False
        print(f"  ! fx 与 fy 差 {abs(fx - fy) / fy * 100:.1f}% > 1%：方形像素不该这样，"
              f"通常是标定板被非等比打了（横竖比例尺分别量一下）。")
    if bad:
        print(f"  ! 有 {len(bad)} 张离群（> 中位 ×{OUTLIER_FACTOR:.0f}）：删掉再 --solve 一遍更准。")
    if n_tilt < TILT_GOAL_N:
        print(f"  ! 倾角 >{TILT_GOAL_DEG:.0f}° 的只有 {n_tilt} 张（想要 ≥{TILT_GOAL_N}）："
              f"畸变和焦距靠斜视的位姿才能分开，"
              f"现在的 RMS 好看也别全信 —— 绕三个轴各转 ±30° 再补拍几张。")
    if ds[-1] / ds[0] < 1.5:
        print(f"  ! 距离只有 {ds[0]:.0f}~{ds[-1]:.0f}mm（没拉开）：远近各来几档，焦距才定得稳。")

    fs = cv2.FileStorage(str(args.out), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        sys.exit(f"写不出 {args.out}——目录存在吗？")
    # 4.6 的 FileStorage.write 不收 list，两个数得分开写
    fs.write("image_width", int(w))
    fs.write("image_height", int(h))
    fs.write("rms_px", float(rms))
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", np.asarray(dist).reshape(1, -1))
    fs.write("views", n)
    fs.write("board_square_mm", config.CHARUCO_SQUARE_MM)
    fs.write("date", time.strftime("%Y-%m-%d %H:%M:%S"))
    fs.write("source_dir", " + ".join(str(p) for p in args.shots))
    fs.release()
    print(f"\n  已写 {args.out}")
    print(f"  {'✓ 达到判据（<0.5px）' if ok else '✗ 还没达到判据'} —— 把这屏数字和结论发我，"
          f"我回填 config 并记进规划")


# ---------------------------------------------------------------- 手眼（批次 D）
# 手眼 = 求"法兰→相机"这个【固定】变换 X（相机用支架固连在法兰上）。原理 AX=XB：
# 臂每换一个位姿，法兰在基座系里转 A、板在相机系里转 B，A 和 B 围着同一个 X。
# 两条铁律，违反哪条都会做出"看着像、用时偏几厘米"的结果，而且事后很难查：
#   ① 板在整场里【绝对不能动】——板挪一下就是往方程里灌毒；
#   ② 位姿集要【转轴多样】——所有位姿只绕同一根轴转是经典退化（比如只左右摆）。
#      实测（2026-10-15 首轮 16 张）最容易不知不觉吃满的是【绕视线轴的滚转】：15 次相对
#      动作里 10 次是滚转，轴散布成了 0.845/0.151/0.004，缺口反倒落在法兰 y（偏摆类）。
#      → 别凭感觉补，看实时 HUD 的 轴 λ 和 已动 计数。
# 采集走拖动模式：人拖臂、程序只读法兰位姿 + 抓帧，全程不发任何运动指令。


def _rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def _rot_axis_angle(R):
    """旋转矩阵 → (转角°, 单位轴)。"""
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    ang = math.degrees(math.acos(c))
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, int(np.argmin(np.abs(w - 1.0)))])
    nn = float(np.linalg.norm(ax))
    return ang, (ax / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0]))


def _pose6_to_T(pose):
    """法兰位姿 [x,y,z,rx,ry,rz]（mm/度）→ 4×4 齐次矩阵（基座←法兰）。

    姿态约定 **R = Rz(rz)·Ry(ry)·Rx(rx)**（ZYX 欧拉角，依次右乘）。这不是照手册抄的：
    2026-10-15 在真机上把 J6 加 30°（J6 就是绕法兰自身 z 轴转），只有这个解释能同时
    满足 R(P') = R(P)·Rz(30°)——误差 0.0002°，12 种候选里第二名差 0.32°；绕 J5 复核
    转角也正好 30.000°（J5 轴在法兰系里 ≈ −x）。换约定前先重跑那个实验。
    """
    x, y, z, rx, ry, rz = (float(v) for v in pose)
    T = np.eye(4)
    T[:3, :3] = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
    T[:3, 3] = (x, y, z)
    return T


_RPC = None


def _flange_pose():
    """只读法兰位姿 [x,y,z,rx,ry,rz]（mm/度）。连不上抛 RuntimeError。

    单独包一层的两个理由：① 全部手眼流程都只读，不经这里发任何运动指令；
    ② 离线自测可以把本函数打桩，不需要真机。
    """
    global _RPC
    if _RPC is None:
        import fr_rpc
        _RPC = fr_rpc.FRRpc()
    try:
        return [float(v) for v in _RPC.flange_pose()]
    except Exception as e:                                   # xmlrpc 连不上/超时
        raise RuntimeError(str(e))


def _classify_motion(M_prev, M_cur):
    """这次相对上一张动的是哪一类 → (类别, 转角°)。M = 相机在板心系里的姿态 R。

    板是竖立的（板心系 y 朝下 → "上" = −y），相机光轴在板心系里的方向是 M·ẑ。
    相对转动轴 a：贴着光轴 = 绕视线滚转；贴着 ±y = 左右摆；其余 = 上下摆。
    滚转在画面上几乎看不出来、拖动手感又最容易重复，所以单独报、实时盯：真机首轮
    15 次相对动作里 10 次是滚转（2026-10-15），把轴散布吃成了 0.845/0.151/0.004。
    """
    d = M_cur @ M_prev.T
    ang, ax = _rot_axis_angle(d)
    view = M_cur @ np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, -1.0, 0.0])
    a_view = math.degrees(math.acos(min(1.0, abs(float(ax @ view)))))
    a_up = math.degrees(math.acos(min(1.0, abs(float(ax @ up)))))
    if a_view < 25.0:
        return "滚转", ang
    if a_up < 35.0:
        return "左右", ang
    return "上下", ang


def _he_axis_match(poses, R_bc):
    """轴对应残差（度）——**完全不用手眼矩阵**，只回答"这批数据自己自洽吗"。

    a_k = 基座系里 G_j·G_i⁻¹ 的转轴（机器人看到的），b_k = **相机→板**相对转动
    B_j·B_i⁻¹ 的转轴。数据自洽 ⟺ 存在【同一个】旋转 R 把每个 b_k 对到 a_k
    （轴是条线，取 ±；这个 R 就是板的姿态 Tb，见下）。
    找不到 = 板姿与法兰位姿对不上：板没拍全 / 板中途动过 / 位姿记录错。
    注意它比"两两转角一致"那句判据强得多——**角度一致对"轴转了个错方向"完全失明**
    （2026-10-15 实测：那批角度只差 0.19°，轴却差 1.9~3.5°）。
    只取转角 >15° 的对：小转角的轴方向本来就被噪声淹（轴噪声 ≈ 姿态噪声/转角）。
    返回 (中位°, 最大°, 对数)，对数不足返回 None。

    ⚠ **传进来的必须是 相机→板（board←cam 的逆），也就是 _solve_board 返回值的转置**。
    这个方向是 2026-10-15 量出来的，不是读文档读来的：板在基座系里不动的几何给出
    A_k = Tb · B_k · Tb⁻¹（Tb = 板→基座，常数）——**只有 B_k 取 相机→板 时两边才共轭**，
    取 板→相机 时不存在这样的常数旋转。对拍（同一位姿集、只换转置）：
       传 相机→板 → 0.000°；传 板→相机 → 11.3°  ← 看着"像有问题"，其实是被转置了
    所以 --handeye-solve 里那句"数据自洽性"曾经一直用错方向（好数据也报 10°+）。
    """
    if len(poses) < 5:
        return None
    Ts = [_pose6_to_T(p) for p in poses]
    a, b = [], []
    for i in range(len(Ts)):
        for j in range(i + 1, len(Ts)):
            Ai = Ts[j][:3, :3] @ Ts[i][:3, :3].T          # 基座系：i → j
            if _rot_axis_angle(Ai)[0] < 15.0:
                continue
            a.append(_rot_axis_angle(Ai)[1])
            b.append(_rot_axis_angle(R_bc[j] @ R_bc[i].T)[1])
    n = len(a)
    if n < 6:
        return None
    a = np.array(a); b = np.array(b)
    rng = np.random.default_rng(0)
    starts = [np.eye(3)] + [np.array(cv2.Rodrigues(rng.normal(size=3))[0]) for _ in range(48)]
    best = (None, -2.0)
    for R0 in starts:
        R = np.array(R0)
        for _ in range(60):
            dots = np.einsum("ij,ij->i", a, b @ R.T)      # a_k · (R·b_k)
            s = np.sign(dots); s[s == 0.0] = 1.0
            H = (b * s[:, None]).T @ a
            U, _S, Vt = np.linalg.svd(H)
            Rn = Vt.T @ U.T
            if np.linalg.det(Rn) < 0:                      # 只要真旋转
                Vt = Vt.copy(); Vt[-1] *= -1.0
                Rn = Vt.T @ U.T
            if np.abs(Rn - R).max() < 1e-13:
                R = Rn
                break
            R = Rn
        o = float(np.mean(np.abs(np.einsum("ij,ij->i", a, b @ R.T))))
        if o > best[1]:
            best = (R, o)
    R = best[0]
    err = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", a, b @ R.T)), -1.0, 1.0)))
    return float(np.median(err)), float(err.max()), n


def _he_board_scale(poses, R_t2c, t_t2c, X, min_span_mm=60.0):
    """量板尺寸对不对，返回 k = 假设的 CHARUCO_SQUARE_MM ÷ 板上方格实际边长。

    k 的方向（容易写反，2026-10-15 从像素级重推 + 仿真实测三处对齐）：
      板 24.0mm、代码假设 25.0mm → 画面里的格子偏小 → PnP 只能把板放【远】一点解释，
      报的距离 = 真距离 ×25/24 = 1.042 倍 ⇒ k = 1.042 > 1（方格比假设的【小】）。
      所以真边长 = CHARUCO_SQUARE_MM / k。本函数收到的 t_t2c 就是这个 k 倍的"真 mm"。

    为什么非要有这一条（2026-10-15 定量）：打印比例这类**均匀尺度错**被 PnP 的
    尺度-距离二义性整段吸收——残差≈0、留一≈0、分区解也不动，① ② ③ **全部失明**。
    合成里把板整体放大 4%（= 方格打成 26.0mm，k=0.962）实测：留一 0.32mm、分区解全过、
    解出的 X 平移差真值 **≈18mm**；而注入 2px 的角点场都还有 2.6mm/px 的斜率能被 ③ 看见。
    偏多少有式子（2026-10-15 扫 1/2/4% 量出来的，板距 470mm 的仿真上 4.7/9.4/18.7mm 全对得上）：
        **ΔX ≈ |1−k| × 板到相机的距离**
    物理上就是：PnP 把板沿视线挪了 |1−k|×板距，手眼把这笔位移记到了 X 的平移上。
    注意它跟位姿集【有没有】定尺度对无关（两组扫出来一样）——定尺度对管的是"能不能读出来"。
    ⇒ 这条是 X 平移精度唯一的软件旁证，采集前也可以直接用卡尺量板复核。

    原理（线性最小二乘，4 个未知量，2026-10-15 在仿真上验过）：
      · 相机中心固连在法兰上：位姿 i 时它在基座系里 c_i = t_fi + R_fi·t_X；
      · 板姿给的"相机 i 原点在相机 j 系里的位置" u_ij（PnP 单位 = k × 真 mm）；
      · 同一个点两种算法：c_i = (R_fj·R_X)·u_ij + c_j，即 c_i − c_j = k⁻¹·(R_fj·R_X)·u_ij；
      · 展开成对 (t_X, 1/k) 的向量方程（都在基座系里比，方向和长度都用上）：
            (R_fj − R_fi)·t_X + (1/k)·(R_fj·R_X)·u_ij = t_fi − t_fj
        R_X 是手眼旋转（旋转方程跟尺度无关，解出来是对的），所以是【线性】的。
        （验算：取 R_fj=I, t_fj=0, t_X=0 → 左边 = R_X·R_X⁻¹(t_fi) = t_fi ✓）
    ⚠ 三个坑，前两个都踩过（干净数据上分别读出 1.15 和 0.54）：
      ① 别拿"法兰平移模长 |t_fi−t_fj|"当分母——法兰一转，偏离它 130mm 的相机走的弧长
         就不是法兰位移了；
      ② 别用"解出来的 t_X"去算相机位移再比模长：那个 t_X 本身就是自洽折中，
         比值会读回 1.00 —— 4% 的板照样读 1.00（这条踩得最狠，等于白写）；
      ③ 也别想把模长方程"抬"成线性的（|ΔR·t_X|² ≠ |t_X|²，ΔR 不是旋转而是两个转动之差）。
      用向量方程 + 一起解出 1/k，才抓得住（仿真：板放大 4%（方格 26.0mm）→ k=0.962、
      X 其实已偏 18mm）。
    只取 |u_ij| ≥ min_span_mm 的对（短基线信噪比差）。
    返回 (k, 可用对数, 最大基线 mm, σ_k(剖面), 定尺度对数)——数据太少/解不出返回 None。
    ⚠ 有没有【定尺度对】（姿态几乎不变+平移>50mm）决定这条判据有没有意义：
      只有全瞄准帧时读数【自己就偏 2.4%】：干净仿真 14 张读 0.9760±0.0139（真值 1.0000）、
      板放大 4%（方格 26.0mm，真 k=0.9598）读 0.9367±0.0134——差的方向对，绝对值偏的 2.4%
      比误差棒 ±1.4% 还大，所以全瞄准时【别拿这个 k 去改常量】。
      补 5~6 条不转腕平移（姿态不变、±60mm）后：干净 0.9991±0.0007（定尺度对 16）、
      板放大 4% 读 0.9597±0.0008（定尺度对 10）✓✓ 都咬到真值 0.3% 以内——
      3.8% 的板错是 44σ，挡得住。
      所以判据以【定尺度对数 ≥2】为前提，误差棒只是参考。
    """
    n = len(poses)
    if n < 6 or X is None:
        return None
    RX = np.asarray(X, float)[:3, :3]
    Tcb, Rf, tf = [], [], []
    for i in range(n):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = np.asarray(R_t2c[i], float), np.asarray(t_t2c[i], float).ravel()[:3]
        Tcb.append(T)
        Tfi = _pose6_to_T(poses[i])
        Rf.append(Tfi[:3, :3])
        tf.append(Tfi[:3, 3])
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            Tji = Tcb[j] @ np.linalg.inv(Tcb[i])
            u = np.asarray(Tji[:3, 3], float)          # 相机 j 系里：相机 i 原点在哪
            du = float(np.linalg.norm(u))
            if du < min_span_mm:
                continue
            th = float(_rot_axis_angle(Rf[j].T @ Rf[i])[0])        # 相对转角（度）
            pairs.append((Rf[j] - Rf[i], tf[j] - tf[i], Rf[j] @ RX @ u, du, th))
    if len(pairs) < 8:
        return None
    A = np.vstack([np.column_stack([dR, Mu]) for dR, _dt, Mu, _du, _th in pairs])
    b = np.concatenate([-dt for _dR, dt, _Mu, _du, _th in pairs])
    # 加权最小二乘（方程是向量式，每条占 3 行）；短基线信噪比差，按 1/|u| 降权
    w = np.repeat([1.0 / max(du, 30.0) for *_r, du, _th in pairs], 3)
    Aw, bw = A * w[:, None], b * w
    x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    q = float(x[3])
    if not np.isfinite(q) or q <= 0.2 or q >= 5.0:
        return None

    # 误差棒必须用【剖面】量，不能拿逐对的 k 散布充数（踩过：全瞄准的位姿集里 q 有一条
    # 几乎不费代价的平方向——逐对散布报 ±0.002，而它跟真值差 2.6%）。做法：把 q 钉在某个值上
    # 只线性拟合 t_X，得 SSE(q)；求 SSE(q) = SSE* + σ̂² 这个水平线的两个交点（二分，不涉网格
    # 分辨率）。曲线是平的（非二次）时交点会跑很远，正好把"看不出"如实报出来。
    def sse_at(qq):
        tXq, *_ = np.linalg.lstsq(Aw[:, :3], bw - Aw[:, 3] * qq, rcond=None)
        r = Aw[:, :3] @ tXq - (bw - Aw[:, 3] * qq)
        return float(r @ r)

    def cross(qq_in, qq_out):
        """在 [qq_in, qq_out] 里二分找 SSE 穿过水平线的地方（两端函数值异号才收敛）"""
        lo, hi = qq_in, qq_out
        g_lo, g_hi = sse_at(lo) - lev, sse_at(hi) - lev
        if g_lo == 0.0:
            return lo
        if g_lo * g_hi > 0:
            return None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            g_mid = sse_at(mid) - lev
            if g_lo * g_mid <= 0:
                hi = mid
            else:
                lo, g_lo = mid, g_mid
        return 0.5 * (lo + hi)

    sse0 = sse_at(q)
    sig2 = sse0 / max(3 * len(pairs) - 4, 1)                         # 每条方程一个残差自由度
    lev = sse0 + sig2
    q_lo = cross(max(q - 0.5 * q, 1e-6), q)                          # 左交点
    q_hi = cross(q, q + 0.5 * q)                                     # 右交点
    if q_lo is None or q_hi is None:                                 # 半个 ±50% 都收不住 → 看不出
        return None
    k_lo, k_hi = 1.0 / q_hi, 1.0 / q_lo                              # k = 1/q 单调减
    # 定尺度对：平移够长（>40mm，信噪比）且【转动项远小于平移】(R_fj−R_fi)·t_X ≤ |u|/20。
    # 这种对里 t_X 几乎退场，方程直接退化成 q|u| ≈ |t_fi−t_fj|，k 不再跟 t_X 纠缠。
    ntX = float(np.linalg.norm(np.asarray(x[:3], float)))
    n_pin = sum(1 for *_r, du, th in pairs
                if du > 40.0 and du > 20.0 * math.radians(th) * ntX)
    return (1.0 / q, len(pairs), float(max(du for *_r, du, _th in pairs)),
            (k_hi - k_lo) / 2.0, n_pin)


def _he_axis_eigs(poses):
    """转动轴多样性：**只吃法兰位姿，与手眼矩阵无关**——解失败时用它分锅。

    对每一对位姿取相对转动的轴（表达在法兰系），堆成散布矩阵 Σ a·aᵀ/N，取特征值（降序）。
    λ3→0 = 所有转动轴躺在一个平面里（只有两根独立轴）。真机踩过（2026-10-15 第一批 16 张）：
    0.845 / 0.151 / **0.0038**，缺口方向 ≈ 法兰 y（[+0.05 +1.00 +0.03]，板心系里 ≈ 板面
    内水平 = 工具分类的「上下」偏摆类）；最富的方向才是法兰 z ≈ 视线（10/15 次动作是滚转）。
    注意别把"最富方向"当成"缺口方向"——判别看缺口的 z 分量，见 _he_gap_advice。
    返回 (特征值降序, 对应单位方向按列) 或 None（有效转动 <3 对）。
    """
    Rs = [_pose6_to_T(p)[:3, :3] for p in poses]
    ax = []
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            ang, a = _rot_axis_angle(Rs[i].T @ Rs[j])
            if ang > 2.0:                      # 太小的相对转动，轴没意义
                ax.append(a)
    if len(ax) < 3:
        return None
    A = np.array(ax)
    w, V = np.linalg.eigh(A.T @ A / len(A))
    return w[::-1], V[:, ::-1]


def _he_gap_advice(miss):
    """缺口方向（法兰系单位向量）→ 人话补法。

    相机顺着法兰 z 装（视线 ≈ 法兰 z），所以只看缺口往哪偏：
      缺口≈法兰 z → 缺滚转：板在画面里几乎不动、原地打转（拧手腕 J6）。
      否则       → 缺偏摆：换个方向看板（上下点头/左右挪），板在画面里明显移动。
    2026-10-15 那批缺口 = 法兰 y（偏摆），而人凭感觉补的都是滚转——所以这句按实测说，
    不按直觉说。判别只认缺口的 z 分量，标反一次就够了。
    """
    if abs(float(miss[2])) > 0.85:
        return ("缺的是【绕视线滚转】：锁住臂只拧手腕 J6，让板在画面里几乎不动地原地打转 20~30°")
    return ("缺的是【偏摆】：换个方向看板——镜头对着板上下点头或左右挪 20~30°，"
            "让板在画面里明显移动/变斜（原地打转那种拧 J6 不算）")


def _he_status_line(n_saved, n_corner, side, blur, dist, tilt, pose, kinds, axes=None):
    """一条 150 字以内的状态行（\\r 原地刷新）。"""
    msg = f"[手眼 {n_saved:>2}/{HE_MIN_SHOTS}] 角点 {n_corner:>2}/24"
    if n_corner >= 4:
        msg += f" | 板宽 {side:>4.0f}px 清 {blur:>5.0f}"
        if dist is not None:
            msg += f" | 距 {dist:>4.0f}mm 倾 {tilt:>3.0f}°"
    if pose is None:
        msg += " | 臂位姿 ×（读不到）"
    else:
        msg += f" | 臂 [{pose[0]:>5.0f} {pose[1]:>5.0f} {pose[2]:>5.0f}]"
    if kinds:
        msg += (f" | 已动 滚{sum(1 for k in kinds if k == '滚转')}"
                f" 左右{sum(1 for k in kinds if k == '左右')}"
                f" 上下{sum(1 for k in kinds if k == '上下')}")
    if axes is not None:
        w = axes[0]
        msg += f" | 轴 {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"    # λ3 逼 0 = 轴共面，解不了
    return msg


def _he_tip(n_corner, side, pose, ang_same, axes=None, margin=None):
    """当前这张能不能存 / 还差什么 —— 附在状态行尾巴上的一句话。"""
    if n_corner < HE_NEED_CORNERS:
        return (f" ⇒ 别存：板只认出 {n_corner}/{HE_NEED_CORNERS} 个角点——缺角的板姿朝向有偏，"
                f"先把整块板摆进画面")
    if margin is not None and margin < HE_EDGE_MARGIN_PX:
        return f" ⇒ 别存：板离画面边只剩 {margin:.0f}px（要 ≥{HE_EDGE_MARGIN_PX}）——往画面中间挪挪"
    if pose is None:
        return " ⇒ 法兰位姿读不到：检查控制器开机、网线、config.ROBOT_IP（现在存不了）"
    if side < HE_BOARD_MIN_PX:
        return f" ⇒ 板 {side:.0f}px < {HE_BOARD_MIN_PX:.0f}px：拖近些再存"
    if ang_same is not None and ang_same < HE_SIMILAR_DEG:
        return f" ⇒ 和已有位姿只差 {ang_same:.0f}°：换个方向再转，别白占一张"
    if axes is not None and axes[0][2] < AXIS_WEAK_EIG:
        w, V = axes
        miss = V[:, 2]
        hard = "现在这样解不了" if w[2] < AXIS_MIN_EIG else "偏弱"
        return (f" ⇒ 转动轴多样性{hard}（轴 {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}，缺口法兰"
                f"[{miss[0]:+.2f} {miss[1]:+.2f} {miss[2]:+.2f}]）：{_he_gap_advice(miss)}，再来 2~3 张")
    return " ⇒ 回车=存这张 · u=撤销上张 · q=结束"


def _he_solve_once(R_g2b, t_g2b, R_t2c, t_t2c, method):
    """跑一次 cv2.calibrateHandEye → 4×4 矩阵 X：**p_法兰 = X · p_相机**。

    方向是量出来的，不是读文档读出来的：合成真值（X_true = 相机心在法兰系 (15,−45,85)mm，
    姿态带滚转）跑完 15 个位姿，OpenCV 返回的矩阵【原样】就对上真值——
    平移差 1.8mm、姿态差 0.08°；要是再取一次逆，差 168mm / 158°，一眼是错的。
    （OpenCV 的参数名叫 R_cam2gripper，但它给的就是"法兰←相机"这个方向，别按名字猜。）
    X 的平移 = 相机光心在法兰系里的位置；X 的第三列 = 相机光轴在法兰系里的方向。
    """
    R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    X = np.eye(4)
    X[:3, :3] = R_c2g
    X[:3, 3] = np.asarray(t_c2g).ravel()
    return X


def _he_resid_white(pobj, pproj, presid):
    """把每帧的角点残差拆成【低阶光滑场】和【白噪声】两部分（分区解的噪声带用它）。

    光滑场 = 每帧都按同一套低阶形状偏（模糊沿画面的缓慢变化、板弯、畸变残余都会造成）；
    白噪声 = 传感器/JPEG/边缘定位的随机部分。
    ⚠ 别拿"光滑/白"的比值当"板弯报警"，2026-10-15 量过，它不干这个活：
      · 白噪声的零假设期望不是 0，是 √(p/(n-p)) = √(6/18) ≈ 0.58（6 参数拟合 24 点）；
      · 真机各批实测 0.60~1.88，跟【板面残差】同涨同落（好批 0.27px→比 0.80；
        报废批 0.33px→1.48；内参第一版 0.82px→1.88）——更像离焦模糊的指示器；
      · 注入光滑径向场时它反而【下降】（0.50→0.32 @1.5px），所以对"畸变没标干净"
        这类系统性错它是瞎的。
    它的正确用途只有一个：给分区解提供"去掉低阶场之后"的重采样池。
    （另有一条留一半角点拟合、预测另一半的检验，四条真机数据都是"留出 RMS ≥ 总 RMS"，
    即没有能泛化的低阶场；那是另一套统计量，别混。）
    返回 (白噪声池 (M,2), 光滑场 RMS 中位 px, 白噪声 RMS 中位 px)。
    """
    if not presid:
        return None, 0.0, 0.0
    pool, smooths, whites = [], [], []
    for pr, rs in zip(pproj, presid):
        xy = np.asarray(pr, float)
        if len(xy) < 10:                      # 点太少，低阶也拟合不出来，整段算白噪声
            pool.append(rs); continue
        x = (xy[:, 0] - xy[:, 0].mean()) / (xy[:, 0].std() + 1e-9)
        y = (xy[:, 1] - xy[:, 1].mean()) / (xy[:, 1].std() + 1e-9)
        B = np.column_stack([np.ones_like(x), x, y, x * y, x * x - 1.0, y * y - 1.0])
        sm = np.zeros_like(rs)
        for c in (0, 1):                      # 两个分量各拟合一次
            coef = np.linalg.lstsq(B, rs[:, c], rcond=None)[0]
            sm[:, c] = B @ coef
        pool.append(rs - sm)
        smooths.append(float(np.sqrt((sm ** 2).sum(axis=1).mean())))
        whites.append(float(np.sqrt(((rs - sm) ** 2).sum(axis=1).mean())))
    return np.vstack(pool), float(np.median(smooths)), float(np.median(whites))


def _he_split_boot(R_g2b, t_g2b, pobj, pproj, presid, splits, method, K, D,
                   n_boot=HE_SPLIT_BOOT, seed=0):
    """分区解的【噪声带】：把角点残差当白噪声重采样，看同样的分区切法本身能差多少。

    为什么必须按数据定带宽（2026-10-15 量的，不是猜的）：
      · 7v7 分区解对角点噪声的放大 ~15 倍：0.1px → 分区差中位 1.6mm，0.2px → 3.5mm，
        0.45px → 5~7mm（仿真位姿集 + 精确投影 + 真 K/D 量的）；
      · 所以 2mm 固定阈值只有 0.1px 的合成数据能达到，真机数据【必然】顶穿——
        那不是数据错，是判据没考虑噪声。
    重采样用【残差减掉低阶光滑场之后】的那部分：不然板弯这类低阶场会被当成噪声，
    带宽平白变宽。⚠ 但这**不足以**让 ③ 看见系统误差：光滑场过不了 PnP 的那部分
    落在残差里的形状并不低阶，照样进池子——实测注入 0.6px 径向偏，带宽 2.7→4.4mm，
    分区差 2.7mm 反而"合格"；扫到 1.5px（X 已偏 3.9mm）比值仍只到 0.69，一次没顶穿。
    结论：这条是粗差探测，不是精度保证（见 cmd_handeye_solve 里的灵敏度说明）。
    返回 [(中位, 95分位, 次数), ...] 与 splits 一一对应；数据太少返回 None。
    """
    pool, sm_rms, wh_rms = _he_resid_white(pobj, pproj, presid)
    if pool is None or len(pool) < 12 or len(splits) == 0:
        return None
    rng = np.random.default_rng(seed)
    out = [[] for _ in splits]
    outr = [[] for _ in splits]        # [DBG] 旋转差带
    for _ in range(n_boot):
        Rp, tp = [], []
        for ob, pr, rs in zip(pobj, pproj, presid):
            j = rng.integers(0, len(pool), len(rs))
            p = (pr + pool[j]).astype(np.float64)
            ok, rv, tv = cv2.solvePnP(ob.astype(np.float64), p, K, D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                break
            Rp.append(cv2.Rodrigues(rv)[0]); tp.append(np.asarray(tv))
        if len(Rp) != len(pobj):
            continue
        for si, (_key, idx_lo, idx_hi) in enumerate(splits):
            Tlo = _he_solve_once([R_g2b[i] for i in idx_lo], [t_g2b[i] for i in idx_lo],
                                 [Rp[i] for i in idx_lo], [tp[i] for i in idx_lo], method)
            Thi = _he_solve_once([R_g2b[i] for i in idx_hi], [t_g2b[i] for i in idx_hi],
                                 [Rp[i] for i in idx_hi], [tp[i] for i in idx_hi], method)
            out[si].append(float(np.linalg.norm(Tlo[:3, 3] - Thi[:3, 3])))
            outr[si].append(float(_rot_axis_angle(Tlo[:3, :3].T @ Thi[:3, :3])[0]))   # [DBG]
    if any(len(o) < 5 for o in out):
        return None
    return [(float(np.median(o)), float(np.percentile(o, 95)), len(o),
             float(np.median(o2)), float(np.percentile(o2, 95)))       # [DBG] 旋转带
            for o, o2 in zip(out, outr)]


def _he_read_csv(csv_path):
    """flange_poses.csv → [(帧文件名, [x,y,z,rx,ry,rz])]，按文件里的顺序。"""
    rows = []
    for line in csv_path.read_text().splitlines()[1:]:
        parts = [p for p in line.split(",") if p != ""]
        if len(parts) == 7:
            rows.append((parts[0].strip(), [float(v) for v in parts[1:]]))
    return rows


def _he_write_csv(csv_path, rows):
    with csv_path.open("w") as f:
        f.write("frame,x,y,z,rx,ry,rz\n")
        for name, pose in rows:
            f.write(",".join([name] + [f"{v:.4f}" for v in pose]) + "\n")


def cmd_handeye(args):
    """拖动模式采手眼数据：人拖臂、程序只读法兰位姿 + 抓帧。全程不发运动指令。"""
    if len(args.shots) > 1:
        sys.exit("--handeye 只能给一个存帧目录")
    shots_dir = args.shots[0]
    shots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = shots_dir / HE_POSES_CSV
    K, D = _try_load_K(args.intr)
    if K is None:
        sys.exit(f"手眼靠内参把板姿解出来，现在读不到 {args.intr} —— 先跑 --solve 生成它")

    print(f"手眼采集（批次 D）：拖臂摆位姿 → 回车存 → 走完 12~15 个位姿按 q")
    print(f"  本工具只做两件事：开相机抓帧、读法兰位姿（RPC 只读）。")
    print(f"  不发任何运动指令——臂用示教器【拖动模式】手动摆。")
    print(f"  存到 {shots_dir}（帧 + {HE_POSES_CSV}）")
    print(f"  铁律①：板从此刻起直到跑 --handeye-solve 都不能再动（碰一下这轮就废）")
    print(f"  铁律②：每张都要【板拍全 24/24】、离画面边 ≥{HE_EDGE_MARGIN_PX}px——缺角的板姿朝向")
    print(f"          会被带偏（2026-10-15 那批 11/16 张缺角 → 整批报废；缺角/贴边/重复现在都存不进去）")
    print(f"  铁律③：位姿别只左右摆——转轴要散开。别凭感觉补方向：盯住 HUD 的 轴 λ 和")
    print(f"          「已动 滚/左右/上下」计数，哪类少补哪类（上一批 10/15 张都是滚转）")
    print()
    try:
        p0 = _flange_pose()
        print(f"  机器人已连上（config.ROBOT_IP = {config.ROBOT_IP}），当前法兰位姿 "
              f"[{p0[0]:.0f} {p0[1]:.0f} {p0[2]:.0f} {p0[3]:.1f} {p0[4]:.1f} {p0[5]:.1f}]")
    except RuntimeError as e:
        sys.exit(f"读不到法兰位姿：{e}\n  检查：控制器开机？网线？ping {config.ROBOT_IP} 通不通？\n"
                 f"  （没有位姿就没法做手眼——本工具不做无位姿的采集）")
    print()

    det = BoardDetector()
    cap = open_camera()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.CAMERA_W
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.CAMERA_H

    # 续采：把已有帧认一遍，补出它们在板心系里的姿态（HUD 分"这类动过没有"要用）
    saved = []                       # [(名, 位姿, M 或 None)]
    kinds = []                       # 每张相对上一张的类别（第一张没有）
    if csv_path.exists():
        for name, pose in _he_read_csv(csv_path):
            if not (shots_dir / name).exists():
                print(f"⚠ {name} 的记录还在、图没了——跳过（要么补图要么删这行）")
                continue
            M = None
            img = cv2.imread(str(shots_dir / name))
            if img is not None:
                cp, idc = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                R_cb, _t = _solve_board(det, cp, idc, K, D)
                if R_cb is not None:
                    M = R_cb.T
            saved.append((name, pose, M))
        for i in range(1, len(saved)):
            if saved[i][2] is not None and saved[i - 1][2] is not None:
                kinds.append(_classify_motion(saved[i - 1][2], saved[i][2])[0])
        if saved:
            print(f"（这个目录里已有 {len(saved)} 张，接着采；想重来就把整个目录清空）")

    n_fail = 0
    pose = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                n_fail += 1
                if n_fail > 30:
                    print("\n相机读不到帧了"); break
                continue
            n_fail = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = det.detect(gray)
            n_corner = 0 if corners is None else len(corners)
            side, blur, dist, tilt, M_cur = 0.0, 0.0, None, None, None
            margin = None
            if n_corner:
                blur = board_blur(gray, corners)
                _x, _y, side = frame_stats(corners, (w, h))
                _p = corners.reshape(-1, 2)
                margin = float(min(_p[:, 0].min(), _p[:, 1].min(),
                                   (w - 1) - _p[:, 0].max(), (h - 1) - _p[:, 1].max()))
                R_cb, t_cb = _solve_board(det, corners, ids, K, D)
                if R_cb is not None:
                    M_cur = R_cb.T
                    tilt = math.degrees(math.acos(min(1.0, abs(float(R_cb[2, 2])))))
                    c_board = np.asarray(det.objp, np.float64).mean(axis=0)
                    dist = float(np.linalg.norm(t_cb + R_cb @ c_board))
            try:
                pose = _flange_pose()
            except RuntimeError:
                pose = None

            ang_same = None
            if M_cur is not None and saved:
                angs = [_rot_axis_angle(M_cur @ s[2].T)[0] for s in saved if s[2] is not None]
                if angs:
                    ang_same = min(angs)

            # 转动轴多样性（只吃法兰位姿）：λ3 逼 0 = 轴共面 = 解不了，实时盯着它
            axes = _he_axis_eigs([s[1] for s in saved]) if len(saved) >= 4 else None

            line = _he_status_line(len(saved), n_corner, side, blur, dist, tilt, pose, kinds, axes)
            line += _he_tip(n_corner, side, pose, ang_same, axes, margin)
            print(f"\r{line:<158}", end="", flush=True)

            cmd = _read_command()
            if cmd is None:
                continue
            if cmd.lower() in ("q", "quit"):
                break
            if cmd.lower() == "u":
                if not saved:
                    continue
                name = saved.pop()[0]
                (shots_dir / name).unlink(missing_ok=True)
                if kinds:
                    kinds.pop()
                _he_write_csv(csv_path, [(n, p) for n, p, _m in saved])
                print(f"\n  撤销：删了 {name}，现在 {len(saved)} 张")
                continue
            if cmd not in ("", " "):
                print("\n  （只认：回车=存 · u=撤销上张 · q=结束）")
                continue

            # ---- 存这张：五道闸 ----
            if n_corner < HE_NEED_CORNERS:
                print(f"\n  ✗ 板上只认出 {n_corner}/{HE_NEED_CORNERS} 个角点——缺角的帧会把板姿朝向带偏"
                      f"（2026-10-15 那批 11/16 张缺角，整批报废）。把整块板摆进画面再存")
                continue
            if margin is not None and margin < HE_EDGE_MARGIN_PX:
                print(f"\n  ✗ 板离画面边只剩 {margin:.0f}px（要 ≥{HE_EDGE_MARGIN_PX}）——"
                      f"贴边的角点定位不准，把板往画面中间挪")
                continue
            if side < HE_BOARD_MIN_PX:
                print(f"\n  ✗ 板宽 {side:.0f}px < {HE_BOARD_MIN_PX:.0f}px，太远了："
                      f"手眼是拿每张的板姿拼出来的，板小 = 板姿不准。拖近些（目标 ≥{HE_BOARD_GOOD_PX:.0f}px）")
                continue
            if pose is None:
                print("\n  ✗ 法兰位姿读不到（控制器/网络）——没有位姿的帧解不了手眼")
                continue
            try:
                p1 = _flange_pose()
                time.sleep(0.35)
                p2 = _flange_pose()
            except RuntimeError as e:
                print(f"\n  ✗ 法兰位姿读不到了：{e}")
                pose = None
                continue
            dp = float(np.linalg.norm(np.array(p1[:3]) - np.array(p2[:3])))
            dr = _rot_axis_angle(_pose6_to_T(p1)[:3, :3].T @ _pose6_to_T(p2)[:3, :3])[0]
            if dp > HE_STILL_MM or dr > HE_STILL_DEG:
                print(f"\n  ✗ 臂还在晃（0.35s 里动了 {dp:.2f}mm / {dr:.2f}°）——稳住手再按回车")
                continue
            if saved:
                pl = saved[-1][1]
                dpv = float(np.linalg.norm(np.array(p2[:3]) - np.array(pl[:3])))
                drv = _rot_axis_angle(_pose6_to_T(p2)[:3, :3].T @ _pose6_to_T(pl)[:3, :3])[0]
                if dpv <= HE_DUP_MM and drv <= HE_DUP_DEG:
                    print(f"\n  ✗ 和上一张的法兰位姿只差 {dpv:.2f}mm / {drv:.2f}°——重复帧、没存"
                          f"（多按了一下回车？上批 006/007 就是这么来的）。要重拍上一张先按 u 撤销")
                    continue

            name = time.strftime("handeye_%Y%m%d_%H%M%S_") + f"{len(saved):03d}.jpg"
            cv2.imwrite(str(shots_dir / name), frame)
            pose = p2
            saved.append((name, pose, M_cur))
            _he_write_csv(csv_path, [(n, p) for n, p, _m in saved])
            note = ""
            if M_cur is not None and len(saved) >= 2 and saved[-2][2] is not None:
                kind, ang = _classify_motion(saved[-2][2], M_cur)
                if ang < HE_SIMILAR_DEG:
                    # 几乎没动就别算"滚转过一次"——计数器一虚，人就会以为覆盖够了。
                    # 0° 的转动轴是任意的（特征值退化），分类本身也没意义。
                    kinds.append("重复")
                    note = f"（只差 {ang:.1f}°，跟上一张几乎一样——白占一张，建议换方向重存）"
                else:
                    kinds.append(kind)
                    note = f"（相对上一张：{kind} {ang:.0f}°）"
            print(f"\n  ✓ 存 {name}  {len(saved)} 张  板宽 {side:.0f}px 距 {dist:.0f}mm 倾 {tilt:.0f}°"
                  f"  臂 [{pose[0]:.1f} {pose[1]:.1f} {pose[2]:.1f}] {note}")
    finally:
        cap.release()
        print("\r" + " " * 158 + "\r", end="")

    # ---- 收尾小结 ----
    print(f"手眼采集结束：{len(saved)} 张 → {shots_dir}")
    if saved:
        n_roll = sum(1 for k in kinds if k == "滚转")
        print(f"  动过的方向：滚转 {n_roll} 次 · 左右 {kinds.count('左右')} 次 · 上下 {kinds.count('上下')} 次")
        poss = sorted(s[1][2] for s in saved)
        print(f"  法兰高度 {poss[0]:.0f}~{poss[-1]:.0f}mm")
        if len(saved) < HE_MIN_SHOTS:
            print(f"  ⚠ 只有 {len(saved)} 张（想 ≥{HE_MIN_SHOTS}，硬底线 {HE_HARD_SHOTS}）——"
                  f"少于 {HE_HARD_SHOTS} 张我拒绝解，建议回去再补几张")
        axes = _he_axis_eigs([s[1] for s in saved]) if len(saved) >= 4 else None
        Rcb = [s[2].T for s in saved if s[2] is not None]
        poss_saved = [s[1] for s in saved if s[2] is not None]
        if len(Rcb) >= 6:
            am = _he_axis_match(poss_saved, Rcb)
            if am is not None:
                med, mx, npair = am
                if med > HE_AXIS_BAD_DEG:
                    tag = "  ← 这批数据不自洽！"
                elif med > HE_AXIS_GOOD_DEG:
                    tag = "  ← 偏大"
                else:
                    tag = " ✓"
                print(f"  数据自洽性（轴对应残差，θ>15° 共 {npair} 对；合成对照 0.00°）："
                      f"中位 {med:.2f}° 最大 {mx:.2f}°{tag}")
                if med > HE_AXIS_BAD_DEG:
                    print(f"  ⚠ 有帧的板姿跟法兰位姿对不上（板没拍全/中途动过/位姿记错）——"
                          f"先别碰板，按每张角点数找出嫌疑张（缺角的现在也存不进去了）")
        if axes is not None:
            w, V = axes
            miss = V[:, 2]
            print(f"  转动轴多样性：轴 {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"
                  f"（λ3 逼 0 = 轴共面 = 平移沿缺口方向不稳）")
            if w[2] < AXIS_MIN_EIG:
                print(f"  ⚠ 转动轴几乎共面！缺口方向 [{miss[0]:+.2f} {miss[1]:+.2f} {miss[2]:+.2f}]（法兰系）"
                      f"{'＝法兰 z＝相机视线' if abs(float(miss[2])) > 0.85 else ''}")
                print(f"    平移沿这个方向会很不稳（解飞先看上面的数据自洽性）。")
                print(f"    补法（按实测缺口方向选，别凭感觉）：{_he_gap_advice(miss)}，补 3~4 张。")
        if n_roll == 0 and len(saved) >= 3 and (axes is None or axes[0][2] >= AXIS_MIN_EIG):
            print(f"  ⚠ 一次【绕视线滚转】都没有——这次缺的正是它：")
            print(f"    板留在画面里别动，只把相机绕自己视线扭 20~30°（= 手腕原地扭一下），再来 2~3 张。")
        if len(saved) >= HE_HARD_SHOTS:
            print(f"\n  解算（判据不过就不写 yaml，放心跑第一遍）：")
            print(f"    python3 -I src/calibrate.py --handeye-solve")
        else:
            print(f"  ⇒ 现在还不解：回去把位姿补到 ≥{HE_HARD_SHOTS} 张，再运行同一个命令")


def cmd_handeye_solve(args):
    """解 AX=XB + 交叉验证。判据：留一法预测误差中位 < 2mm（规划 D 批）。"""
    det = BoardDetector()
    K, D = _try_load_K(args.intr)
    if K is None:
        sys.exit(f"读不到内参 {args.intr} —— 手眼必须先有内参")
    shots_dir = args.shots[0]
    csv_path = shots_dir / HE_POSES_CSV
    if not csv_path.exists():
        sys.exit(f"没有 {csv_path} —— 先用 --handeye 采一轮")
    rows = _he_read_csv(csv_path)
    if len(rows) < HE_HARD_SHOTS:
        sys.exit(f"只有 {len(rows)} 张（硬底线 {HE_HARD_SHOTS}）——回去接着采")

    # ---- 每张解板姿（板→相机），顺带算板面自洽（跟内参无关的硬指标）----
    # pobj/pproj/presid：每帧的对象点、投影点、残差（proj-corner）——分区解的噪声带要用
    names, poses, R_t2c, t_t2c, per_px, missing, cxy, ncorner = [], [], [], [], [], [], [], []
    pobj, pproj, presid = [], [], []
    for name, pose in rows:
        img = cv2.imread(str(shots_dir / name))
        if img is None:
            missing.append(name); continue
        corners, ids = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if corners is None or len(corners) < MIN_CORNERS:
            missing.append(name); continue
        ncorner.append(len(corners))
        R, t = _solve_board(det, corners, ids, K, D)
        if R is None:
            missing.append(name); continue
        obj_i = det.objp[ids.ravel()].astype(np.float64)
        proj, _ = cv2.projectPoints(obj_i, cv2.Rodrigues(R)[0], t.reshape(3, 1), K, D)
        per_px.append(float(cv2.norm(corners.astype(np.float64),
                                    proj.astype(np.float64), cv2.NORM_L2) / math.sqrt(len(proj))))
        pobj.append(obj_i); pproj.append(proj.reshape(-1, 2))
        presid.append(corners.reshape(-1, 2).astype(np.float64) - proj.reshape(-1, 2))
        # 分区解要用：板心在画面里的位置 + 距离（板心 = 认出来那几个角点的平均）
        cp = corners.reshape(-1, 2).mean(axis=0)
        cxy.append([float(cp[0]), float(cp[1]), float(np.linalg.norm(t))])
        names.append(name); poses.append(pose); R_t2c.append(R); t_t2c.append(t)
    n = len(names)
    if n < HE_HARD_SHOTS:
        sys.exit(f"有效帧只有 {n} 张（硬底线 {HE_HARD_SHOTS}）；认不出的：{missing}")
    if missing:
        print(f"⚠ 有 {len(missing)} 张认不出板，已跳过：{missing}")

    print("=" * 78)
    print(f"手眼解算（AX=XB）· {n} 张 · 板竖直立在基座系里不动")
    print("=" * 78)
    med_px = sorted(per_px)[n // 2]
    print(f"  每视角板面误差 中位 {med_px:.3f}px（跟内参无关的硬指标：合成平板 0.03~0.11px，"
          f"弓 1.5mm → 1.0~1.6px）")
    if med_px > 1.0:
        print(f"    ⚠ 板面不自洽：要么板没贴平（纸贴在软的东西上/被磕弯），要么这几张糊了。")
    n_partial = [i for i in range(n) if ncorner[i] < HE_NEED_CORNERS]
    if n_partial:
        print(f"    ⚠ 有 {len(n_partial)}/{n} 张【板没拍全】（<{HE_NEED_CORNERS} 个角点）："
              f"{' '.join(names[i][-11:] for i in n_partial)}")
        print(f"      缺角 → 板姿朝向会被带偏（板面误差看不出来，它只反映拍到的那些角点），"
              f"后果直接体现在下面的轴对应残差里。")

    # ---- 数据自洽性：轴对应残差（只看法兰位姿和板姿，依然不用手眼矩阵）。这条必须先看。
    # 2026-10-15 教训：那批 16 张里 10 张板没拍全，这里报 1.9~3.5°（合成对照 0.00°），
    # 而"两两转角一致"只差 0.19° —— 角度判据对"轴朝错了方向"完全失明，别把它当免死金牌。
    # ⚠ 这里要传 **相机→板**（R_t2c 的转置）：板不动的几何只在这个方向上给出常数共轭
    #   A_k = Tb·B_k·Tb⁻¹。传原方向的话，一批完美数据也会报 10°+（仿真里量出来的）。
    am = _he_axis_match(poses, [R.T for R in R_t2c])
    am_med = None
    if am is not None:
        am_med, am_max, am_n = am
        if am_med > HE_AXIS_BAD_DEG:
            tag = "  ← 数据本身不自洽！"
        elif am_med > HE_AXIS_GOOD_DEG:
            tag = "  ← 偏大"
        else:
            tag = " ✓"
        print(f"\n  数据自洽性（轴对应残差，θ>15° 共 {am_n} 对；合成对照 0.00°）："
              f"中位 {am_med:.2f}°  最大 {am_max:.2f}°{tag}")
        if am_med > HE_AXIS_BAD_DEG:
            print(f"    ✗ 没有任何一个旋转能把【机器人看到的转轴】和【相机看到的转轴】对上——")
            print(f"      这不是「运动不够多样」，是板姿/位姿对不上。按可能性排：")
            print(f"      ① 板没拍全（缺角 → 板姿朝向有偏）→ 每张都要 24/24，板离画面边留余量；")
            print(f"      ② 板中途被动过；③ 某几张糊了；④ 按回车时臂还在动，位姿和画面不同步。")
            print(f"      这一条修好之前，别去改运动多样性，也别信后面任何判据。")
        elif am_med > HE_AXIS_GOOD_DEG:
            print(f"    ⚠ 偏大（> {HE_AXIS_GOOD_DEG}°）：多半有个别张缺角/糊了——重点查角点少的那几张。")

    # ---- 转动轴多样性：**只吃法兰位姿**。它决定平移有多不稳，但通常不是"解不了"的主因
    #（主因先看上面的数据自洽性）——
    axes = _he_axis_eigs(poses)
    ax3 = None
    if axes is not None:
        w, V = axes
        ax3 = float(w[2])
        miss = V[:, 2]
        tag = " ✓" if ax3 >= AXIS_MIN_EIG else "  ← 退化！"
        print(f"\n  转动轴多样性（只看机器人位姿，与手眼矩阵无关）：轴 {w[0]:.3f} / {w[1]:.3f} / "
              f"{w[2]:.3f}{tag}")
        if ax3 < AXIS_MIN_EIG:
            print(f"    ✗ 所有相对转动的轴几乎躺在一个平面里，缺口方向 [{miss[0]:+.2f} {miss[1]:+.2f} "
                  f"{miss[2]:+.2f}]（法兰系）{'＝法兰 z ＝相机视线' if abs(float(miss[2])) > 0.85 else ''}")
            print(f"      → 平移沿这个方向会很不稳（θ>25° 的对越多越稳），但它只解释解不稳，")
            print(f"        解释不了「解飞」——解飞先看上面的数据自洽性。")
            print(f"        补法（按实测缺口方向选，别凭感觉）：{_he_gap_advice(miss)}，再来 2~3 张。")
        elif ax3 < AXIS_WEAK_EIG:
            print(f"    ⚠ 轴多样性偏弱（{ax3:.3f} < {AXIS_WEAK_EIG}）[DBG 缺口法兰 "
                  f"[{miss[0]:+.2f} {miss[1]:+.2f} {miss[2]:+.2f}]]：{_he_gap_advice(miss)}，"
                  f"补 2~3 张会稳很多。")

    # ---- 输入：基座←法兰 的 R,t + 相机←板 的 R,t ----
    R_g2b, t_g2b, T_b_f, T_c_b = [], [], [], []
    for i, p in enumerate(poses):
        T = _pose6_to_T(p)
        T_b_f.append(T)
        R_g2b.append(T[:3, :3].copy())
        t_g2b.append(T[:3, 3].reshape(3, 1).copy())
        Tcb = np.eye(4)
        Tcb[:3, :3], Tcb[:3, 3] = R_t2c[i], t_t2c[i]
        T_c_b.append(Tcb)

    methods = {"Tsai": cv2.CALIB_HAND_EYE_TSAI, "Park": cv2.CALIB_HAND_EYE_PARK,
               "Horaud": cv2.CALIB_HAND_EYE_HORAUD, "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
               "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS}
    sols = {}
    print(f"\n  五种解法各跑一遍（T = 法兰←相机；相机心 = T 的平移，光轴 = T 的 z 轴）")
    for mname, m in methods.items():
        T = _he_solve_once(R_g2b, t_g2b, R_t2c, t_t2c, m)
        sols[mname] = T
        ax = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        print(f"    {mname:<11} 相机心 [{T[0, 3]:>7.1f} {T[1, 3]:>7.1f} {T[2, 3]:>7.1f}]mm"
              f"   |t| {np.linalg.norm(T[:3, 3]):>6.1f}mm  光轴 [{ax[0]:+.3f} {ax[1]:+.3f} {ax[2]:+.3f}]")
    m_primary = HE_LOO_METHOD if HE_LOO_METHOD in methods else "Park"
    base = sols[m_primary]
    dt = [np.linalg.norm(sols[m][:3, 3] - base[:3, 3]) for m in sols]
    dra = [_rot_axis_angle(sols[m][:3, :3].T @ base[:3, :3])[0] for m in sols]
    print(f"    解法间离散：平移最差 {max(dt):.2f}mm，旋转最差 {max(dra):.3f}°"
          f"（五种算法互相独立，这里要是大就说明数据本身有鬼）")

    # ---- 板尺寸自检：打印比例这类均匀尺度错只有这条能量出来（见 _he_board_scale）----
    bscale = _he_board_scale(poses, R_t2c, t_t2c, base)
    scale_ok = True
    if bscale is None:
        print(f"\n  板尺寸自检：这条这次【看不出】（位姿对不足、或尺度信息太弱——"
              f"差在【姿态不变、纯平移】的位姿太少：全在瞄准就不行）。")
        print(f"      不挡你，但这次板尺寸就只剩『采集前用卡尺量方格间距』一条防线（目标 "
              f"{config.CHARUCO_SQUARE_MM:.1f}mm，量 x/y 两个方向）。")
    else:
        k, nk, span, sig, npin = bscale
        print(f"\n  板尺寸自检（k = 假设的 {config.CHARUCO_SQUARE_MM:.1f}mm ÷ 板上方格实际边长："
              f"k>1 = 实际比假设的【小】，打印被缩过）")
        if npin < 2:
            print(f"    k = {k:.4f} ± {sig:.4f}（{nk} 对，最大位移 {span:.0f}mm）  "
                  f"← 这次看不出：定尺度的对只有 {npin} 对（姿态几乎不变+平移>50mm 才算）")
            print(f"      全瞄准的位姿集里 k 跟 t_X 分不开，读数自己就偏（仿真实测：干净数据读"
                  f" k=0.976 而真值 1.000，板错 4% 读 0.937 而真值 0.960——差的方向对、"
                  f"绝对值偏 2.4%，误差棒 ±1.4% 盖不住这个偏）。")
            print(f"      所以【别拿这个 k 去改常量】。不挡你，但这次板尺寸只剩『采集前用卡尺量方格"
                  f"间距』一条防线（目标 {config.CHARUCO_SQUARE_MM:.1f}mm，量 x/y 两个方向）。")
        elif sig > HE_SCALE_MAX_SIG:
            verdict = (f"← 这次看不出（误差棒 ±{sig * 100:.1f}% 太宽）")
            print(f"    k = {k:.4f} ± {sig:.4f}（{nk} 对，最大位移 {span:.0f}mm，"
                  f"定尺度对 {npin}）  {verdict}")
            print(f"      不挡你；但采集时补几条不转腕的平移位姿（±60mm）这条就能到 ±0.1%。")
        else:
            if abs(k - 1.0) > max(0.01, 3.0 * sig):
                scale_ok = False
                verdict = (f"← 板尺寸跟假设不一致！按 {config.CHARUCO_SQUARE_MM:.1f}mm 假设解，"
                           f"板上方格实际是 {config.CHARUCO_SQUARE_MM / k:.2f}mm")
            else:
                verdict = "✓ 板尺寸跟假设一致（打印比例没跑偏）"
            print(f"    k = {k:.4f} ± {sig:.4f}（{nk} 对，最大位移 {span:.0f}mm，"
                  f"定尺度对 {npin}）  {verdict}")
        if not scale_ok:
            d_board = float(np.median([np.linalg.norm(np.asarray(t).ravel()) for t in t_t2c]))
            print(f"      ⇒ 均匀尺度错被 PnP 的尺度-距离二义性整段吸收：板面残差、留一、分区解"
                  f"全看不出来，但 X 的平移会整体偏 ≈ |1−k| × 板距 = "
                  f"{abs(1.0 - k) * d_board:.1f}mm（板距中位 {d_board:.0f}mm）。")
            print(f"      这条式子不是估的：2026-10-15 拿仿真量过（板距 470mm，注入 1/2/4% 尺度错 → "
                  f"偏 4.7/9.4/18.7mm，对得上；且跟位姿集是不是带定尺度对无关）。")
            print(f"      修法（二选一）：① 用卡尺量板上方格实际多宽，把 config.CHARUCO_SQUARE_MM "
                  f"改成 {config.CHARUCO_SQUARE_MM / k:.2f} 再重解（内参不用重算——平面板的 fx "
                  f"与方格绝对尺寸无关）；② 按『实际大小/100%』重打一块再采。")

    # ---- 交叉验证 ③：分区解。留一只看得见"每张独立的随机噪声"，看不见"系统性的模型错"——
    # 2026-10-15 合成实测：合成画面里有个约 0.6px 的"角点被往外拽"的偏（效果像多了一点畸变），
    # 留一对此一无所知（报 0.25mm 很好看），可解出的手眼偏真值 1.7mm。
    # 分区解专治这类：把位姿按【板在画面左/右、上/下、远/近】各切两半分别解，
    # 模型错会随区域变（左半解的和右半解的不一样），随机噪声只是让两边都抖一点。
    T_f_c = base
    print(f"\n  交叉验证③（分区解，专抓留一看不见的系统性错——板弯/打印缩放/畸变没标干净都属这类）")
    # 先把三个分区切法定下来（按画面位置/距离），噪声带要用同一套切法
    splits = []
    for axis_name, key in (("左右", 0), ("上下", 1), ("远近", 2)):
        vals = np.array([cxy[i][key] for i in range(n)])
        idx_lo = [i for i in range(n) if vals[i] < np.median(vals)]
        idx_hi = [i for i in range(n) if vals[i] >= np.median(vals)]
        splits.append((axis_name, key, idx_lo, idx_hi))
    band = _he_split_boot(R_g2b, t_g2b, pobj, pproj, presid,
                          [(a, lo, hi) for a, _k, lo, hi in splits], methods[m_primary], K, D,
                          n_boot=60)   # [DBG] 25→60
    split_ok = True
    for si, (axis_name, key, idx_lo, idx_hi) in enumerate(splits):
        if min(len(idx_lo), len(idx_hi)) < 5:
            print(f"    {axis_name}：一半少于 5 张，跳过（样本太少解不稳）")
            continue
        Tlo = _he_solve_once([R_g2b[i] for i in idx_lo], [t_g2b[i] for i in idx_lo],
                             [R_t2c[i] for i in idx_lo], [t_t2c[i] for i in idx_lo], methods[m_primary])
        Thi = _he_solve_once([R_g2b[i] for i in idx_hi], [t_g2b[i] for i in idx_hi],
                             [R_t2c[i] for i in idx_hi], [t_t2c[i] for i in idx_hi], methods[m_primary])
        d_t = float(np.linalg.norm(Tlo[:3, 3] - Thi[:3, 3]))
        d_r, _ax = _rot_axis_angle(Tlo[:3, :3].T @ Thi[:3, :3])
        # 判据：固定 2mm 是【地板】，真正的线是"数据自己的噪声能解释多少"（95 分位）
        thr = HE_CROSS_MM
        btxt = ""
        if band is not None:
            b_med, b_95, b_n, br_med, br_95 = band[si]                     # [DBG]
            thr = max(HE_CROSS_MM, b_95)
            btxt = (f"，噪声带 {b_med:.1f}/{b_95:.1f}mm（重采样 {b_n} 次）"
                    f" [DBG 旋转带 {br_med:.3f}/{br_95:.3f}°: d_r vs max(0.5,br95) = "
                    f"{'✓' if d_r < max(0.5, br_95) else '✗'}]")
        flag = "" if d_t < thr and d_r < 0.5 else "   ← 两边对不上！"
        if flag:
            split_ok = False
        print(f"    {axis_name}：{len(idx_lo)} 张 vs {len(idx_hi)} 张 → 两个解差 {d_t:.2f}mm / {d_r:.3f}°{btxt}{flag}")
    if not split_ok:
        print(f"    ✗ 分区解对不上：数据里有一层【跟位置/距离相关的系统误差】，留一是抓不到它的。")
        print(f"      排查：板是不是弯了/贴得不平（同一块板在不同视角下形状不一样）？打印缩放对过没有？")
    # 灵敏度到底有多少——2026-10-15 拿合成真值量出来的（角点位移、两种注入方式互相对拍）：
    #   光滑径向场 b px → X 平移偏 2.6·b mm（b=0.6→1.7mm、1.0→2.7mm、1.5→3.9mm）；
    #   而 ③ 的"观测差/噪声带"比值在任何 b 上都停在 0.5~0.7，**从不顶穿**——因为这种场
    #   过不了 PnP 的那部分会掉进残差池、把带宽一起撑大（板面残差 0.09→0.27px 同步涨）。
    #   也就是说：①②③ 是【粗差探测】，不是 2mm 精度保证；~2px 以下的系统误差它们看不见。
    #   绝对精度只能靠 E1 的示教点复核（跟角点统计无关的一条路）。别把这段删了当没看见。
    print(f"    灵敏度（实测，别指望它抓小错）：光滑角点场 ~2px 以下本题材料都抓不到——"
          f"注入 0.6/1.0/1.5px 时 X 其实已偏 1.7/2.7/3.9mm，而本条只在 2px 上下才顶穿。")
    print(f"      ①②③ 的定位是【粗差探测】（缺角/板动过/板弯/位姿退化）；绝对精度由 E1 示教复核兜底。")

    # ---- [DBG] 随机半边 bootstrap：任意 17/18 切分的解摆幅 = "自然水平" ----
    rngd = np.random.default_rng(1)
    dtl, drl = [], []
    for _ in range(25):
        perm = rngd.permutation(n)
        lo, hi = [int(i) for i in perm[:n // 2]], [int(i) for i in perm[n // 2:]]
        Tlo = _he_solve_once([R_g2b[i] for i in lo], [t_g2b[i] for i in lo],
                             [R_t2c[i] for i in lo], [t_t2c[i] for i in lo], methods[m_primary])
        Thi = _he_solve_once([R_g2b[i] for i in hi], [t_g2b[i] for i in hi],
                             [R_t2c[i] for i in hi], [t_t2c[i] for i in hi], methods[m_primary])
        dtl.append(float(np.linalg.norm(Tlo[:3, 3] - Thi[:3, 3])))
        drl.append(float(_rot_axis_angle(Tlo[:3, :3].T @ Thi[:3, :3])[0]))
    dtl, drl = np.array(dtl), np.array(drl)
    print(f"\n  [DBG] 随机半边（25 次 17/18 随机切分，{m_primary}）：平移差 中位 {np.median(dtl):.2f} / "
          f"最大 {dtl.max():.2f}mm；旋转差 中位 {np.median(drl):.3f} / 最大 {drl.max():.3f}°；"
          f"d_r≥0.5° 的比例 {float((drl >= 0.5).mean()) * 100:.0f}%")

    # ---- 交叉验证 ①：每张各把板原点映到基座系，看散不散（板没动 → 应该聚成一点）----
    T_b_b = [T_b_f[i] @ T_f_c @ T_c_b[i] for i in range(n)]
    p_base = np.array([T[:3, 3] for T in T_b_b])
    mean = p_base.mean(axis=0)
    dev = np.linalg.norm(p_base - mean, axis=1)
    rms = float(np.sqrt((dev ** 2).mean()))
    angs = []
    for i in range(n):
        for j in range(i + 1, n):
            angs.append(_rot_axis_angle(T_b_b[i][:3, :3].T @ T_b_b[j][:3, :3])[0])
    print(f"\n  交叉验证①：按解出来的 T 把每张的板原点投回基座系（板其实没动 → 应该聚成一点）")
    print(f"    位置散 RMS {rms:.2f}mm  最大 {dev.max():.2f}mm   姿态两两最大夹角 {max(angs):.2f}°")

    # ---- 交叉验证 ②：留一法。抽掉一张、用其余张解 T，再拿它去预测抽掉那张该看到什么板姿。
    # 这是唯一能诚实地问"新位姿上还准不准"的办法：全量解的表观误差必然好看。
    loo = []
    for k in range(n):
        idx = [i for i in range(n) if i != k]
        Tk = _he_solve_once([R_g2b[i] for i in idx], [t_g2b[i] for i in idx],
                            [R_t2c[i] for i in idx], [t_t2c[i] for i in idx],
                            methods[m_primary])
        pk = np.array([(T_b_f[i] @ Tk @ T_c_b[i])[:3, 3] for i in idx]).mean(axis=0)
        pred = (np.linalg.inv(Tk) @ (np.linalg.inv(T_b_f[k]) @ np.append(pk, 1.0)))[:3]
        loo.append(float(np.linalg.norm(pred - t_t2c[k])))
    loo = np.array(loo)
    med_loo, max_loo = float(np.median(loo)), float(loo.max())
    loo_ok = med_loo < HE_CROSS_MM
    ok = loo_ok and split_ok and scale_ok
    print(f"  交叉验证②（留一法，{m_primary} 解）：板姿预测误差 中位 {med_loo:.2f}mm  "
          f"最大 {max_loo:.2f}mm   判据 中位 < {HE_CROSS_MM}mm {'✓' if loo_ok else '✗'}"
          + ("" if loo_ok == ok else "（但③分区解/板尺寸自检没过 → 总判据仍然 ✗）"))
    order = sorted(range(n), key=lambda i: -loo[i])
    worst = [f"{names[i]}({loo[i]:.1f})" for i in order[:4]]
    print(f"    最差的几张：{' · '.join(worst)}")
    if not ok:
        if not scale_ok:
            print(f"    ✗ 没到判据 —— **主因：板尺寸自检没过（见上面「板尺寸自检」那行）**：")
            print(f"      这不是数据脏，是【板上方格的真实边长跟 config 里的假设对不上】——"
                  f"X 的平移会整体按那个比例偏。先按那行给的修法改常量再重解，别去重采。")
        elif am_med is not None and am_med > HE_AXIS_BAD_DEG:
            print(f"    ✗ 没到判据 —— **主因已定位：数据本身不自洽（见上面「轴对应残差」那行）**：")
            print(f"      板没拍全 / 板被动过 / 位姿记错。重采一轮，且每张都要 24/24、板离画面边留余量。")
        elif ax3 is not None and ax3 < AXIS_MIN_EIG:
            print(f"    ✗ 没到判据 —— 转动轴退化（见上面那行）：平移沿缺口方向不稳。")
            print(f"      补法（按实测缺口方向选，别凭感觉）：{_he_gap_advice(miss)}。")
        else:
            print(f"    ✗ 没到判据。按可能性排：① 板没拍全/太小/糊 → 对照每张角点数（缺角的直接删）；")
            print(f"      ② 板动过（场里有人碰过板/桌角）→ 重采；③ 位姿转轴太单一 → 按上面那行的缺口方向补；")
            print(f"      ④ 臂位姿读数与画面不同步（拖到一半就按回车）→ 等臂停稳再存。")

    # ---- 写文件：判据不过就不写 —— 一份没达标的手眼躺在 data/ 里，
    # 下游（E1 集成）会当真的用，那是比没有更坏的事。----
    if ok:
        fs = cv2.FileStorage(str(args.out), cv2.FILE_STORAGE_WRITE)
        fs.write("R_flange_cam", T_f_c[:3, :3])
        fs.write("t_flange_cam_mm", T_f_c[:3, 3].reshape(3, 1))
        fs.write("direction", "p_flange = R @ p_cam + t   （相机系坐标 → 法兰系；"
                              "t = 相机光心在法兰系里的位置）")
        fs.write("views", n)
        fs.write("method", m_primary)
        fs.write("split_ok", int(split_ok))
        fs.write("loo_median_mm", med_loo)
        fs.write("loo_max_mm", max_loo)
        fs.write("scatter_rms_mm", rms)
        fs.write("mean_reproj_px", med_px)
        fs.write("date", time.strftime("%Y-%m-%d %H:%M:%S"))
        fs.write("source_dir", str(shots_dir))
        fs.release()
        print(f"\n  已写 {args.out}")
        print(f"  ✓ 达到判据（①板尺寸自检、②留一中位 < 2mm、③分区解一致 全过）——"
              f" 把这屏也发我，我回填 config 并记规划")
    else:
        print(f"\n  不写 {args.out}（判据没过，不能让它冒充标定结果）。")
        if args.out.exists():
            print(f"  ⚠ {args.out.name} 现在就躺在那儿——**那是旧的/没达标的解，别拿去用**（直接删掉它）。")


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser(
        description="A2 相机内参（ChArUco 板）：--capture 采集 / --solve 解算；"
                    "批次 D 手眼：--handeye 拖动采集 / --handeye-solve 解算")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", action="store_true", help="开预览，回车存帧、q 结束")
    g.add_argument("--solve", action="store_true", help="读存帧目录，算内参并存 yaml")
    g.add_argument("--handeye", action="store_true",
                   help="手眼采集：拖臂摆位姿（只读法兰位姿 + 抓帧，不发运动指令）")
    g.add_argument("--handeye-solve", action="store_true", help="解手眼 AX=XB 并交叉验证")
    ap.add_argument("--shots", type=Path, nargs="+", default=None,
                    help=f"存帧目录（内参默认 {SHOTS_DIR}，手眼默认 {HE_SHOTS_DIR}）；"
                         f"--solve 可以给多个目录，把几批并起来标一遍")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"输出（内参 {OUT_YAML}；手眼 {HE_OUT_YAML}）")
    ap.add_argument("--intr", type=Path, default=OUT_YAML,
                    help=f"手眼读的内参 yaml（默认 {OUT_YAML}）")
    args = ap.parse_args()
    if args.shots is None:
        args.shots = [HE_SHOTS_DIR] if (args.handeye or args.handeye_solve) else [SHOTS_DIR]
    if args.out is None:
        args.out = HE_OUT_YAML if args.handeye_solve else OUT_YAML

    print(f"OpenCV {cv2.__version__}（aruco {'老' if hasattr(cv2.aruco, 'CharucoBoard_create') else '新'} API）")
    print(f"标定板：{config.CHARUCO_SQUARES[0]}列×{config.CHARUCO_SQUARES[1]}行 · "
          f"方格 {config.CHARUCO_SQUARE_MM}mm · 小码 {config.CHARUCO_MARKER_MM}mm · "
          f"{config.CHARUCO_DICT}（共 {config.CHARUCO_SQUARES[0] * config.CHARUCO_SQUARES[1] // 2} 个小码、"
          f"{(config.CHARUCO_SQUARES[0] - 1) * (config.CHARUCO_SQUARES[1] - 1)} 个内角点）")
    if args.capture:
        cmd_capture(args)
    elif args.solve:
        cmd_solve(args)
    elif args.handeye:
        cmd_handeye(args)
    else:
        cmd_handeye_solve(args)


if __name__ == "__main__":
    main()
