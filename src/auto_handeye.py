# -*- coding: utf-8 -*-
"""D 批手眼采集 · 自动版（程序控臂自己走位姿）

为什么要有它：首轮 16 张报废的两个主犯（见 memory: handeye-data-selfconsistency）——
① 11/16 张板没拍全（缺角 → 板姿朝向偏）、② 位姿多样性没量着补（15 次动作 10 次滚转，
轴散布 0.845/0.151/0.004）。这两件事在"人拖臂 + 回车存帧"里全靠手感和自觉，
程序里可以每一步都闭环验证、按实测缺口方向决定下一张怎么转。

跟拖动版（`calibrate.py --handeye`）的关系：**帧文件、flange_poses.csv 格式、
各道门限常量完全一致**，存出来的目录直接喂给 `calibrate.py --handeye-solve`，
解算端一行都不用改。

--go 时按顺序做这些事：
  0. 前检：急停/错误码/自动模式/8083 状态帧在流 + 板现在就在画面里 → 打印议程 → 等你在终端输 GO
  1. 使能 → **首动自检**：垂直上 5mm 再回落（这就是规划里 B3 的"第一次真动"）
  2. 学局部雅可比：沿法兰 x/y/z 各走 ±20mm 的探针，量出「mm → 画面 px / 距离 mm」的换算
  3. 对准：把板心拉进画面中心、距离拉到 ~440mm（迭代修正，每步 ≤40mm）
  4. 走议程：14 个相对位姿——偏摆（绕板心画弧，视线方向变）、滚转（绕视线原地转）、
     平移（近/远/侧移）三类都有。每个位姿：先回对准位姿（顺便验"板没动过"）→
     分步走弧（每步都确认板还在画面里）→ 再对准 → 停稳 → 过门 → 存帧
     门 = 24/24 角点 + 离画面边 ≥60px + 板宽 ≥350px + 与【所有】已存帧不重复
     之后再加 6 条【定尺度平移】（姿态冻结、纯平移 ±60mm，见 _walk_strokes）——
     全瞄准的位姿集测不出标定板自己的尺寸（打印比例），这 6 条是唯一的软件防线
  5. 议程走完若轴散布 λ3 < 0.08：按实测缺口方向自适应补位姿（最多到 --max-shots 张）
  6. 收尾：回出发位姿 → 打印 数据自洽性 + 轴散布 + 下一步解算命令。臂保持使能，不自动下电

安全（都写死在程序里，不靠临场记性）：
  · 速度 ≤ config.MAX_SPEED_PCT（默认 12%）；单条位移 ≤ 95mm（config 的闸是 100）
  · 局部盒：以出发位姿为中心 ±260mm(x/y) ±240mm(z)，且绝对 z ≥ 70mm。超了拒发
    （议程走完的包络实测 x168/y138/z169mm，剩下的余量留给对准修正）
  · 逆解一律走 GetInverseKinRef（ref=当前关节）**钉关节家族**，解出来的关节原样下传给
    fr_arm（不再有第二发隐式逆解）；解出来的解若跳变 > 35° 或肘部塌陷 → 用八组枚举
    （GetInverseKin cfg 0~7）兜底挑"健康且连续"的那组；都没有 → 跳过这个位姿
  · **肘部高度守卫**（2026-10-16 撞机事故的量）：目标关节的肘部（J3 轴线）估计 z <
    120mm 一律拒发——深家族对准 ≈40mm（就是撞支架的那组），正常家族 ≈358mm
  · 每条运动后回读关节和命令比对（差 > 5° 停、> 1.5° 打标），关节全量进 joints_log.csv
    （家族/姿态变化在只记位姿的日志里完全隐形——这次事故的教训）
  · 每条运动后验"到位没"：实测位姿 vs 目标 > 1.5mm/0.4° → 停（被碰撞/急停打断的典型样子）
  · 走弧时每步都要求板还在画面里（≥12 角点），不然立刻停
  · 运动后画面流会滞后 ~10s 才"追上"真实画面（2026-10-15 深夜实测：伺服一动，USB 流就卡
    一阵，期间抓到的还是运动前的旧画面——"翻译不变性"怪案的真凶；手拖不触发，所以拖动
    批次全对）。每次运动后的第一抓自动等"跳变"确认追上才继续（_grab_fresh），等不到会 ⚠ 打标
  · 相机装在 J6 定子侧（转 J6 相机不转，用户眼睛实锤）——存进 csv 的位姿是【定子】帧
    （= 法兰 ∘ Rz(−j6)），原始法兰位姿 + j6 记在 flange_poses_raw_j6.csv
  · 每次回到对准位姿都比对"板在相机里的位姿"，差 > 2mm/0.5° = 板动过了 → 停（铁律①）
  · 任何异常 / Ctrl-C：先 Pause 停住，打印现场；绝不自动切模式、绝不自动下使能、绝不改碰撞等级

用法：
  python3 -I src/auto_handeye.py                 # 只读预演（连上读位姿、开相机看板、打印议程和包围盒）
  python3 -I src/auto_handeye.py --dry-joints    # 只读预演关节轨迹：回家过渡 + 整条议程的 IK/关节/肘部高度（不碰相机）
  python3 -I src/auto_handeye.py --go            # 真跑：手放急停按钮上，程序会先要你输 GO
  python3 -I src/auto_handeye.py --go-home       # 回家翻转：从当前位姿分步摆到正常家族对准位姿（真动，手放急停）
  python3 -I src/auto_handeye.py --disable       # 收工后下使能（本程序不替你下）
  python3 -I src/auto_handeye.py --selftest      # 只跑离线数学校验，不连机器人

预演/真跑前：关掉别的占相机的进程；示教器切到【自动模式】（本程序绝不自己切模式）。
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # python3 -I 下也能 import

import config                                                       # noqa: E402
import calibrate as C                                               # noqa: E402
from fr_arm import Arm, SafetyError                                 # noqa: E402
from fr_rpc import FRRpcError                                       # noqa: E402

# ---------------------------------------------------------------- 参数

TARGET_DIST_MM = 440.0     # 板心距的目标（板宽 ≈ 440px，在 350~550 的建议带里）
DIST_TOL_MM = 30.0         # 距离对准容差
CENTER_TOL_PX = 100.0      # 板心离画面中心容差（平移类位姿故意让它偏一点，不硬拉回来）
PROBE_MM = 20.0            # 探针位移
PROBE_MAX_ITER = 3         # 探针不合格时的重试次数
MOVE_MAX_MM = 95.0         # 程序自己的单条位移上限（比 config.MAX_MOVE_MM=100 更严）
ARC_STEP_DEG = 8.0         # "绕板心画弧"每步最多转多少度（440mm 处 8° ≈ 61mm 弦长）
ROLL_STEP_DEG = 12.0       # 滚转也要分步：圆心（板心）不在视线轴上，一口气转 25° 会有
                           # ~50mm 的位移，跟偏摆最后一步叠一起就顶到 100mm 的闸了（离线量过）
BOX_DXY_MM = 260.0         # 局部盒半宽（x/y）：议程实测包络峰值 x168 y138 mm（5 种朝向扫过），
                           # 再留 ~90mm 给对准修正
BOX_DZ_MM = 240.0          # 局部盒半高（z）：议程实测 z 峰值 169mm —— 20° 弧往下就 150mm 了
Z_FLOOR_MM = 70.0          # 绝对地板线：法兰 z 不许低于它
SPEED_DEFAULT = 12.0       # %（≤ config.MAX_SPEED_PCT=20）
TILT_DEG = 20.0            # 偏摆基准角（±1 个因子）
ROLL_DEG = 25.0            # 滚转基准角
JOINT_JUMP_DEG = 35.0      # 逆解关节跳变上限：每步位移 ≤55mm，健康解只该动几度；
                           # 跳到几十度 = 换构型（可能反手），跳过这个位姿
MOVE_VERIFY_MM = 1.5       # 到位自检：实测 vs 目标
MOVE_VERIFY_DEG = 0.4
BOARD_MOVED_MM = 2.0       # 回对准位姿时板位姿的允许漂移（超过 = 板动过了）
BOARD_MOVED_DEG = 0.5
MIN_SEP_DEG = 10.0         # 自适应补的位姿至少要和已有的差这么多
PATH_STEP_MM = 60.0        # 长距离回程的直线插值步长
PATH_STEP_DEG = 8.0        # 同上，姿态步长
AIM_MAX_ITER = 6
AIM_CLAMP_MM = 40.0        # 对准时单次横向修正上限
AIM_CLAMP_Z = 60.0         # 对准时单次轴向修正上限
SETTLE_TIMEOUT_S = 25.0
LIFT_TEST_MM = 5.0         # 首动自检：垂直上 5mm
MAX_SHOTS = 26             # 14 议程 + 6 定尺度平移 + 最多 6 张自适应补位姿
STROKE_MM = 60.0           # 定尺度平移的标称行程（±x/±y/±z，姿态冻结）
STROKE_MIN_MM = 40.0       # 收缩到比它还小就不发：<40mm 的位姿对不算"定尺度对"（解算端要 >40）
STROKE_PAD_PX = 30.0       # 收缩时给画面余量留的富余（板心距离≠板角距离，另加 25% 上浮）
SETTLE_MM = C.HE_STILL_MM  # 停稳判据复用拖动版的手眼常量（0.3mm / 0.15°）
SETTLE_DEG = C.HE_STILL_DEG

# ---- 画面"追上"判据（2026-10-15 深夜 probe_wait 实测：一次 20mm 运动后，画面流先给 ~10s
# 的旧画面，然后在 +9~+12s 之间一跳跳到新构图——跳变那两帧逐像素均差 25.3，追上后回到 ~2）
FRESH_JUMP_MEAN = 5.0      # 相邻两抓逐像素均差 ≥ 它 = "画面跳变"（= 流追上了）
FRESH_STILL_MEAN = 3.0     # ≤ 它 = 静止噪声帧（夜里实测 1.7~2.4）
FRESH_STABLE_N = 3         # 跳变后再连稳这么多帧才算追稳
FRESH_STILL_MIN_S = 20.0   # 没跳变但一直稳：至少等这么久才按"已追上"收（位移 <26px 的
                           # 小移动对不上跳变判据；20s 是实测卡窗 12s 的 ~1.7 倍余量）
FRESH_STILL_N = 20         # …且至少稳住这么多帧（防低帧率时"稳 3 帧"就收）
FRESH_WAIT_MAX_S = 50.0    # 硬上限：超了也交帧，但 ⚠ 打标（这帧可能还是旧的）
FRESH_TICK_S = 0.25        # 抓帧循环节拍

# 相机光心在法兰系里的大致位置（mm）——只用来当"绕板心画弧"的圆心先验。
# 数来自首轮五解法里 x/y 一致的那几个（Park/Horaud：约 105,-41,-43；z 分量各解法
# 飘得厉害，因为那批数据本来就不能信）。它只是**前馈**：圆心偏三四十毫米，
# 后果是绕完弧板心偏几十像素，对准循环一趟就收拾了。
CAM_OFFSET_DEFAULT = (105.0, -41.0, -43.0)

# 正常（肘朝上）家族的对准关节角：2026-10-16 撞机后的"回家"路线里验证过的那组
# （fk 回 [414.2,-106.7,352.8,48.2,-86.6,40.4]，肘部 z≈358mm；翻转路径按 200 路点
# FK 模拟过：腕/相机全程高位）。--go-home 的目标就是它。深家族的同位姿解
# [13.24,-200.18,78.71,-67.46,-14.76,-83.9]（肘 z≈40mm，就是撞铝型材支架的那组）永远不用。
NORMAL_AIM_JOINTS = [13.24, -128.69, -78.71, 18.48, -14.76, -83.9]
DEFAULT_AIM_POSE = [414.2, -106.7, 352.8, 48.2, -86.6, 40.4]   # 上次真跑的对准位姿（板没动过）

# 肘部（J3 轴线）高度守卫——撞机事故量出来的判据（memory: fr3-elbow-geometry）：深家族
# 对准时肘部 z≈40mm（离横梁只有几毫米），正常家族 ≈358mm。任何【目标关节】的肘部估计
# < 120mm 一律拒发。估计法 = 固定 J1/J2/J4/J5/J6、J3 扫一串值，法兰点共圆、圆心即肘。
ELBOW_MIN_Z_MM = 120.0
ELBOW_FIT_RES_MM = 1.0       # 圆拟合残差超过它 → 这次估计不可信 → 守卫跳过并打标
ELBOW_FIT_R_MIN = 40.0       # 圆半径合理区间（真机实测 203.8~333.4mm）
ELBOW_FIT_R_MAX = 900.0
ELBOW_SWEEP_DEG = (150.0, 120.0, 90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0, -120.0, -150.0)

# 回家翻转（--go-home）参数
HOME_STEP_DEG = 10.0         # 关节过渡每步最多转多少度（J3 要翻 ~161.6° → 17 步）
HOME_SPEED_PCT = 10.0        # 翻转速度上限（比采集还慢）
HOME_MIN_TCP_Z = 300.0       # 过渡中法兰 z 下限（验证过的路线最低 353mm）
HOME_MIN_CAM_BOARD = 350.0   # 相机光心↔板心距离下限 mm（验证过最低 437 = 对准本身）
HOME_MIN_WRIST_BOARD = 330.0 # 法兰原点↔板心距离下限 mm（验证过 ≥393）
MOVE_VERIFY_JOINT_DEG = 5.0  # 运动后关节回读差超过它 → 停（>1.5° 先警告记一笔）

DRY_ROW_SLEEP_S = 0.12       # --dry-joints 每行之间歇一口气：2026-10-16 预演两次把控制器
                             # 的 :20003/:8083 敲到同时拒连（都在 ~600 次调用处，fr_rpc 的
                             # 连接复用修复后仍复现）——调用量能省则省，节流照留；预演本来
                             # 就该慢慢看数字，不赶时间
DRY_SWEEP_MIN_DJ = 20.0      # 预演议程的轻量模式：单步 Δj ≤ 它就不实测肘部（每行实测 = 11 发
                             # FK，是调用量的大头；家族翻转在 J2 上有 71.5° 天然间距，小动
                             # 不可能换家族）。>20° 的大动、兜底换解的行照测；真跑路径永远实测。

_HC_NO_ELBOW = object()      # _home_checks 的哨兵：调用方没给预计算的肘部值（该重算）

# 议程：(名字, 偏摆x 因子, 偏摆y 因子, 滚转 因子, 轴向mm(+近/−远), 侧向x mm, 侧向y mm)
# 因子乘 --tilt / --roll。偏摆 = 绕垂直于视线的轴"绕板心画弧"（板在画面里移动/变斜），
# 滚转 = 绕视线原地打转（板在画面里几乎不动）。首轮实测缺口在法兰 y（偏摆类），
# 所以偏摆占大头；滚转留着是因为它把"最富方向"钉死，λ3 才上得去。
TEMPLATE = [
    ("抬 +1     (绕板心弧)",  +1.00,  0.00,  0.00,   0.0,   0.0,   0.0),
    ("摆 +1     (绕板心弧)",   0.00, +1.00,  0.00,   0.0,   0.0,   0.0),
    ("抬 -1     (绕板心弧)",  -1.00,  0.00,  0.00,   0.0,   0.0,   0.0),
    ("摆 -1     (绕板心弧)",   0.00, -1.00,  0.00,   0.0,   0.0,   0.0),
    ("滚 +1     (原地打转)",   0.00,  0.00, +1.00,   0.0,   0.0,   0.0),
    ("滚 -1     (原地打转)",   0.00,  0.00, -1.00,   0.0,   0.0,   0.0),
    ("抬+1 滚+1",             +1.00,  0.00, +1.00,   0.0,   0.0,   0.0),
    ("摆-1 滚-1",              0.00, -1.00, -1.00,   0.0,   0.0,   0.0),
    ("近40 + 摆+0.75",         0.00, +0.75,  0.00, +40.0,   0.0,   0.0),
    ("远55 + 抬+0.75",        +0.75,  0.00,  0.00, -55.0,   0.0,   0.0),
    ("侧移 x+45",              0.00,  0.00,  0.00,   0.0, +45.0,   0.0),
    ("侧移 y-45",              0.00,  0.00,  0.00,   0.0,   0.0, -45.0),
    ("抬-0.5 摆+0.5 滚+0.5",  -0.50, +0.50, +0.50,   0.0,   0.0,   0.0),
    ("抬+0.75 侧移x-40",      +0.75,  0.00,  0.00,   0.0, -40.0,   0.0),
]

# 毒位姿黑名单（模板第 i 条，1 起；2026-10-16 三次掉线的铁证，别删）：
# 第 3 条 "抬 -1 (绕板心弧)" 整族位姿会把控制器打到崩溃——黑匣子实录（第 3 次）：
# "[3] 第3步" 的 GetInverseKinRef 请求被接收，正常 2~6ms 的 IK 在服务器里跑了 79ms、
# 然后连接被远端无响应关闭（RemoteDisconnected），随后 :20003/:8083 全拒连、法兰灯
# 回到"灭红绿蓝交替"（=通信未建立）。第 2 次预演也死在同一行（[3]第3步），而两次的
# 累计调用量差了 ~500 次 → 不是调用量问题，是这族位姿在旧固件(000-000-022)的
# ik_ref 里触发崩溃（同类请求打 [1]/[2] 的行都安然无恙）。这族位姿本来也全是
# "没有健康解/会被跳过"，留着只有崩溃风险没有收益 → 预演与真跑都整条跳过
# （含它的"←回对准"——那只是又回一次对准位姿，别的条目会做）。
AGENDA_SKIP_TEMPLATE = {3}


class Stop(RuntimeError):
    """硬停：现场有不对劲的地方，Pause + 打印，不再继续走位姿。"""


class SkipPose(RuntimeError):
    """软跳：这个位姿不走（跳解 / 逆解无解 / 太难），换下一个。"""


# ---------------------------------------------------------------- 数学小件

def _rot_axis_deg(axis, deg):
    """绕 x/y/z 转 deg 度的 3×3（跟 calibrate._rot_x/y/z 同一套约定）。"""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def _T_to_pose6(T):
    """4×4 → [x,y,z,rx,ry,rz]（mm/度）。是 calibrate._pose6_to_T 的逆（ZYX 欧拉角）。

    R = Rz(rz)·Ry(ry)·Rx(rx) 时：ry = asin(−R20)、rx = atan2(R21, R22)、rz = atan2(R10, R00)。
    --selftest 会拿 200 个随机位姿来回换算验它，自己的数学不在真机上试。
    """
    R = np.asarray(T[:3, :3], float)
    ry = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    rx = math.atan2(float(R[2, 1]), float(R[2, 2]))
    rz = math.atan2(float(R[1, 0]), float(R[0, 0]))
    x, y, z = (float(v) for v in T[:3, 3])
    return [x, y, z, math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def _trans_delta(v):
    """工具系里纯平移 v=(dx,dy,dz) mm → 4×4。"""
    T = np.eye(4)
    T[:3, 3] = (float(v[0]), float(v[1]), float(v[2]))
    return T


def _conn_ish(err):
    """这个异常像不像"连接断了/服务没了"（而不是机器人明确回了一个 Fault）？

    用途：_solve_pinned 里区分"这一组逆解无解"（正常，换下一组）和"控制器整个联系不上"
    （如实上抛，别把断线伪装成"无解→跳过"把整段议程悄悄跳完）。
    """
    s = str(err)
    return any(k in s for k in ("Connection", "refused", "timed out", "Timeout",
                                "RemoteDisconnected", "Broken pipe", "reset"))


def _sub_steps(tx, ty, rz, axial, lat_x, lat_y):
    """一个议程条目 → 若干步（每步都是"相对对准位姿"的累计量，不是增量）。

    为什么必须拆：单条 MoveL 的位移有闸（config 100mm / 程序 95mm），而
    ① 20° 的弧一口气走 = 弦长 2·440·sin10° = 153mm；
    ② 滚转绕的圆心是板心，它不在视线轴上（在工具系里偏 ~113mm），所以
       "原地打转 25°"本身也带 ~50mm 的平移——它和弧的最后一步叠起来离线量到 103mm。
    拆法：偏摆先按 ARC_STEP_DEG 分步弧行，滚转再按 ROLL_STEP_DEG 叠上去，平移压在最后一步。
    """
    out = []
    nt = max(1, int(math.ceil(max(abs(tx), abs(ty)) / ARC_STEP_DEG)))
    for k in range(1, nt + 1):
        f = k / nt
        out.append([tx * f, ty * f, 0.0, 0.0, 0.0, 0.0])
    if abs(rz) > 1e-9:
        nr = max(1, int(math.ceil(abs(rz) / ROLL_STEP_DEG)))
        for k in range(1, nr + 1):
            out.append([tx, ty, rz * k / nr, 0.0, 0.0, 0.0])
    # 平移一律压在最后一步上（没有滚转时最后一步就是偏摆的收尾那步——别再加一步空的：
    # 那样会连发两条一模一样的指令，"走 0mm"）
    out[-1][3], out[-1][4], out[-1][5] = axial, lat_x, lat_y
    return out


def _entry_delta(tx, ty, rz, axial, lat_x, lat_y, center_p):
    """议程条目 → 相对 4×4（左乘在对准位姿 T_aim 上）。

    旋转一律【绕 center_p 这个点】做（center_p = 板心在工具系里的估计位置）：
    绕视线转 = 板心在轴上不变 → 原地打转；绕垂直于视线的轴转 = 板心被当圆心 → 相机画弧，
    距离和"看没看住板"都自动保持。之后再做工具系平移（轴向/侧向）。
    """
    R = np.eye(3)
    for ax, dg in (("x", tx), ("y", ty), ("z", rz)):
        if abs(dg) > 1e-9:
            R = R @ _rot_axis_deg(ax, dg)
    A = np.eye(4)
    A[:3, :3] = R
    if center_p is not None:
        p = np.asarray(center_p, float)
        A[:3, 3] = p - R @ p
    return A @ _trans_delta((lat_x, lat_y, axial))


def _math_selftest(verbose=True):
    """离线数学校验：欧拉角来回换算、绕板心旋转的不动点、议程生成。不连机器人。"""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        pose = [rng.uniform(-600, 600), rng.uniform(-600, 600), rng.uniform(100, 800),
                rng.uniform(-175, 175), rng.uniform(-85, 85), rng.uniform(-175, 175)]
        T = C._pose6_to_T(pose)
        T2 = C._pose6_to_T(_T_to_pose6(T))
        worst = max(worst, float(np.abs(T - T2).max()))
    if worst > 1e-9:
        raise SystemExit(f"× 欧拉角来回换算对不上：{worst:.2e}")
    # 绕板心旋转：圆心必须是不动点（Δ·p = p）
    p = np.array(CAM_OFFSET_DEFAULT, float) + np.array([0.0, 0.0, TARGET_DIST_MM])
    for rots in ((20, 0, 0), (0, -20, 0), (0, 0, 25), (15, 15, 15)):
        D = _entry_delta(*rots, 0.0, 0.0, 0.0, p)
        q = D[:3, :3] @ p + D[:3, 3]
        if float(np.linalg.norm(q - p)) > 1e-9:
            raise SystemExit(f"× 绕板心旋转的圆心被移走了 {np.linalg.norm(q - p):.2e}mm")
    # 纯平移条目不能带旋转
    D = _entry_delta(0, 0, 0, 40.0, -45.0, 0.0, p)
    if float(np.abs(D[:3, :3] - np.eye(3)).max()) > 1e-12:
        raise SystemExit("× 纯平移条目里混进了旋转")
    if verbose:
        print(f"✓ 离线数学校验过：欧拉角来回差 {worst:.1e}（200 个随机位姿）；绕板心旋转圆心不动")


# ---------------------------------------------------------------- 主流程

class Auto:
    def __init__(self, args):
        self.args = args
        self.shots = Path(args.shots)
        self.det = None
        self.K = self.D = None
        self.cap = None
        self.arm = None
        self.saved = []            # [(帧文件名, pose6, R 相机←板)]
        self.kinds = []            # 每张相对上一张的动作分类（拖动版 HUD 里那一栏）
        self.T0 = None             # 出发位姿
        self.T_aim = None          # 对准后的位姿（议程基准）
        self.p_board = None        # 板心在工具系里的估计（绕板心画弧的圆心）
        self.cam_offset = tuple(float(v) for v in args.cam_offset)
        self.img_w, self.img_h = config.CAMERA_W, config.CAMERA_H
        self.J = None              # 2×2 px/mm：法兰 x/y 位移 → 画面 x/y 位移
        self.J_dist = None         # 学 J 时的板距 mm
        self.ref_board = None      # 对准位姿上第一次量的 (R_cb, t_cb)：用来验"板没动过"
        self.target_dist = float(args.target_mm)
        self.f_tilt = float(args.tilt)
        self.f_roll = float(args.roll)
        self._last_frame = None    # 最近一帧（画面"追上"判据的锚）
        self._wait_fresh = False   # 刚运动完：下次抓帧先等画面追上
        self.fresh_log = []        # [(等待s, 见到跳变?, 跳变幅度)]——收尾给统计
        self.raw = []              # [(帧名, 原始法兰位姿6, j6)]——定子修正的底账
        self._jlog = None          # joints_log.csv 的句柄（真跑才开）
        self._jlw = None
        self.ez_min = None         # 全程目标关节的肘部估计最低值（收尾统计）

    # ---------- 资源 ----------

    def _check_disk(self):
        if self.args.go:                               # 预演绝不写盘、连目录都不建
            self.shots.mkdir(parents=True, exist_ok=True)
        old = sorted(self.shots.glob("*.jpg"))
        if old and self.args.go:
            stamp = time.strftime("%Y%m%d_%H%M")
            sys.exit(
                f"存帧目录里已经有 {len(old)} 张 jpg：{self.shots}\n"
                f"  自动版是从头存一套完整的，先把它挪走再跑：\n"
                f"    mv {self.shots} {self.shots.parent}/{self.shots.name}_废旧_{stamp}\n"
                f"  （旧的那批先别删——里面还有 24/24 的帧，说不定能挑出来用）")
        self.K, self.D = C._try_load_K(self.args.intr)
        if self.K is None:
            sys.exit(f"读不到内参 {self.args.intr} —— 手眼靠它把板姿解出来，先跑 calibrate.py --solve")
        return old

    def _connect(self):
        self.arm = Arm()
        self.arm.connect()
        self.det = C.BoardDetector()

    def _close(self):
        if self._jlog is not None:
            try:
                self._jlog.close()
            except Exception:
                pass
            self._jlog = None
        if self.arm is not None:
            try:
                self.arm.close()
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    # ---------- 只读 ----------

    def _pose(self):
        last = None
        for _ in range(4):
            try:
                return [float(v) for v in self.arm.r.flange_pose()]
            except Exception as e:                      # noqa: BLE001
                last = e
                time.sleep(0.15)
        raise Stop(f"读法兰位姿读不到（RPC 连不上/超时）：{last}")

    def _joints(self):
        try:
            return [float(v) for v in self.arm.joint_deg()]
        except Exception as e:                          # noqa: BLE001
            raise Stop(f"读关节角读不到：{e}")

    def _st(self, name, default=None):
        try:
            return getattr(self.arm.st, name)
        except AttributeError:
            return default

    def _grab_raw(self):
        for _ in range(6):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.05)
        raise Stop("相机读帧失败（连读 6 次没成）——相机是不是被别处抢走了？")

    def _grab(self):
        """抓一帧；刚运动过就先等画面流"追上"（这台相机的固定脾气，见 FRESH_* 常量）。"""
        if self._wait_fresh:
            frame = self._grab_fresh()
        else:
            frame = self._grab_raw()
        self._last_frame = frame
        return frame

    def _grab_fresh(self):
        """连抓帧直到画面出现"跳变"（= 流真的追上了），或按稳定/超时兜底。返回追上那帧。

        probe_wait 实测（2026-10-15 深夜）：20mm 运动后流先给 ~10s 旧画面，追上那一刻
        相邻两帧均差 25.3（静止时只有 ~2），所以跳变非常好认；跳变后再连稳 3 帧收工。
        位移 <26px 的小移动对不上跳变判据——等满 FRESH_STILL_MIN_S 且一直稳就按追上收。
        """
        t0, prev, stable, jumped, mag = time.time(), self._last_frame, 0, False, 0.0
        while True:
            frame = self._grab_raw()
            if prev is not None:
                d = float(cv2.absdiff(frame, prev).mean())
                if d >= FRESH_JUMP_MEAN:
                    jumped, stable, mag = True, 0, max(mag, d)
                elif d <= FRESH_STILL_MEAN:
                    stable += 1
            prev = frame
            el = time.time() - t0
            if jumped and stable >= FRESH_STABLE_N:
                print(f"      · 画面追上：等 {el:.1f}s（跳变均差 {mag:.1f}）", flush=True)
                break
            if el >= FRESH_WAIT_MAX_S:
                print(f"      ⚠ 画面等了 {FRESH_WAIT_MAX_S:.0f}s 没等到跳变——"
                      f"这一帧可能还是运动前的旧画面！", flush=True)
                break
            if (not jumped) and stable >= FRESH_STILL_N and el >= FRESH_STILL_MIN_S:
                print(f"      · 画面 {el:.1f}s 没见跳变（位移小或流正常）——按已追上继续",
                      flush=True)
                break
            time.sleep(FRESH_TICK_S)
        self._wait_fresh = False
        self.fresh_log.append((el, jumped, mag))
        return prev

    def _metrics(self, frame):
        """一帧给出全部判据：角点数/板宽/离边/清晰度/板姿/距离/板心像素偏移。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.det.detect(gray)
        n = 0 if corners is None else int(len(corners))
        m = {"n": n, "side": 0.0, "blur": 0.0, "margin": 0.0, "dist": None,
             "tilt": None, "off": (0.0, 0.0), "center_px": None,
             "R_cb": None, "t_cb": None, "span": (0.0, 0.0)}
        if not n:
            return m
        m["blur"] = C.board_blur(gray, corners)
        _x, _y, m["side"] = C.frame_stats(corners, (self.img_w, self.img_h))
        pts = corners.reshape(-1, 2)
        m["span"] = (float(pts[:, 0].ptp()), float(pts[:, 1].ptp()))   # 角点包围盒 (宽, 高)
        m["margin"] = float(min(pts[:, 0].min(), pts[:, 1].min(),
                                (self.img_w - 1) - pts[:, 0].max(),
                                (self.img_h - 1) - pts[:, 1].max()))
        R, t = C._solve_board(self.det, corners, ids, self.K, self.D)
        if R is not None:
            m["R_cb"], m["t_cb"] = R, np.asarray(t, float).ravel()
            m["tilt"] = math.degrees(math.acos(min(1.0, abs(float(R[2, 2])))))
            c_b = np.asarray(self.det.objp, np.float64).mean(axis=0)
            p_cam = m["t_cb"] + R @ c_b
            m["dist"] = float(np.linalg.norm(p_cam))
            px, _ = cv2.projectPoints(p_cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                      self.K, self.D)
            m["center_px"] = px.reshape(2)
            m["off"] = (px.reshape(2) - np.array([self.K[0, 2], self.K[1, 2]]))
        return m

    # ---------- 肘部高度守卫（撞机事故的产物） ----------

    @staticmethod
    def _fnum(v, w=4, dec=0):
        """数值打印小件：None（拟合不可信）打成 nan，不炸格式。"""
        return f"{float('nan') if v is None else v:>{w}.{dec}f}"

    def _elbow_z(self, joints):
        """肘部（J3 轴线）高度的 FK 圆拟合估计（纯只读）。返回 dict(z, res, r, n) 或 None。

        技法（2026-10-16 首用于解困，见 memory: fr3-elbow-geometry）：固定 J1/J2/J4/J5/J6，
        让 J3 扫一串值，法兰点共圆；圆心即 J3 轴线上一点，其 z 就是"肘部高度"。
        真机实测残差 0.000mm、半径 203~333mm（随远端构型变）。
        返回 None = 拟合不可信（点太少/共线/残差大/半径离谱）——调用方决定怎么处理。
        """
        base = [float(v) for v in joints]
        pts = []
        for off in ELBOW_SWEEP_DEG:
            j = list(base)
            j[2] = max(-160.0, min(160.0, base[2] + off))
            try:
                p = self.arm.fk(j)
            except Exception:                              # noqa: BLE001
                return None
            pts.append([float(v) for v in p[:3]])
        P = np.asarray(pts, float)
        keep = [0]                                         # 去重：贴太近的点不算数
        for i in range(1, len(P)):
            if min(float(np.linalg.norm(P[i] - P[k])) for k in keep) > 5.0:
                keep.append(i)
        if len(keep) < 6:
            return None
        P = P[keep]
        c = P.mean(0)
        _u, S, Vt = np.linalg.svd(P - c, full_matrices=False)
        if S[1] < 1.0 or S[2] > 2.0 * ELBOW_FIT_RES_MM:    # 共线 / 不共面 → 不可信
            return None
        u, v = Vt[0], Vt[1]
        q = np.column_stack([(P - c) @ u, (P - c) @ v])
        A = np.column_stack([2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))])
        cx, cy, c0 = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)[0]
        r = math.sqrt(max(float(c0) + cx * cx + cy * cy, 0.0))
        res = float(np.abs(np.linalg.norm(q - [cx, cy], axis=1) - r).max())
        if res > ELBOW_FIT_RES_MM or not (ELBOW_FIT_R_MIN <= r <= ELBOW_FIT_R_MAX):
            return None
        center = c + cx * u + cy * v
        return {"z": float(center[2]), "res": res, "r": float(r), "n": len(keep)}

    def _sol_ok(self, j, j_ref, sweep_min_dj=None):
        """候选解过不过闸：返回 (过不过, Δj°, 肘部 z 或 None)。

        闸就两条（和撞机教训一一对应）：Δj ≤ JOINT_JUMP_DEG（不换构型/不腕翻转）、
        肘部 z ≥ ELBOW_MIN_Z_MM（不压进底座横梁）。顺序是先算便宜的 Δj——只有过了的
        才花 11 次 FK 去量肘部。
        sweep_min_dj 给了值（预演轻量模式）：Δj ≤ 它的小动直接算过、不实测（家族翻转在
        J2 上有 71.5° 天然间距，小动不可能换家族）；真跑路径不传 → 永远实测。
        肘部拟合不可信（None）不否决，只打标。
        """
        dj = max(abs(a - b) for a, b in zip(j, j_ref))
        if dj > JOINT_JUMP_DEG:
            return False, dj, None
        if sweep_min_dj is not None and dj <= sweep_min_dj:
            return True, dj, None
        ez = self._elbow_z(j)
        ez_z = None if ez is None else float(ez["z"])
        if ez_z is not None and ez_z < ELBOW_MIN_Z_MM:
            return False, dj, ez_z
        return True, dj, ez_z

    def _solve_pinned(self, pose, j_ref, tag="", light=False):
        """逆解并钉家族：ik_ref(ref=当前关节) 主答 → 不过闸就八组枚举兜底。

        返回 (关节×6, Δj°, 肘部 z 或 None, 来源) ；找不到"健康且连续"的解 → None。
        来源 = "ik_ref" 或 "cfgN"（枚举第 N 组）。
        light=True（预演议程用）：单步 ≤ DRY_SWEEP_MIN_DJ 的行不实测肘部（省 11 发 FK/行）；
        真跑路径永远 light=False（全闸照跑）。

        为什么要有枚举兜底：2026-10-16 dry 预演实测，ik_ref 在链式调用里会把位姿解到
        深家族（[2] 回对准 Δmax 71.5° = |−128.69−(−200.18)|，正是两家族的 J2 间距）——
        它只是"择近启发式"，一旦中途翻过去，按它的输出接着解，后面整条链都会翻。
        枚举兜底把"健康 + 连续"变成挑选判据本身：家族由物理量（肘部高度）钉死，
        不再由启发式的输出背书。找不到健康解 → 返回 None → 调用方跳过（不发指令）。
        """
        primary, dj_p, ez_p = None, None, None
        try:
            j = self.arm.ik_ref(pose, j_ref)
            if len(j) == 6 and max(abs(float(v)) for v in j) > 1e-6:
                primary = [float(v) for v in j]
        except FRRpcError as e:
            if _conn_ish(e):                       # 连接死了 → 别拿"无解"顶包，如实抛
                raise
        if primary is not None:
            ok, dj_p, ez_p = self._sol_ok(
                primary, j_ref, sweep_min_dj=DRY_SWEEP_MIN_DJ if light else None)
            if ok:
                return primary, dj_p, ez_p, "ik_ref"
        cands = []
        for cfg in range(8):
            try:
                j = self.arm.ik(pose, cfg)
            except FRRpcError as e:
                if _conn_ish(e):
                    raise
                continue                            # 这一组无解/被拒：换下一组
            if len(j) != 6 or max(abs(float(v)) for v in j) < 1e-6:
                continue
            if primary is not None and max(abs(a - b)
                                           for a, b in zip(j, primary)) < 0.05:
                continue                            # 和主答同一个解，别重复量
            cands.append((f"cfg{cfg}", [float(v) for v in j]))
        best = None
        # 按 Δj 升序逐个过闸，取第一个健康的——与"过闸者里最连续"等价，但每行只给到
        # 必要的那 1 个候选实测肘部（老写法给全部候选都测，是预演调用量的大头；
        # 2026-10-16 两次预演各在 ~600 次调用处把控制器打掉线的教训）
        cands.sort(key=lambda c_: max(abs(a - b) for a, b in zip(c_[1], j_ref)))
        for src, j in cands:
            dj = max(abs(a - b) for a, b in zip(j, j_ref))
            if dj > JOINT_JUMP_DEG:
                break                                # 排序后一超闸，后面的只会更远
            ok, dj, ez_z = self._sol_ok(
                j, j_ref, sweep_min_dj=DRY_SWEEP_MIN_DJ if light else None)
            if ok:
                best = (j, dj, ez_z, src)
                break
        def _ezs(v):                                 # None=轻量模式没实测（或拟合不可信）
            return "—" if v is None else self._fnum(v)
        if best is None:
            why = (f"ik_ref Δmax {dj_p:.0f}°、肘z {_ezs(ez_p)}" if primary is not None
                   else "ik_ref 没给解")
            print(f"      ⚠ {tag}：{why}；枚举 8 组也没有健康解 → 这条跳过", flush=True)
        elif primary is not None:
            print(f"      （{tag}：ik_ref 没过闸（Δmax {dj_p:.0f}°、肘z {_ezs(ez_p)}）"
                  f"→ 枚举兜底选了 {best[3]}：Δmax {best[1]:.0f}°、肘z "
                  f"{_ezs(best[2])}）", flush=True)
        else:
            print(f"      （{tag}：ik_ref 没给解 → 枚举取 {best[3]}）", flush=True)
        return best

    def _plan_home(self, j_cur, j_target):
        """当前关节 → 目标关节的纯关节线性分步表（每步 ≤ HOME_STEP_DEG，末步精确到位）。"""
        dj = max(abs(a - b) for a, b in zip(j_target, j_cur))
        n = max(1, int(math.ceil(dj / HOME_STEP_DEG)))
        return [[a + (k / n) * (b - a) for a, b in zip(j_cur, j_target)]
                for k in range(1, n + 1)]

    def _home_checks(self, joints, pb_base, elbow_z=_HC_NO_ELBOW):
        """一组关节下的安全量：肘部 z、法兰 z、法兰/相机光心离板心的距离（全只读）。

        elbow_z 给值时不再重算肘部（_solve_pinned 已经量过——一次 11 发 FK，别量两遍）。
        """
        c = {"tcp_z": None, "elbow_z": None, "cam_board": None, "tcp_board": None}
        try:
            p = self.arm.fk(joints)
        except Exception:                                  # noqa: BLE001
            return c
        c["tcp_z"] = float(p[2])
        c["tcp_board"] = float(np.linalg.norm(np.asarray(p[:3], float) - pb_base))
        T = C._pose6_to_T([float(v) for v in p])
        cam = T[:3, 3] + T[:3, :3] @ np.asarray(self.cam_offset, float)
        c["cam_board"] = float(np.linalg.norm(cam - pb_base))
        if elbow_z is _HC_NO_ELBOW:
            ez = self._elbow_z(joints)
            c["elbow_z"] = None if ez is None else float(ez["z"])
        else:
            c["elbow_z"] = elbow_z
        return c

    # ---------- 前检 / 状态 ----------

    def _preflight(self, need_auto):
        rows = [("8083 状态帧在流动", self.arm.st.age < 2.0, f"距今 {self.arm.st.age:.1f}s"),
                ("急停没有按下", self._st("estop") == 0, f"estop={self._st('estop')}"),
                ("控制器无错误码", self._st("error_code") == 0, f"error_code={self._st('error_code')}"),
                ("当前不在运动", self._st("motion_done") == 1, f"motion_done={self._st('motion_done')}"),
                ("不在拖动模式", self._st("program_state") != 4,
                 f"program_state={self._st('program_state')}（4=拖动）")]
        if need_auto:
            rm = self._st("robot_mode")
            rows.append(("模式 = 自动（robot_mode 0）", rm == 0,
                         f"robot_mode={rm}（0=自动 1=手动 2=拖动）→ 在示教器上切到自动模式，"
                         f"本程序绝不自己切"))
        ok = True
        for name, good, detail in rows:
            print(f"  {'✓' if good else '×'} {name}：{detail}")
            ok = ok and good
        return ok

    # ---------- 运动 ----------

    def _box_violation(self, T_target):
        """目标位姿违不违反安全盒？返回一句人话或 None。预演和真跑共用这一条规则。"""
        t, t0 = np.asarray(T_target[:3, 3], float), np.asarray(self.T0[:3, 3], float)
        for i, nm in enumerate("xyz"):
            lim = BOX_DXY_MM if i < 2 else BOX_DZ_MM
            if abs(t[i] - t0[i]) > lim:
                return f"{nm} 超局部盒（差 {t[i]-t0[i]:+.0f}mm > {lim:.0f}）"
        if t[2] < Z_FLOOR_MM:
            return f"z={t[2]:.0f}mm 低于地板线 {Z_FLOOR_MM:.0f}mm"
        return None

    def _box_check(self, T_target, tag):
        bad = self._box_violation(T_target)
        if bad:
            raise Stop(f"{tag}：目标 {bad}——拒发")

    def _move_abs(self, T_target, tag, quiet=False):
        """走到绝对位姿：局部盒 → 位移 → 钉家族逆解 → 跳变/肘部双闸 → 发指令（非阻塞）
        → 等停稳 → 位姿自检 → 关节回读自检 + 记一行关节日志。

        ⚠ 2026-10-16 撞机教训（memory: auto-handeye-collector-status）：① 逆解改用
        GetInverseKinRef（ref=当前关节）显式钉家族——旧的自动解≈"就近解"，在肘部塌陷的
        家族里会把臂身压进底座横梁；② 逆解出来的关节【原样下传】给 arm.move_l，不再有
        第二发隐式逆解；③ 肘部（J3 轴线）高度估计 < ELBOW_MIN_Z_MM → 拒发；④ ik_ref
        自己偶尔也会翻家族（dry 预演实测）→ _solve_pinned 的八组枚举兜底，健康解才走。
        """
        pose = _T_to_pose6(T_target)
        self._box_check(T_target, tag)
        cur = self._pose()
        d = float(np.linalg.norm(np.asarray(pose[:3]) - np.asarray(cur[:3])))
        if d > MOVE_MAX_MM:
            raise Stop(f"{tag}：单条位移 {d:.0f}mm > {MOVE_MAX_MM:.0f}mm（程序 bug，宁可停）")
        j_cur = self._joints()
        sol = self._solve_pinned(pose, j_cur, tag=tag)
        if sol is None:
            raise SkipPose(f"{tag}：ik_ref + 八组枚举都没有'健康且连续'的解"
                           f"（Δj≤{JOINT_JUMP_DEG:.0f}° 且肘部 ≥{ELBOW_MIN_Z_MM:.0f}mm）"
                           f"——跳过这个位姿（没发任何指令）")
        j_new, dj, ez_z, src = sol
        if ez_z is not None:
            self.ez_min = ez_z if self.ez_min is None else min(self.ez_min, ez_z)
        if not quiet:
            print(f"    → {tag}：走 {d:>4.0f}mm（关节最大动 {dj:>4.0f}°"
                  + (f"，肘 z≈{ez_z:.0f}" if ez_z is not None else "，肘 z≈?（拟合没过）")
                  + "）", flush=True)
        try:
            self.arm.move_l(pose, speed=self.args.speed, blocking=False, joints=j_new)
        except SafetyError as e:
            raise Stop(f"{tag}：被 Arm 的安全闸拦下（{e}）")
        got = self._settle(tag)
        dp = float(np.linalg.norm(np.asarray(got[:3]) - np.asarray(pose[:3])))
        dr = C._rot_axis_angle(C._pose6_to_T(got)[:3, :3].T
                               @ C._pose6_to_T(pose)[:3, :3])[0]
        if dp > MOVE_VERIFY_MM or dr > MOVE_VERIFY_DEG:
            raise Stop(f"{tag}：没走到位（差 {dp:.1f}mm / {dr:.2f}°）——"
                       f"是不是被碰撞检测/急停打断了？现场看一眼再重跑")
        j_act = self._joints()
        dja = max(abs(a - b) for a, b in zip(j_act, j_new))
        if dja > MOVE_VERIFY_JOINT_DEG:
            raise Stop(f"{tag}：关节回读和命令差 {dja:.1f}°（> {MOVE_VERIFY_JOINT_DEG:.0f}°）"
                       f"——姿态和命令的不是一回事，停")
        if dja > 1.5:
            print(f"      ⚠ 关节回读差 {dja:.1f}°（超过平常 ~0 的噪声，记一笔）", flush=True)
        self._jlog_row(tag, j_new, j_act, dja, ez_z, got)
        self._wait_fresh = True          # 刚动过：下次抓帧先等画面流追上
        return got

    def _jlog_row(self, tag, j_t, j_a, dja, ez_z, tcp):
        """关节全量日志（旁车）：只记位姿的话，家族/姿态变化在日志里完全隐形（撞机教训）。"""
        if not self.args.go:
            return
        if self._jlog is None:
            self.shots.mkdir(parents=True, exist_ok=True)
            p = self.shots / "joints_log.csv"
            new = not p.exists()
            self._jlog = open(p, "a", newline="")
            self._jlw = csv.writer(self._jlog)
            if new:
                self._jlw.writerow(["time", "tag"]
                                   + [f"jT{i}" for i in range(1, 7)]
                                   + [f"jA{i}" for i in range(1, 7)]
                                   + ["dj_max", "elbow_z_mm"]
                                   + ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"])
        self._jlw.writerow([time.strftime("%H:%M:%S"), tag]
                           + [f"{v:.3f}" for v in j_t] + [f"{v:.3f}" for v in j_a]
                           + [f"{dja:.3f}", "" if ez_z is None else f"{ez_z:.1f}"]
                           + [f"{v:.3f}" for v in tcp])
        self._jlog.flush()

    def _move_to(self, T_target, tag, board_check=True, quiet=False):
        """走一段可能很长的路：按 60mm / 8° 分段直线插值（位置线性、姿态走 Rodrigues）。

        为什么不能直接一条 MoveL：单条位移有闸（程序 95mm / config 100mm），而"回对准位姿"
        这种回程在 440mm 半径上走 20° 弧就是 139mm（仿真里第一次就撞上了）。分段后每段
        ≤60mm，板上还在画面里（中间点顺手看一眼）。短距离时它就是一条 _move_abs。
        """
        T_cur = C._pose6_to_T(self._pose())
        d = float(np.linalg.norm(np.asarray(T_target[:3, 3]) - np.asarray(T_cur[:3, 3])))
        ang = C._rot_axis_angle(np.asarray(T_target[:3, :3]) @ T_cur[:3, :3].T)[0]
        n = max(1, int(math.ceil(d / PATH_STEP_MM)), int(math.ceil(ang / PATH_STEP_DEG)))
        if n == 1:
            return self._move_abs(T_target, tag, quiet=quiet)
        rvec = cv2.Rodrigues(np.asarray(T_target[:3, :3]) @ T_cur[:3, :3].T)[0]
        if not quiet:
            print(f"    → {tag}：{d:.0f}mm / {ang:.0f}° 分 {n} 段走", flush=True)
        for k in range(1, n + 1):
            f = k / n
            T_k = np.eye(4)
            T_k[:3, :3] = cv2.Rodrigues(rvec * f)[0] @ T_cur[:3, :3]
            T_k[:3, 3] = T_cur[:3, 3] + f * (np.asarray(T_target[:3, 3]) - np.asarray(T_cur[:3, 3]))
            self._move_abs(T_k, f"{tag} 第{k}/{n}段", quiet=True)
            if board_check and k < n:
                m = self._metrics(self._grab())
                if m["n"] < 12:
                    raise Stop(f"{tag}：走到第 {k}/{n} 段板只剩 {m['n']} 角点——停")
        return self._pose()

    def _settle(self, tag):
        """等臂真停稳：motion_done=1 且连读两次位姿都不动（复用拖动版的 0.3mm/0.15°）。"""
        t0, last = time.time(), None
        while time.time() - t0 < SETTLE_TIMEOUT_S:
            if self.arm.st.age > 3.0:
                raise Stop(f"{tag}：8083 状态流断了 {self.arm.st.age:.1f}s——停")
            md = self._st("motion_done", 0)
            p = self._pose()
            R = C._pose6_to_T(p)[:3, :3]
            if last is not None:
                dp = float(np.linalg.norm(np.asarray(p[:3]) - np.asarray(last[0][:3])))
                dr = C._rot_axis_angle(R.T @ last[1])[0]
                if md == 1 and dp <= SETTLE_MM and dr <= SETTLE_DEG:
                    return p
            last = (p, R)
            time.sleep(0.3)
        raise Stop(f"{tag}：等停稳超时 {SETTLE_TIMEOUT_S:.0f}s")

    # ---------- 对准 ----------

    def _aim(self, tag):
        """把板心拉进画面中心、距离拉到 target_dist；返回 (帧, metrics)。"""
        if self.J is None:
            raise Stop("还没学雅可比就要求对准（程序顺序错了）")
        for it in range(AIM_MAX_ITER + 1):
            frame = self._grab()
            m = self._metrics(frame)
            if m["n"] < 6 or m["R_cb"] is None:
                raise Stop(f"{tag}：认不出板（{m['n']} 角点）——乱动更危险，停下来看现场")
            off = np.asarray(m["off"], float)
            d = float(m["dist"])
            c_ok = float(np.linalg.norm(off)) <= CENTER_TOL_PX
            d_ok = abs(d - self.target_dist) <= DIST_TOL_MM
            g_ok = (m["n"] >= C.HE_NEED_CORNERS and m["margin"] >= C.HE_EDGE_MARGIN_PX)
            if c_ok and d_ok and g_ok:
                print(f"    · {tag} 对准 OK：板心偏 {np.linalg.norm(off):>3.0f}px、"
                      f"距 {d:>4.0f}mm、离边 {m['margin']:>3.0f}px（修正 {it} 次）", flush=True)
                return frame, m
            if it == AIM_MAX_ITER:
                break
            J = self.J * (self.J_dist / d)              # px/mm ∝ 1/距离
            dxy = np.linalg.solve(J, -off)
            n_ = float(np.linalg.norm(dxy))
            if n_ > AIM_CLAMP_MM:
                dxy = dxy * (AIM_CLAMP_MM / n_)
            dz = float(np.clip(d - self.target_dist, -AIM_CLAMP_Z, AIM_CLAMP_Z))
            if c_ok:
                dxy = np.zeros(2)
            if d_ok:
                dz = 0.0
            cur = self._pose()
            self._move_abs(C._pose6_to_T(cur) @ _trans_delta((dxy[0], dxy[1], dz)),
                           f"{tag} 对准修正{it+1}", quiet=True)
        raise SkipPose(f"{tag}：{AIM_MAX_ITER} 次修正还是对不准"
                       f"（最后偏 {np.linalg.norm(off):.0f}px、距 {d:.0f}mm）")

    def _probe(self):
        """探针：沿法兰 x/y/z 各走 ±PROBE_MM，量出 mm→px / mm→距离 的换算。"""
        base = C._pose6_to_T(self._pose())
        f0 = self._grab()
        m0 = self._metrics(f0)
        if m0["R_cb"] is None:
            raise Stop(f"探针前认不出板（{m0['n']} 角点）——先把板摆进画面再跑")
        d0, c0 = float(m0["dist"]), np.asarray(m0["center_px"], float)
        cols = []
        for axis, vec in (("x", (PROBE_MM, 0.0, 0.0)), ("y", (0.0, PROBE_MM, 0.0))):
            got = None
            for it in range(PROBE_MAX_ITER):
                self._move_abs(base @ _trans_delta(vec), f"探针 +{PROBE_MM:.0f}mm 沿法兰 {axis}")
                m = self._metrics(self._grab())
                self._move_abs(base, "探针回位", quiet=True)
                if m["R_cb"] is not None:
                    got = np.asarray(m["center_px"], float)
                    break
                print(f"    · {axis} 方向探针这一趟没认出板，重来（{it+1}/{PROBE_MAX_ITER}）",
                      flush=True)
            if got is None:
                raise Stop(f"{axis} 方向探针连走 {PROBE_MAX_ITER} 趟都认不出板——停")
            cols.append((got - c0) / PROBE_MM)
        J = np.column_stack(cols)
        detJ = float(np.linalg.det(J))
        if abs(detJ) < 1e-6:
            raise Stop(f"探针学出来的雅可比是退化的（det={detJ:.2e}）——相机没装正？停")
        self.J, self.J_dist = J, d0
        exp = float(self.K[0, 0]) / d0
        got = float(np.sqrt(abs(detJ)))
        print(f"    · 雅可比：{PROBE_MM:.0f}mm→{np.linalg.norm(cols[0]):.1f}px / "
              f"{np.linalg.norm(cols[1]):.1f}px；量级 {got:.2f}px/mm vs 期望 fx/d={exp:.2f}"
              f"（{'差得多，记着看' if abs(got/exp-1) > 0.25 else '对得上'}）", flush=True)
        # z 方向：只是验方向（+法兰 z 应该靠近板、距离变小）
        self._move_abs(base @ _trans_delta((0.0, 0.0, PROBE_MM)), "探针 +20mm 沿法兰 z")
        mz = self._metrics(self._grab())
        self._move_abs(base, "探针回位", quiet=True)
        if mz["R_cb"] is None:
            raise Stop("z 方向探针这趟认不出板——停")
        dz = float(mz["dist"]) - d0
        print(f"    · z 探针：距离 {d0:.0f} → {mz['dist']:.0f}mm（期望 −{PROBE_MM:.0f}）"
              f"{'' if -1.5*PROBE_MM < dz < -0.5*PROBE_MM else '  ← 方向/量级不对，对准循环会吃亏'}",
              flush=True)

    # ---------- 采集 ----------

    def _capture(self, frame, m, tag):
        """过门 → 停稳 → 存帧 + 立刻更新 csv。返回 (存了没, 原因)。"""
        if m["n"] < C.HE_NEED_CORNERS:
            return False, f"角点 {m['n']}/24（缺角的板姿朝向会偏，拖动版也拒收）"
        if m["margin"] < C.HE_EDGE_MARGIN_PX:
            return False, f"离画面边只剩 {m['margin']:.0f}px（要 ≥{C.HE_EDGE_MARGIN_PX}）"
        if m["side"] < C.HE_BOARD_MIN_PX:
            return False, f"板宽 {m['side']:.0f}px < {C.HE_BOARD_MIN_PX:.0f}px"
        p1 = None
        for _ in range(3):
            p1 = self._pose()
            time.sleep(0.35)
            p2 = self._pose()
            dp = float(np.linalg.norm(np.asarray(p1[:3]) - np.asarray(p2[:3])))
            dr = C._rot_axis_angle(C._pose6_to_T(p1)[:3, :3].T
                                   @ C._pose6_to_T(p2)[:3, :3])[0]
            if dp <= C.HE_STILL_MM and dr <= C.HE_STILL_DEG:
                break
        else:
            return False, f"停稳验了三次还没过（{dp:.2f}mm/{dr:.3f}°）"
        for nm, pl, _R in self.saved:                   # 与【所有】已存帧比，不只上一张
            dp = float(np.linalg.norm(np.asarray(p2[:3]) - np.asarray(pl[:3])))
            dr = C._rot_axis_angle(C._pose6_to_T(p2)[:3, :3].T
                                   @ C._pose6_to_T(pl)[:3, :3])[0]
            if dp <= C.HE_DUP_MM and dr <= C.HE_DUP_DEG:
                return False, f"跟 {nm} 是重复位姿（{dp:.2f}mm/{dr:.2f}°）"
        name = time.strftime("handeye_%Y%m%d_%H%M%S_") + f"{len(self.saved):03d}.jpg"
        if not cv2.imwrite(str(self.shots / name), frame):
            raise Stop(f"帧写不进去：{self.shots / name}")
        # 相机装在 J6 定子侧（五/六轴之间）——2026-10-15 用户眼睛实锤：转 J6 画面不转。
        # 手眼模型要"相机刚性固定在其上的那一帧"，所以把法兰位姿转成定子位姿再进 csv：
        #   定子位姿 = 法兰位姿 ∘ Rz(−j6)
        # 固定的六轴零偏在相对运动里自动抵消，不用标定。原始法兰位姿记进旁车文件。
        j6 = float(self._joints()[5])
        Rz4 = np.eye(4)
        Rz4[:3, :3] = _rot_axis_deg("z", -j6)
        p2s = _T_to_pose6(C._pose6_to_T(p2) @ Rz4)
        self.raw.append((name, p2, j6))
        self._write_raw()
        self.saved.append((name, p2s, m["R_cb"].T))
        C._he_write_csv(self.shots / C.HE_POSES_CSV, [(n, p) for n, p, _ in self.saved])
        M = m["R_cb"].T                                  # 相机在板心系里的姿态
        if len(self.saved) >= 2:
            self.kinds.append(C._classify_motion(self.saved[-2][2], M))
        else:
            self.kinds.append(None)
        axes = C._he_axis_eigs([p for _n, p, _M in self.saved]) if len(self.saved) >= 4 else None
        line = C._he_status_line(len(self.saved), m["n"], m["side"], m["blur"], m["dist"],
                                 m["tilt"], p2, [k for k in self.kinds if k], axes)
        print(f"\r{line}\n", flush=True)
        return True, "已存"

    def _write_raw(self):
        """旁车文件：原始法兰位姿 + j6（主 csv 里存的是定子位姿）。

        主 csv 格式必须和拖动版逐字节兼容（解算端读它），j6 修正的底账另开一份：
        万一以后发现"定子侧"判错了，用它能原样倒回去。
        """
        with open(self.shots / "flange_poses_raw_j6.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "x", "y", "z", "rx", "ry", "rz", "j6"])
            for name, p, j6 in self.raw:
                w.writerow([name] + [f"{v:.4f}" for v in p] + [f"{j6:.4f}"])

    def _return_to_aim(self):
        """回对准位姿，并顺手验"板没动过"（铁律①）。"""
        self._move_to(self.T_aim, "回对准位姿", board_check=True, quiet=True)
        m = self._metrics(self._grab())
        if m["R_cb"] is None:
            raise Stop("回到对准位姿后认不出板了——现场变了，停")
        if self.ref_board is not None:
            R0, t0 = self.ref_board
            dt = float(np.linalg.norm(m["t_cb"] - t0))
            dr = C._rot_axis_angle(m["R_cb"].T @ R0)[0]
            if dt > BOARD_MOVED_MM or dr > BOARD_MOVED_DEG:
                raise Stop(f"板动过了（相机里差 {dt:.1f}mm / {dr:.2f}°）——"
                           f"铁律①：这一轮数据作废，别继续加帧")
        return m

    def _walk_entry(self, tag, tx, ty, rz, axial, lat_x, lat_y):
        """走一个议程条目：分步（偏摆弧 → 滚转 → 平移）+ 每步验板还在画面里 + 再对准。"""
        steps = _sub_steps(tx, ty, rz, axial, lat_x, lat_y)
        for k, (txk, tyk, rzk, axk, lxk, lyk) in enumerate(steps, 1):
            last = (k == len(steps))
            D = _entry_delta(txk, tyk, rzk, axk, lxk, lyk, self.p_board)
            self._move_abs(self.T_aim @ D, f"{tag} 第{k}/{len(steps)}步",
                           quiet=(len(steps) > 1 and not last))
            m = self._metrics(self._grab())
            if m["n"] < 12:
                raise Stop(f"{tag}：第 {k}/{len(steps)} 步后板只剩 {m['n']} 角点——走飞了，停")
        return self._aim(tag)

    def _stroke_deltas(self, m_aim):
        """按对准帧的画面余量，定出这组"不转腕平移"的行程（工具系 mm）。

        横向 s 让角点走 s·fx/距 像素、轴向 s 让板【两边各】涨 side·s/(2·距) 像素，都拿画面余量
        反推，保证 24/24 + 离边 ≥60px 的门过得去（过了也还有 STROKE_PAD_PX 富余）。
        ⚠ 系数别超 1：2026-10-15 仿真里写的是 1.25/2.5（= 多吃 25% 余量），结果那个方向余量小的
        −y 那条每次都丢掉 4 个角点被门拒掉，六条只剩五条——虚报的行程等于白跑一趟。
        """
        room = max(float(m_aim["margin"]) - (C.HE_EDGE_MARGIN_PX + STROKE_PAD_PX), 0.0)
        d = float(m_aim["dist"]) if m_aim["dist"] else self.target_dist
        s_lat = min(STROKE_MM, room * d / float(self.K[0, 0]))
        s_ax = min(STROKE_MM, 2.0 * room * d / max(float(m_aim["side"]), 1.0))
        out = []
        if s_lat >= STROKE_MIN_MM:
            out += [(+s_lat, 0.0, 0.0), (-s_lat, 0.0, 0.0), (0.0, +s_lat, 0.0), (0.0, -s_lat, 0.0)]
        if s_ax >= STROKE_MIN_MM:
            out += [(0.0, 0.0, +s_ax), (0.0, 0.0, -s_ax)]
        return out, s_lat, s_ax

    def _walk_strokes(self, m_aim):
        """定尺度平移：姿态冻结在 T_aim 上，纯平移 ±60mm（x/y 横向 + z 轴向）。

        为什么非要有这一组（2026-10-15 定量，见 C._he_board_scale）：打印比例这类
        **均匀尺度错**被 PnP 的尺度-距离二义性整段吸收——板面残差、留一、分区解全都看不见它，
        可它按同比例把 X 的平移带偏（合成实测：板缩 4% → X 平移差 18mm，而留一只报 0.32mm）。
        全瞄准的位姿集里尺度信息≈0（k 和 t_X 分不开：同一份真值两个读数差 2.8%，误差棒兜不住）；
        补 6 条"姿态不变 + 大平移"的对之后，k 从 0.9375±0.0097 变成 0.9612±0.0005
        （真值 0.9623）——准到 0.1%，4% 的错就是 60σ。这一组还顺手把 X 的平移跟尺度假设解耦。
        注意：这里【不能】调 _aim（对准会重新转腕，姿态就不冻结了），也别调 _return_to_aim
        （那个"板没动过"检查只在 T_aim 处成立——纯平移时板在相机里的位姿本来就该变）。
        """
        deltas, s_lat, s_ax = self._stroke_deltas(m_aim)
        if not deltas:
            print(f"    ⚠ 定尺度平移：画面余量只够 ~{max(s_lat, s_ax):.0f}mm（要 ≥"
                  f"{STROKE_MIN_MM:.0f}mm）——这组跳过。板尺寸自检这次会看不出，"
                  f"只能靠采集前用卡尺量方格间距兜底")
            return 0
        print(f"    [定尺度] 不转腕平移 x/y ±{s_lat:.0f}mm"
              + (f"、z ±{s_ax:.0f}mm" if s_ax >= STROKE_MIN_MM else "（z 余量不够，只做横向）")
              + f"，共 {len(deltas)} 条 —— 板尺寸自检全靠这一组")
        n = 0
        for k, d in enumerate(deltas, 1):
            tag = (f"[定尺度 {k}/{len(deltas)}] 平移 [{d[0]:+.0f} {d[1]:+.0f} {d[2]:+.0f}]mm")
            try:
                self._move_to(self.T_aim @ _trans_delta(d), tag, board_check=True, quiet=True)
                self._settle(tag)
            except SkipPose as e:
                print(f"    · {tag} 跳过：{e}")
                continue
            frame = self._grab()
            m = self._metrics(frame)
            ok, why = self._capture(frame, m, tag)
            if not ok:
                print(f"    · {tag} 没过门（{why}）")
                continue
            n += 1
        return n

    # ---------- 议程 ----------

    def _plan_poses(self, T_aim, p_board, tilt, roll):
        out = []
        for nm, tx, ty, rz, ax_, lx, ly in TEMPLATE:
            D = _entry_delta(tx * tilt, ty * tilt, rz * roll, ax_, lx, ly, p_board)
            out.append((nm, _T_to_pose6(T_aim @ D)))
        return out

    def _walk_rot_base(self, g, ang, tag):
        """绕【基座系的 g 轴、过板心】转 ang 度——自适应补位姿专用的"绕板心画弧"。

        直接用 _entry_delta 只绕工具轴转；这里的缺口方向是基座系量出来的，所以单写一个。
        圆心取板心（p_board 在基座系里的位置），板就一直待在画面的原地；
        28° 一口气转会让法兰走 2·440·sin14° ≈ 213mm（超单条位移闸），所以照样分步。
        """
        pb = self.T_aim[:3, 3] + self.T_aim[:3, :3] @ self.p_board
        n = max(1, int(math.ceil(abs(ang) / ARC_STEP_DEG)))
        for k in range(1, n + 1):
            th = math.radians(ang * k / n)
            Kx = np.array([[0.0, -g[2], g[1]], [g[2], 0.0, -g[0]], [-g[1], g[0], 0.0]], float)
            Rd = np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)
            T = np.eye(4)
            T[:3, :3] = Rd @ self.T_aim[:3, :3]
            T[:3, 3] = pb - T[:3, :3] @ self.p_board
            self._move_abs(T, f"{tag} 第{k}/{n}步", quiet=(k != n))
            m = self._metrics(self._grab())
            if m["n"] < 12:
                raise Stop(f"{tag}：第 {k}/{n} 步后板只剩 {m['n']} 角点——走飞了，停")
        return self._aim(tag)

    def _gap_base(self, poses):
        """基座系里转动轴散布的最小特征方向 = 缺的那个方向（原来在法兰系里算的，
        这里要拿它去生成新位姿，所以搬到基座系）。"""
        Rs = [C._pose6_to_T(p)[:3, :3] for p in poses]
        ax = []
        for i in range(len(Rs)):
            for j in range(i + 1, len(Rs)):
                ang, a = C._rot_axis_angle(Rs[i].T @ Rs[j])
                if ang > 2.0:
                    ax.append(a)
        if len(ax) < 3:
            return None
        A = np.array(ax)
        w, V = np.linalg.eigh(A.T @ A / len(A))
        return V[:, 0]

    # ---------- 主流程 ----------

    def run(self):
        _math_selftest()
        old = self._check_disk()
        self._connect()
        print(f"已连上 {self.arm.ip}（只读）；内参 {self.args.intr}")
        print("前检：")
        p_ok = self._preflight(need_auto=True)
        if not p_ok:
            if self.args.go:
                sys.exit("× 前检没过，不发任何指令。照上面的 × 逐条处理（模式要在示教器上自己切）。")
            print("  （预演模式：前检没过只提示，不拦）")

        # 起始位姿的肘部高度总闸（撞机事故后新加）：深家族对准 ≈40mm、正常家族 ≈358mm。
        # 起始就在塌陷家族里 → --go 直接拒发（先 --go-home 或手动点动抬出来）。
        j_start = self._joints()
        ez0 = self._elbow_z(j_start)
        if ez0 is None:
            print("  ⚠ 起始肘部估计不可信（圆拟合没过）——这一步只打标；每条运动前还会再查")
        else:
            ok_el = ez0["z"] >= ELBOW_MIN_Z_MM
            print(f"  {'✓' if ok_el else '×'} 起始肘部高度估计 z≈{ez0['z']:.0f}mm"
                  f"（闸 {ELBOW_MIN_Z_MM:.0f}mm；深家族≈40、正常家族≈358）")
            if not ok_el:
                if self.args.go:
                    sys.exit("× 起始就在肘部塌陷的家族里——先跑 --go-home（或手动把 J2 点到 "
                             "≈−129）再重跑")
                print("  （预演模式：只警告不拦）")

        # 开相机
        try:
            self.cap = C.open_camera()
        except SystemExit as e:
            sys.exit(str(e))
        self.args_w, self.args_h = config.CAMERA_W, config.CAMERA_H
        frame = self._grab()
        m0 = self._metrics(frame)
        self.T0 = C._pose6_to_T(self._pose())
        pose0 = _T_to_pose6(self.T0)
        print(f"当前法兰位姿 [{pose0[0]:.0f} {pose0[1]:.0f} {pose0[2]:.0f} "
              f"{pose0[3]:.1f} {pose0[4]:.1f} {pose0[5]:.1f}]")
        print(f"画面里：角点 {m0['n']}/24，板宽 {m0['side']:.0f}px"
              + (f"，距 {m0['dist']:.0f}mm，倾 {m0['tilt']:.0f}°，离边 {m0['margin']:.0f}px，"
                 f"板心偏 ({m0['off'][0]:+.0f},{m0['off'][1]:+.0f})px"
                 if m0["R_cb"] is not None else "（认不出板）"))

        # 议程预演（用当前位姿当"对准位姿"的替身——λ 只看旋转，不看位置）
        d_plan = m0["dist"] if m0["dist"] else self.target_dist
        p_plan = np.array(self.cam_offset, float) + np.array([0.0, 0.0, d_plan])
        plan = self._plan_poses(self.T0, p_plan, self.f_tilt, self.f_roll)
        axes = C._he_axis_eigs([p for i, (_nm, p) in enumerate(plan, 1)
                                if i not in AGENDA_SKIP_TEMPLATE])   # 拉黑的不算进 λ
        print(f"\n议程（{len(TEMPLATE) - len(AGENDA_SKIP_TEMPLATE)} 条可走、"
              f"{len(AGENDA_SKIP_TEMPLATE)} 条拉黑（毒位姿），偏摆 ±{self.f_tilt:.0f}° / "
              f"滚转 ±{self.f_roll:.0f}° / 平移 ±40~60mm）：")
        for i, (nm, p) in enumerate(plan, 1):
            if i in AGENDA_SKIP_TEMPLATE:
                print(f"  {i:>2}. {nm:<26} ⏭ 已拉黑（会把控制器打到崩溃）——不跑")
                continue
            a = C._rot_axis_angle(C._pose6_to_T(p)[:3, :3] @ np.linalg.inv(self.T0[:3, :3]))
            print(f"  {i:>2}. {nm:<26} 绕基座轴 [{a[1][0]:+.2f} {a[1][1]:+.2f} {a[1][2]:+.2f}]"
                  f" 转 {a[0]:>4.1f}°")
        if axes is not None:
            w = axes[0]
            print(f"  议程做完的转动轴散布（λ 只吃旋转，所以这个预告跟真跑一致）："
                  f"λ = {w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}（要 ≥{C.AXIS_WEAK_EIG}）")
            if w[2] < C.AXIS_WEAK_EIG:
                print(f"  ⚠ 议程自己就不够多样（λ3={w[2]:.3f}）——"
                      f"{C._he_gap_advice(axes[1][:, 2])}")
            else:
                print(f"  ✓ 议程覆盖够（λ3={w[2]:.3f} ≥ {C.AXIS_WEAK_EIG}）")
        # 定尺度平移（真跑在议程之后、补位姿之前做）：行程按当前画面余量算，预演先给个数
        st_deltas, s_lat, s_ax = self._stroke_deltas(m0)
        # 预演是按【当前】位姿估的，可真是跑到这一步之前板已经被对准拉正了，余量会大得多
        # （2026-10-15 实测过这个误报：板心偏 (0,+192)px 时预演说"发不了"，对准后余量
        #  (1080−325)/2 ≈ 377px 绰绰有余）。对准后的余量 ≈ min((W−角点宽)/2,(H−角点高)/2)。
        sw, sh = m0["span"]
        m_room = min((self.img_w - sw) / 2.0, (self.img_h - sh) / 2.0)
        st_aligned, s_lat_a, s_ax_a = self._stroke_deltas({**m0, "margin": m_room})
        if st_aligned:
            used, sl, sa = ((st_deltas, s_lat, s_ax) if st_deltas
                            else (st_aligned, s_lat_a, s_ax_a))
            print(f"   + 定尺度平移 {len(used)} 条（姿态冻结在出发位姿，不转腕）"
                  + ("" if st_deltas else "【板对准拉正后】") + f"：x/y ±{sl:.0f}mm"
                  + (f"、z ±{sa:.0f}mm" if sa >= STROKE_MIN_MM else "（z 余量不够）")
                  + "  ← 板尺寸自检（打印比例）全靠这一组")
            if not st_deltas:
                print(f"     （按【当前这个】位姿估只有 ~{max(s_lat, s_ax):.0f}mm——板心还没拉正，"
                      f"真跑时对准那一步会先把它拉到画面中心，届时余量 {m_room:.0f}px）")
        else:
            print(f"   ⚠ 定尺度平移发不了：画面余量只够 ~{max(s_lat, s_ax):.0f}mm（要 ≥"
                  f"{STROKE_MIN_MM:.0f}mm）。板尺寸自检会看不出——采集前务必用卡尺量一遍方格间距")
        st_plan = [(f"定尺度 [{d[0]:+.0f} {d[1]:+.0f} {d[2]:+.0f}]",
                    _T_to_pose6(self.T0 @ _trans_delta(d)))
                   for d in (st_deltas or st_aligned)]

        print(f"\n安全包围盒：以出发位姿为中心 x/y ±{BOX_DXY_MM:.0f}mm、z ±{BOX_DZ_MM:.0f}mm，"
              f"绝对 z ≥ {Z_FLOOR_MM:.0f}mm；单条位移 ≤{MOVE_MAX_MM:.0f}mm；速度 {self.args.speed:.0f}%"
              f"（≤{config.MAX_SPEED_PCT}）")
        bad = [(nm, self._box_violation(C._pose6_to_T(p))) for nm, p in plan + st_plan]
        bad = [(nm, w) for nm, w in bad if w]
        if bad:
            print(f"\n  ⚠ 盒预检（以【出发位姿】为基准；真跑基准是「对准位姿」，会差几十毫米）："
                  f"{len(bad)}/{len(plan) + len(st_plan)} 个目标越界——")
            for nm, w in bad[:6]:
                print(f"      {nm.strip()}：{w}")
            print(f"      ⇒ 先把板挪到让臂够得着、或改 --cam-offset 让对准位姿偏回来，再真跑")
        else:
            print(f"  ✓ 盒预检：{len(plan) + len(st_plan)} 个目标都还在盒里"
                  f"（以出发位姿为基准估算）")

        if not self.args.go:
            print(f"\n（预演到此为止，什么都没动、什么都没写。）")
            print(f"目标存帧目录：{self.shots}"
                  + (f"——⚠ 里面有 {len(old)} 张旧 jpg，真跑前要先挪走" if old else "（干净）"))
            print(f"真跑：python3 -I src/auto_handeye.py --go     ← 手放急停按钮上")
            return 0

        # ---- 真跑 ----
        print("\n" + "=" * 72)
        print("接下来臂会自己动。请你：① 把急停按钮握在手里 ② 盯着臂和板 ③ 有不对就拍急停。")
        print("跑起来之后不需要你按任何键：全程约 25~35 分钟（每次运动后画面流要 ~10s 才追上，"
              "程序自动等——这是这台相机的固定脾气），中途只在出事/结束时才停下来。")
        try:
            s = input("确认后输入 GO 回车开跑（别的任何输入 = 退出）：").strip()
        except EOFError:
            sys.exit("读不到终端输入（stdin 不是终端？）——没开跑")
        if s != "GO":
            print("没开跑。")
            return 1

        t_start = time.time()
        try:
            print("\n[1/6] 使能…")
            rc = self.arm.enable()
            print(f"      RobotEnable(1) → {rc}")
            time.sleep(1.0)
            if self._st("error_code") != 0:
                raise Stop(f"使能后 error_code={self._st('error_code')}——"
                           f"（见 memory: fairino-fr3-slave-offline，先查重载电缆）")

            print("[2/6] 首动自检：垂直上 5mm 再回落（B3 的第一次真动）…")
            T_lift = C._pose6_to_T(self._pose())
            T_lift[2, 3] += LIFT_TEST_MM
            got = self._move_abs(T_lift, f"首动自检 +{LIFT_TEST_MM:.0f}mm（基座 z）")
            print(f"      ✓ 动了：z {self.T0[2, 3]:.1f} → {got[2]:.1f}mm")
            m_lift = self._metrics(self._grab())      # 这一抓会先等画面流追上
            _jum = bool(self.fresh_log and self.fresh_log[-1][1])
            print(f"      {'✓' if _jum else '⚠'} 画面自检："
                  + (f"看到跳变——画面真的跟着腕子动了（角点 {m_lift['n']}/24）" if _jum
                     else "没见跳变（5mm 位移本就小，或流还有问题——下一步探针会硬查）"))
            self._move_abs(self.T0, "首动自检回位")
            print("      ✓ 回位 OK")

            print("[3/6] 学局部雅可比（三趟探针；每趟运动后等画面追上，约 2 分钟）…")
            self._probe()

            print("[4/6] 对准（把板心拉进画面中心、距离拉到 "
                  f"{self.target_dist:.0f}mm）…")
            frame, m = self._aim("出发位姿")
            m_aim = dict(m)                                     # 定尺度平移要按它的画面余量定行程
            self.T_aim = C._pose6_to_T(self._pose())
            self.p_board = np.array([self.cam_offset[0], self.cam_offset[1],
                                     self.cam_offset[2] + float(m["dist"])])
            self.ref_board = (m["R_cb"], np.asarray(m["t_cb"], float))
            pa = _T_to_pose6(self.T_aim)
            print(f"      对准位姿 [{pa[0]:.1f} {pa[1]:.1f} {pa[2]:.1f} "
                  f"{pa[3]:.1f} {pa[4]:.1f} {pa[5]:.1f}]；"
                  f"画弧圆心先验（工具系）[{self.p_board[0]:.0f} {self.p_board[1]:.0f} "
                  f"{self.p_board[2]:.0f}]mm")

            # 定尺度平移是必做的（板尺寸自检的唯一软件防线），所以先把它的名额留出来
            n_stroke_want = len(self._stroke_deltas(m_aim)[0])
            budget = max(C.HE_MIN_SHOTS, self.args.max_shots - n_stroke_want)
            print(f"[5/6] 走议程（{len(TEMPLATE)} 条中 {len(AGENDA_SKIP_TEMPLATE)} 条拉黑"
                  f"（毒位姿），目标 ≥{C.HE_MIN_SHOTS} 张，"
                  f"议程最多 {budget} 张——给 {n_stroke_want} 条定尺度平移留了名额）…")
            fails = 0
            for i, (nm, tx, ty, rz, ax_, lx, ly) in enumerate(TEMPLATE, 1):
                if len(self.saved) >= budget:
                    break
                if i in AGENDA_SKIP_TEMPLATE:            # 毒位姿：绝不再碰（见常数注释）
                    print(f"    · [{i}/{len(TEMPLATE)}] {nm.strip()}：⏭ 已拉黑——这族位姿"
                          f"会把控制器打到崩溃，整条跳过（不发指令）")
                    continue
                tag = f"[{i}/{len(TEMPLATE)}] {nm.strip()}"
                try:
                    self._return_to_aim()
                    frame, m = self._walk_entry(tag, tx * self.f_tilt, ty * self.f_tilt,
                                                rz * self.f_roll, ax_, lx, ly)
                except SkipPose as e:
                    print(f"    · 跳过：{e}")
                    fails += 1
                    continue
                ok, why = self._capture(frame, m, tag)
                if not ok:
                    print(f"    · 这张没过门（{why}）——跳过")
                    fails += 1
            print(f"      议程走完：存 {len(self.saved)} 张，跳过 {fails} 条")

            n_st = self._walk_strokes(m_aim)
            print(f"      定尺度平移：存 {n_st} 张"
                  + ("（板尺寸自检这次有信息了）" if n_st >= 3 else
                     "（少于 3 张——板尺寸自检多半看不出，改用卡尺量）"))

            print("[6/6] 自适应补位姿（按实测缺口方向，最多补到 "
                  f"{self.args.max_shots} 张）…")
            while len(self.saved) < self.args.max_shots:
                poses = [p for _n, p, _M in self.saved]
                axes = C._he_axis_eigs(poses)
                if axes is None or axes[0][2] >= C.AXIS_WEAK_EIG:
                    print(f"      λ3 = {axes[0][2]:.3f} ≥ {C.AXIS_WEAK_EIG}，不用补了"
                          if axes is not None else "      样本太少，没法判轴散布，先不补")
                    break
                g = self._gap_base(poses)
                if g is None:
                    break
                print(f"      λ3 = {axes[0][2]:.3f} < {C.AXIS_WEAK_EIG}，缺口（基座系）"
                      f"[{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}] —— 绕它转一个")
                made = False
                for sgn, deg in ((+1, 28.0), (-1, 28.0), (+1, 35.0), (-1, 35.0)):
                    ang = deg * sgn
                    # 先算目标姿态，用它和已有位姿的夹角挡掉"太像的"（MIN_SEP_DEG）
                    th = math.radians(ang)
                    Kx = np.array([[0.0, -g[2], g[1]], [g[2], 0.0, -g[0]],
                                   [-g[1], g[0], 0.0]], float)
                    Rd = np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)
                    R_new = Rd @ self.T_aim[:3, :3]
                    sep = min(C._rot_axis_angle(R_new @ C._pose6_to_T(q)[:3, :3].T)[0]
                              for q in poses)
                    if sep < MIN_SEP_DEG:
                        continue
                    tag = f"补({ang:+.0f}° 绕缺口)"
                    try:
                        self._return_to_aim()
                        frame, m = self._walk_rot_base(g, ang, tag)
                    except SkipPose as e:
                        print(f"    · 这个方向走不通（{e}），换个符号/角度")
                        continue
                    ok, why = self._capture(frame, m, tag)
                    if ok:
                        made = True
                        break
                    print(f"    · {tag} 没过门（{why}），换个角度")
                if not made:
                    print("      补不下去了（方向都试过）——就用手上的这些去解")
                    break

            print("\n收尾：回对准位姿 → 回出发位姿…")
            if not self.args.no_return:
                self._move_to(self.T_aim, "回对准位姿", board_check=True, quiet=True)
                self._move_to(self.T0, "回出发位姿", board_check=False, quiet=True)
                print("      ✓ 回到出发位姿")

            self._summary(time.time() - t_start)
            return 0

        except KeyboardInterrupt:
            print("\n\n你按了 Ctrl-C —— 先 Pause 把运动停住…")
            self._pause_now()
            self._after_stop("Ctrl-C", len(self.saved))
            return 2
        except Stop as e:
            print(f"\n\n■ 停下来：{e}")
            self._pause_now()
            self._after_stop(str(e), len(self.saved))
            return 3
        except SafetyError as e:
            print(f"\n\n■ 安全闸拦下：{e}")
            self._pause_now()
            self._after_stop(f"安全闸：{e}", len(self.saved))
            return 3
        except FRRpcError as e:
            print(f"\n\n■ RPC 报错（指令被拒/超时）：{e}")
            self._pause_now()
            self._after_stop(f"RPC：{e}", len(self.saved))
            return 3

    def _pause_now(self):
        if self.arm is None:
            return
        try:
            print(f"      Pause → {self.arm.pause()}")
            p = self._pose()
            print(f"      现在停在这：[{p[0]:.1f} {p[1]:.1f} {p[2]:.1f} "
                  f"{p[3]:.1f} {p[4]:.1f} {p[5]:.1f}]")
        except Exception as e:                          # noqa: BLE001
            print(f"      （Pause/读位姿没成：{e}）")

    def _after_stop(self, why, n_saved):
        print(f"      已存 {n_saved} 张在 {self.shots}（csv 已同步，随时可解）")
        print(f"      臂还【保持使能】停着。要下电：python3 -I src/auto_handeye.py --disable")
        print(f"      再来一次：先想清楚为什么停（{why}），必要时把臂拖回出发位姿再跑 --go")

    def _summary(self, dt):
        poses = [p for _n, p, _M in self.saved]
        Rcb = [M for _n, _p, M in self.saved]     # self.saved 里存的是 R_cb.T = 相机→板，
        #                                           正是 _he_axis_match 要的方向（别"顺手"改回去）
        print("\n" + "=" * 72)
        print(f"采集结束：{len(self.saved)} 张，用时 {dt/60:.1f} 分钟 → {self.shots}")
        if len(poses) < 5:
            print("× 张数太少（<5），解不了。为什么跳过这么多看上面的日志。")
            return
        am = C._he_axis_match(poses, Rcb)
        if am is not None:
            med, mx, n = am
            good = med <= 0.7
            print(f"数据自洽性（轴对应残差，X-free）：中位 {med:.2f}° / 最大 {mx:.2f}°，{n} 对 "
                  f"{'✓ 这关过了（拖动版首轮这里 3.53°/11.37°）' if good else '× 还是大——先别信解，看日志里哪几张没过门'}")
        axes = C._he_axis_eigs(poses)
        if axes is not None:
            w, V = axes
            print(f"转动轴散布（法兰系）：λ = {w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f} "
                  f"{'✓' if w[2] >= C.AXIS_WEAK_EIG else '× 偏弱'}"
                  f"（首轮 0.845/0.151/0.004）；缺口法兰[{V[0,2]:+.2f} {V[1,2]:+.2f} {V[2,2]:+.2f}]")
            if w[2] < C.AXIS_WEAK_EIG:
                print(f"   → {C._he_gap_advice(V[:, 2])}")
        kinds = [k for k in self.kinds if k]
        print(f"动作分类计数：滚转 {sum(1 for k in kinds if k == '滚转')} / "
              f"左右 {sum(1 for k in kinds if k == '左右')} / 上下 {sum(1 for k in kinds if k == '上下')}"
              f"（首轮 10/15 是滚转）")
        if self.raw:
            j6s = sorted(j for _n, _p, j in self.raw)
            print(f"J6 定子修正：j6 范围 {j6s[0]:.1f}°~{j6s[-1]:.1f}°（中位 "
                  f"{j6s[len(j6s) // 2]:.1f}°）——csv 里存的是定子位姿（相机所在那一帧）；"
                  f"原始法兰位姿+j6 见 flange_poses_raw_j6.csv")
        if self.fresh_log:
            els = sorted(e for e, _j, _m in self.fresh_log)
            nj = sum(1 for _e, j, _m in self.fresh_log if j)
            print(f"画面追赶（这台相机的固定脾气）：等 {len(els)} 次，中位 "
                  f"{els[len(els) // 2]:.1f}s、最长 {els[-1]:.1f}s；见到跳变 {nj} 次"
                  + (f"，{len(els) - nj} 次没见（位移小或流卡）" if len(els) - nj else ""))
        if self.ez_min is not None:
            print(f"肘部高度守卫：全程目标关节的肘部估计最低 {self.ez_min:.0f}mm"
                  f"（闸 {ELBOW_MIN_Z_MM:.0f}）")
        if self._jlog is not None:
            print(f"关节全量日志：{self.shots / 'joints_log.csv'}"
                  f"（每条运动的命令关节/回读/肘部估计——家族变化藏不住了）")
        print(f"\n下一步：python3 -I src/calibrate.py --handeye-solve")
        print(f"  看两行再决定用不用：① 数据自洽性中位 ≤0.7°  ② λ3 ≥ {C.AXIS_WEAK_EIG}")
        print(f"  臂还【保持使能】。要下电：python3 -I src/auto_handeye.py --disable")

    # ---------- 只读预演（--dry-joints） / 回家翻转（--go-home） ----------

    def dry_joints(self):
        """只读预演：不碰相机、不发任何运动指令、不写盘。

        两段计划全打成表：① 从当前位姿到正常家族对准位姿的关节过渡；② 整条议程
        （14 条 + 6 条定尺度平移）逐子步的 ik_ref 关节、肘部估计、法兰高度、离板距离。
        板没动过时，这些数字就该是真跑时会发生的数字——动臂之前先审它。

        --dry-agenda-only 跳过 ① 和八组枚举表（它们已单独验证过，省调用）；--dry-from N
        从议程第 N 条起（分块跑；或重启后直扑第 N 条，测它是不是"一碰就死"的那条）。
        议程段走轻量模式省控制器调用（小动不实测肘部）——2026-10-16 两次预演都在
        ~600 次调用处把控制器打掉线的教训；真跑路径不受影响（全闸照跑）。
        """
        self._connect()
        print(f"已连 {self.arm.ip}（只读预演：不发任何运动指令、不碰相机、不写盘）")
        self._preflight(need_auto=False)
        j_cur = self._joints()
        pose_cur = _T_to_pose6(C._pose6_to_T(self._pose()))
        print(f"\n当前法兰位姿 [{pose_cur[0]:.1f} {pose_cur[1]:.1f} {pose_cur[2]:.1f} "
              f"{pose_cur[3]:.1f} {pose_cur[4]:.1f} {pose_cur[5]:.1f}]")
        print(f"当前关节 [{j_cur[0]:.2f} {j_cur[1]:.2f} {j_cur[2]:.2f} "
              f"{j_cur[3]:.2f} {j_cur[4]:.2f} {j_cur[5]:.2f}]")
        ez_c = self._elbow_z(j_cur)
        if ez_c is None:
            print("肘部估计：这次圆拟合不可信（跳过）")
        else:
            print(f"肘部估计 z≈{ez_c['z']:.0f}mm（闸 {ELBOW_MIN_Z_MM:.0f}；深家族≈40、"
                  f"正常家族≈358）{'✓' if ez_c['z'] >= ELBOW_MIN_Z_MM else ' ⚠ 在塌陷区'}"
                  f"  拟合残差 {ez_c['res']:.3f}mm、半径 {ez_c['r']:.1f}mm")

        aim_pose = [float(v) for v in (self.args.aim_pose or DEFAULT_AIM_POSE)]
        src = "命令行 --aim-pose" if self.args.aim_pose else "上次真跑记录（板没动过）"
        print(f"\n议程基准对准位姿（{src}）：[{aim_pose[0]:.1f} {aim_pose[1]:.1f} "
              f"{aim_pose[2]:.1f} {aim_pose[3]:.1f} {aim_pose[4]:.1f} {aim_pose[5]:.1f}]")

        j_target = list(NORMAL_AIM_JOINTS)
        pk = self.arm.fk(j_target)
        dpk = float(np.linalg.norm(np.asarray(pk[:3]) - np.asarray(aim_pose[:3])))
        drk = C._rot_axis_angle(C._pose6_to_T(pk)[:3, :3].T
                                @ C._pose6_to_T(aim_pose)[:3, :3])[0]
        print(f"目标（正常家族对准关节，撞机后验证过）[{j_target[0]:.2f} {j_target[1]:.2f} "
              f"{j_target[2]:.2f} {j_target[3]:.2f} {j_target[4]:.2f} {j_target[5]:.2f}]"
              f"  fk 回对准位姿差 {dpk:.2f}mm / {drk:.2f}°")
        if dpk > 3.0 or drk > 1.0:
            sys.exit("× 验证过的目标关节和对准位姿对不上（> 3mm / 1°）——常数被改过？停下来查")

        pb_tool = np.array([self.cam_offset[0], self.cam_offset[1],
                            self.cam_offset[2] + self.target_dist], float)
        pb_base = (C._pose6_to_T(aim_pose)[:3, 3]
                   + C._pose6_to_T(aim_pose)[:3, :3] @ pb_tool)

        if getattr(self.args, "dry_agenda_only", False):
            print("\n（--dry-agenda-only：八组枚举表与回家过渡表之前预演已验过，跳过不重跑）")
        else:
            print("\n八组逆解枚举（对准位姿；只作对照——目标固定在验证过的那组）：")
            for cfg in range(8):
                time.sleep(DRY_ROW_SLEEP_S)              # 节流（同 _dry_agenda）
                try:
                    j = self.arm.ik(aim_pose, cfg)
                except Exception as e:                      # noqa: BLE001
                    print(f"  cfg {cfg}：控制器拒绝（{str(e)[:60]}）")
                    continue
                if len(j) != 6 or max(abs(v) for v in j) < 1e-6:
                    print(f"  cfg {cfg}：无效解")
                    continue
                pj = self.arm.fk(j)
                dj_ = float(np.linalg.norm(np.asarray(pj[:3]) - np.asarray(aim_pose[:3])))
                ez = self._elbow_z(j)
                tag = (" ← 目标（就是它）" if max(abs(a - b)
                                                 for a, b in zip(j, j_target)) < 0.5 else "")
                warn = " ⚠ 肘部塌陷区" if (ez is not None and ez["z"] < ELBOW_MIN_Z_MM) else ""
                print(f"  cfg {cfg}：J2 {j[1]:>8.2f}  J3 {j[2]:>8.2f}  J4 {j[3]:>8.2f}  "
                      f"肘 z={self._fnum(ez and ez['z'])}  回位姿差 {dj_:.2f}mm{tag}{warn}")

            steps = self._plan_home(j_cur, j_target)
            print(f"\n[返家过渡] 当前 → 正常家族对准关节：{len(steps)} 步（每步 ≤{HOME_STEP_DEG:.0f}°；"
                  f"--go-home 就是照这张表走 MoveJ）：")
            print("   步       J2       J3       J4      肘z   法兰z  相机↔板  法兰↔板")
            any_bad = False
            for k, j in enumerate(steps, 1):
                time.sleep(DRY_ROW_SLEEP_S)              # 节流（同 _dry_agenda）
                c = self._home_checks(j, pb_base)
                fl = []
                if c["elbow_z"] is not None and c["elbow_z"] < ELBOW_MIN_Z_MM:
                    fl.append("⚠肘低")
                if c["tcp_z"] is not None and c["tcp_z"] < HOME_MIN_TCP_Z:
                    fl.append("⚠法兰低")
                if c["cam_board"] is not None and c["cam_board"] < HOME_MIN_CAM_BOARD:
                    fl.append("⚠近板")
                if c["tcp_board"] is not None and c["tcp_board"] < HOME_MIN_WRIST_BOARD:
                    fl.append("⚠腕近板")
                any_bad = any_bad or bool(fl)
                print(f"  {k:>3}  {j[1]:8.2f} {j[2]:8.2f} {j[3]:8.2f}  "
                      f"{self._fnum(c['elbow_z'], 5)}  {self._fnum(c['tcp_z'], 6)}  "
                      f"{self._fnum(c['cam_board'], 7)}  {self._fnum(c['tcp_board'], 7)}"
                      + ("  " + " ".join(fl) if fl else ""))
            print("  " + ("⚠ 上面有带标记的步——先别跑 --go-home，逐条看" if any_bad
                          else "✓ 全程肘部高位、离板远——干净"))

        self._dry_agenda(aim_pose, j_target, pb_tool, pb_base,
                         start=max(1, int(getattr(self.args, "dry_from", 1))))
        print("\n（只读预演到此为止：没有发过任何运动指令，也没有写任何文件。）")
        return 0

    def _dry_agenda(self, aim_pose, j_target, pb_tool, pb_base, start=1):
        """议程预演：mirror _walk_entry / _walk_strokes 的位姿生成，逐子步打关节/肘部/高度。

        轻量模式（light=True）：单步 ≤ DRY_SWEEP_MIN_DJ 的行不实测肘部（显示 "—"）、
        兜底候选按 Δj 升序只测到第一个健康的——省控制器调用（2026-10-16 两次掉线教训），
        判闸语义不变。start=N：从议程第 N 条（编号含"←回对准"，和真跑一致）起跑，分块用。
        """
        T_aim = C._pose6_to_T(aim_pose)
        st = {"ez_lo": None, "dj_hi": 0.0, "nbad": 0, "n": 0, "nskip": 0, "nswap": 0,
              "nlight": 0, "j2": []}

        # 先铺满整条计划（每条 子步… + ←回对准，再 6 条定尺度），再按 start 截断——
        # 编号分块前后对得上，"从第 N 条起"指的就是真跑里的第 N 条
        plan = []
        for i, (nm, tx, ty, rz, ax_, lx, ly) in enumerate(TEMPLATE, 1):
            if i in AGENDA_SKIP_TEMPLATE:            # 毒位姿：整条不入计划（真跑同款）
                print(f"    · [{i}] {nm.strip()}：⏭ 已拉黑（毒位姿，会把控制器打到崩溃）"
                      f"——计划里整条略过（含它的 ←回对准）")
                continue
            steps = _sub_steps(tx * self.f_tilt, ty * self.f_tilt, rz * self.f_roll, ax_, lx, ly)
            for k, (a, b, c_, d_, e_, f_) in enumerate(steps, 1):
                D = _entry_delta(a, b, c_, d_, e_, f_, pb_tool)
                w = f"第{k}/{len(steps)}步" if len(steps) > 1 else "单步"
                plan.append((f"[{i:>2}] {nm.strip()} {w}", T_aim @ D))
            plan.append((f"[{i:>2}] ←回对准", T_aim))
        for d in ((+STROKE_MM, 0.0, 0.0), (-STROKE_MM, 0.0, 0.0), (0.0, +STROKE_MM, 0.0),
                  (0.0, -STROKE_MM, 0.0), (0.0, 0.0, +STROKE_MM), (0.0, 0.0, -STROKE_MM)):
            plan.append((f"[定尺度] 平移 [{d[0]:+.0f} {d[1]:+.0f} {d[2]:+.0f}]mm",
                         T_aim @ _trans_delta(d)))
        if start > 1:
            if start > len(plan):
                print(f"  --dry-from {start} 超过议程总数 {len(plan)}——没有条目要跑")
                return
            print(f"\n（--dry-from {start}：从头截断，只跑第 {start}→{len(plan)} 条）")
            plan = plan[start - 1:]
        j_prev = list(j_target)

        def step(tag, T_t, j_prev_):
            time.sleep(DRY_ROW_SLEEP_S)                  # 节流：别把控制器敲掉线
            pose_t = _T_to_pose6(T_t)
            sol = self._solve_pinned(pose_t, j_prev_, tag=tag,
                                     light=True)         # 预演轻量模式（真跑永远全闸）
            st["n"] += 1
            if sol is None:
                st["nbad"] += 1
                st["nskip"] += 1
                print(f"  {tag:<36} × 没有健康解 —— 真跑会跳过这条（不发指令）")
                return j_prev_                           # 没动：链上保持上一条的关节
            j, dj, ez_z, src = sol
            c = self._home_checks(j, pb_base, elbow_z=ez_z)
            fl = []
            if dj > 20.0:
                fl.append(f"⚠大动 {dj:.0f}°")
            if c["cam_board"] is not None and c["cam_board"] < 300.0:
                fl.append("⚠近板")
            st["nbad"] += bool(fl)
            st["nswap"] += (src != "ik_ref")
            st["dj_hi"] = max(st["dj_hi"], dj)
            if ez_z is None and dj <= DRY_SWEEP_MIN_DJ:
                st["nlight"] += 1
            if c["elbow_z"] is not None:
                st["ez_lo"] = c["elbow_z"] if st["ez_lo"] is None else min(st["ez_lo"], c["elbow_z"])
            st["j2"].append(j[1])
            note = "" if src == "ik_ref" else f"  ← 兜底 {src}"
            ez_s = self._fnum(ez_z, 5) if ez_z is not None else "    —"
            print(f"  {tag:<36} Δmax {dj:5.1f}°  肘z {ez_s}  "
                  f"法兰z {self._fnum(c['tcp_z'], 6)}  相机↔板 {self._fnum(c['cam_board'], 5)}"
                  + ("  " + " ".join(fl) if fl else "") + note)
            if fl or src != "ik_ref":                    # 打标的行把六个关节都摊开
                print("        " + "  ".join(f"J{i+1} {a:>7.1f}→{b:>7.1f}"
                                             for i, (a, b) in enumerate(zip(j_prev_, j))))
            return j

        print(f"\n[议程预演] 对准位姿 → {len(TEMPLATE) - len(AGENDA_SKIP_TEMPLATE)} 条议程"
              f"（另 {len(AGENDA_SKIP_TEMPLATE)} 条已拉黑：毒位姿）+ 6 条定尺度平移"
              f"（定尺度按最大 {STROKE_MM:.0f}mm 行程预演；自适应补位姿按实测缺口现场定，"
              f"不在表里；小动不实测肘部=轻量模式，显示 —）：")
        for tag, T_t in plan:
            j_prev = step(tag, T_t, j_prev)
        j2r = (f"J2 范围 [{min(st['j2']):.1f}, {max(st['j2']):.1f}]°" if st["j2"]
               else "J2：全跳过了？")
        note_l = (f"；{st['nlight']} 条小动（≤{DRY_SWEEP_MIN_DJ:.0f}°）没实测肘部"
                  f"（显示 — 就是它，闸不受影响）" if st["nlight"] else "")
        ezl = (f"{self._fnum(st['ez_lo'], 4)}mm" if st["ez_lo"] is not None
               else "—（没有一行实测过肘部）")
        print(f"  汇总：{st['n']} 条计划移动；跳过 {st['nskip']} 条、ik_ref 翻家族被枚举"
              f"兜底救回 {st['nswap']} 条；肘部估计最低 "
              f"{ezl}（闸 {ELBOW_MIN_Z_MM:.0f}）{note_l}；"
              f"最大单步关节变化 {st['dj_hi']:.1f}°（闸 {JOINT_JUMP_DEG:.0f}）；{j2r}")
        print("  " + (f"⚠ 有 {st['nbad']} 条带标记、{st['nskip']} 条会被跳过——逐条看上面"
                      if st["nbad"] else
                      "✓ 全部位姿都有健康解、没触发标记（板没动的话，真跑就按这个轨迹走）"))

    def go_home(self):
        """回家翻转（真动！）：当前位姿 → 正常家族对准位姿，分步 MoveJ，每步全查。

        撞机事故后的"回家"通道：目标 = NORMAL_AIM_JOINTS（验证过的那组，fk 回上次真跑的
        对准位姿）；过渡 = 纯关节线性（200 路点 FK 模拟过：腕/相机全程高位、离板 ≥437mm）。
        MoveJ 指令形状在本机固件上【没验过】，所以第一步先做 J6 ±3° 自测（转 J6 相机不转，
        最无害），形状不对就退出、改用示教器手动按 --dry-joints 的表点动。
        """
        self._connect()
        print(f"已连 {self.arm.ip}（--go-home：会发运动指令！）")
        p_ok = self._preflight(need_auto=True)
        if not p_ok:
            sys.exit("× 前检没过——不发任何指令（模式要在示教器上切到【自动】）。")
        j_cur = self._joints()
        ez_c = self._elbow_z(j_cur)
        if ez_c is None:
            sys.exit("× 当前肘部估计拟合不可信（先别自动翻转）。手动按 --dry-joints 的表点动。")
        print(f"起始肘部估计 z≈{ez_c['z']:.0f}mm（闸 {ELBOW_MIN_Z_MM:.0f}）")
        if ez_c["z"] < ELBOW_MIN_Z_MM:
            sys.exit("× 起始就在肘部塌陷的家族里——先手动点动抬出来（--dry-joints 有表）再跑")
        aim_pose = [float(v) for v in (self.args.aim_pose or DEFAULT_AIM_POSE)]
        j_target = list(NORMAL_AIM_JOINTS)
        pk = self.arm.fk(j_target)
        dpk = float(np.linalg.norm(np.asarray(pk[:3]) - np.asarray(aim_pose[:3])))
        drk = C._rot_axis_angle(C._pose6_to_T(pk)[:3, :3].T
                                @ C._pose6_to_T(aim_pose)[:3, :3])[0]
        if dpk > 3.0 or drk > 1.0:
            sys.exit(f"× 目标关节和对准位姿对不上（{dpk:.2f}mm / {drk:.2f}°）——先跑 --dry-joints 看")
        dj0 = max(abs(a - b) for a, b in zip(j_target, j_cur))
        if dj0 < 1.5:
            print(f"✓ 已在目标位姿/家族（最大关节差 {dj0:.2f}°），什么都不用做。")
            return 0
        steps = self._plan_home(j_cur, j_target)
        pb_tool = np.array([self.cam_offset[0], self.cam_offset[1],
                            self.cam_offset[2] + self.target_dist], float)
        pb_base = (C._pose6_to_T(aim_pose)[:3, 3]
                   + C._pose6_to_T(aim_pose)[:3, :3] @ pb_tool)
        home_v = min(float(self.args.speed), HOME_SPEED_PCT)
        print(f"\n计划：{len(steps)} 步 MoveJ（每步 ≤{HOME_STEP_DEG:.0f}°，速度 {home_v:.0f}%），"
              f"J3 翻 {abs(j_target[2] - j_cur[2]):.0f}°、J4 翻 {abs(j_target[3] - j_cur[3]):.0f}°。"
              f"每步查：等停稳 → 关节回读（> {MOVE_VERIFY_JOINT_DEG:.0f}° 停）→ "
              f"肘部 z / 法兰 z / 离板距离。")
        try:
            s = input("手放急停上，确认输入 GO 回车（别的任何输入 = 退出）：").strip()
        except EOFError:
            sys.exit("读不到终端输入——没开跑")
        if s != "GO":
            print("没开跑。")
            return 1
        log = None
        try:
            self.shots.mkdir(parents=True, exist_ok=True)
            log = open(self.shots / time.strftime("joints_home_%Y%m%d_%H%M.csv"),
                       "w", newline="")
            lw = csv.writer(log)
            lw.writerow(["time", "row"] + [f"jT{i}" for i in range(1, 7)]
                        + [f"jA{i}" for i in range(1, 7)]
                        + ["dj_max", "elbow_z", "tcp_z", "cam_board", "tcp_board"])

            print("\n[0] 使能…")
            rc = self.arm.enable()
            print(f"    RobotEnable(1) → {rc}")
            time.sleep(1.0)
            if self._st("error_code") != 0:
                raise Stop(f"使能后 error_code={self._st('error_code')}——现场看一眼")

            print("[1] MoveJ 形状自测：J6 ±3° 再回位（转 J6 相机不转，最无害的一步）…")
            d6 = -3.0 if j_cur[5] > 170.0 else 3.0
            j_t6 = list(j_cur)
            j_t6[5] += d6
            self.arm.move_j(j_t6, speed=5.0, blocking=True)
            self._settle("MoveJ 自测")
            j_r = self._joints()
            d5 = j_r[5] - j_cur[5]
            d_oth = max(abs(j_r[i] - j_cur[i]) for i in (0, 1, 2, 3, 4))
            if abs(d5 - d6) > 0.5 or d_oth > 0.5:
                raise Stop(f"MoveJ 形状不对：本想只动 J6 {d6:+.0f}°，实测 ΔJ6={d5:+.2f}°、"
                           f"其他轴最大动 {d_oth:.2f}°——这条通道先别用，改手动配对点动"
                           f"（表在 --dry-joints 的输出里）")
            print(f"    ✓ 形状对：ΔJ6={d5:+.2f}°（其他轴 ≤ {d_oth:.2f}°）")
            self.arm.move_j(j_cur, speed=5.0, blocking=True)
            self._settle("MoveJ 自测回位")

            print(f"[2] 过渡 {len(steps)} 步…")
            for k, j_k in enumerate(steps, 1):
                self.arm.move_j(j_k, speed=home_v, blocking=True)
                self._settle(f"过渡 {k}/{len(steps)}")
                j_a = self._joints()
                dja = max(abs(a - b) for a, b in zip(j_a, j_k))
                c = self._home_checks(j_a, pb_base)
                if dja > MOVE_VERIFY_JOINT_DEG:
                    raise Stop(f"第 {k} 步关节回读差 {dja:.1f}°——停")
                if c["elbow_z"] is not None and c["elbow_z"] < ELBOW_MIN_Z_MM:
                    raise Stop(f"第 {k} 步肘部 z={c['elbow_z']:.0f}mm < "
                               f"{ELBOW_MIN_Z_MM:.0f}——停")
                if c["tcp_z"] is not None and c["tcp_z"] < HOME_MIN_TCP_Z:
                    raise Stop(f"第 {k} 步法兰 z={c['tcp_z']:.0f}mm < {HOME_MIN_TCP_Z:.0f}——停")
                if c["cam_board"] is not None and c["cam_board"] < HOME_MIN_CAM_BOARD:
                    raise Stop(f"第 {k} 步相机离板 {c['cam_board']:.0f}mm < "
                               f"{HOME_MIN_CAM_BOARD:.0f}——停")
                if c["tcp_board"] is not None and c["tcp_board"] < HOME_MIN_WRIST_BOARD:
                    raise Stop(f"第 {k} 步法兰离板 {c['tcp_board']:.0f}mm < "
                               f"{HOME_MIN_WRIST_BOARD:.0f}——停")
                print(f"  {k:>2}/{len(steps)}  回读差 {dja:.2f}°  肘z {self._fnum(c['elbow_z'], 4)}  "
                      f"法兰z {self._fnum(c['tcp_z'], 5)}  相机↔板 {self._fnum(c['cam_board'], 5)}mm",
                      flush=True)
                lw.writerow([time.strftime("%H:%M:%S"), f"step {k}"]
                            + [f"{v:.3f}" for v in j_k] + [f"{v:.3f}" for v in j_a]
                            + [f"{dja:.3f}"]
                            + ["" if v is None else f"{v:.1f}" for v in
                               (c["elbow_z"], c["tcp_z"], c["cam_board"], c["tcp_board"])])
                log.flush()
            j_end = self._joints()
            dj_e = max(abs(a - b) for a, b in zip(j_end, j_target))
            print(f"\n✓ 到位：和目标的关节差最大 {dj_e:.2f}°")
            print(f"日志：{log.name}")
            print("下一步：python3 -I src/auto_handeye.py --go（手放急停；板没动的话板就在"
                  "画面里，可以直接开跑）")
            return 0
        except KeyboardInterrupt:
            print("\n\n你按了 Ctrl-C —— 先 Pause 停住…")
            self._pause_now()
            return 2
        except (Stop, SafetyError) as e:
            print(f"\n\n■ 停下来：{e}")
            self._pause_now()
            self._go_home_after_stop()
            return 3
        except FRRpcError as e:
            print(f"\n\n■ RPC 报错：{e}")
            print("    （很可能就是 MoveJ 指令形状在本机固件上没验过——别硬试；"
                  "改手动配对点动，表在 --dry-joints 的输出里）")
            self._pause_now()
            self._go_home_after_stop()
            return 3
        finally:
            if log is not None:
                log.close()

    def _go_home_after_stop(self):
        try:
            j = self._joints()
            print(f"      现在关节 [{j[0]:.2f} {j[1]:.2f} {j[2]:.2f} "
                  f"{j[3]:.2f} {j[4]:.2f} {j[5]:.2f}]")
        except Exception:                                   # noqa: BLE001
            pass
        print("      臂保持使能停着。搞定原因后可以再跑一次 --go-home（会从当前关节重新分步）。")


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser(description="D 批手眼采集 · 自动（程序控臂走位姿）")
    ap.add_argument("--go", action="store_true",
                    help="真跑。默认（不加）只做只读预演：连上读位姿+相机看板+打印议程，不发指令")
    ap.add_argument("--disable", action="store_true", help="只做一件事：下使能后退出")
    ap.add_argument("--selftest", action="store_true", help="只跑离线数学校验，不连机器人")
    ap.add_argument("--shots", type=Path, default=C.HE_SHOTS_DIR,
                    help=f"存帧目录（默认 {C.HE_SHOTS_DIR}）")
    ap.add_argument("--intr", type=Path, default=C.OUT_YAML,
                    help=f"内参 yaml（默认 {C.OUT_YAML}）")
    ap.add_argument("--speed", type=float, default=SPEED_DEFAULT,
                    help=f"运动速度百分比（默认 {SPEED_DEFAULT:.0f}，上限 config.MAX_SPEED_PCT）")
    ap.add_argument("--tilt", type=float, default=TILT_DEG, help=f"偏摆基准角（默认 {TILT_DEG:.0f}°）")
    ap.add_argument("--roll", type=float, default=ROLL_DEG, help=f"滚转基准角（默认 {ROLL_DEG:.0f}°）")
    ap.add_argument("--max-shots", type=int, default=MAX_SHOTS, help=f"最多存几张（默认 {MAX_SHOTS}）")
    ap.add_argument("--target-mm", type=float, default=TARGET_DIST_MM,
                    help=f"对准的目标板距 mm（默认 {TARGET_DIST_MM:.0f}）")
    ap.add_argument("--cam-offset", type=float, nargs=3, default=list(CAM_OFFSET_DEFAULT),
                    metavar=("X", "Y", "Z"),
                    help="相机光心在法兰系里的大致位置 mm——只当绕板心画弧的圆心先验")
    ap.add_argument("--no-return", action="store_true", help="收尾不回出发位姿（默认回）")
    ap.add_argument("--dry-joints", action="store_true",
                    help="只读预演关节轨迹：回家过渡 + 整条议程的 IK/关节/肘部高度全打出来"
                         "（不发指令、不碰相机、不写盘）")
    ap.add_argument("--dry-agenda-only", action="store_true",
                    help="预演只跑议程段（八组枚举表/回家过渡表跳过——它们已单独验证过，省调用）")
    ap.add_argument("--dry-from", type=int, default=1, metavar="N",
                    help="议程预演从第 N 条起（分块跑；或重启后直扑第 N 条，测它是不是一碰就死的触发器）")
    ap.add_argument("--go-home", action="store_true",
                    help="回家翻转：把臂从当前位置分步（MoveJ，每步 ≤10°）摆到正常家族对准位姿"
                         "（真发运动指令！手放急停）")
    ap.add_argument("--aim-pose", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                    help=f"议程基准对准位姿（默认用上次真跑记录 {DEFAULT_AIM_POSE}；板挪过才需要给）")
    args = ap.parse_args()

    if args.selftest:
        _math_selftest()
        return 0

    if args.disable:
        a = Arm()
        a.connect()
        try:
            print(f"现在：急停={getattr(a.st, 'estop', None)} error_code={getattr(a.st, 'error_code', None)} "
                  f"模式={getattr(a.st, 'robot_mode', None)}")
            print(f"RobotEnable(0) → {a.disable()}")
        finally:
            a.close()
        return 0

    if args.dry_joints:
        auto = Auto(args)
        try:
            return auto.dry_joints()
        finally:
            auto._close()
    if args.go_home:
        auto = Auto(args)
        try:
            return auto.go_home()
        finally:
            auto._close()

    auto = Auto(args)
    try:
        return auto.run()
    finally:
        auto._close()


if __name__ == "__main__":
    sys.exit(main())
