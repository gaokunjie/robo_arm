# -*- coding: utf-8 -*-
"""P1 第一枪 · 零位移 MoveL 自检（第一次让机器人收指令）

要验证的链路：PC 发 xmlrpc → 控制器接受 → 机器人执行 → 8083 读到"运动到位"

【不带参数跑 = 完全只读，机器人收不到任何指令】
要真发指令，必须显式加 --confirm-motion。

加了 --confirm-motion 之后做这几件事：
    1. 只读：连上、存档当前位姿/关节角/安全状态
    2. 使能：电机上电，臂从"软"变"绷住"。**臂不移动**
    3. 发一条 MoveL，目标是【第 1 步读回来的当前位置】，位移 = 0
    4. 盯着 8083 看 motion_done 有没有掉到 0 再回 1
         —— 看到就说明控制器【真的执行了】，不只是"收下了命令"
         —— 没看到不能证明没执行（零位移可能一瞬间就完事），要看 3/4 的旁证
    5. 再读位姿，验：RPC 无错、无新错误码、位姿没动（< 0.05mm）
    6. 打印结论后停下

它【不会】做的事：
    · 不算任何目标坐标 —— 目标就是"你现在在哪"，不碰桌面、不碰杯子
    · 不走第二段、不重试、不自动改模式
    · 任何一步不过就停下报告

运行：
    python3 -I src/selfcheck_motion.py                      # 只读，绝对安全
    python3 -I src/selfcheck_motion.py --confirm-motion     # 使能 + 零位移 MoveL
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I 下不会自动带脚本目录

import config                                                   # noqa: E402
from fr_arm import Arm, SafetyError                             # noqa: E402

PASS, FAIL, WARN = "✓", "✗", "!"


def hr(t=""):
    print(f"\n{'─' * 66}")
    if t:
        print(t)


def fmt(v, n=3):
    return "[" + ", ".join(f"{x:.{n}f}" for x in v) + "]"


def preflight(arm):
    """现场安全前置检查。任一不过就不该发指令。"""
    st = arm.st
    checks = [
        ("急停未按下", st.estop == 0, f"estop={st.estop}"),
        ("无错误码", st.error_code == 0, f"error_code={st.error_code}"),
        ("当前不在运动", st.motion_done == 1, f"motion_done={st.motion_done}"),
    ]
    ok = True
    for name, good, detail in checks:
        print(f"  {PASS if good else FAIL} {name}  ({detail})")
        ok = ok and good
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-motion", action="store_true",
                    help="真的使能并发一条零位移 MoveL。不加这个参数则完全只读。")
    args = ap.parse_args()

    print("=" * 66)
    print(f"P1 零位移 MoveL 自检 · {config.ROBOT_IP} · 固件 {config.FIRMWARE}")
    print("=" * 66)
    if not args.confirm_motion:
        print("模式：**只读**（没加 --confirm-motion，机器人收不到任何指令）")

    arm = Arm()
    arm.connect()

    # ---------- 1. 存档 ----------
    hr("[1] 发指令前的存档")
    pose0 = arm.tcp_pose()
    joint0 = arm.joint_deg()
    print(f"  TCP  = {fmt(pose0, 2)}   ← 工具号 {config.TOOL_ID}，即法兰位姿")
    print(f"  关节 = {fmt(joint0)}")

    hr("[2] 现场安全前置检查")
    if not preflight(arm):
        print("\n  ✗ 前置检查没过，不发任何指令。先处理上面标 ✗ 的那条。")
        arm.close()
        return 1

    mode = getattr(arm.st, "robot_mode", None)
    print(f"  · 模式 robot_mode={mode} ({'自动' if mode == 0 else '手动' if mode == 1 else '拖动' if mode == 2 else '?'})")
    if mode == 1:
        print(f"    {WARN} 现在是**手动模式**。手动模式下控制器常常拒绝 RPC 发来的运动指令，")
        print(f"      下面那条 MoveL 有可能直接被拒——被拒不算故障，正好说明得先切自动模式。")
        print(f"      切换方式：示教器上切到自动模式。脚本不自动切——那会改机器人状态，得你点头。")

    if not args.confirm_motion:
        hr("[3] 本可以做什么（现在不做）")
        print("  加 --confirm-motion 后会：使能 → 发一条 MoveL 到【下面这个位姿】→ 验位姿没动")
        print(f"    目标位姿 = {fmt(pose0, 2)}   （就是当前位置，位移 0）")
        print(f"  速度会被夹到 {config.MAX_SPEED_PCT}%（旧版 MoveL 只收百分比），位移闸 {config.MAX_MOVE_MM} mm 也照样生效。")
        hr()
        print("本次运行全部只读，机器人未收到任何指令。")
        arm.close()
        return 0

    # ---------- 3. 使能 ----------
    hr("[3] 使能（电机上电；臂会从软变绷住，但不移动）")
    rc = arm.enable()
    print(f"  RobotEnable(1) 返回 {rc}")
    time.sleep(1.0)
    if not preflight(arm):
        print("\n  ✗ 使能之后安全检查没过，停在这里，不发运动指令。")
        arm.close()
        return 1

    # ---------- 4. 零位移 MoveL ----------
    hr("[4] 零位移 MoveL")
    print(f"  目标 = 当前位置 = {fmt(pose0, 2)}")
    print(f"  位移 = {sum((a - b) ** 2 for a, b in zip(pose0[:3], arm.tcp_pose()[:3])) ** 0.5:.4f} mm")
    print("  发送中…")

    t0 = time.time()
    try:
        joints_sent = arm.move_l(pose0, speed=config.MAX_SPEED_PCT, blocking=False)
    except SafetyError as e:
        print(f"  {FAIL} 被安全闸拦下：{e}")
        arm.close()
        return 1
    except Exception as e:
        print(f"  {FAIL} 发指令就失败了：{type(e).__name__}: {e}")
        print("      → 这多半是参数格式或模式被拒，不是机械故障。把上面这行原文发我。")
        arm.close()
        return 1

    # 盯 motion_done：掉到 0 说明控制器真的开始执行了
    saw_start, t_start = False, None
    while time.time() - t0 < 3.0:
        if arm.st.motion_done == 0:
            saw_start, t_start = True, time.time() - t0
            break
        time.sleep(0.005)
    print(f"  发送耗时 {(time.time() - t0) * 1000:.0f} ms；逆解关节 = {fmt(joints_sent)}")
    if saw_start:
        print(f"  {PASS} 看到 motion_done 掉到 0（在第 {t_start * 1000:.0f} ms）—— 控制器执行了这条指令")
    else:
        print(f"  {WARN} 3 秒内没看到 motion_done 掉到 0。")
        print(f"     零位移动作可能快到读不到（帧率 100ms），也可能被拒了——看下面两关怎么过。")

    try:
        arm.wait_done(timeout=5.0)
        print(f"  {PASS} motion_done 回到 1（用时 {time.time() - t0:.2f}s）")
    except TimeoutError as e:
        print(f"  {FAIL} {e}")

    # ---------- 5. 事后核对 ----------
    hr("[5] 事后核对")
    time.sleep(0.3)
    pose1 = arm.tcp_pose()
    d = sum((a - b) ** 2 for a, b in zip(pose0[:3], pose1[:3])) ** 0.5
    print(f"  TCP = {fmt(pose1, 2)}")
    print(f"  {PASS if d < 0.05 else FAIL} 位移 = {d:.4f} mm（零位移指令，应该就是 0）")

    st = arm.st
    print(f"  {PASS if st.error_code == 0 else FAIL} 错误码 = {st.error_code}")
    print(f"  {PASS if st.estop == 0 else FAIL} 急停 = {st.estop}")

    hr()
    if saw_start and d < 0.05 and st.error_code == 0:
        print("结论：**链路通了**。控制器接受了我们的参数格式、执行了指令、位姿没动、无错误。")
        print("      下一次才让臂真的走一小段（垂直向上 5mm，是最不容易撞到东西的方向）。")
        print("      臂现在还处于【使能】状态，是绷住的。要松下来就跑 RobotEnable(0)。")
    else:
        print("结论：**有东西不对**，先别往下走。把上面整段输出发我。")
    arm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
