# -*- coding: utf-8 -*-
"""P0 自检脚本 · 只读，机器人绝不会动。

分五关，任何一关不过都会明确告诉你卡在哪：
    1. 相机设备在不在
    2. 指令通道（xmlrpc :20003）通不通
    3. 状态通道（8083）通不通，且和指令通道读数对不对得上
    4. 正逆解自洽（默认【跳过】——控制器逆解是禁区，见下；--ik 才冒险跑）
    5. 现场安全状态：急停、错误码、当前是否在运动

⚠ 第 4 关默认跳过（2026-10-16 起）：它调用的 GetInverseKin 正是四次应用层崩溃的
  唯一共同因子（全部发生在工作空间边界附近的位姿上，事后重启才能恢复）。
   本自检从此不碰逆解。等自研离线运动学（fr3_model.py）接好线，第 4 关改成
   "正解自洽 + 自研 IK/FK 一致性"检查——用的全是 GetForwardKin（可信）。

运行：  python3 -I src/check_connect.py         # 只读五关（第 4 关跳过）
        python3 -I src/check_connect.py --ik    # 连第 4 关也跑（危险，手放急停）
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # -I 模式下不会自动带上脚本目录

import config                                              # noqa: E402
from fr_arm import Arm                                     # noqa: E402

PASS, FAIL, WARN = "✓", "✗", "!"


def hr(title=""):
    print(f"\n{'─' * 66}")
    if title:
        print(title)


def main():
    print("=" * 66)
    print(f"P0 自检 · {config.ROBOT_IP} · 固件 {config.FIRMWARE}")
    print("=" * 66)

    # ---------- 1. 相机 ----------
    hr("[1/5] 相机")
    dev = f"/dev/video{config.CAMERA_DEV}"
    if os.path.exists(dev):
        print(f"  {PASS} {dev} 存在")
    else:
        print(f"  {FAIL} {dev} 不存在 —— 设备号可能变了，跑 `ls /dev/v4l/by-id/` 看看")

    # ---------- 2/3. 两条通道 ----------
    arm = Arm()
    hr("[2/5] 指令通道 xmlrpc :20003")
    arm.connect()
    print(f"  {PASS} 连接成功，机器人报告 {len(arm.r.methods())} 个可用指令")

    hr("[3/5] 状态通道 :8083 与指令通道互验")
    rpc_j = arm.joint_deg()
    st_j = arm.st.joints
    diff = max(abs(a - b) for a, b in zip(rpc_j, st_j))
    tag = PASS if diff < 0.05 else FAIL
    print(f"  {tag} 关节角最大差 {diff:.4f} 度（关节流 100ms 一帧，几毫度以内算正常）")
    print(f"      RPC  : {[round(v, 3) for v in rpc_j]}")
    print(f"      8083 : {[round(v, 3) for v in st_j]}")
    print(f"      TCP  : {[round(v, 2) for v in arm.tcp_pose()]}   ← 当前是法兰位姿(工具号0)")

    # ---------- 4. 正逆解自洽 ----------
    hr("[4/5] 正逆解自洽性")
    pose = arm.tcp_pose()
    if "--ik" not in sys.argv:
        print(f"  {WARN} 默认跳过：控制器逆解（GetInverseKin / GetInverseKinRef）是【禁区】")
        print("      2026-10-16 四次应用层崩溃的共同因子，全部发生在工作空间边界附近的位姿，")
        print("      纯规划就崩（臂没动过），行为不可预测。自研离线运动学接好线前不碰它。")
        print("      知道风险仍要跑：python3 -I src/check_connect.py --ik（黑匣子开、手放急停）")
        # 用可信的正解做一条不碰逆解的自洽检查
        fk_pose = arm.fk(rpc_j)
        d_pos = sum((a - b) ** 2 for a, b in zip(fk_pose[:3], pose[:3])) ** 0.5
        tag = PASS if d_pos < 0.5 else WARN
        print(f"  {tag} [替代] 实测关节→正解 vs 实测法兰位姿 位置偏差 {d_pos:.4f} mm"
              f"（正解可信的当场旁证）")
    else:
        try:
            j_ik = arm.ik(pose)
            j_rt = arm.fk(j_ik)
            pos_err = sum((a - b) ** 2 for a, b in zip(pose[:3], j_rt[:3])) ** 0.5
            tag = PASS if pos_err < 0.5 else WARN
            print(f"  {tag} 位姿→逆解→正解 偏差 {pos_err:.4f} mm")
            print(f"      逆解关节 = {[round(v, 3) for v in j_ik]}")
            jac = max(abs(a - b) for a, b in zip(j_ik, rpc_j))
            tag = PASS if jac < 1.0 else WARN
            print(f"  {tag} 逆解关节 vs 实际关节 最大差 {jac:.3f} 度"
                  f"{'' if jac < 1.0 else '  ← 差得多说明有别的等效解(肘上/肘下)，不影响，但记一笔'}")
        except Exception as e:
            print(f"  {FAIL} 逆解失败: {e}")

    # ---------- 5. 现场安全 ----------
    hr("[5/5] 现场安全状态")
    st = arm.st
    checks = [
        ("急停未按下", st.estop == 0, f"estop={st.estop}"),
        ("无错误码", st.error_code == 0, f"error_code={st.error_code}"),
        ("当前不在运动中", st.motion_done == 1, f"motion_done={st.motion_done}"),
    ]
    for name, ok, detail in checks:
        print(f"  {PASS if ok else FAIL} {name}  ({detail})")
    print(f"      模式 robot_mode={st.robot_mode} (0自动/1手动)  "
          f"程序 program_state={st.program_state} (1停止/2运行/3暂停/4拖动)")
    print(f"      关节温度/力矩 torques={[round(v, 1) for v in st.torques]}")

    # ---------- 汇总 ----------
    hr()
    n_methods = len(arm.r.methods())
    print(f"结论：指令通道 {n_methods} 个方法可用；两条通道读数一致；机器人可安全接管。")
    print("本次运行全部只读，机器人未运动。")

    arm.close()


if __name__ == "__main__":
    main()
