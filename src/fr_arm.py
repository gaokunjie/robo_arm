# -*- coding: utf-8 -*-
"""法奥 FR3 · 机械臂封装（唯一允许发运动指令的入口）

三层结构：
    fr_rpc.FRRpc        —— 指令下行（xmlrpc :20003）
    fr_state.StateReader —— 状态上行（8083 状态帧，100ms 一帧）
    Arm（本文件）        —— 把两者凑成"发一条、等一条到位"的阻塞式接口，外加三道安全闸

三道安全闸（任何一条不过就抛异常，绝不发指令）：
    1. 速度闸：speed 夹到 MAX_SPEED_PCT 以内（本机旧版 MoveL 只收百分比，给不了 mm/s）
    2. 位移闸：单条运动相对当前位置的位移 ≤ MAX_MOVE_MM
    3. 空间闸：目标 TCP 必须落在 WORKSPACE 盒子里（ENFORCE_WORKSPACE 打开后才生效）

为什么运动到位要读 8083 而不是 RPC：本机固件没有 GetRobotMotionDone 这类查询指令。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 支持 python3 -I 直接运行本文件

import config
from fr_rpc import FRRpc, FRRpcError
from fr_state import StateReader


class SafetyError(RuntimeError):
    """安全闸拦下的操作。机器人没有收到任何指令。"""


def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a[:3], b[:3])) ** 0.5


class Arm:
    def __init__(self, ip=None):
        self.ip = ip or config.ROBOT_IP
        self.r = FRRpc(self.ip)
        self.st = StateReader(self.ip)
        self._started = False

    # ---------------- 连接 ----------------
    def connect(self):
        """连上并启动状态流。不做任何运动、不使能。"""
        self.st.start()
        t0 = time.time()
        while time.time() - t0 < 5.0:                # 等第一帧，确认状态通道真的活着
            if self.st.frames:
                break
            time.sleep(0.05)
        if not self.st.frames:
            raise ConnectionError(f"8083 状态通道没收到帧: {self.st.last_error}")
        self._started = True
        return True

    def close(self):
        self.st.stop()

    # ---------------- 只读 ----------------
    def tcp_pose(self):
        return list(self.r.tcp_pose())

    def joint_deg(self):
        return list(self.r.joint_deg())

    def ik(self, pose, config_=-1):
        return self.r.ik(pose, config_)

    def ik_ref(self, pose, ref):
        """带参考关节的逆解（钉家族用）：ref 一般传【当前关节】。见 fr_rpc.ik_ref。"""
        return self.r.ik_ref(pose, ref)

    def fk(self, joints):
        return self.r.fk(joints)

    def errors(self):
        """两级错误信息：8083 的 error_code + webapp 的 error_info 原文。"""
        return {"error_code": getattr(self.st, "error_code", None)}

    # ---------------- 状态 ----------------
    def wait_done(self, timeout=None):
        self.st.wait_motion_done(timeout)
        return self.st.snapshot()[0]

    def _guard(self, target_pose, speed):
        """三道闸。返回夹过速度的 speed。"""
        # 闸 3：空间盒
        if config.ENFORCE_WORKSPACE:
            for i, ax in enumerate(("x", "y", "z")):
                lo, hi = config.WORKSPACE[ax]
                if not (lo <= target_pose[i] <= hi):
                    raise SafetyError(
                        f"目标 {ax}={target_pose[i]:.1f} 超出工作空间 [{lo}, {hi}]")
        # 闸 2：单条位移
        cur = self.tcp_pose()
        d = _dist(target_pose, cur)
        if d > config.MAX_MOVE_MM:
            raise SafetyError(
                f"单条位移 {d:.1f}mm 超过上限 {config.MAX_MOVE_MM}mm —— 拆成多段，或确认坐标没错")
        # 闸 1：速度。本机旧版 MoveL 的速度单位是【百分比】，不是 mm/s
        v = min(abs(float(speed)), config.MAX_SPEED_PCT)
        return v

    # ---------------- 运动 ----------------
    def move_l(self, target_pose, speed=None, acc=None, blocking=True, joints=None):
        """直线运动到目标位姿（基座系 mm/deg）。

        target_pose 是【环心 TCP】的位姿；MoveL 要求同时给关节角和笛卡尔位姿。

        joints: 可选。调用方已经逆解并验过的关节角——给了就【原样下发，不再自己逆解】
        （采集器的"钉家族 + 送验过的关节"通道；撞机事故的教训之一就是这里会悄悄
        再解一次、绕过调用方的闸）。不给时自己解：用 ik_ref(pose, 当前关节)，
        钉在当前家族上（2026-10-16 前用的是 GetInverseKin 自动解，家族不可控）。

        speed / acc 都是【百分比 0~100】（本机旧版 MoveL 没有 velAccParamMode，
        给不了 mm/s）。默认取 config.MAX_SPEED_PCT / MAX_ACC_PCT，会被闸门再夹一次。
        """
        speed = speed if speed is not None else config.MAX_SPEED_PCT
        acc = acc if acc is not None else config.MAX_ACC_PCT
        v = self._guard(target_pose, speed)
        if joints is None:
            joints = self.ik_ref(target_pose, self.joint_deg())
        if len(joints) != 6:
            raise SafetyError(f"逆解没给出 6 个关节角: {joints}")
        # blendR=0（非阻塞）：-1 的"阻塞"语义会让服务端把这次 xmlrpc 调用挂到运动结束
        # （2026-10-16 实测：8° 要 2.9s、48° 要 ~17s＞客户端 3s 超时——而且【超时≠没动】，
        # 客户端放弃后控制器照跑完）。等结束改用状态制 wait_done（8083 帧）。
        self.r.move_l([float(x) for x in joints], target_pose, config.TOOL_ID, config.USER_ID,
                      v, acc, 100, 0.0)
        if blocking:
            self.wait_done()
        return joints

    def move_j(self, target_joints, speed=None, blocking=True):
        """关节运动（一般只用于大范围换位，取放过程用 move_l 走直线）。"""
        speed = speed if speed is not None else config.MAX_SPEED_PCT
        speed = min(abs(float(speed)), config.MAX_SPEED_PCT)
        pose = self.fk(target_joints)
        # blendT=0（非阻塞）理由同 move_l：-1 阻塞语义会把 3s 超时撞穿（2026-10-16 实测）
        self.r.move_j(target_joints, pose, config.TOOL_ID, config.USER_ID,
                      speed, 30.0, 100, 0.0)
        if blocking:
            self.wait_done()
        return pose

    # ---------------- 使能 / 急停 ----------------
    def enable(self):
        return self.r.enable(1)

    def disable(self):
        return self.r.enable(0)

    def pause(self):
        return self.r.stop()

    def resume(self):
        return self.r.resume()

    def estop_pressed(self):
        return getattr(self.st, "estop", None) == 1


if __name__ == "__main__":
    a = Arm()
    a.connect()
    print(f"已连接 {a.ip}；8083 帧数 {a.st.frames}")
    print(f"  TCP   = {[round(v, 2) for v in a.tcp_pose()]}")
    print(f"  关节  = {[round(v, 2) for v in a.joint_deg()]}")
    print(f"  8083  = joints {a.st.joints}")
    print(f"  急停  = {a.estop_pressed()}   错误码 = {a.st.error_code}")
    a.close()
