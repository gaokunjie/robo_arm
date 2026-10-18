# -*- coding: utf-8 -*-
"""法奥 FR3 · 8083 状态反馈读取（后台线程，永远只保留最新一帧）

为什么需要它：本机固件没有 GetRobotMotionDone / GetRobotErrorCode 这类 RPC，
"运动到没到位""有没有报错""急停有没有被按"全都要从 8083 的状态帧里读。

帧格式（《机器人控制器 8083 端口状态反馈用户手册》4.0.0）：
    帧头 0x5A5A(uint16) | CNT(uint8) | LEN(uint16) | DATA(LEN 字节) | 校验和(uint16)
    校验和 = 帧头到数据内容所有字节按 uint16 求和
数据内容是紧凑排列(无对齐填充)的定长结构，字段顺序见 _FMT。

用法：
    st = StateReader(); st.start()
    st.motion_done   # 1=到位
    st.error_code    # 当前错误码
    st.estop         # 1=急停按下
"""

import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 支持 python3 -I 直接运行本文件
import config

# 逐字段布局（'<' = 小端 + 无对齐）。每项 = (字段名, struct 格式, 是否数组)
# 注意：本机固件(000-000-022)的帧只有 234 字节，到 gripper_motion_done 为止；
#       4.0.0 手册里的伺服/扩展轴/夹爪后半段字段这台机器还没有（实测帧长核对过）。
_LAYOUT = [
    ("program_state",       "B",   False),   # 1-停止 2-运行 3-暂停 4-拖动
    ("error_code",          "B",   False),
    ("robot_mode",          "B",   False),   # 0-自动 1-手动 2-拖动
    ("joints",              "6d",  True),    # 关节角 deg
    ("tcp",                 "6d",  True),    # 工具位姿
    ("tool_num",            "i",   False),
    ("torques",             "6d",  True),    # N·m
    ("program_name",        "20s", False),
    ("prog_total_line",     "B",   False),
    ("prog_cur_line",       "B",   False),
    ("cl_do_h",             "B",   False),
    ("cl_do_l",             "B",   False),
    ("tl_do_l",             "B",   False),
    ("cl_di_h",             "B",   False),
    ("cl_di_l",             "B",   False),
    ("tl_di_l",             "B",   False),
    ("ft_data",             "6d",  True),
    ("ft_active",           "B",   False),
    ("estop",               "B",   False),   # 1-急停按下
    ("motion_done",         "i",   False),   # 1-到位 0-未到位
    ("gripper_motion_done", "B",   False),
]
_FIELDS = [f[0] for f in _LAYOUT]
_ARRAY_FIELDS = {f[0] for f in _LAYOUT if f[2]}

# 预编译每个字段的 (名字, Struct, 偏移)
_OFFS = []
_off = 0
for _name, _fmt, _is_arr in _LAYOUT:
    _st = struct.Struct("<" + _fmt)
    _OFFS.append((_name, _is_arr, _st))
    _off += _st.size
_SZ = _off
assert _SZ == 234, f"状态帧布局对不上: 算出 {_SZ} 字节，实测 234 字节"


class StateReader(threading.Thread):
    daemon = True

    def __init__(self, ip=None, port=8083):
        super().__init__(name="fr-state-8083")
        self.ip = ip or config.ROBOT_IP
        self.port = port
        self._lock = threading.Lock()
        self._data = {}
        self._t = 0.0
        self._stop = threading.Event()
        self.ok = False
        self.last_error = None
        self.frames = 0

    # ---------------- 线程体 ----------------
    def run(self):
        while not self._stop.is_set():
            try:
                self._loop()
            except Exception as e:
                self.ok = False
                self.last_error = f"{type(e).__name__}: {e}"
                time.sleep(0.5)

    def _loop(self):
        s = socket.create_connection((self.ip, self.port), timeout=2.0)
        s.settimeout(2.0)
        self.ok = True
        self.last_error = None
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("8083 连接被关闭")
            buf += chunk
            while True:
                i = buf.find(b"\x5a\x5a")
                if i < 0 or len(buf) < i + 5:
                    break
                _head, _cnt, dlen = struct.unpack_from("<HBH", buf, i)
                if len(buf) < i + 5 + dlen + 2:
                    break
                body = buf[i + 5:i + 5 + dlen]
                buf = buf[i + 5 + dlen + 2:]
                if dlen == _SZ:
                    self._ingest(body)
        s.close()

    def _ingest(self, body):
        d = {}
        pos = 0
        for name, is_arr, st in _OFFS:
            v = st.unpack_from(body, pos)
            pos += st.size
            if is_arr:
                d[name] = [round(x, 4) for x in v]
            elif name == "program_name":
                d[name] = v[0].strip(b"\x00").decode("utf-8", "ignore")
            else:
                d[name] = v[0]
        with self._lock:
            self._data = d
            self._t = time.time()
            self.frames += 1

    # ---------------- 读取接口 ----------------
    def snapshot(self):
        with self._lock:
            return dict(self._data), self._t

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        with self._lock:
            if name in self._data:
                return self._data[name]
        raise AttributeError(name)

    @property
    def age(self):
        with self._lock:
            return time.time() - self._t if self._t else float("inf")

    def wait_motion_done(self, timeout=None, poll=0.02):
        """等运动到位（robot_motion_done == 1）。返回等待秒数。"""
        timeout = timeout or config.MOTION_TIMEOUT_S
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.motion_done == 1:
                    return time.time() - t0
            except AttributeError:
                pass
            time.sleep(poll)
        raise TimeoutError(f"等待运动到位超时 {timeout}s")

    def stop(self):
        self._stop.set()


if __name__ == "__main__":
    st = StateReader()
    st.start()
    time.sleep(1.5)
    print(f"帧长校验: 解析需要 {_SZ} 字节")
    d, t = st.snapshot()
    if not d:
        print("没收到有效帧:", st.last_error)
    else:
        print(f"已收帧数 {st.frames}, 距今 {time.time()-t:.2f}s")
        for k in ("program_state", "error_code", "robot_mode", "estop", "motion_done"):
            print(f"  {k} = {d[k]}")
        print("  joints =", d["joints"])
        print("  tcp    =", d["tcp"])
    st.stop()
