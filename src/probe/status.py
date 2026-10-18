# -*- coding: utf-8 -*-
"""只读状态快照：完整 8083 帧 + RPC 关键读数。不发任何指令。"""
import http.client
import sys
import time
import xmlrpc.client

sys.path.insert(0, "/home/kelly/桌面/1014/src")
from fr_state import StateReader

IP = "192.168.58.3"


class T(xmlrpc.client.Transport):
    def __init__(self, timeout=5.0):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host):
        return http.client.HTTPConnection(host, timeout=self.timeout)


p = xmlrpc.client.ServerProxy(f"http://{IP}:20003", transport=T())

st = StateReader(IP)
st.start()
t0 = time.time()
while time.time() - t0 < 5.0 and not st.frames:
    time.sleep(0.05)
print(f"8083 帧数 = {st.frames}")
snap, ts = st.snapshot()
for k, v in snap.items():
    if isinstance(v, list):
        v = [round(x, 3) for x in v]
    print(f"  {k:<22} = {v}")
print(f"  帧年龄              = {time.time() - ts:.2f} s")


def show(label, name, *a):
    try:
        v = getattr(p, name)(*a)
        if isinstance(v, (list, tuple)):
            v = [round(x, 3) for x in v]
        print(f"  {label:<22} = {v}")
    except Exception as e:
        print(f"  {label:<22} = [{type(e).__name__}] {e}")


print("\nRPC 只读：")
show("GetActualTCPPose", "GetActualTCPPose", 1)
show("GetActualJointPosDegree", "GetActualJointPosDegree", 1)
show("GetActualToolFlangePose", "GetActualToolFlangePose", 1)
show("GetRobotState(程序状态)", "GetRobotState")
show("GetRobotErrorCode(如有)" , "GetRobotErrorCode")
st.stop()
