# -*- coding: utf-8 -*-
"""只读前置核查：关节限位 + 两个"万一被误解释"的位姿可不可达。不发运动指令。"""
import http.client, xmlrpc.client
class T(xmlrpc.client.Transport):
    def __init__(self, timeout=5.0):
        super().__init__(); self.timeout = timeout
    def make_connection(self, host):
        return http.client.HTTPConnection(host, timeout=self.timeout)
p = xmlrpc.client.ServerProxy("http://192.168.58.3:20003", transport=T())

def show(label, fn, *a):
    try:
        print(f"  {label}: {getattr(p, fn)(*a)!r}")
    except Exception as e:
        print(f"  {label}: [{type(e).__name__}] {e}")

print("== 关节软限位 [min,max] ==")
v = list(p.GetJointSoftLimitDeg(1))
for i in range(6):
    print(f"  J{i+1}: {v[2*i]:>8.2f} ~ {v[2*i+1]:>8.2f}")

pose  = [222.97, -492.73, 452.66, -84.82, 12.15, 133.36]   # 当前位姿
joint = [-43.793, -168.178, 38.230, -102.278, -6.410, 64.251]  # 当前关节

print("\n== 可解性自检（GetInverseKinHasSolution(0,{x,y,z,rx,ry,rz,config})）==")
show("当前位姿本身（应该=1）", "GetInverseKinHasSolution", 0, pose + [-1])
show("把当前【关节】当成位姿（危险解）", "GetInverseKinHasSolution", 0, joint + [-1])

print("\n== 逆解对照 ==")
show("当前位姿的逆解", "GetInverseKin", 0, *pose, -1)
print(f"  实际关节角     : {joint}")
