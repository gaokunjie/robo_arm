# -*- coding: utf-8 -*-
"""只读：检查"当前关节角当成位姿"这个错位解释是否可达。不可达 => 错位也是安全的。"""
import http.client, xmlrpc.client
class T(xmlrpc.client.Transport):
    def __init__(self, timeout=5.0):
        super().__init__(); self.timeout = timeout
    def make_connection(self, host):
        return http.client.HTTPConnection(host, timeout=self.timeout)
p = xmlrpc.client.ServerProxy("http://192.168.58.3:20003", transport=T())

pose  = [222.97, -492.73, 452.66, -84.82, 12.15, 133.36]
joint = [-43.793, -168.178, 38.230, -102.278, -6.410, 64.251]

print("== GetInverseKinHasSolution(type, {位姿6}, {参考关节6}) ==")
for label, pz in (("当前位姿本身（对照，应 True）", pose),
                  ("当前关节当成位姿（错位解释）", joint)):
    try:
        r = p.GetInverseKinHasSolution(0, pz, joint)
        print(f"  {label}: {r!r}")
    except Exception as e:
        print(f"  {label}: [{type(e).__name__}] {e}")

print("\n== 错位位姿的 TCP 离基座多远 ==")
x, y, z = joint[0], joint[1], joint[2]
print(f"  x={x} y={y} z={z}  →  距基座轴 { (x*x+y*y)**0.5 :.1f} mm，离基座平面 {z:.1f} mm")
for i, (lo, hi) in enumerate([(-175,175),(-265,85),(-160,160),(-265,85),(-175,175),(-175,175)]):
    v = pose[i]
    print(f"  若把当前位姿当关节 J{i+1}={v:>8.2f}  限位[{lo},{hi}]  {'✗ 超限' if not (lo<=v<=hi) else '✓ 在限内'}")
