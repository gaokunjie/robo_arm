# -*- coding: utf-8 -*-
"""只读探针：问控制器自己 MoveL 的签名。不发任何运动指令。"""
import sys, http.client, xmlrpc.client

IP = "192.168.58.3"

class T(xmlrpc.client.Transport):
    def __init__(self, timeout=4.0):
        super().__init__(); self.timeout = timeout
    def make_connection(self, host):
        return http.client.HTTPConnection(host, timeout=self.timeout)

p = xmlrpc.client.ServerProxy(f"http://{IP}:20003", transport=T())

def show(fn, *a):
    try:
        print(f"  {fn}{a if a else ''} -> {getattr(p, fn)(*a)!r}")
    except Exception as e:
        print(f"  {fn}{a if a else ''} -> [{type(e).__name__}] {e}")

print("== 内省 ==")
show("system.methodHelp", "MoveL")
show("system.methodSignature", "MoveL")
show("system.methodHelp", "MoveJ")
show("system.methodSignature", "MoveJ")
print("\n== 方法名单里带 Move 的 ==")
try:
    ms = sorted(m for m in p.system.listMethods() if "Move" in m or "move" in m)
    print("  " + "\n  ".join(ms))
except Exception as e:
    print("  ", e)
