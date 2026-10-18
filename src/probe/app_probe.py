# -*- coding: utf-8 -*-
"""应用层状态探针（只读！）：拉 WebApp /action/sta basic 全量字段 + get_robot_type。

背景：2026-10-16 凌晨控制器在 :20003 收到毒位姿请求后应用层崩溃；用户断电重启后
法兰 LED 仍「灭/红/绿/蓝」交替。实测 :80（WebApp）活着、:20003/:8083 不监听。
本脚本只发 POST /action/login、/action/sta、/action/get —— 全部是读操作，不写、不动作。
"""
import http.client
import json
import time

IP = "192.168.58.3"


def session():
    c = http.client.HTTPConnection(IP, 80, timeout=4.0)
    c.request("POST", "/action/login", body="username=admin&password=123",
              headers={"Content-Type": "application/x-www-form-urlencoded"})
    r = c.getresponse()
    r.read()
    ck = "; ".join(x.split(";")[0] for x in (r.msg.get_all("Set-Cookie") or []))
    print(f"[登录] HTTP {r.status}  cookie={'有' if ck else '无'}")
    return c, ck


def post(c, ck, path, body):
    c.request("POST", path, body=body,
              headers={"Content-Type": "application/json", "Cookie": ck})
    r = c.getresponse()
    raw = r.read().decode("utf-8", "replace")
    return r.status, raw


c, ck = session()

st, raw = post(c, ck, "/action/sta", '{"cmd":"basic"}')
print(f"[sta] HTTP {st}")
try:
    d = json.loads(raw)
    print(json.dumps(d, ensure_ascii=False, indent=1))
except Exception as e:  # noqa: BLE001
    print(f"解析失败：{e}\n原文前 3000 字：\n{raw[:3000]}")

st, raw = post(c, ck, "/action/get", '{"cmd":"get_robot_type"}')
print(f"\n[get_robot_type] HTTP {st}: {raw[:500]}")

st, raw = post(c, ck, "/action/get", '{"cmd":"get_status_flag"}')
print(f"\n[get_status_flag] HTTP {st}: {raw[:800]}")

c.close()
print(f"\n（探测结束 {time.strftime('%H:%M:%S')}；全程只读）")
