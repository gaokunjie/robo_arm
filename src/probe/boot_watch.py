# -*- coding: utf-8 -*-
"""开机过程直播监看：ping + 端口(80/20003/8083) 每 2s 一采；状态一变就打一行（带绝对时间）。

用途：用户断电重启的全程取证 ——
  ① 机器真断电了没有（ping / :80 有没有断档）；
  ② 应用层起来了没有（:20003/:8083 开没开）；
  ③ 起来后又崩没崩（开了又关 = 崩溃环）。
纯只读：ICMP + TCP connect，连上立刻关。
输出直接打 stdout（外面用重定向存文件）。
"""
import socket
import subprocess
import sys
import time

IP = "192.168.58.3"
PORTS = [80, 20003, 8083]       # 网页层 / 指令口 / 状态口
APP = (1, 2)                    # PORTS 中属于「应用层」的下标
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 1200.0
FAST, SLOW = 2.0, 5.0


def ping_ok():
    r = subprocess.run(["ping", "-n", "-c", "1", "-W", "1", IP],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def port_ok(p):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect((IP, p))
        return True
    except Exception:
        return False
    finally:
        s.close()


def snap():
    return ping_ok(), tuple(port_ok(p) for p in PORTS)


def app_up(ps):
    return ps[APP[0]] and ps[APP[1]]


def app_any(ps):
    return ps[APP[0]] or ps[APP[1]]


def line(ok, ps, note=""):
    print(f"{time.strftime('%H:%M:%S')}  t+{time.time() - t0:6.1f}s  "
          f"ping={'✓' if ok else '×'}  "
          + "  ".join(f"{p}:{'开' if v else '关'}" for p, v in zip(PORTS, ps))
          + note, flush=True)


t0 = time.time()
last = None
seen_up = 0
last_print = 0.0
print(f"== 开机直播 {time.strftime('%H:%M:%S')}（盯 {int(DUR)}s）==", flush=True)
try:
    while time.time() - t0 < DUR:
        s = snap()
        ok, ps = s
        seen_up = seen_up + 1 if app_up(ps) else 0
        if s != last:
            note = ""
            if last is not None:
                po, pv = last
                if po and not ok:
                    note += "  ← 机器掉线了（断电/重启中）"
                elif not po and ok:
                    note += "  ← 机器回来了（重启后网络先起）"
                if not app_any(pv) and app_up(ps):
                    note += "  ★★ 应用层起来了！"
                elif app_any(pv) and not app_any(ps):
                    note += "  ← 应用层又没了（崩回去了）"
            line(ok, ps, note)
            last = s
            last_print = time.time()
        elif time.time() - last_print >= 30.0:
            line(ok, ps, "  ·")
            last_print = time.time()
        time.sleep(SLOW if seen_up >= 30 else FAST)
except KeyboardInterrupt:
    pass
print(f"== 直播结束 {time.strftime('%H:%M:%S')} ==", flush=True)
