# -*- coding: utf-8 -*-
"""查故障原文 / 清除可复位故障（法奥 WebApp 内部接口，纯 HTTP）。

为什么需要单独一个工具：
1. 本机固件（000-000-022）的 xmlrpc 只给一个 error_code 字节，两张手册附表还互相打架
   （8083 附表说 4=奇异位姿、主手册说 4=碰撞，实测都对不上）；写给人看的故障原文在
   WebApp 的 /action/sta 里，只有 HTTP 接口能拿到。
2. 121 个 xmlrpc 方法里【没有】任何清错方法（2026-10-16 实测 system.listMethods）——
   清错只能走 WebApp 的 /action/set（cmd 107 / ResetAllError()），本工具就是它。

用法：
    python3 -I src/clear_error.py            # 只读：打印 error_info / alarm_info 原文
    python3 -I src/clear_error.py --clear    # 输 CLEAR 确认后清除"可复位"故障

安全边界：本工具只【读状态】和【清故障】；不碰使能、不碰模式、不碰任何运动。
清错用"清完再读"复核（不看 set_feedback 的返回值当真，旧固件返回值不可靠）。
"""
import argparse
import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                  # noqa: E402


def _opener():
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def login(op):
    """登录必须 form 编码（用 JSON 一律 401）；302 = 成功。"""
    data = urllib.parse.urlencode({"username": "admin", "password": "123"}).encode()
    req = urllib.request.Request(f"http://{config.ROBOT_IP}/action/login", data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with op.open(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:      # 302 重定向走这里
        return e.code


def sta(op):
    req = urllib.request.Request(f"http://{config.ROBOT_IP}/action/sta",
                                 data=json.dumps({"cmd": "basic"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with op.open(req, timeout=5) as r:
        return json.load(r)


def set_cmd(op, num, content):
    req = urllib.request.Request(f"http://{config.ROBOT_IP}/action/set",
                                 data=json.dumps({"cmd": num, "data": {"content": content}}).encode(),
                                 headers={"Content-Type": "application/json"})
    with op.open(req, timeout=8) as r:
        return r.status, r.read().decode("utf-8", "replace")[:200]


def show(d, tag):
    err = d.get("error_info") or []
    alm = d.get("alarm_info") or []
    print(f"[{tag}] error_info = {err}")
    print(f"[{tag}] alarm_info = {alm}")
    print(f"[{tag}] rbtEnableState={d.get('rbtEnableState')} mode={d.get('mode')} "
          f"program_state={d.get('program_state')} btn_box_stop={d.get('btn_box_stop_signal')}")
    j = d.get("joints") or {}
    if j:
        print(f"[{tag}] joints = " + ", ".join(f"{k}={j[k]:.3f}" for k in sorted(j)))
    return err, alm


def main():
    ap = argparse.ArgumentParser(description="查故障原文 / 清除可复位故障（WebApp 接口）")
    ap.add_argument("--clear", action="store_true", help="清除可复位故障（需输 CLEAR 确认）")
    args = ap.parse_args()

    op = _opener()
    code = login(op)
    print(f"登录 /action/login → {code}{'（成功）' if code in (200, 302) else '⚠ 登录不顺，后面多半失败'}")
    d = sta(op)
    err, alm = show(d, "清前")
    if not err and not alm:
        print("\n无故障，不需要清除。")
        return 0
    if not args.clear:
        print("\n有故障在上面。要清除：加 --clear（或 WebApp 界面上的清除报错按钮）。")
        return 4

    print("\n即将清除报错（只清故障，不碰使能/模式/运动）。")
    if input("  输 CLEAR 回车 = 清除；其他 = 取消：").strip().upper() != "CLEAR":
        print("已取消。")
        return 1
    st, body = set_cmd(op, 107, "ResetAllError()")
    print(f"/action/set cmd=107 → http {st}、响应 {body!r}")

    # 等 set_feedback["107"]（返回值不放响应里，要回读）
    fb = None
    for _ in range(12):
        time.sleep(0.5)
        d2 = sta(op)
        f = d2.get("set_feedback") or {}
        fb = f.get("107", f.get(107))
        if fb is not None:
            break
    print(f"set_feedback[107] = {fb}")

    d3 = sta(op)
    err2, alm2 = show(d3, "清后")
    if not err2 and not alm2:
        print("\n✓ 故障已清除。")
        return 0
    print("\n✗ 故障还在——多半是「不可复位」类：点 WebApp 界面按钮也不会消，"
          "按故障原文处理（重启/断电见 fairino-fr3-cold-boot-recovery 的套路）。")
    return 4


if __name__ == "__main__":
    sys.exit(main())
