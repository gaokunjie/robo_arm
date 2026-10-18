# -*- coding: utf-8 -*-
"""所有只读规划的统一入口：全局注入 FK 缓存 + 进度可见。

## 为什么必须有这个（2026-10-18）

控制器 `GetForwardKin` 是 ~20 次/秒的**硬地板**（第 5 次控制器崩溃后立的，
`FK_MIN_THROTTLE_S = 0.05`，不可调低）。一次 40 段规划要几千次 FK ⇒ 几分钟。
三个后果，全部踩过：

  ① 前台跑必然被 shell 超时打断；
  ② **中断时缓存白攒** —— 那才是真正的损失（每次都从头烧控制器）；
  ③ 节流的 sleep 在 `fk_throttled` 外面，挂在 `Arm.fk` 上的缓存**完全无效**。

本文件把三件事一次做对：
  · `fr3_math.attach_cache(FKCache(...))` —— 缓存进到节流内部，命中 0ms；
  · 缓存**落盘**，跨进程、跨会话累积（`data/fk_cache.json`）；
  · 每 5 秒打印一行"已用时 / 新查 / 命中"，**绝不让屏幕长时间零输出**。

用法（前台短任务）：
    python3 -u -I src/run_cached.py --lookdown-plan --elev 40
长任务请配合 src/bg_plan.sh 放后台，**不要**在前台等。

⚠ 只读：本入口不改变任何调用方的语义；see_cup/grasp 的运动指令仍由它们自己的闸门把关。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fr3_math                                             # noqa: E402
from fr_fkcache import FKCache                               # noqa: E402
from fr_rpc import FRRpc                                     # noqa: E402
import config                                                # noqa: E402


class Progress(threading.Thread):
    """后台心跳：每 5 秒打印一行，保证长时间任务也"看得见在动"。"""

    def __init__(self, fc, period=5.0):
        super().__init__(daemon=True)
        self.fc, self.period, self.stop_ = fc, period, False

    def run(self):
        t0 = time.monotonic()
        while not self.stop_:
            time.sleep(self.period)
            s = self.fc.stats()
            rate = (s["misses"] - getattr(self, "_last", 0)) / self.period
            self._last = s["misses"]
            print(f"    …{time.monotonic() - t0:6.0f}s  缓存条目 {s['entries']}  "
                  f"新查 {s['misses']}（{rate:.0f}/s，理论上限 20/s）  "
                  f"命中 {s['hits']}（{s['hit_rate'] * 100:.0f}%）", flush=True)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ("--help", "-h"):
        print(__doc__)
        return
    target = "see_cup.py"
    if argv and argv[0] == "--script":
        target = argv[1]
        argv = argv[2:]
    rpc = FRRpc(ip=config.ROBOT_IP)
    fc = FKCache(rpc.fk)
    fr3_math.attach_cache(fc)
    print(f"[缓存] 载入 {len(fc.data)} 条 → data/fk_cache.json；"
          f"命中直接返回，未命中才付 {fr3_math.FK_MIN_THROTTLE_S}s 节流", flush=True)

    prog = Progress(fc)
    prog.start()
    sys.argv = [target] + argv
    try:
        mod = __import__(Path(target).stem)
        mod.main()
    except SystemExit as exc:
        print(f"[run_cached] 目标脚本 exit({exc.code})")
    finally:
        prog.stop_ = True
        fc.save(force=True)
        s = fc.stats()
        print(f"[缓存] 收尾：条目 {s['entries']}  新查 {s['misses']}  命中 {s['hits']}"
              f"（{s['hit_rate'] * 100:.0f}%）", flush=True)


if __name__ == "__main__":
    main()
