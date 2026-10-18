# -*- coding: utf-8 -*-
"""控制器 FK 的**持久化缓存** + 带墙钟预算的查询。

## 为什么必须有这个（2026-10-18 立的纪律）

`GetForwardKin` 是网络调用，且 `fr3_math.FK_MIN_THROTTLE_S = 0.05`（≈20 次/秒）是
第 5 次控制器崩溃后立的**硬地板**，不可调低。于是：

    一次数值优化 = 数百~数千次 FK  ⇒  分钟级
    一个 9×6 起点的扫描          =  数小时级

**任何把控制器当计算内核的设计都是错的**——2026-10-18 连续三次把命令跑到超时，
其中两次是我自己的 shell 超时，一次是用户手动中断。

本模块给出正解：

1. **缓存**：查过的 (关节 → 位姿) 永久落盘 `data/fk_cache.json`。
   重复查询 0 成本 ⇒ **实验成本可以跨会话累积**，同一个问题问第二遍是免费的。
   这是从"每次实验都重新烧控制器"变成"越做越便宜"的关键。
2. **墙钟预算**：`Budget` 到点即抛。宁可带着部分结果失败，也不许无声地烧一小时。
3. **命中率上报**：缓存没帮上忙时要看得见（`stats()`），否则会误以为已经很快了。

只读：本模块只调 `GetForwardKin`，不发任何运动指令。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import fr3_math

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "fk_cache.json"
LOCK_PATH = ROOT / "data" / "fk_cache.lock"


class Budget:
    """墙钟预算。到点抛 `Timeout`，让调用方带着部分结果优雅退出。"""

    def __init__(self, seconds, what=""):
        self.t0 = time.monotonic()
        self.seconds = float(seconds)
        self.what = what

    def elapsed(self):
        return time.monotonic() - self.t0

    def left(self):
        return self.seconds - self.elapsed()

    def check(self, where=""):
        if self.left() <= 0:
            raise Timeout(f"预算耗尽（{self.seconds:.0f}s，已用 {self.elapsed():.0f}s）"
                         f" @ {where or self.what}")
        return True


class Timeout(RuntimeError):
    pass


def _key(q):
    return ",".join(f"{float(v):.4f}" for v in q)


class FKCache:
    """带持久化缓存的控制器 FK。

    用法：
        fc = FKCache(rpc.fk)
        p = fc.pose([13.5, -129, 82.8, -71.3, -21.6, 46.9])   # → [x,y,z,rx,ry,rz]
    """

    def __init__(self, fk_fn, path=CACHE_PATH, autosave_every=25, verbose=False):
        self._fk_raw = fk_fn
        self._fk = fr3_math.fk_throttled(fk_fn)          # 硬节流，绕不过
        self.path = Path(path)
        self.verbose = verbose
        self.autosave_every = int(autosave_every)
        self._dirty = 0
        self.data = {}
        self.hits = 0
        self.misses = 0
        self.load()

    # ---------------- 持久化 ----------------
    def load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:                                   # noqa: BLE001
                self.data = {}
        return len(self.data)

    def save(self, force=False):
        """原子落盘。临时文件名带 pid —— 两个进程同时跑只读规划时，
        共用同一个 `<name>.tmp` 会互相抢：先 replace 的把文件拿走，
        后一个就 `FileNotFoundError`（2026-10-18 实踩，导致整轮扫描白跑）。
        """
        if not self._dirty and not force:
            return
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        tmp.replace(self.path)                                 # 原子替换，防半截文件
        self._dirty = 0

    # ---------------- 查询 ----------------
    def pose(self, q, budget=None):
        """法兰位姿 [x,y,z,rx,ry,rz]（与控制器同约定）。缓存命中则不发网络调用。"""
        k = _key(q)
        hit = self.data.get(k)
        if hit is not None:
            self.hits += 1
            return list(hit)
        if budget is not None:
            budget.check("FKCache.pose")
        v = list(self._fk(q))[:6]
        self.data[k] = v
        self.misses += 1
        self._dirty += 1
        if self._dirty >= self.autosave_every:
            self.save()
        if self.verbose and self.misses % 50 == 0:
            print(f"    [fkcache] 新查询 {self.misses}，命中 {self.hits}", flush=True)
        return v

    def T(self, q, budget=None):
        from fr3_math import pose6_to_T
        return pose6_to_T(self.pose(q, budget))

    def stats(self):
        tot = self.hits + self.misses
        return dict(entries=len(self.data), hits=self.hits, misses=self.misses,
                    hit_rate=(self.hits / tot) if tot else 0.0)
