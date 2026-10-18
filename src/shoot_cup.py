# -*- coding: utf-8 -*-
"""在观察位拍一帧 → 识别杯上 ArUco → 解算杯轴（真实杯位，覆盖 config 的估计值）。

## 为什么这一步是全链的关键裁决

`config.CUP_AXIS_XY` 现在填的是**用户卷尺报的 r=590mm + 沿用上次视觉方位**的估计值。
它只是"把相机先瞄到附近"的初值。本脚本用**冻结的手眼**（2026-10-16 18:43，
用户确认摄像头此后未被拆装）把杯轴解出来，然后：

  · **独立验算 1**：|xy| 应该接近 590mm（用户卷尺）⇒ 这是对【手眼+FK+PnP 全链】的
    绝对精度裁决，与视觉自洽无关；
  · **独立验算 2**：推出的承载面高应该接近 −85mm（臂座 85mm）⇒ 验 z 链；
  · 两者都过 ⇒ 才可以覆写 config 并进入插入。

只读：拍照与解算都不发运动指令（相机要等画面追上运动后的旧帧，用 `Cam.fresh()`）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                              # noqa: E402
import cup_pose                                             # noqa: E402
import fr3_math as M                                       # noqa: E402
from fr_arm import Arm                                     # noqa: E402
from fr_cam import Cam                                      # noqa: E402
from fr_fkcache import FKCache                             # noqa: E402
from detect_cup import Detector                            # noqa: E402

OUT_DIR = config.DATA_DIR / "cup_shots"
N_FRAMES = 8          # 连拍帧数：这台相机的"追上"判据不可靠，用检出结果决定


def main():
    arm = Arm()
    arm.connect()
    fc = FKCache(arm.r.fk)
    M.attach_cache(fc)
    K, D, X = cup_pose.load_calib()
    q = np.asarray(arm.joint_deg(), float)
    T = fc.T(q)
    print(f"关节 {np.round(q,2).tolist()}")
    print(f"法兰 {np.round(T[:3,3],1).tolist()}")

    cam = Cam()
    import cv2
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = OUT_DIR / ts
    d.mkdir(parents=True, exist_ok=True)

    # ⚠ 连拍多帧逐帧检测，用能解算的那一帧。
    #   原因（2026-10-18 实踩）：这台相机伺服运动后会先吐约 10s 旧画面，
    #   `Cam.fresh()` 的"没见跳变"兜底并不可靠 —— 同一位姿两次运行分别检出 1 个和 0 个码。
    #   ⇒ 不靠判据猜，直接多拍几帧、用哪一帧由【检出结果】决定。
    det = Detector()
    best, best_n, per = None, 0, []
    for i in range(int(N_FRAMES)):
        fr = cam.raw()
        cv2.imwrite(str(d / f"raw{i:02d}.jpg"), fr)
        dd = det.detect(fr)
        n = len([x for x in dd if x.get("id") in (1, 2)])
        per.append(n)
        print(f"   第 {i+1} 帧：检出杯码 {n} 个", flush=True)
        if n > best_n:
            best_n, best = n, fr
        time.sleep(1.0)
    print(f"   各帧杯码数：{per}  → 取最好的一帧（{best_n} 个）")
    if best is None:
        print("✗ 所有帧都没检出杯码")
        arm.close(); cam.close(); return
    f = best
    cv2.imwrite(str(d / "shot.jpg"), f)
    dets = det.detect(f)
    cup = [x for x in dets if x.get("id") in (1, 2)]
    print(f"\n检出 ArUco {len(dets)} 个（杯码 {len(cup)} 个）：")
    for x in cup:
        c = np.asarray(x["corners"], float)
        print(f"   id={x['id']} 中心={np.round(c.mean(0),0).tolist()} "
              f"跨度={(c.max(0)-c.min(0)).max():.0f}px")
    if not cup:
        print("✗ 没检出杯码 —— 不猜、不覆写 config。")
        arm.close()
        cam.close()
        return

    shots = [{"tag": ts, "T": T, "dets": dets}]   # T 必须是 4×4（solve_pnp 里 Tb = sh["T"] @ X）
    res = cup_pose.solve_pnp(shots, K, D, X, say=lambda *a, **k: None)
    if res is None:
        print("✗ solve_pnp 无解（被质量闸拒）")
        arm.close()
        cam.close()
        return
    ax = res.get("axis_xy") if isinstance(res, dict) else None
    print(f"\nsolve_pnp 全部字段：")
    for k, v in (res.items() if isinstance(res, dict) else []):
        if isinstance(v, (int, float, str, tuple, list, bool)) or v is None:
            print(f"   {k:18s} {v}")
    if ax is not None:
        r = float(np.hypot(ax[0], ax[1]))
        print(f"\n=== 独立验算 ===")
        print(f"  |xy| = {r:.1f}mm   用户卷尺 590mm   差 {r-590:+.1f}mm")
        tz = res.get("implied_table_z")
        if tz is not None:
            print(f"  推出承载面 z = {tz:.1f}mm   卡尺 −85mm   差 {tz-config.TABLE_Z:+.1f}mm")
        (d / "result.json").write_text(json.dumps(
            {"tag": ts, "axis_xy": list(map(float, ax)),
             "res": {k: (v if isinstance(v, (int, float, str, list)) else str(v))
                     for k, v in (res.items() if isinstance(res, dict) else [])}},
            ensure_ascii=False), encoding="utf-8")
        print(f"\n→ {d}")
    arm.close()
    cam.close()


if __name__ == "__main__":
    main()
