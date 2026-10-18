# -*- coding: utf-8 -*-
"""批次 A · A1 码静态辨识 —— 采集 + CSV 记录（杯子不需要上机器人）

A1 要回答的问题只有一个：**杯子上贴着 6 枚 60° 均布的码（2026-10-15 起；旧版 3 枚 120°），
从镜头方向看过去，任意朝向下是否都至少有一枚能被认出来？** 顺带定死镜像/翻转、记录有效像素尺寸。

协议（两种模式）：
    · 默认 **自动模式**：每 --interval 秒自动拍一帧，你只要缓慢、匀速地转杯子。
      比"12 个角度逐个按回车"样本更多、也更省事；朝向不用精确，覆盖到就行。
    · `--manual`：按一次回车拍一帧（想逐个角度精确录时用）。

判据（规划 §0.5）：有效样本里检出率 ≥ 35/36 量级（自动模式下按"每帧"统计）。

输出：
    data/a1_detect_static.csv   每帧一行（含 cv2 版本、ID、像素边长、倾斜代理值）
    data/a1_shots/              每帧存图（复盘用；--no-save 可关）

跑法：
    python3 -I src/a1_static_test.py                 # 自动，2 秒一帧，拍 40 帧
    python3 -I src/a1_static_test.py --interval 1.5 --n 48 --tag far
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                     # noqa: E402
from detect_cup import Detector, _expected_ids                    # noqa: E402

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv-python")


class Camera:
    """常开取流（config.CAMERA_DEV 只允许一个取流方，别开第二个进程）。"""

    def __init__(self):
        self.cap = cv2.VideoCapture(config.CAMERA_DEV, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit(f"/dev/video{config.CAMERA_DEV} 打不开——是不是 camera_stream.py 还占着？")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_H)

    def read(self):
        """连读两帧取后一帧，避免拿到旧缓冲。"""
        self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            sys.exit("取流中断")
        return frame

    def close(self):
        self.cap.release()


def frame_row(i, tag, dets, want):
    """一帧 → 一行 CSV。倾斜代理 = 最短边/最长边（斜视越狠越小）。"""
    ids = sorted({d["id"] for d in dets})
    hit = sorted(set(ids) & want)
    spans = [d["span_px"] for d in dets]
    ratios = []
    for d in dets:
        s = sorted(d["side_px"])
        ratios.append(round(s[0] / s[-1], 3) if s[-1] > 0 else 0)
    return {
        "frame": i,
        "tag": tag,
        "t": round(time.time(), 3),
        "n_detected": len(dets),
        "ids": "|".join(str(x) for x in ids),
        "n_expected": len(hit),
        "ok": int(bool(hit)),
        "span_max_px": round(max(spans), 1) if spans else "",
        "span_min_px": round(min(spans), 1) if spans else "",
        "min_side_ratio": min(ratios) if ratios else "",
        "extra_ids": "|".join(str(x) for x in sorted(set(ids) - want)),
    }


def main():
    ap = argparse.ArgumentParser(description="A1 静态辨识采集")
    ap.add_argument("--interval", type=float, default=2.0, help="自动模式拍摄间隔（秒）")
    ap.add_argument("--n", type=int, default=40, help="拍多少帧后自动停")
    ap.add_argument("--manual", action="store_true", help="回车拍一帧（交互）")
    ap.add_argument("--watch", action="store_true", help="实时对准模式：持续打印检测状态、不存帧（Ctrl-C 退出）")
    ap.add_argument("--tag", default="", help="本次标签（如 near/mid/far），进 CSV 和文件名")
    ap.add_argument("--no-save", action="store_true", help="不存帧图")
    args = ap.parse_args()

    tag = args.tag or time.strftime("%H%M%S")
    out_csv = config.DATA_DIR / "a1_detect_static.csv"
    shot_dir = config.DATA_DIR / "a1_shots"
    if not args.no_save:
        shot_dir.mkdir(parents=True, exist_ok=True)

    cam, det = Camera(), Detector()
    want = _expected_ids()
    print(f"A1 采集 · 相机 /dev/video{config.CAMERA_DEV} {config.CAMERA_W}×{config.CAMERA_H} · "
          f"检测器 cv2 {det.cv2_version} · {config.ARUCO_DICT} · 期望 ID {sorted(want)}")
    if args.manual:
        print("模式：交互（回车拍一帧，q 结束）")
    else:
        print(f"模式：自动（每 {args.interval}s 一帧，共 {args.n} 帧）——请缓慢、匀速转动杯子")
    print(f"输出：{out_csv}   标签：{tag}")

    if args.watch:
        print("实时对准模式：把杯子举到镜头前，直到出现 ✓（Ctrl-C 退出，不存帧）")
        try:
            while True:
                dets = det.detect(cam.read())
                ids = sorted({d["id"] for d in dets})
                spans = [d["span_px"] for d in dets]
                if spans:
                    msg = (f"✓ 检出 {len(dets)} 枚  id={ids}  命中期望 {sorted(set(ids) & want)}"
                           f"  span≈{max(spans):.0f}px")
                else:
                    msg = "· 未检出（调整杯子位置/距离；杯明明在画面里仍不检出 → 怀疑镜像或失焦）"
                print("\r" + msg.ljust(78), end="", flush=True)
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\n（退出对准模式）")
        finally:
            cam.close()
        return

    rows = []
    is_new = not out_csv.exists()
    try:
        if args.manual:
            i = 0
            while True:
                if input(f"[{i}] 回车=拍  q=结束 > ").strip().lower() == "q":
                    break
                i += 1
                rows.append(_shot(cam, det, i, tag, want, shot_dir, args.no_save))
        else:
            for i in range(1, args.n + 1):
                time.sleep(args.interval)
                r = _shot(cam, det, i, tag, want, shot_dir, args.no_save)
                rows.append(r)
                if not r["ok"]:
                    print(f"  {i:>3}  ✗ 未检出  (span {r['span_max_px'] or '-'})   "
                          f"← 注意：若杯明明在画面里，先怀疑镜像/失焦/距离")
                else:
                    print(f"  {i:>3}  ✓ {r['ids']:<8} span {r['span_max_px']:>6}px  "
                          f"倾斜比 {r['min_side_ratio']}")
    except KeyboardInterrupt:
        print("\n（手动中断）")
    finally:
        cam.close()

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if is_new and rows:
            w.writeheader()
        w.writerows(rows)

    if rows:
        ok = sum(r["ok"] for r in rows)
        print(f"\n本次 {len(rows)} 帧，检出 {ok}，检出率 {ok / len(rows) * 100:.0f}%   （已追加到 {out_csv}）")
    else:
        print("没拍到任何帧。")


def _shot(cam, det, i, tag, want, shot_dir, no_save):
    frame = cam.read()
    dets = det.detect(frame)
    if not no_save:
        cv2.imwrite(str(shot_dir / f"a1_{tag}_{i:03d}.jpg"), frame)
    return frame_row(i, tag, dets, want)


if __name__ == "__main__":
    main()
