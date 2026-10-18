# -*- coding: utf-8 -*-
"""视觉 · 单帧 ArUco 检测（批次 A / A1：只“看见”，不“换算”）

边界（重要）：输入图像 → 输出“哪些码、在哪（像素）、多大（像素）”。
像素→基座坐标的换算属于批次 D 标定之后；**不做 solvePnP 位姿解算**——
码贴在锥面上，四角连成的弦长 ≠ 标称边长，尺度会系统性偏 7~9%（规划 §3.3），
位姿在 P3/P4 用“射线 ∩ 桌面平面 + 码中心”解决。

兼容两代 OpenCV ArUco API：
    ≥ 4.7  ArucoDetector（新版）
    ≤ 4.6  detectMarkers （老版；本机 `python3 -I` 下是 4.6.0）
检测结果里**记录 cv2 版本**——换检测器 = 换仪器，数据里必须留痕。

用法：
    python3 -I src/detect_cup.py --camera                      # 抓一帧检测（config.CAMERA_DEV）
    python3 -I src/detect_cup.py 图片.png                      # 检测图片文件
    python3 -I src/detect_cup.py --camera --save data/a1.jpg --json data/a1.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I 下不会自动带脚本目录

import config                                                    # noqa: E402

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("需要 opencv-python（含 aruco 模块）与 numpy")


def _dictionary():
    code = getattr(cv2.aruco, config.ARUCO_DICT, None)
    if code is None:
        sys.exit(f"当前 cv2({cv2.__version__}) 里没有 {config.ARUCO_DICT}")
    for name in ("getPredefinedDictionary", "Dictionary_get"):
        f = getattr(cv2.aruco, name, None)
        if f is None:
            continue
        try:
            return f(code)
        except Exception:
            continue
    sys.exit("找不到构造 ArUco 字典的方法")


class Detector:
    """进程内复用一个实例（构造 DetectorParameters 不便宜）。"""

    def __init__(self):
        self.dic = _dictionary()
        self.cv2_version = cv2.__version__
        self._new = None
        if hasattr(cv2.aruco, "ArucoDetector"):                  # OpenCV ≥ 4.7
            self._new = cv2.aruco.ArucoDetector(self.dic, cv2.aruco.DetectorParameters())
        else:                                                    # OpenCV ≤ 4.6
            self._params = cv2.aruco.DetectorParameters_create()

    def detect(self, img):
        """→ [{id, center:(u,v), corners:4×2, side_px:四边长, span_px:边长均值}]

        角点顺序 = 码自身的 左上、右上、右下、左下（OpenCV 约定）；
        center 是四角均值（亚像素近似，够 A1 用；mm 级精度靠批次 D 的射线∩平面）。
        """
        if self._new is not None:
            corners, ids, _ = self._new.detectMarkers(img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(img, self.dic, parameters=self._params)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            c = np.asarray(c, dtype=float).reshape(4, 2)
            sides = [float(np.linalg.norm(c[(k + 1) % 4] - c[k])) for k in range(4)]
            out.append({
                "id": int(i),
                "corners": [[round(u, 2), round(v, 2)] for u, v in c],
                "center": [round(float(c[:, 0].mean()), 2), round(float(c[:, 1].mean()), 2)],
                "side_px": [round(s, 2) for s in sides],
                "span_px": round(sum(sides) / 4.0, 2),
            })
        return out


def grab(dev=None, warm=8):
    """从 config.CAMERA_DEV 抓一帧（MJPG + config 里的分辨率）。"""
    dev = config.CAMERA_DEV if dev is None else dev
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"/dev/video{dev} 打不开（别的进程占着？见 HANDOFF 摄像头一节）")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_H)
    frame = None
    for _ in range(warm):                    # 前几帧曝光/白平衡不稳，丢掉
        ok, frame = cap.read()
    cap.release()
    if frame is None or not ok:
        sys.exit("打开了相机但读不到帧")
    return frame


def _expected_ids():
    return {config.ARUCO_ID_CUP_A, config.ARUCO_ID_CUP_B}


def main():
    ap = argparse.ArgumentParser(description="单帧 ArUco 检测（不做坐标换算）")
    ap.add_argument("image", nargs="?", help="图片文件；不给就配 --camera")
    ap.add_argument("--camera", action="store_true", help=f"从 /dev/video{config.CAMERA_DEV} 抓一帧")
    ap.add_argument("--save", metavar="路径", help="把这一帧存成图片（连同检测框另存 *_marked.jpg）")
    ap.add_argument("--json", metavar="路径", help="把检测原始结果写 JSON（含角点）")
    args = ap.parse_args()

    if args.camera and args.image:
        sys.exit("--camera 和 图片 二选一")
    if not args.camera and not args.image:
        sys.exit("用法：detect_cup.py --camera  或  detect_cup.py 图片.png")

    t0 = time.time()
    if args.camera:
        img = grab()
        src = f"camera:/dev/video{config.CAMERA_DEV}"
    else:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"读不到图片：{args.image}")
        src = args.image

    det = Detector()
    found = det.detect(img)
    h, w = img.shape[:2]
    print(f"来源 {src}   画面 {w}×{h}   用时 {(time.time()-t0)*1000:.0f} ms")
    print(f"检测器 cv2 {det.cv2_version} · {config.ARUCO_DICT} · 期望 ID {sorted(_expected_ids())}")

    if not found:
        print("\n未检出任何码。")
    else:
        print(f"\n{'ID':>4} {'中心 u,v(px)':>20} {'边长均值':>9} {'四边长(px)':>28}")
        for d in sorted(found, key=lambda x: x["id"]):
            u, v = d["center"]
            print(f"{d['id']:>4} {f'({u:.1f}, {v:.1f})':>20} {d['span_px']:>9.1f}"
                  f"   {[f'{s:.1f}' for s in d['side_px']]}")

    ids = {d["id"] for d in found}
    want = _expected_ids()
    hit = ids & want
    print(f"\n期望 ID 命中 {len(hit)}/{len(want)}{'：' + str(sorted(hit)) if hit else ''}"
          + (f"；另有非期望 ID {sorted(ids - want)}" if ids - want else ""))

    if args.save:
        cv2.imwrite(args.save, img)
        vis = img.copy()
        for d in found:
            cv2.polylines(vis, [np.int32(d["corners"])], True, (0, 0, 255), 2)
            cv2.putText(vis, str(d["id"]), tuple(np.int32(d["center"])), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)
        marked = os.path.splitext(args.save)[0] + "_marked.jpg"
        cv2.imwrite(marked, vis)
        print(f"已存 {args.save}（框注版 {marked}）")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"source": src, "image_hw": [h, w], "cv2": det.cv2_version,
                       "dict": config.ARUCO_DICT, "time": time.time(), "detections": found},
                      f, ensure_ascii=False, indent=1)
        print(f"已存 {args.json}")


if __name__ == "__main__":
    main()
