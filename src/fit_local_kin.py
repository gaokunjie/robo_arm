# -*- coding: utf-8 -*-
"""拟合 + 验证 FR3 本地运动学模型（纯离线，不连机器人）。

输入 `data/fk_samples.json`（`collect_fk_samples.py` 只读采的），输出 `data/fr3_local_kin.json`。

## 为什么需要留出验证

上一版（`fit_urdf_kin.py`，18 点 J2 扫描弧）报 RMS 2.9mm 看着很好，但两个**独立位形**
的残差是 20.5mm / 4.65mm —— 样本全在一条弧上，参数被那条弧的方向定死了，
垂直于它的方向上没有任何约束。**拟合 RMS 只衡量"在样本上"，不衡量"在新位形上"。**
⇒ 这里的判据是**留出集**误差，不是训练 RMS。

## 判据（与实验步骤书 G2 一致：模型 FK vs 真机 FK < 1mm）

分两组分别报：
  · 全体留出 RMS / max
  · **朝下位形子集**的留出误差（套取/抬升只在这一组里用，权重最高）
朝下子集为空时脚本会明说"工作区位形无验证"，不许当合格用。

用法：
    python3 -I src/fit_local_kin.py                  # 5 折交叉验证 + 留出
    python3 -I src/fit_local_kin.py --signs-scan    # 顺便扫零位符号组合
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fr3_local as L                                       # noqa: E402

DATA = L.ROOT / "data" / "fk_samples.json"


def load(path=DATA):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return (np.asarray(d["joints_deg"], float),
            np.asarray(d["flange_xyz_mm"], float),
            np.asarray(d.get("tool_down", [False] * d["n"]), bool),
            d)


def kfold_idx(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(idx, f), idx[f]) for f in np.array_split(idx, k)]


def screen_signs(Q, P, top=8):
    """**零参数**粗筛 64 种符号组合：β=0、t=0、off=0，只看裸 URDF 的残差。

    为什么先做这一步：符号判错的后果是**几百 mm** 的整体偏差（实测 400mm+），
    而零位偏置带来的修正只有十几 mm。两者差两个数量级 ⇒ 符号完全可以先用零参数
    目标函数分辨出来，不必为 64 组都跑一遍完整拟合（那要几十分钟）。

    返回按 RMS 升序的 [(rms, signs), ...]。
    """
    e = np.linalg.norm(P, axis=1).mean()
    out = []
    for s in product([1, -1], repeat=6):
        signs = np.array(s, float)
        r = np.linalg.norm(L.Model(signs=signs).residuals(Q, P), axis=1)
        out.append((float(np.sqrt((r ** 2).mean())), signs))
    out.sort(key=lambda t: t[0])
    print(f"  （对照：样本自身离基座原点平均 {e:.0f}mm；符号全错的残差量级 ≈ 400mm）")
    for r, s in out[:top]:
        print(f"    signs {tuple(int(v) for v in s)}  裸URDF RMS {r:8.3f}mm", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="拟合 FR3 本地运动学模型")
    ap.add_argument("--signs-scan", action="store_true", help="扫 64 种零位符号组合")
    ap.add_argument("--top", type=int, default=8, help="粗筛后进入完整拟合的组合数")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    Q, P, DOWN, meta = load()
    n = len(Q)
    print(f"样本 {n} 个（采集 {meta.get('date', '?')}，节流 {meta.get('throttle_s')}s，"
          f"只读无运动）")
    print(f"  朝下位形 {int(DOWN.sum())} 个", flush=True)

    if args.signs_scan:
        print("\n[1/2] 零参数粗筛 64 种符号组合…", flush=True)
        print("  背景：老脚本在退化数据上选了 (1,1,1,1,1,1)，但真机 j2 为负时手臂抬起、")
        print("       URDF 里是水平的 —— **J2 极可能反号**。符号判错会让残差整体差几百 mm。")
        ranked = screen_signs(Q, P, top=args.top)

        print(f"\n[2/2] 对前 {args.top} 名做 {args.k} 折交叉验证…", flush=True)
        cv = kfold_idx(n, args.k, seed=1)
        rows = []
        for rank, (raw, signs) in enumerate(ranked[:args.top], 1):
            try:
                errs, trs = [], []
                for tr, te in cv:
                    m, r_tr = L.fit(Q[tr], P[tr], signs=signs)
                    trs.append(r_tr)
                    errs.append(np.linalg.norm(m.residuals(Q[te], P[te]), axis=1))
            except Exception as exc:                          # noqa: BLE001
                print(f"    {rank:2d}. signs {tuple(int(v) for v in signs)}  失败：{exc}",
                      flush=True)
                continue
            e = np.concatenate(errs)
            rms = float(np.sqrt((e ** 2).mean()))
            rows.append((rms, float(e.max()), float(np.mean(trs)), signs))
            print(f"    {rank:2d}. signs {tuple(int(v) for v in signs)}  "
                  f"留出RMS {rms:8.3f}mm  max {e.max():8.3f}mm  训练 {np.mean(trs):8.3f}mm",
                  flush=True)
        rows.sort(key=lambda r: r[0])
        if not rows:
            raise SystemExit("符号扫描全部失败")
        rms, mx, tr, signs = rows[0]
        print(f"\n  最佳 signs {tuple(int(v) for v in signs)}：留出 RMS {rms:.3f}mm")
        if rms > 1.0:
            print(f"  ⚠ 留出 RMS {rms:.3f}mm 未达 <1mm 判据（G2）—— 模型还不足以"
                  "对\"够得着/够不着\"下结论。见 §输出 里的下一步。")
    else:
        signs = (1,) * 6

    print(f"\n用 signs = {tuple(int(v) for v in signs)} 做 {args.k} 折交叉验证…")
    folds = kfold_idx(n, args.k)
    tr_all, ho_all, ho_down = [], [], []
    for k, (tr, te) in enumerate(folds, 1):
        m, r_tr = L.fit(Q[tr], P[tr], signs=signs)
        ho = m.residuals(Q[te], P[te])
        e = np.linalg.norm(ho, axis=1)
        tr_all.append(r_tr)
        ho_all.append(e)
        sel = DOWN[te]
        if sel.any():
            ho_down.append(e[sel])
        print(f"  折 {k}: 训练 RMS {r_tr:6.3f}  留出 RMS {np.sqrt((e**2).mean()):6.3f}"
              f"  留出 max {e.max():6.3f}mm  (留出 {len(te)}，其中朝下 {int(sel.sum())})")

    ho_all = np.concatenate(ho_all)
    print(f"\n=== 留出总账（{args.k} 折，每点都当过一次没见过的）===")
    print(f"  训练 RMS 均值 {np.mean(tr_all):.3f}mm")
    print(f"  留出 RMS {np.sqrt((ho_all**2).mean()):.3f}mm   留出 max {ho_all.max():.3f}mm"
          f"   p95 {np.percentile(ho_all, 95):.3f}mm")
    if ho_down:
        hd = np.concatenate(ho_down)
        print(f"  朝下位形子集（{len(hd)} 个）: RMS {np.sqrt((hd**2).mean()):.3f}mm  "
              f"max {hd.max():.3f}mm  ← 套取/抬升只在这一组里用")
    else:
        print("  ⚠ 朝下位形子集为空 —— **工作区位形没有任何验证**，本模型不得用于下"
              "\"够得着/够不着\"结论。先跑 collect_fk_samples.py 补朝下样本。")

    print("\n在全体样本上重拟合并落盘…")
    model, rms = L.fit(Q, P, signs=signs)
    model.save()
    print(f"  β={model.beta:.3f}°  t={np.round(model.t,2).tolist()} mm")
    print(f"  零位偏置 o1..o6 = {np.round(model.off,3).tolist()}°")
    print(f"  全体样本残差 RMS {rms:.3f}mm")
    print(f"→ {L.PARAMS_PATH}")

    print("\n=== 闭式可达性（只依赖 URDF 链长，与拟合无关，可独立复核）===")
    reach, zs = L.planar_reach_limit()
    print(f"  臂展 {reach:.0f}mm，肩枢 z={zs:.0f}mm")
    print("  '法兰正对 (d,z) 且工具竖直向下' 可达 ⟺ d² + (z−140)² ≤ 622²")


if __name__ == "__main__":
    main()
