# -*- coding: utf-8 -*-
"""扫滚转 ρ，找一个"从当前姿态每个关节跳变都小"的观察位（只读）。

为什么必须扫 ρ：`see_cup.py` 的注释记着 2026-10-17 13:19 的实撞 ——
"瞄点只挪 30mm，ρ 固定时求解器只能落到 J4 −84° 那族、被 cap 闸全拦"。
腕是大 freedom：同一目标位姿，ρ 不同 ⇒ J4/J6 怎么分担 166° 的滚转不同 ⇒
分支不同 ⇒ 与当前姿态的关节距离可以差 200° 以上。

判据：max|Δq_i| ≤ 50°（闸门是 60°，留余量），且全部关节在软限位内。
"""
import sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config, cup_pose, fr3_math
from fr_fkcache import FKCache
from fr_rpc import FRRpc

rpc = FRRpc(ip=config.ROBOT_IP)
fc = FKCache(rpc.fk); fr3_math.attach_cache(fc)
K, D, X = cup_pose.load_calib(); Xinv = np.linalg.inv(X)
lim = np.asarray(rpc.joint_soft_limits(), float).reshape(6, 2)
BB = [(lim[i,0], lim[i,1]) for i in range(6)]
q_now = np.asarray(rpc.joint_deg(), float)
print(f"当前关节 {np.round(q_now,2).tolist()}", flush=True)

aim = np.array([config.CUP_AXIS_XY[0], config.CUP_AXIS_XY[1], config.marker_plane_z()])
toward = -aim[:2]/np.linalg.norm(aim[:2])
import math

def solve(T_fl):
    def cost(q):
        T = fc.T(q)
        ce = fr3_math.rotvec_deg(T[:3,:3] @ T_fl[:3,:3].T)
        pe = T[:3,3] - T_fl[:3,3]
        viol = np.clip(lim[:,0]-q,0,None) + np.clip(q-lim[:,1],0,None)
        return float(pe@pe + ce@ce + 1e4*float(viol@viol))
    best=None
    for seed in (q_now, np.array([q_now[0],-60.,40.,-100.,-60.,q_now[5]])):
        r=minimize(cost, seed, method="L-BFGS-B", bounds=BB,
                   options=dict(maxiter=220, eps=0.6, ftol=1e-16, gtol=1e-16))
        T=fc.T(r.x)
        ep=float(np.linalg.norm(T[:3,3]-T_fl[:3,3]))
        er=float(np.linalg.norm(fr3_math.rotvec_deg(T[:3,:3]@T_fl[:3,:3].T)))
        ok=np.all(r.x>=lim[:,0]-1e-6) and np.all(r.x<=lim[:,1]+1e-6) and ep<5 and er<2
        if ok and (best is None or ep<best[1]): best=(r.x.copy(),ep,er)
        if best is not None: break
    return best

print(f"\n{'仰角':>4s}{'距离':>6s}{'ρ':>5s}{'残差mm/°':>13s}{'max|dq|':>9s}{'各关节Δq':>34s}  判读")
print("-"*96, flush=True)
results=[]
t0=time.time()
for elev in (30.,45.,60.):
    el=math.radians(elev)
    for dist in (300.,380.,460.):
        for roll in range(0,360,30):
            z1=(aim-np.array([toward[0]*dist*math.cos(el),
                              toward[1]*dist*math.cos(el),
                              dist*math.sin(el)]))
            C=aim+np.array([toward[0]*dist*math.cos(el),
                            toward[1]*dist*math.cos(el),
                            dist*math.sin(el)])+z1*0
            camC=C; z=(aim-camC)/np.linalg.norm(aim-camC)
            up=np.array([0.,0.,-1.])
            xa=np.cross(up,z); xa/=np.linalg.norm(xa); ya=np.cross(z,xa)
            if roll:
                cr,sr=math.cos(math.radians(roll)),math.sin(math.radians(roll))
                xr=cr*xa+sr*ya; xa,ya=xr,np.cross(z,xr)
            Tc=np.eye(4); Tc[:3,0]=xa; Tc[:3,1]=ya; Tc[:3,2]=z; Tc[:3,3]=camC
            T_fl=Tc@Xinv
            got=solve(T_fl)
            if got is None:
                print(f"{elev:4.0f}{dist:6.0f}{roll:5d}{'—':>13s}{'—':>9s}{'—':>34s}  ✗ IK/限位", flush=True)
                continue
            q,ep,er=got
            dq=np.abs(q-q_now); mx=float(dq.max())
            good=mx<=50.0
            results.append((mx,elev,dist,roll,q.copy(),ep,er,T_fl.copy()))
            print(f"{elev:4.0f}{dist:6.0f}{roll:5d}{ep:8.2f}/{er:5.2f}{mx:9.1f}"
                  f"{np.round(dq,0).astype(int).tolist()!s:>34s}  {'✓' if good else '✗'}", flush=True)
results.sort(key=lambda r:r[0])
print(f"\n用时 {time.time()-t0:.0f}s")
if results:
    mx,elev,dist,roll,q,ep,er,T_fl=results[0]
    print(f"★ 最小跳变：仰角{elev:.0f}° 距离{dist:.0f}mm ρ={roll}°  max|dq|={mx:.1f}°  残差 {ep:.2f}mm/{er:.2f}°")
    print(f"  关节 {np.round(q,2).tolist()}")
    print(f"  法兰 {np.round(T_fl[:3,3],1).tolist()}")
    print(f"  当前法兰 {np.round(fc.T(q_now)[:3,3],1).tolist()}")
    print("\n满足 max|dq|<=50° 的候选数：", sum(1 for r in results if r[0]<=50.0))
