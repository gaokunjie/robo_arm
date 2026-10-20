# -*- coding: utf-8 -*-
"""法奥 FR3 · 轻量 RPC 客户端（直连 xmlrpc :20003）

为什么不用官方 Python SDK：官方 SDK(master) 的连接自检要求 GetControllerIP，
本机固件(000-000-022)没有这个方法，自检失败后 SDK 所有调用一律返回 -4；
而且本机固件是 121 个方法的老一代接口，SDK 里的 613 个方法大半不存在。

参数格式一律以《机器人控制器通讯指令协议用户手册》(307 页, 见 参考手册/) 为准：
本机固件是【平铺参数】风格，例如逆解是 GetInverseKin(type,x,y,z,rx,ry,rz,config) 8 个参数。

返回值约定（2026-10-15 实测，别照新版手册判）：本机旧固件的写指令【成功返回 1，不是 0】；
失败大多直接抛 xmlrpc Fault（类型/参数错都走 Fault，原文可读）。已用 WebApp 独立复核：
SetSpeed(35) 返回 1 且 vel_radio 0.2→0.35、复位 SetSpeed(20) 返回 1 且回到 0.2；
RobotEnable(1) 返回 1 且 rbtEnableState 确实变 1；WaitMs(100)（不可能失败的纯延时）也是 1。
手册的错误码表（0=成功、3=参数个数异常…）是 V4 版的，对不上这台旧固件——判成功别写 ==0。
读取类指令返回数值数组。
"""

import http.client
import os
import time
import xmlrpc.client

import config


# ---------------- 黑匣子（RPC 调用记录）----------------
# 2026-10-16 两次 dry 预演都在 ~600 次调用处把控制器打掉线（连接复用修复后仍死）
# → 需要"第几次、什么调用、什么时候"的精确记录。默认关闭；设环境变量
#   FR_RPC_LOG=/path/to/log
# 才启用。每行：单调秒 序号 名字(参数截断80字) 耗时ms 结果；另有 conn 新建/复用 事件
# （复用行=keep-alive 真在用；突然冒"新建"=服务器把连接关了或基类偷偷重试了）。
_LOG = {"fh": None, "n": 0, "t0": None, "tried": False}


def _log_open():
    if _LOG["tried"]:
        return
    _LOG["tried"] = True
    _LOG["t0"] = time.monotonic()
    path = os.environ.get("FR_RPC_LOG")
    if not path:
        return
    try:
        _LOG["fh"] = open(path, "a", encoding="utf-8")
        _log_line(f"=== 黑匣子开启 pid={os.getpid()} ===")
    except OSError as e:
        print(f"⚠ FR_RPC_LOG 打开失败（不记了）: {e}")


def _log_line(msg):
    fh = _LOG["fh"]
    if fh is None:
        return
    try:
        fh.write(f"{time.monotonic() - _LOG['t0']:9.3f} {msg}\n")
        fh.flush()
    except OSError:
        pass


class _TimeoutTransport(xmlrpc.client.Transport):
    """给 xmlrpc 加上超时——默认没有超时，机器人在跑运动指令时会阻塞到天荒地老。

    ⚠ 2026-10-16 教训（dry 预演把控制器服务敲掉线）：**必须保留基类的连接复用**。
    基类 make_connection 会把连接缓存进 self._connection 走 HTTP/1.1 keep-alive；
    这里最初图省事每次 new 一个 HTTPConnection，几百次只读调用就是几百条 TCP 连接
    砸向嵌入式控制器（那次 :20003 和 :8083 两个服务随后同时拒绝连接）。
    下面这版照着基类抄，只多一个超时参数。

    ⚠ 2026-10-20：**出错后必须扔掉这条连接**。控制器偶发 HTTP 408 时，响应和
    请求可能已经错位；继续在旧 socket 上发下一条，会把一次瞬时故障放大成连续
    超时。reset() 由 FRRpc.call 在非 Fault 异常后调用，下一条请求走全新连接。
    """

    def __init__(self, timeout=6.0):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host):
        # 与基类一致：同 host 复用已有连接；只是创建时带上超时
        if self._connection and host == self._connection[0]:
            _log_line("conn 复用")
            return self._connection[1]
        chost, self._extra_headers, x509 = self.get_host_info(host)
        self._connection = host, http.client.HTTPConnection(chost, timeout=self.timeout)
        _log_line("conn 新建")
        return self._connection[1]

    def reset(self):
        """关闭并丢弃缓存的 keep-alive 连接（出错恢复用）。"""
        old, self._connection = self._connection, None
        if old is not None:
            try:
                old[1].close()
            except Exception:                                  # noqa: BLE001
                pass
            _log_line("conn 关闭（出错后重建）")


class FRRpcError(RuntimeError):
    """RPC 失败。uncertain=True 表示「指令可能已经执行，但响应丢了」。

    典型场景：HTTP 408 / socket 超时。此时**绝不能盲目重发**——先看机器人状态，
    已到位就继续，没动才重发。Fault（机器人明确拒绝）是确定性的，不重试。
    """

    def __init__(self, msg, *, uncertain=False):
        super().__init__(msg)
        self.uncertain = bool(uncertain)


class FRRpc:
    def __init__(self, ip=None, timeout=6.0):
        _log_open()
        self.ip = ip or config.ROBOT_IP
        self.url = f"http://{self.ip}:20003"
        self._transport = _TimeoutTransport(timeout)
        self._p = xmlrpc.client.ServerProxy(self.url, transport=self._transport)

    # ---------------- 底层 ----------------
    def call(self, name, *args):
        _log_open()
        t = time.monotonic()
        arg_s = str(args)[1:-1].replace("\n", " ")[:80] if args else ""
        try:
            r = getattr(self._p, name)(*args)
            _LOG["n"] += 1
            _log_line(f"#{_LOG['n']:04d} {name}({arg_s}) {(time.monotonic() - t) * 1000:7.1f}ms ok")
            return r
        except xmlrpc.client.Fault as e:
            _LOG["n"] += 1
            _log_line(f"#{_LOG['n']:04d} {name}({arg_s}) {(time.monotonic() - t) * 1000:7.1f}ms "
                      f"Fault: {str(e.faultString)[:80]}")
            raise FRRpcError(f"{name} 被机器人拒绝: {e.faultString}") from e
        except Exception as e:
            self._transport.reset()
            _LOG["n"] += 1
            _log_line(f"#{_LOG['n']:04d} {name}({arg_s}) {(time.monotonic() - t) * 1000:7.1f}ms "
                      f"ERR: {type(e).__name__}: {str(e)[:60]}")
            raise FRRpcError(
                f"{name} 调用失败: {type(e).__name__}: {e}", uncertain=True) from e

    def _call_read(self, name, *args):
        """只读调用的重试退避：连接类错误（拒绝/重置/超时/断线）重试 3 次，间隔 0.5/1.5/3s。

        xmlrpc Fault（参数错、无解这类"机器人明确回复了"的错）是确定性的，立刻抛。
        ⚠ 写指令【绝不】走这条路——运动的指令不重发（重发 = 可能走两遍）。
        """
        last = None
        for k, back in enumerate((0.0, 0.5, 1.5, 3.0)):
            if back:
                time.sleep(back)
            try:
                return self.call(name, *args)
            except FRRpcError as e:
                last = e
                if "被机器人拒绝" in str(e):        # 服务器应答了：确定性错误，不重试
                    raise
                if k == 3:
                    raise
        raise last                                   # 不可达，给静态检查一个交代

    def methods(self):
        return sorted(self.call("system.listMethods"))

    # ---------------- 只读 ----------------
    def tcp_pose(self):
        """当前工具位姿 [x,y,z,rx,ry,rz]（mm / deg）。"""
        return list(self._call_read("GetActualTCPPose", 1))

    def joint_deg(self):
        return list(self._call_read("GetActualJointPosDegree", 1))

    def flange_pose(self):
        return list(self._call_read("GetActualToolFlangePose", 1))

    def ik(self, pose, config_=-1, type_=0):
        """逆解：位姿 → 关节角。协议手册 6.3.26，注释见下。"""
        j = self._call_read("GetInverseKin", type_,
                            float(pose[0]), float(pose[1]), float(pose[2]),
                            float(pose[3]), float(pose[4]), float(pose[5]), config_)
        # 返回可能是 [j1..j6]，也可能带 errcode 前缀，两种都兼容
        return list(j[1:]) if len(j) == 7 else list(j)

    def ik_ref(self, pose, ref):
        """带参考关节的逆解：返回"离参考最近"的那组解（协议手册 6.3.27）。

        2026-10-16 用试形法实测的形状 = (INT, pose×6, ref×6)——【第 1 个参数是 int】：
          · 发 2 个参数        → "Format string requests 3 items from array, but array has only 2"
          · 发 (pose, ref, 0) → "Value of type ARRAY supplied where type INT was expected"
          · 发 (0, pose, ref) → 返回关节角 ✓（ref=当前关节时就返回当前家族的解）
        用途：**钉关节家族**。自动解（config=−1）只是"近似就近"，撞机事故的根因之一
        就是没人能说清它到底给了哪一族；ref 明确写成当前关节后，家族是白纸黑字的输入。
        ⚠ 但它只是"择近启发式"：2026-10-16 dry 预演实测它会偶发地把位姿解到深家族
        （Δ71.5°=|−128.69−(−200.18)|），采集器侧的八组枚举兜底就是给它擦屁股的。
        """
        j = self._call_read("GetInverseKinRef", 0,
                            [float(v) for v in pose], [float(v) for v in ref])
        return list(j[1:]) if len(j) == 7 else list(j)

    def fk(self, joints):
        """正解：关节角 → 位姿。协议手册 6.3.24，同样是 6 个平铺参数。"""
        p = self._call_read("GetForwardKin", *[float(v) for v in joints])
        return list(p[1:]) if len(p) == 7 else list(p)

    def joint_soft_limits(self):
        v = self.call("GetJointSoftLimitDeg", 1)
        return [(v[2 * i], v[2 * i + 1]) for i in range(6)]

    def default_trans_vel(self):
        return self.call("GetDefaultTransVel")

    # ---------------- 控制 ----------------
    def enable(self, state=1):
        return self.call("RobotEnable", int(state))

    def set_mode(self, state):
        """0 = 自动模式，1 = 手动模式。"""
        return self.call("Mode", int(state))

    def set_speed(self, pct):
        """全局速度百分比 0~100。"""
        pct = int(min(max(pct, 0), 100))
        return self.call("SetSpeed", pct)

    def stop(self):
        """暂停（= 停住当前运动）。本机固件没有 StopMotion，暂停/继续是 Pause / Resume。

        ⚠ 本机固件 Pause 要 1 个参数：发 0 个会被拒，原文
        "Format string requests 1 items from array"。参数语义手册没写（V4 手册里查不到
        这个 xmlrpc 方法），这里按实测的形状发 1。
        ⚠⚠ 在【运动中】真停一次还没验过：第一次真用会发生在带急停的现场——如果 Pause
        返回奇怪的值或没停住，直接拍急停。
        """
        return self.call("Pause", 1)

    def resume(self):
        return self.call("Resume")

    # ---------------- 运动 ----------------
    def move_l(self, joints, pose, tool, user, speed, acc=100.0, ovl=100.0,
               blend_r=0.0, search=0):
        # blend_r/blend_t 语义（手册）：-1=在此停止【阻塞】——服务端会把本次调用挂到运动结束
        # （老固件实测：8° 要 2.9s，48° 要 ~17s 直接撞穿客户端 3s 超时；且【超时≠没动】）。
        # 所以调用层（fr_arm）一律传 0.0 非阻塞，等结束用状态制 wait_done。
        """直线运动。**本机固件是 V4.0.0 之前的旧 MoveL，收 12 组参数。**

        2026-10-14 实测定的，别照手册的 33 个平铺标量发——那个会被控制器拒绝，
        原文是：Format string requests exactly 12 items from array, but array has 33 items.
        手册里的 MoveL 标着"V4.0.0 更新版"，本机固件 000-000-022 早于它。

        12 组顺序 = 手册 V4 全量参数表去掉旧版没有的三个标量（blendMode / oacc /
        velAccParamMode）后按类型分组。2026-10-15 用 src/probe/type_scan.py 逐槽实测：
            {j1~j6} {x,y,z,rx,ry,rz} tool user speed acc ovl blendR
            {exaxis1~4} search offset_flag {dt_x~dt_rz}
        实测关键点（别凭印象写）：槽 9 是【数组】exaxis——手册 V4 表里 exaxisPos1~4
        就排在 search_flag 之前，旧版同理；之前误把手册的"search {ep}"顺序抄反过一次。
        旁证：手册写 GetInverseKinHasSolution 收 3 组，发 2 组 → 控制器回
            Format string requests 3 items from array, but array has only 2 items.
        move_j 也是同样"按组"的写法，11 个参数。

        ⚠ speed / acc / ovl 是【百分比 0~100】，不是 mm/s——旧版没有 velAccParamMode，
          给不了物理速度。想要 mm/s 得先实测出百分比↔mm/s 的换算。
        ⚠ 类型是【混合】的，2026-10-14/15 试错 + 逐槽扫描定死：3/4/10/11 = int，
          5~8 = float，1/2/9/12 = 数组。踩过的坑（都别再犯）：
            · tool/user 发 float  → "DOUBLE supplied where INT was expected"（首跑栽这）
            · 槽 9 发 int         → "'A' specifier requires type ARRAY"（9 是 exaxis 数组）
            · ovl 想按手册 V4 发 int → 实测旧版要 float（V4 表把它标成 int 是改过的）
          （GetInverseKin 的 type/config 发 int 是能用的，说明固件确实认 int 型，
            所以不是"一律都要 double"这么简单。）
        ⚠ search / offset_flag / 偏移组 的顺序已按实测钉死（9 数组 exaxis → 10 search →
          11 offset_flag → 12 {dt_x~dt_rz}），但内容一律传 0，没验过它们各自真起作用。
          将来要用焊丝寻位或偏移，先在空负载下单独验一次再上。
        blend_r=-1 表示到位阻塞（非阻塞传 0~1000 的平滑半径 mm）。
        返回：成功 = 1（本机固件成功码，见文件头备注；别按手册的 0 判成失败）。
        """
        return self.call("MoveL",
                         [float(v) for v in joints],           # 1  {j1~j6}
                         [float(v) for v in pose],             # 2  {x,y,z,rx,ry,rz}
                         int(tool), int(user),                 # 3 4  ← 实测 int，别写 float（首跑栽这）
                         float(speed), float(acc), float(ovl), float(blend_r),  # 5 6 7 8
                         [0.0, 0.0, 0.0, 0.0],                 # 9  {exaxis1~4} ← 实测是数组，不是 search
                         int(search),                          # 10 search（焊丝寻位，不发就 0）
                         0,                                    # 11 offset_flag（不偏移）
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])       # 12 {dt_x~dt_rz}

    def move_j(self, joints, pose, tool, user, speed, acc=100.0, ovl=100.0, blend_t=0.0):
        """关节运动（协议手册 2.1，29 槽 → xmlrpc 按组收成 11 个实参）。

        ⚠ 类型实测（2026-10-16）：**ovl 必须 float**。协议手册表 2-1 写 ovl 是 int，
        但本机老固件（000-000-022）传 int 直接被拒：
        「Value of type INT supplied where type DOUBLE was expected」。
        官方 SDK 的 cast 表也是 float（tool/user 才是 int），照它。

        blend_t 同 move_l：0=非阻塞（-1 会让服务端阻塞到运动结束——8° 就 2.9s，撞 3s 超时）。
        """
        return self.call("MoveJ",
                         [float(v) for v in joints], [float(v) for v in pose],
                         int(tool), int(user), float(speed), float(acc), float(ovl),
                         [0.0, 0.0, 0.0, 0.0], float(blend_t), 0,
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ---------------- 坐标系 / 安全 ----------------
    def set_tool_coord(self, tool_id, pose):
        return self.call("SetToolCoord", int(tool_id), [float(v) for v in pose], 0, 0)

    def set_collision_level(self, level):
        """碰撞等级 0~5（越大越不灵敏），本机固件的 SetAnticollision(mode, level, config)。"""
        return self.call("SetAnticollision", 0, int(level), 0)

    def set_collision_strategy(self, strategy):
        return self.call("SetCollisionStrategy", int(strategy))

    # ---------------- IO ----------------
    def set_do(self, idx, status):
        return self.call("SetDO", int(idx), int(status), 0, 0)

    def get_di(self, idx):
        return self.call("GetDI", int(idx), 1)
