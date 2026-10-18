# -*- coding: utf-8 -*-
"""MoveL 12 组参数 · 类型+数组长度 全自动扫描（零位移，纯探测）

控制器报错有两种家族，都带"我们发的类型"和"它要的类型"两侧：
  A) Value of type INT supplied where type DOUBLE was expected.
  B) Value to be decomposed is of type DOUBLE, but the 'A' specifier requires type ARRAY
所以按【位置升序】逐槽探，且把未测槽统一填成跟"本次试探类型"不同的基准类型，
第一条错报里的"发送类型"就唯一指向当前槽 —— 一个槽一次调用定死。

2026-10-15 实测拼出的完整签名（12 组）＝手册 V4 全量表去掉旧版没有的 blendMode /
oacc / velAccParamMode 三个标量，按类型分组：
    {j6} {pose6} tool user speed acc ovl blendR {exaxis1~4} search offset_flag {dt_x~dt_rz}
    A6    A6      I    I    D    D    D   D      A[4]         I      I            A[6]
本轮要验的就是这条：先整发一发（对了就收工），不成再升序逐槽定死。
（勘误：前一轮"槽 5~11 里至少还有一个要数组"其实**是对的**——那个数组就是槽 9
  = exaxis；我一度误判成探针假警报。槽 9 发 int 报"要 ARRAY"今天又复现两次。）
（★ 上面若打印 "MoveL 返回 1"：那不是报错——本机旧固件写指令成功码就是 1，已用
 SetSpeed+WebApp vel_radio 回读独立验证，WaitMs/RobotEnable 同样返回 1；失败会抛
 xmlrpc Fault。详见 fr_rpc.py 文件头。）

运行：python3 -I src/probe/type_scan.py   （从哪跑都行，脚本自己把项目 src/ 插进 sys.path）
"""
import http.client
import re
import sys
import time
import xmlrpc.client

sys.path.insert(0, "/home/kelly/桌面/1014/src")
from fr_state import StateReader

IP = "192.168.58.3"
SCAN = [5, 6, 7, 8, 9, 10, 11, 12]
MAX_CALLS = 26
BASE = "D"                      # 未测槽的基准类型（必须 ≠ 本次试探类型）
SCALAR_VALUES = {3: 0, 4: 0, 5: 20, 6: 30, 7: 100, 8: -1, 10: 0, 11: 0}
LEN_TRIES = [4, 6, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12]


class _T(xmlrpc.client.Transport):
    def __init__(self, timeout=3.0):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host):
        return http.client.HTTPConnection(host, timeout=self.timeout)


p = xmlrpc.client.ServerProxy(f"http://{IP}:20003", transport=_T())


def parse_fault(s):
    """→ (发送类型, 期望类型) 或 None。类型归一成 A/I/D。"""
    m = re.search(r"Value of type (\w+) supplied where type (\w+)", s)
    if m:
        return _norm(m.group(1)), _norm(m.group(2))
    m = re.search(r"is of type (\w+), but the '(\w+)' specifier requires type (\w+)", s)
    if m:
        return _norm(m.group(1)), _norm(m.group(3))
    return None


def _norm(w):
    return {"ARRAY": "A", "DOUBLE": "D", "INT": "I"}.get(w.upper(), w.upper())


def make_args(types, lens, joints, pose):
    out = []
    for i in range(1, 13):
        if i == 1:
            out.append([float(v) for v in joints])
        elif i == 2:
            out.append([float(v) for v in pose])
        elif types[i] == "A":
            out.append([0.0] * lens[i])
        elif types[i] == "I":
            out.append(int(SCALAR_VALUES[i]))
        else:
            out.append(float(SCALAR_VALUES[i]))
    return out


def send(types, lens):
    joints = [float(v) for v in p.GetActualJointPosDegree(1)]
    pose = [float(v) for v in p.GetActualTCPPose(1)]
    args = make_args(types, lens, joints, pose)
    layout = " ".join((f"{types[i]}[{lens[i]}]" if types[i] == "A" else types[i])
                      for i in range(1, 13))
    try:
        r = p.MoveL(*args)
        return ("ok", r, layout)
    except xmlrpc.client.Fault as e:
        return ("fault", e.faultString, layout)


def main():
    st = StateReader(IP)
    st.start()
    t0 = time.time()
    while time.time() - t0 < 5.0 and not st.frames:
        time.sleep(0.05)
    if not st.frames:
        print(f"✗ 8083 无帧：{st.last_error}")
        return 1
    print(f"前置检查：estop={st.estop}  error_code={st.error_code}  motion_done={st.motion_done}")
    if st.estop or st.error_code or st.motion_done != 1:
        print("✗ 现场状态不适合发指令，退出。")
        st.stop()
        return 1

    types = {1: "A", 2: "A", **{i: BASE for i in range(3, 13)}}   # 槽 1/2 恒为 {j} {pose} 数组
    types[3] = types[4] = "I"          # 已实测：tool/user = int
    types[5] = types[6] = types[7] = types[8] = "D"   # 实测：speed/acc/ovl/blendR = double
    types[9] = "A"                     # 实测（三次）：exaxis1~4；发 int 时报"要 ARRAY"
    types[10] = types[11] = "I"        # search / offset_flag（按手册 V4 表序推）
    types[12] = "A"                    # dt_x~dt_rz 6 个偏移量
    lens = {1: 6, 2: 6, 9: 4, 12: 6}
    calls = 0
    done = False

    # 第 0 步：先整发一发"当前最可能签名"——过了就一发收工，等于给 fr_rpc.move_l 背书
    kind, res0, layout0 = send(types, lens)
    calls = 1
    if kind == "ok":
        print(f"\n✓✓✓ 一发命中！MoveL 返回 {res0}")
        print(f"    最终布局：{layout0}")
        done = True
    else:
        print(f"\n整发未过（正常，继续逐槽定）：{res0}\n     布局：{layout0}")

    for k in SCAN:
        if done:
            break
        for j in SCAN:                 # 未测槽回填基准，已定的保持
            if j > k and types[j] != "A":
                types[j] = BASE
        types[k] = "I"
        kind, res, layout = send(types, lens)
        calls += 1
        if kind == "ok":
            print(f"\n✓✓✓ 成功！MoveL 返回 {res}")
            print(f"    最终布局：{layout}")
            done = True
            break
        got = parse_fault(res)
        if not got:
            print(f"\n{k:>3}  ？类型报错之外的消息，停下：{res}\n     布局：{layout}")
            st.stop()
            return 1
        sent, want = got
        if sent == "I":
            if want == "A":
                print(f"{k:>3}  槽要 ARRAY —— 试长度")
                solved = False
                for L in LEN_TRIES:
                    types[k] = "A"
                    lens[k] = L
                    kind, res, layout = send(types, lens)
                    calls += 1
                    if kind == "ok":
                        print(f"\n✓✓✓ 成功！MoveL 返回 {res}")
                        print(f"    最终布局：{layout}")
                        solved, done = True, True
                        break
                    got2 = parse_fault(res)
                    if got2 and got2[0] == "D":
                        print(f"{k:>3}  数组长度 {L} 通过")
                        solved = True
                        break
                    print(f"{k:>3}  长度 {L} 不对：{res[:90]}")
                if not solved:
                    print("  ✗ 长度没试出来，把上面发我。")
                    st.stop()
                    return 1
                if done:
                    break
            else:
                types[k] = want
                print(f"{k:>3}  试探 I → 要 {want}（定为 {want}）")
        elif sent == "D":
            types[k] = "I"
            print(f"{k:>3}  试探 I → 对了（定为 I）")
        else:
            print(f"{k:>3}  ？意外：发送类型报成 {sent}，停下。{res}")
            st.stop()
            return 1
        if calls > MAX_CALLS:
            print("调用次数超过上限，停下。")
            st.stop()
            return 1

    if not done:
        print("\n扫完仍未成功。把上面整段发我。")
        st.stop()
        return 1

    time.sleep(0.5)
    pose1 = [float(v) for v in p.GetActualTCPPose(1)]
    print(f"    事后 TCP = {[round(v, 2) for v in pose1]}")
    print(f"    总调用次数 {calls}")
    st.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
