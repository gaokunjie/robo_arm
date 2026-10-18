# 探针归档（2026-10-14 从 /tmp/fr_probe 收编）

这些是"直接问控制器"的一次性探针，**不走 `src/fr_arm.py` 的安全闸**
（当初就是为了绕开封装排查 MoveL 参数格式）。所以它们只允许用于
【纯只读】或【零位移】场景；正式运动一律走 `fr_arm`。

| 文件 | 干什么 | 会发运动吗 |
|---|---|---|
| `type_scan.py` | MoveL 12 槽参数类型/长度升序扫描（**B1 就用它**） | 发 MoveL，但目标 = 发送瞬间的当前位姿，**位移恒为 0** |
| `status.py` | 8083 整帧 + 关键 RPC 只读快照 | 否 |
| `precheck.py` | 关节软限位、逆解、"错位解释"可达性 | 否 |
| `reach.py` | `GetInverseKinHasSolution` 只读核查 | 否 |
| `methods.py` / `introspect.py` | 列 RPC 方法表 / 探测 `system.*` | 否 |
| `boot_watch.py` | 断电/重启直播监看：ping + 80/20003/8083 每 2s 采样，状态一变打时间戳（用法：`python3 -I …/boot_watch.py [秒数]`） | 否（纯网络观测） |
| `app_probe.py` | 网页层（:80）只读探针：登录 + `/action/sta` 全量 + `get_robot_type`/`get_status_flag` | 否 |

后两个是 2026-10-16 控制器恢复取证时收编的（网页层诊断，见 `HANDOFF.md` 顶部 §2）。

跑法（脚本内有旧绝对路径，能跑就不动它）：

```bash
python3 -I /home/kelly/桌面/1014/src/probe/type_scan.py
```

⚠ `type_scan.py` 每次发指令前自己三查 `estop` / `error_code` / `motion_done`，
任一不过拒发；但**"发 MoveL"这件事本身需要用户手放急停**——这是项目铁律。
