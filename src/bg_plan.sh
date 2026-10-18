#!/usr/bin/env bash
# 后台跑只读规划：永不阻塞前台，因此**不再有超时**。
#
#   用法：  bash src/bg_plan.sh <日志名> <见下方命令...>
#   查看：  tail -f /tmp/opencode/<日志名>.log
#   结果：  /tmp/opencode/<日志名>.done  （含退出码与缓存统计）
#
# 为什么放后台：控制器 FK 是 ~20 次/秒的硬地板，一次 40 段规划要几千次调用
# = 几分钟。前台跑必然被 shell 超时打断，而且中断时**缓存白攒**（那才是真正的损失）。
set -u
NAME="$1"; shift
LOG="/tmp/opencode/${NAME}.log"
DONE="/tmp/opencode/${NAME}.done"
SCRIPT="${1:-see_cup.py}"; shift || true
mkdir -p /tmp/opencode
rm -f "$DONE" "$LOG"
nohup bash -c "
  cd /home/kelly/桌面/1014
  python3 -u -I src/run_cached.py --script '$SCRIPT' $* > '$LOG' 2>&1
  echo \"exit=\$?\" >> '$LOG'
  touch '$DONE'
" >/dev/null 2>&1 &
echo "已在后台启动：$SCRIPT $*"
echo "  日志：$LOG   （tail -f 查看）"
echo "  完成标志：$DONE"
