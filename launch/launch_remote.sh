#!/usr/bin/env bash
# ============================================================
# 启动 remote_host（机器人端）
# 终端1: 控制服务（brainbot 上游）
# 终端2: AI 策略服务器（本工程四层大脑）
# ============================================================
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"

echo "[launch_remote] 启动 AI 策略服务器 (七层认知大脑) ..."
python3 -m src.brainbot_adapters.policy_server --host 127.0.0.1 --port 5555 &
POLICY_PID=$!

echo "[launch_remote] 策略服务器 PID=$POLICY_PID (tcp://127.0.0.1:5555)"

# 若已安装上游 brainbot 控制服务，可同时启动（此处可选）
if command -v brainbot-command-service >/dev/null 2>&1; then
    echo "[launch_remote] 启动 brainbot 命令服务 ..."
    brainbot-command-service &
fi

trap "kill $POLICY_PID 2>/dev/null" EXIT
wait
