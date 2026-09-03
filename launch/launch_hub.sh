#!/usr/bin/env bash
# ============================================================
# 启动 hub_host（中心端）
# 终端3: brainbot 命令服务（上游）
# 终端4: 可视化仪表盘（brainbot_webviz，需上游脚本）
# 备用: 本工程内置的闭环演示
# ============================================================
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"

echo "[launch_hub] 运行内置七层闭环演示 ..."
python3 scripts/demo_closed_loop.py

# 若安装了上游 brainbot_webviz，可在此启动仪表盘
if command -v brainbot-command-service >/dev/null 2>&1; then
    echo "[launch_hub] 提示: 可另开终端运行 brainbot webviz (上游)"
fi
