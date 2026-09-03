#!/usr/bin/env bash
# 运行全部测试
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"
exec python3 -m pytest tests/ -v -p no:cacheprovider "$@"
