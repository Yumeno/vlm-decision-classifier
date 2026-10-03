#!/bin/bash
# 自前の判定サーバー(CPU のみ。vLLM :8000 の前段、127.0.0.1:8012)
# 必要な環境変数: VENV_EVAL(本リポジトリを pip install -e した venv)、REPO_DIR(リポジトリのルート)
# 任意: LOG_DIR(既定 ~/dgemma-logs)
set -u
: "${VENV_EVAL:?VENV_EVAL を指定}" "${REPO_DIR:?REPO_DIR を指定}"
LOG_DIR=${LOG_DIR:-$HOME/dgemma-logs}; mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
exec "$VENV_EVAL/bin/python" -m classifier_demo dgemma-server --vllm http://127.0.0.1:8000 --model dgemma \
  --host 127.0.0.1 --port 8012 > "$LOG_DIR/dgemma-server.log" 2>&1
