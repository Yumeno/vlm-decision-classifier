#!/bin/bash
# 使い方: run_eval.sh <label> <seed> <modes> <note>
# 本番設定: 自前 dgemma-server(:8012)、keyed、yn_style=yn、instruction=default、16問1回(max_per_read 0)、samples 1、名寄せあり。
# 必要な環境変数: VENV_EVAL、REPO_DIR。任意: LOG_DIR(既定 ~/dgemma-logs)
set -u
: "${VENV_EVAL:?VENV_EVAL を指定}" "${REPO_DIR:?REPO_DIR を指定}"
LOG_DIR=${LOG_DIR:-$HOME/dgemma-logs}; mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
LABEL=$1; SEED=$2; MODES=$3; NOTE=$4
"$VENV_EVAL/bin/python" -m classifier_demo evaluate --manifest dataset/manifest.jsonl --model dgemma \
  --base-url http://127.0.0.1:8000/v1 --dgemma-url http://127.0.0.1:8012 \
  --dgemma-samples 1 --dgemma-seed "$SEED" --dgemma-yn-style yn \
  --modes "$MODES" --warmup 0 --image-format jpeg --max-edge 1024 --dataset-version v1.0.0 \
  --runtime-info doc/experiments/runtime/dgemma-vllm.json --runtime-label "$LABEL" \
  --note "$NOTE" --output-dir "results/$LABEL" > "$LOG_DIR/$LABEL.log" 2>&1
rc=$?
echo "rc=$rc"
tail -5 "$LOG_DIR/$LABEL.log"
exit $rc
