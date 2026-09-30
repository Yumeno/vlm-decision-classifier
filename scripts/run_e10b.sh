#!/usr/bin/env bash
# E10b: 実データ31枚で json と json_schema を比較(準備あり、方式ごと)。モデルごとにパッチ版 llama-server を起動して回す。
# 使い方: LLAMACPP_DIR=... LMSC_DIR=... [OUT=results/e10b] bash scripts/run_e10b.sh
# 注: 記録した実行は、同じコマンドを個人パス直書きにした使い捨て版で回した(コミット前)。コマンドの中身は同じ。
REPO=${REPO:-$(pwd)}
LB=${LLAMACPP_DIR:?}
MD=${LMSC_DIR:?lmstudio-community のモデルフォルダ}
OUT=${OUT:-$REPO/results/e10b}
mkdir -p $OUT; LOG=$OUT/progress.log; cd $REPO
echo "$(date -u +%FT%TZ) START E10b" >> $LOG
stop_server() { P=$(netstat -ano | grep ':1235 ' | grep LISTENING | awk '{print $5}' | head -1); [ -n "$P" ] && taskkill //F //PID $P > /dev/null 2>&1; sleep 4; }
while read sid alias repo f mp rt; do
  stop_server
  ($LB/patched/build/bin/Release/llama-server.exe -m $MD/$repo/$f --mmproj $MD/$repo/$mp --alias $alias -ngl 99 -c 8192 -np 4 --kv-unified -sm none -mg 0 --port 1235 > $OUT/server_$sid.log 2>&1 &)
  for i in $(seq 1 60); do curl -s http://127.0.0.1:1235/health | grep -q ok && break; sleep 2; done
  echo "$(date -u +%FT%TZ) loaded $sid gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $LOG
  .venv/Scripts/python.exe -m classifier_demo evaluate --manifest dataset/manifest.jsonl --base-url http://127.0.0.1:1235/v1 --model $alias --modes json,json_schema --warmup 1 --prime --axis-concurrency 1 --image-format jpeg --max-edge 1024 --dataset-version v1.0.0 --runtime-info doc/experiments/runtime/$rt.json --runtime-label E10b_$sid --note "E10b json vs json_schema on dataset" --output-dir $OUT/$sid > $OUT/$sid.stdout.log 2>&1
  echo "$(date -u +%FT%TZ) done $sid exit=$?" >> $LOG
done <<'T'
S3 qwen3.5-2b-q4 Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf mmproj-Qwen3.5-2B-BF16.gguf small-S3
S5 qwen3.5-0.8b-q4 Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf mmproj-Qwen3.5-0.8B-BF16.gguf small-S5
S7 gemma-4-e4b-q4 gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf mmproj-gemma-4-E4B-it-BF16.gguf small-S7
S9 gemma-4-e2b-q4 gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf mmproj-gemma-4-E2B-it-BF16.gguf small-S9
F1 qwen3.5-9b Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf mmproj-Qwen3.5-9B-BF16.gguf final-qwen-patched
T
stop_server
echo "$(date -u +%FT%TZ) ALL DONE" >> $LOG
