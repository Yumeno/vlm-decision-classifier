#!/usr/bin/env bash
# E10 出力形式ストレステスト(白色ノイズ画像 N枚 × 方式)。モデルごとにパッチ版 llama-server を起動して回す。
# 使い方: LLAMACPP_DIR=... LMSC_DIR=... [OUT=results/e10] [N=100] [MODELS="S3 S4 S5 S6 S9 F1"] bash scripts/run_e10.sh
REPO=${REPO:-$(pwd)}
LB=${LLAMACPP_DIR:?}
MD=${LMSC_DIR:?lmstudio-community のモデルフォルダ}
OUT=${OUT:-$REPO/results/e10}
N=${N:-100}
MODELS=${MODELS:-"S3 S4 S5 S6 S9 F1"}
mkdir -p $OUT; LOG=$OUT/progress.log; cd $REPO
echo "$(date -u +%FT%TZ) START N=$N MODELS=$MODELS" >> $LOG
stop_server() { P=$(netstat -ano | grep ':1235 ' | grep LISTENING | awk '{print $5}' | head -1); [ -n "$P" ] && taskkill //F //PID $P > /dev/null 2>&1; sleep 4; }
start_server() { # repo file mmproj alias name
  ($LB/patched/build/bin/Release/llama-server.exe -m $MD/$1/$2 --mmproj $MD/$1/$3 --alias $4 -ngl 99 -c 8192 -np 4 --kv-unified -sm none -mg 0 --port 1235 > $OUT/server_$5.log 2>&1 &)
  for i in $(seq 1 60); do curl -s http://127.0.0.1:1235/health | grep -q ok && return 0; sleep 2; done; return 1; }
while read sid alias repo f mp rt; do
  [ -z "$sid" ] && continue
  case " $MODELS " in *" $sid "*) ;; *) continue;; esac
  [ -f "$MD/$repo/$f" ] || { echo "$(date -u +%FT%TZ) MISSING $sid $f" >> $LOG; continue; }
  name=$sid
  [ -f $OUT/$name/summary.md ] && { echo "$(date -u +%FT%TZ) skip $name" >> $LOG; continue; }
  stop_server
  echo "$(date -u +%FT%TZ) before $name gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $LOG
  start_server $repo $f $mp $alias $name || { echo "$(date -u +%FT%TZ) SERVER FAIL $name" >> $LOG; continue; }
  echo "$(date -u +%FT%TZ) loaded $name gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ') apps3090: $(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep -c 0D:00)" >> $LOG
  .venv/Scripts/python.exe scripts/format_stress.py --base-url http://127.0.0.1:1235/v1 --model $alias --modes choice,json,json_schema,bundled,bundled_yn --n $N --warmup 1 --prime --runtime-info doc/experiments/runtime/$rt.json --runtime-label $name --note "E10 format stress $name" --output-dir $OUT/$name > $OUT/$name.stdout.log 2>&1
  echo "$(date -u +%FT%TZ) done $name exit=$?" >> $LOG
done <<'LIST'
S3 qwen3.5-2b-q4 Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf mmproj-Qwen3.5-2B-BF16.gguf small-S3
S4 qwen3.5-2b-q8 Qwen3.5-2B-GGUF Qwen3.5-2B-Q8_0.gguf mmproj-Qwen3.5-2B-BF16.gguf small-S4
S5 qwen3.5-0.8b-q4 Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf mmproj-Qwen3.5-0.8B-BF16.gguf small-S5
S6 qwen3.5-0.8b-q8 Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q8_0.gguf mmproj-Qwen3.5-0.8B-BF16.gguf small-S6
S9 gemma-4-e2b-q4 gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf mmproj-gemma-4-E2B-it-BF16.gguf small-S9
F1 qwen3.5-9b Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf mmproj-Qwen3.5-9B-BF16.gguf final-qwen-patched
LIST
stop_server
echo "$(date -u +%FT%TZ) ALL DONE" >> $LOG
