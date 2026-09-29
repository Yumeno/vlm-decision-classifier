#!/usr/bin/env bash
# 最終の取り直し(F1〜F3)。doc/experiments/final-runbook.md の手順どおり。
REPO=${REPO:-$(pwd)}  # リポジトリのルートで実行する
LB=${LLAMACPP_DIR:?llama.cpp の vanilla/ と patched/ を置いたフォルダを LLAMACPP_DIR に}
QM=${QWEN_DIR:?Qwen3.5-9B-GGUF のフォルダを QWEN_DIR に}
GM=${GEMMA_DIR:?gemma-4-12B-it-GGUF のフォルダを GEMMA_DIR に}
OUT=${OUT:-$REPO/results/final}
# 再測定用の絞り込み(既定は従来どおり)。例: PRIME=0 CONDITIONS=F1 EDGES=1024 REPS=1 OUT=results/final_reprime
PRIME=${PRIME:-1}  # 0 なら --prime を付けない
PRIME_FLAG=""; [ "$PRIME" = 1 ] && PRIME_FLAG="--prime"
CONDITIONS=${CONDITIONS:-"F1 F2 F3"}
EDGES=${EDGES:-"1024 768"}
REPS=${REPS:-"1 2 3"}
mkdir -p $OUT
LOG=$OUT/progress.log
echo "$(date -u +%FT%TZ) START PRIME=$PRIME CONDITIONS=$CONDITIONS EDGES=$EDGES REPS=$REPS" >> $LOG
cd $REPO
stop_server() {
  P=$(netstat -ano | grep ':1235 ' | grep LISTENING | awk '{print $5}' | head -1)
  [ -n "$P" ] && taskkill //F //PID $P > /dev/null 2>&1
  sleep 4
}
start_server() { # $1=cond
  case $1 in
    F1) BIN=$LB/patched; M=$QM/Qwen3.5-9B-Q4_K_M.gguf; MP=$QM/mmproj-Qwen3.5-9B-BF16.gguf; A=qwen3.5-9b;;
    F2) BIN=$LB/vanilla; M=$QM/Qwen3.5-9B-Q4_K_M.gguf; MP=$QM/mmproj-Qwen3.5-9B-BF16.gguf; A=qwen3.5-9b;;
    F3) BIN=$LB/patched; M=$GM/gemma-4-12B-it-Q4_K_M.gguf; MP=$GM/mmproj-gemma-4-12B-it-BF16.gguf; A=gemma-4-12b-it;;
  esac
  ($BIN/build/bin/Release/llama-server.exe -m $M --mmproj $MP --alias $A -ngl 99 -c 8192 -np 4 --kv-unified -sm none -mg 0 --port 1235 > $OUT/server_$2.log 2>&1 &)
  for i in $(seq 1 60); do curl -s http://127.0.0.1:1235/health | grep -q ok && return 0; sleep 2; done
  return 1
}
for rep in $REPS; do
 for cond in $CONDITIONS; do
  case $cond in F1) RT=final-qwen-patched; MID=qwen3.5-9b;; F2) RT=final-qwen-vanilla; MID=qwen3.5-9b;; F3) RT=final-gemma-patched; MID=gemma-4-12b-it;; esac
  for edge in $EDGES; do
   for v in a b; do
    name=${cond}_e${edge}_${v}_r${rep}
    if [ -f $OUT/$name/summary.md ]; then echo "$(date -u +%FT%TZ) skip $name (done)" >> $LOG; continue; fi
    stop_server
    echo "$(date -u +%FT%TZ) before $name gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $LOG
    if ! start_server $cond $name; then echo "$(date -u +%FT%TZ) SERVER FAIL $name" >> $LOG; continue; fi
    echo "$(date -u +%FT%TZ) loaded $name gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ') apps3090: $(nvidia-smi --query-compute-apps=pid,process_name,gpu_bus_id --format=csv,noheader | grep -c 0D:00)" >> $LOG
    if [ $v = a ]; then VA="--modes choice,json,bundled --rank-threshold 0.5 --bundled-multi rank"; else VA="--modes choice,bundled --confirm --bundled-multi yn"; fi
    .venv/Scripts/python.exe -m classifier_demo evaluate --manifest dataset/manifest.jsonl --base-url http://127.0.0.1:1235/v1 --model $MID $VA --warmup 1 $PRIME_FLAG --axis-concurrency 1 --image-format jpeg --max-edge $edge --dataset-version v1.0.0 --runtime-info doc/experiments/runtime/$RT.json --runtime-label $name --note "final re-run $name (doc/experiments/final-runbook.md)" --output-dir $OUT/$name > $OUT/$name.stdout.log 2>&1
    echo "$(date -u +%FT%TZ) done $name exit=$?" >> $LOG
   done
  done
 done
done
stop_server
echo "$(date -u +%FT%TZ) ALL DONE" >> $LOG
