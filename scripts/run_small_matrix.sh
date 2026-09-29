#!/usr/bin/env bash
# 小型モデルの比較(S1〜S9)。最終の取り直し(final-runbook)と同じ手順、繰り返し1回。
REPO=${REPO:-$(pwd)}
LB=${LLAMACPP_DIR:?}
MD=${LMSC_DIR:?lmstudio-community のモデルフォルダ}
OUT=$REPO/results/small
mkdir -p $OUT; LOG=$OUT/progress.log; cd $REPO
stop_server() { P=$(netstat -ano | grep ':1235 ' | grep LISTENING | awk '{print $5}' | head -1); [ -n "$P" ] && taskkill //F //PID $P > /dev/null 2>&1; sleep 4; }
start_server() { # repo file mmproj alias name
  ($LB/patched/build/bin/Release/llama-server.exe -m $MD/$1/$2 --mmproj $MD/$1/$3 --alias $4 -ngl 99 -c 8192 -np 4 --kv-unified -sm none -mg 0 --port 1235 > $OUT/server_$5.log 2>&1 &)
  for i in $(seq 1 60); do curl -s http://127.0.0.1:1235/health | grep -q ok && return 0; sleep 2; done; return 1; }
while read sid alias repo f mp; do
  [ -z "$sid" ] && continue
  [ -f "$MD/$repo/$f" ] || { echo "$(date -u +%FT%TZ) MISSING $sid $f" >> $LOG; continue; }
  for v in a b; do
    name=${sid}_${v}
    [ -f $OUT/$name/summary.md ] && { echo "$(date -u +%FT%TZ) skip $name" >> $LOG; continue; }
    stop_server
    echo "$(date -u +%FT%TZ) before $name gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $LOG
    start_server $repo $f $mp $alias $name || { echo "$(date -u +%FT%TZ) SERVER FAIL $name" >> $LOG; continue; }
    echo "$(date -u +%FT%TZ) loaded $name gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ') apps3090: $(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | grep -c 0D:00)" >> $LOG
    if [ $v = a ]; then
      .venv/Scripts/python.exe -m classifier_demo probe --base-url http://127.0.0.1:1235/v1 --model $alias > $OUT/${sid}_probe.log 2>&1
      echo "$(date -u +%FT%TZ) probe $sid: $(grep -E 'logprobs present|thinking detected' $OUT/${sid}_probe.log | tr '\n' ' ')" >> $LOG
      VA="--modes choice,json,bundled --rank-threshold 0.5 --bundled-multi rank"
    else VA="--modes choice,bundled --confirm --bundled-multi yn"; fi
    .venv/Scripts/python.exe -m classifier_demo evaluate --manifest dataset/manifest.jsonl --base-url http://127.0.0.1:1235/v1 --model $alias $VA --warmup 1 --prime --axis-concurrency 1 --image-format jpeg --max-edge 1024 --dataset-version v1.0.0 --runtime-info doc/experiments/runtime/small-$sid.json --runtime-label $name --note "small-model comparison $name" --output-dir $OUT/$name > $OUT/$name.stdout.log 2>&1
    echo "$(date -u +%FT%TZ) done $name exit=$?" >> $LOG
  done
done <<'LIST'
S1 qwen3.5-4b-q4 Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf mmproj-Qwen3.5-4B-BF16.gguf
S2 qwen3.5-4b-q8 Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf mmproj-Qwen3.5-4B-BF16.gguf
S3 qwen3.5-2b-q4 Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf mmproj-Qwen3.5-2B-BF16.gguf
S4 qwen3.5-2b-q8 Qwen3.5-2B-GGUF Qwen3.5-2B-Q8_0.gguf mmproj-Qwen3.5-2B-BF16.gguf
S5 qwen3.5-0.8b-q4 Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf mmproj-Qwen3.5-0.8B-BF16.gguf
S6 qwen3.5-0.8b-q8 Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q8_0.gguf mmproj-Qwen3.5-0.8B-BF16.gguf
S7 gemma-4-e4b-q4 gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf mmproj-gemma-4-E4B-it-BF16.gguf
S8 gemma-4-e2b-q8 gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q8_0.gguf mmproj-gemma-4-E2B-it-BF16.gguf
S9 gemma-4-e2b-q4 gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf mmproj-gemma-4-E2B-it-BF16.gguf
LIST
stop_server
echo "$(date -u +%FT%TZ) ALL DONE" >> $LOG
