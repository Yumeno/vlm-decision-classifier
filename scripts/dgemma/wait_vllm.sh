#!/bin/bash
# vLLM(:8000)が起動するまで待つ。最大500秒。プロセスが死んだらログ末尾を出して失敗。
LOG_DIR=${LOG_DIR:-$HOME/dgemma-logs}
for i in $(seq 1 100); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = "200" ]; then echo "READY after $((i*5))s"; exit 0; fi
  pgrep -f 'vllm serve' >/dev/null || { echo DIED; tail -5 "$LOG_DIR/vllm-serve.log"; exit 1; }
  sleep 5
done
echo TIMEOUT
