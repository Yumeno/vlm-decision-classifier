#!/bin/bash
# DiffusionGemma 4bit AWQ を 3090 だけで vLLM に載せる(issue #4)。vision_prefix_lm パッチ適用済みの venv を使う。
# 必要な環境変数:
#   VENV_VLLM  vLLM(ab5266769 + vision_prefix_lm パッチ)の venv のディレクトリ
#   MODEL_DIR  pixelkaiser/diffusiongemma-26B-A4B-it-AWQ-MLP-W4A16-G64-S32-L1024(rev a9557bd)のローカルコピー
#   GPU_UUID   RTX 3090 の UUID(nvidia-smi -L で出る GPU-xxxxxxxx-... )
# 任意: LOG_DIR(既定 ~/dgemma-logs)
set -u
: "${VENV_VLLM:?VENV_VLLM を指定}" "${MODEL_DIR:?MODEL_DIR を指定}" "${GPU_UUID:?GPU_UUID を指定}"
LOG_DIR=${LOG_DIR:-$HOME/dgemma-logs}; mkdir -p "$LOG_DIR"
source "$VENV_VLLM/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_UUID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
# 報告した全件評価は --gpu-memory-utilization 0.80(KV 2.9 GiB / 13,772 tokens)。
# 0.94(KV 6.26 GiB / 29,730 tokens)は追加検証(issue #12)用で、報告した実行には使っていない。
exec vllm serve "$MODEL_DIR" \
  --served-model-name dgemma \
  --host 127.0.0.1 --port 8000 \
  --diffusion-config '{"canvas_length": 256}' \
  --max-logprobs 32 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --reasoning-parser gemma4 \
  --override-generation-config '{"max_new_tokens": null}' \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --attention-backend TRITON_ATTN \
  --max-num-seqs 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  > "$LOG_DIR/vllm-serve.log" 2>&1
