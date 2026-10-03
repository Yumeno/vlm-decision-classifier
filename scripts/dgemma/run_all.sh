#!/bin/bash
# 全件評価の一連(vLLM 再起動直後=画像キャッシュ空の状態で seed0 を先に流す)。
# 必要な環境変数: VENV_EVAL、REPO_DIR(run_eval.sh と同じ)
D=$(cd "$(dirname "$0")" && pwd)
bash "$D/wait_vllm.sh" || exit 1
bash "$D/run_eval.sh" dgemma_own_seed0 0 dgemma_choice,dgemma_json "DiffusionGemma own server (keyed, yn, default instruction, aliasing, 16q/1 read, samples 1, seed 0) paired with plain JSON; vLLM restarted just before (empty prefix cache); no prime, warmup 0" || exit 1
bash "$D/run_eval.sh" dgemma_own_seed1 1 dgemma_choice "same as seed0 run but seed 1, choice only; images already in prefix cache (timing is warm)" || exit 1
bash "$D/run_eval.sh" dgemma_own_seed2 2 dgemma_choice "same as seed0 run but seed 2, choice only; images already in prefix cache (timing is warm)" || exit 1
