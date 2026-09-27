# Phase 4 実行手順(E3/E4/E5: llama.cpp 画像キャッシュ改造の有無)

- 対象: Qwen3.5 9B(E1 と同じ GGUF・mmproj)、データセット v1.0.0、taxonomy 0.4.1。
- 比較するのは、**同じ上流コミット・同じビルド条件**で作った未改造版と改造版の `llama-server`。差は `doc/patches/llamacpp-mtmd-checkpoint.patch` の1行だけ。
  - E3 と E4 の比較 = パッチの効果。
  - E1b(LM Studio)と E3 の比較 = サーバーの違い。パッチの効果と混ぜない。
- 速度比較は RTX 3090 だけで行い、測定中は 3090 でほかの推論を動かさない(作者と合意、2026-09-27)。
- **GPU を使う前に作者に確認する**(CLAUDE.md)。

## 1. ビルド(未改造版・改造版で同じ手順)

置き場所はリポジトリの外(ビルドツリーが大きいため)。以下は `~/Desktop/llamacpp-vdc/` に置いた例。

```bash
# 同じコミットを2か所に取得する
git init vanilla
git -C vanilla remote add origin https://github.com/ggml-org/llama.cpp.git
git -C vanilla fetch --depth 1 origin f95b0d95394d5e311ba8228689972843178c5e28
git -C vanilla checkout FETCH_HEAD
# patched も同様に取得してから、パッチを当てる
git -C patched apply <このリポジトリ>/doc/patches/llamacpp-mtmd-checkpoint.patch
```

```powershell
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$cuda  = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_PATH = $cuda
$env:PATH = "$cuda\bin;" + $env:PATH
& $cmake -B build -G "Visual Studio 17 2022" -A x64 -T "cuda=$cuda" `
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" `
    -DCMAKE_CUDA_COMPILER="$cuda\bin\nvcc.exe" -DCUDAToolkit_ROOT="$cuda" `
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
& $cmake --build build --config Release --target llama-server --parallel 8
```

- 古い CUDA(12.1 など)が PATH の先頭にあると失敗するので、`-T cuda=` と `-DCMAKE_CUDA_COMPILER` の両方で 12.8 に固定する。
- 実績(2026-09-27): CUDA 12.8.93、MSVC 19.44.35228.0、VS 2022 BuildTools。`llama-server --version` は `0.4.1-dev (build 1, commit f95b0d9)`。

## 2. 起動(1台ずつ。切り替えるときは前のサーバーを終了してから)

```
llama-server -m <models>/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf --mmproj <models>/lmstudio-community/Qwen3.5-9B-GGUF/mmproj-Qwen3.5-9B-BF16.gguf -ngl 99 -c 8192 -np 4 -sm none -mg 0 --port 1235
```

- `-c 8192 -np 4` は E1/E1b の LM Studio の設定(コンテキスト 8192、並列数 4)に合わせた。KV はスロット共有(`--kv-unified` が既定で有効)。
- `-sm none -mg 0` で RTX 3090 だけに載せる。**`-mg` の番号は llama.cpp(CUDA)の並び**で、`nvidia-smi` の番号とは違う(この PC では CUDA0 = RTX 3090、CUDA1 = RTX 4060 Ti。`llama-server --list-devices` で確認)。
- 起動後に `nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv` で 3090(bus 0D)だけ使用量が増えたことを確かめ、実行環境 JSON に書く。

## 3. 測定(サーバーごとに)

1. probe: `.venv/Scripts/python.exe -m classifier_demo probe --base-url http://127.0.0.1:1235/v1 --model <id>`(thinking が出ないこと、logprobs があること)
2. E5(小ベンチマーク): `.venv/Scripts/python.exe scripts/benchmark_cache.py --base-url http://127.0.0.1:1235/v1 --model <id> --label vanilla --output results/E5/vanilla.json`(改造版は `--label patched`)
3. E3/E4(全件、選択式のみ): `.venv/Scripts/python.exe -m classifier_demo evaluate --manifest dataset/manifest.jsonl --base-url http://127.0.0.1:1235/v1 --model <id> --modes choice --warmup 1 --dataset-version v1.0.0 --runtime-info doc/experiments/runtime/qwen-llamaserver-vanilla.json --runtime-label qwen-llamaserver-vanilla --output-dir results/E3_qwen-llamaserver-vanilla`(改造版は `patched`、`results/E4_…`)
4. サーバーを終了し、3090 に残っているプロセスがないことを確認する。

## 4. 記録

- `run.json`・`cases.csv`・`summary.md`(E3/E4)と E5 の JSON を `doc/experiments/` にコピーする(個人パスがないことを確認する)。
- E3 と E4 のケース別の予測の差(判定が変わったケース)を確認し、変わったものは個別に調べる。改善の数値だけを記事に載せない。
