# 実験の再現

このリポジトリで行った実験(E1〜E10、F1〜F3、S1〜S9)を再現するための手順と、結果の表、開発の経緯をまとめています。使い方は [`README.md`](../README.md)、CLI のオプションは [`cli.md`](cli.md) を参照してください。実験の条件・結果・限界の一次資料は [`experiments/report.md`](experiments/report.md) です。

## どれが本表か

実験は順に積み重ねたため、古い測定と新しい測定が混ざっています。読むときは次の順で見てください。

| 見たいもの | 見る場所 |
|---|---|
| **本表**(方式ごとの時間・精度。Qwen 9B / Gemma 12B と小型モデル) | `reprime`: [`experiments/reprime/summary.md`](experiments/reprime/summary.md)。F1〜F3・S1〜S9 を、準備なし・準備ありの2条件で各1回測り直したもの |
| 出力形式の崩れやすさ(ノイズ画像、実データでの `json_schema`) | `E10` / `E10b`: [`experiments/e10/summary.md`](experiments/e10/summary.md) |
| 経緯(古い測定。比較に使わない部分がある) | E1〜E5(LM Studio / llama-server、画像キャッシュ改造)、E7・E9(束ね質問、ホットロード)、旧 F1〜F3(`final/`)・旧 S1〜S9(`small/`)。旧 F1〜F3・S1〜S9 は `--prime` が通常JSONだけに効いていなかった測定で、**時間の比較は公平ではない**(精度は同じ) |

実験の一覧とコマンドは、次の各節にあります。

## 現状

Phase 5(レポート・公開準備)まで進み、E1〜E5(Qwen/Gemma、通常JSONとの比較、llama.cpp画像キャッシュ改造の比較)、E9(ホットロード・確認のオン・オフ・軸の並列)、E7(束ね質問、送信画像の形式と長辺)、E7e(束ね質問の複数選択)の実験が済んでいます。その後、記事の本表として、1つのコミット・同じ条件で全条件を取り直し(最終の取り直し F1〜F3、各3回)、さらに小型モデル(Qwen3.5 4B/2B/0.8B、Gemma 4 E4B/E2B。S1〜S9)を比べました。`--prime` が通常JSONだけに効いていなかった不公平(PR #20で修正)が後で分かったため、F1〜F3とS1〜S9は修正後のコードで「準備なし・準備あり」の2条件に測り直しています(各1回。本表はこちら)。実測結果は [`experiments/report.md`](experiments/report.md) にあります(本表は §1.2、E1〜E9・E7は経緯として §4)。

## 前提

- 前提は、データセット v1.0.0（`dataset/manifest.jsonl`）、taxonomy 0.4.1（現在の `taxonomy/default.yaml`）、リポジトリのルートから実行、`--dataset-version v1.0.0`。結果は `results/<名前>/` に出し、公開用のものを `doc/experiments/` に写しています。
- **E1/E1-J/E2/E2-J は taxonomy 0.4.0** で測ったので、再現には当時の `taxonomy/default.yaml`（git履歴。SHA256 `ec7b2a2f…` で確認）を `--taxonomy` に渡す必要があります。現在の `default.yaml` は 0.4.1 です（E1b/E2b と同じ）。
- 再現するときは、GPUに他のプロセスがいない状態が望ましいです（速度が揺れるため）。
- E1b以降は、RTX 3090に他のプロセスがない状態で測りました（E1/E1-J/E2/E2-Jは別プロジェクトのllama-serverが常駐した状態。report §2）。

## 実験の一覧

| 実験ID | 何を測ったか | コマンドまたは手順書 | 結果フォルダ |
|---|---|---|---|
| E1 / E1-J | Qwen3.5 9B（LM Studio）の選択式と通常JSON。taxonomy 0.4.0 | [`phase3-runbook.md`](experiments/phase3-runbook.md) | `E1_qwen-lmstudio/` |
| E2 / E2-J | Gemma 4 12B（LM Studio）の選択式と通常JSON。taxonomy 0.4.0 | 同上（Gemmaの引数はrunbook） | `E2_gemma-lmstudio/` |
| E1b / E2b | E1/E2の再評価（taxonomy 0.4.1）。`--runtime-label qwen-lmstudio-tax041`（Gemmaは `gemma-lmstudio-tax041`）、`--output-dir results/E1b_qwen-lmstudio`（`E2b_gemma-lmstudio`） | 同上（`--taxonomy` は既定の0.4.1） | `E1b_qwen-lmstudio/`、`E2b_gemma-lmstudio/` |
| E3 / E4 | llama-server（未改造/改造）でQwenの選択式（全件）。パッチの効果 | [`phase4-runbook.md`](experiments/phase4-runbook.md) §3 | `E3_qwen-llamaserver-vanilla/`、`E4_qwen-llamaserver-patched/` |
| E5 | 同じ画像で質問だけ変える／画像切替A→B→Aの小ベンチマーク | 下記 | `E5_qwen-llamaserver/`（`vanilla.json`/`patched.json`） |
| E9a〜E9e | ホットロード、確認のオン・オフ、軸の並列（改造版llama-server） | 下記 | `E9a_…`〜`E9e_…` |
| E7a〜E7e | 束ね質問、送信画像の形式・長辺、束ね質問の複数選択（Y/N欄） | 下記 | `E7a_…`〜`E7e_…` |
| F1〜F3（最終の取り直し、旧測定） | Qwen改造・Qwen未改造・Gemma改造 × 長辺1024/768 × 回a/b × 3回（36回）。`--prime` がJSONだけに効いていなかった測定（時間比較は不公平、精度は同じ） | [`final-runbook.md`](experiments/final-runbook.md)、`scripts/run_final_matrix.sh` | `final/` |
| F1〜F3（測り直し。本表） | 同じ条件を、準備の不公平の修正後に準備なし・準備ありで各1回（24実行） | 下記、`scripts/run_final_matrix.sh` | `reprime/final_noprime/`、`reprime/final_prime/` |
| S1〜S9（小型モデル、旧測定） | 小型モデル9種の比較（F1〜F3と同じ手順、繰り返し1回。準備の不公平あり） | `scripts/run_small_matrix.sh` | `small/` |
| S1〜S9（測り直し。本表） | 同じ条件を準備なし・準備ありで各1回（36実行） | 下記、`scripts/run_small_matrix.sh` | `reprime/small_noprime/`、`reprime/small_prime/` |

## 各実験のコマンド

コマンドは、リポジトリのルートから実行します。

### E1 / E1-J / E2 / E2-J（Phase 3。LM Studio）

コマンドは [`phase3-runbook.md`](experiments/phase3-runbook.md) §手順5 のとおりです。Qwenの例（`--confirm --image-format png` が必要）:

```powershell
.venv\Scripts\python.exe -m classifier_demo evaluate --manifest dataset\manifest.jsonl --model qwen3.5-9b --modes choice,json --warmup 1 --confirm --image-format png --dataset-version v1.0.0 --runtime-info doc\experiments\runtime\qwen-lmstudio.json --runtime-label qwen-lmstudio --output-dir results\E1_qwen-lmstudio
```

Gemmaは `--model gemma-4-12b-it --runtime-info doc\experiments\runtime\gemma-lmstudio.json --runtime-label gemma-lmstudio --output-dir results\E2_gemma-lmstudio`。E1b/E2b は同じコマンドで、上表の `--runtime-label` と `--output-dir` にします。

### E3 / E4（Phase 4。llama-server 未改造/改造）

未改造版と改造版を同じコミット・同じビルド手順で作り（[`phase4-runbook.md`](experiments/phase4-runbook.md) §1）、1台ずつ起動して測ります（§2〜§3）。`--confirm --image-format png` が必要:

```powershell
.venv\Scripts\python.exe -m classifier_demo evaluate --manifest dataset\manifest.jsonl --base-url http://127.0.0.1:1235/v1 --model qwen3.5-9b --modes choice --warmup 1 --confirm --image-format png --dataset-version v1.0.0 --runtime-info doc\experiments\runtime\qwen-llamaserver-vanilla.json --runtime-label qwen-llamaserver-vanilla --output-dir results\E3_qwen-llamaserver-vanilla
```

改造版（E4）は `--runtime-info doc\experiments\runtime\qwen-llamaserver-patched.json --runtime-label qwen-llamaserver-patched --output-dir results\E4_qwen-llamaserver-patched`。

### E5（画像キャッシュ改造の小ベンチマーク）

サーバーごとに1回ずつ（PNGで測りました）:

```powershell
.venv\Scripts\python.exe scripts\benchmark_cache.py --base-url http://127.0.0.1:1235/v1 --model qwen3.5-9b --label vanilla --image-format png --output results\E5\vanilla.json
```

改造版は `--label patched --output results\E5\patched.json`。

### E9a〜E9e（ホットロード・確認・軸の並列。改造版llama-server）

サーバーは [README の導入](../README.md#3-サーバーを起動する)の起動引数（`-ngl 99 -c 8192 -np 4 --kv-unified -sm none -mg 0`。E9eだけ `--no-cache-idle-slots` を追加）で、**条件ごとに再起動**します。共通部分（**`--image-format png` が必要**）:

```powershell
.venv\Scripts\python.exe -m classifier_demo evaluate --manifest dataset\manifest.jsonl --base-url http://127.0.0.1:1235/v1 --model qwen3.5-9b --modes choice,json --warmup 1 --prime --image-format png --dataset-version v1.0.0 --runtime-info doc\experiments\runtime\qwen-llamaserver-patched.json <条件> --output-dir results\<名前>
```

| 実験 | `<条件>` |
|---|---|
| E9a（確認オン・逐次） | `--confirm --axis-concurrency 1` |
| E9b（確認オン・並列4、準備1本）、E9b2（同・準備4本同時）、E9e（同・準備4本同時。サーバーを `--no-cache-idle-slots` で起動） | `--confirm --axis-concurrency 4` |
| E9c（確認オフ・逐次） | `--axis-concurrency 1 --rank-threshold 0.5` |

（E9b と E9b2 は、後者だけ並列準備の修正後のコミット `acfe504` で測っています。準備の本数は `--prime` と `--axis-concurrency N` の併用で N 本になります。）

### E7a〜E7e（束ね質問・送信画像・Y/N欄。改造版llama-server）

確認オフ（閾値0.5）。サーバーは条件ごとに再起動。共通部分:

```powershell
.venv\Scripts\python.exe -m classifier_demo evaluate --manifest dataset\manifest.jsonl --base-url http://127.0.0.1:1235/v1 --model qwen3.5-9b --modes choice,json,bundled --warmup 1 --prime --axis-concurrency 1 --rank-threshold 0.5 --dataset-version v1.0.0 --runtime-info doc\experiments\runtime\qwen-llamaserver-patched.json <条件> --output-dir results\<名前>
```

| 実験 | `<条件>` | 備考 |
|---|---|---|
| E7b（番号付き・PNG1024） | `--image-format png --max-edge 1024` | |
| E7c（JPEG1024） | `--image-format jpeg --max-edge 1024` | |
| E7d（JPEG768） | `--image-format jpeg --max-edge 768` | |
| E7e（束ね質問の複数選択をY/N欄に） | `--image-format jpeg --max-edge 1024 --bundled-multi yn` | |
| E7a（記号のみ・空白区切り、PNG1024） | コミット `cc427e6` をチェックアウトして実行 | 束ね質問の回答形式を、のちに番号付きへ変えた（形式のずれが起きたため）ので、現在のコードでは再現できない。結果は比較には使っていない（report §4.7） |

測定時のコミット・時刻・条件は、各フォルダの `run.json`（`tool_commit`、`started`、`note` など）と [`report.md`](experiments/report.md) §2、[`doc/worklog.md`](worklog.md) にあります。
実験記録(`run.json` の `tool_commit` など)や worklog に残るコミット ID は、非公開の開発リポジトリのものです。この公開リポジトリでの対応するコミットは [`commit-map.tsv`](commit-map.tsv)(左が開発リポジトリ、右が公開リポジトリ)で引けます。

### F1〜F3（最終の取り直し）

旧測定（`--prime` の不公平あり。下記の測り直しが本表）です。手順の詳細は [`final-runbook.md`](experiments/final-runbook.md)。3条件（F1 Qwen改造、F2 Qwen未改造、F3 Gemma改造）× 長辺2 × 回2 × 繰り返し3 = 36回を、繰り返しを一番外側にして回します（回ごとにサーバーを再起動）。llama.cppの `vanilla/`（未改造）と `patched/`（改造）をビルドして置いたフォルダ、Qwen・GemmaのGGUFフォルダを環境変数で渡します。Windows の Git Bash で、リポジトリのルートから:

```bash
LLAMACPP_DIR=<llama.cpp の置き場所> QWEN_DIR=<Qwen の GGUF のフォルダ> GEMMA_DIR=<Gemma の GGUF のフォルダ> bash scripts/run_final_matrix.sh
```

集計（各回の `results/final/<回>/` の `cases.csv`・`run.json` から、時間の平均±標準偏差・精度・失敗・繰り返しの揺れを出す）:

```powershell
.venv\Scripts\python.exe scripts\aggregate_final.py --input results\final --output doc\experiments\final
```

出力は `doc/experiments/final/summary.md` と `runs/`（各回の生データ）、`progress.log`。実行環境は `doc/experiments/runtime/final-*.json`（`--runtime-info` に渡したもの）です。

#### F1〜F3の測り直し（準備なし・準備あり。本表）

PR #20 で `--prime` を方式ごとの準備にしたあと、同じ手順で、準備なし（`PRIME=0`）と準備あり（`PRIME=1`）を1回ずつ測ります（`REPS=1`。絞り込みの `CONDITIONS`・`EDGES` は既定で全条件・長辺1024と768）。出力先は `OUT` で切り替えます（Git Bash、リポジトリのルートから）:

```bash
# 準備なし（画像の読み込みを判定時間に含む）
PRIME=0 REPS=1 OUT=results/reprime/noprime LLAMACPP_DIR=<llama.cpp の置き場所> QWEN_DIR=<Qwen の GGUF のフォルダ> GEMMA_DIR=<Gemma の GGUF のフォルダ> bash scripts/run_final_matrix.sh
# 準備あり（--prime。準備の時間は prime_ms に別記録）
PRIME=1 REPS=1 OUT=results/reprime/prime LLAMACPP_DIR=<llama.cpp の置き場所> QWEN_DIR=<Qwen の GGUF のフォルダ> GEMMA_DIR=<Gemma の GGUF のフォルダ> bash scripts/run_final_matrix.sh
```

各実行は `results/reprime/<noprime|prime>/<F1|F2|F3>_e<長辺>_<a|b>_r1/`。集計は `scripts/aggregate_final.py`（`--input` に上の出力先、`--output` に任意のフォルダ）で出せます。結果は `doc/experiments/reprime/summary.md` と、`final_noprime/`・`final_prime/` の `runs/`（`run.json`・`cases.csv`・`summary.md`）、`progress.log` です（`results/` の生出力そのものはコミットしていません）。

### S1〜S9（小型モデル）

旧測定（`--prime` の不公平あり）の手順です。F1〜F3と同じ手順（改造版llama-server、JPEG・長辺1024、`--prime`、回a/b）を、繰り返し1回で、小型モデル9種について回します。モデルは `lmstudio-community` のGGUF（Q4_K_M / Q8_0）とmmproj BF16で、フォルダ構成はスクリプト内の一覧（`LIST`）を参照してください。

```bash
LLAMACPP_DIR=<llama.cpp の置き場所> LMSC_DIR=<lmstudio-community のモデルフォルダ> bash scripts/run_small_matrix.sh
```

出力は `doc/experiments/small/summary.md`（表）と `runs/`。実行環境は `doc/experiments/runtime/small-S*.json`。

測り直し（本表）は、`PRIME` と `OUT` を環境変数で切り替えて、準備なし・準備ありを1回ずつ流します:

```bash
PRIME=0 OUT=results/reprime/small_noprime LLAMACPP_DIR=<llama.cpp の置き場所> LMSC_DIR=<lmstudio-community のモデルフォルダ> bash scripts/run_small_matrix.sh
PRIME=1 OUT=results/reprime/small_prime LLAMACPP_DIR=<llama.cpp の置き場所> LMSC_DIR=<lmstudio-community のモデルフォルダ> bash scripts/run_small_matrix.sh
```

結果は `doc/experiments/reprime/summary.md` と `small_noprime/`・`small_prime/`（`runs/`・`progress.log`・`S*_probe.log`）です。

### E10（出力形式の頑健性ストレステスト。追加課題）

白色ノイズ画像N枚（seed 0..N-1、1024x1024、決定的に生成）で、選択式・通常JSON・`json_schema`（制約付きデコード）・束ね質問（rank / yn）の形式不正率と時間を比べます。正解は使わず、実画像の失敗率ではなく方式間の頑健性の比較です（`evaluate --modes` にも `json_schema` を指定できます）。`--warmup 1 --prime` で方式ごとに自分の先頭の準備を送ります。

```bash
LLAMACPP_DIR=<llama.cpp の置き場所> LMSC_DIR=<lmstudio-community のモデルフォルダ> N=100 MODELS="S3 S4 S5 S6 S9 F1" bash scripts/run_e10.sh
```

結果は `results/e10/<モデル>/`（`cases.csv`・`run.json`・`summary.md`）。記録した結果は [`doc/experiments/e10/`](experiments/e10/summary.md)（`noise/`）にあります。

**E10b（実データ31枚で通常JSONと `json_schema` を比較）**:

```bash
LLAMACPP_DIR=<llama.cpp の置き場所> LMSC_DIR=<lmstudio-community のモデルフォルダ> bash scripts/run_e10b.sh
```

結果は `results/e10b/<モデル>/`。記録した結果は `doc/experiments/e10/dataset/`。記録した実行は、同じコマンドを個人パス直書きにした使い捨て版で回しました（コマンドの中身は同じ。`run.json` の `tool_commit` を参照）。

### DiffusionGemma（issue #4。vLLM、WSL 内で実行）

`google/diffusiongemma-26B-A4B-it` の 4bit AWQ 版（`pixelkaiser/diffusiongemma-26B-A4B-it-AWQ-MLP-W4A16-G64-S32-L1024`、rev `a9557bd`）を、専用の WSL2 ディストロの vLLM で `dgemma` として serve（`:8000`）し、その前段に本リポジトリの `dgemma-server`（`:8012`）を置きます。どちらも WSL 内の `127.0.0.1` に bind し Windows から到達できないため、**評価は WSL 内で実行**します（リポジトリは WSL 内にクローンするか、Windows 側のものを WSL のマウント経由で参照）。条件は [`experiments/runtime/dgemma-vllm.json`](experiments/runtime/dgemma-vllm.json)、結果は [`experiments/dgemma/README.md`](experiments/dgemma/README.md)。サーバーの起動は利用者が行います。

起動の順序:

1. **WSL2 ディストロ**: 専用のディストロ（Ubuntu 24.04 相当）に GPU 対応の PyTorch 環境と、vLLM（commit `ab5266769e702434a0968d47319d0731d8ccac35`）を venv で用意します（RTX 3090 単独、24GB）。vLLM のインストール手順は上流に従います。
2. **画像品質パッチ**: vLLM の画像トークンは因果で処理されるため、openjev の `vision_prefix_lm` パッチを、venv 内の vLLM に適用します（パッチ本体: https://github.com/razorback16/openjev/blob/dcd20947b5ddad5be4a8f5aed6aa6dd245653823/docker/patches/vision_prefix_lm.py 。本リポジトリには含めない。適用手順はそのファイルの説明に従う）。適用した事実は `dgemma-vllm.json` の `vision_prefix_lm patch` に記録します。
3. **vLLM** を `scripts/dgemma/serve_vllm.sh` で起動（環境変数 `VENV_VLLM`=パッチ適用済み venv、`MODEL_DIR`=モデルのローカルコピー、`GPU_UUID`=3090 の UUID（`nvidia-smi -L`））。全フラグ: `--served-model-name dgemma --host 127.0.0.1 --port 8000 --diffusion-config '{"canvas_length": 256}' --max-logprobs 32 --limit-mm-per-prompt '{"image": 4, "video": 0}' --reasoning-parser gemma4 --override-generation-config '{"max_new_tokens": null}' --enable-prefix-caching --enable-prompt-tokens-details --async-scheduling --attention-backend TRITON_ATTN --max-num-seqs 2 --max-model-len 4096 --gpu-memory-utilization 0.80`（環境変数は `CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=$GPU_UUID`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`、`HF_HUB_OFFLINE=1`、`VLLM_NO_USAGE_STATS=1`）。報告した実行は 0.80（KV 2.9 GiB / 13,772 tokens。画像プロンプト約10枚分）。0.94（KV 6.26 GiB / 29,730 tokens）には後から上げた（追加検証 [issue #12](https://github.com/Yumeno/vlm-decision-classifier/issues/12) 用）ので、報告した実行には使っていません。`scripts/dgemma/wait_vllm.sh` で起動完了を待てます。
4. **dgemma-server** を `scripts/dgemma/serve_dgemma.sh` で起動（`VENV_EVAL`=本リポジトリを入れた venv、`REPO_DIR`=リポジトリのルート。`--vllm http://127.0.0.1:8000 --model dgemma --port 8012`、canvas は既定の 256）。
5. **evaluate** を実行（下記）。

**推奨設定（issue #12）**: PR #13 の報告した実行は `GPU_MEM_UTIL=0.80`・画像トークン予算 280（既定）で測った。その後の測定（issue #12 のコメント）では、`GPU_MEM_UTIL=0.94` は画像キャッシュなしのリクエストが約8倍遅く（3,734 ms 対 約500 ms。GPU の空きが 0.75 GB）、`0.88` なら速度を保てた（約500 ms、KV 22,891 トークン）ので **`GPU_MEM_UTIL=0.88`** を推奨する。画像トークンはサーバー全体の `--mm-processor-kwargs`（`serve_vllm.sh` の `MST`）で設定し、**`MST=140`** は精度が 280 とほぼ同じで、キャッシュなしの画像で約20%速かった（70 は outfit・color が落ちる）。リクエスト単位の `mm_processor_kwargs`（クライアントの `--dgemma-max-soft-tokens`）は、vLLM が `--trust-request-mm-kwargs` なしでは拒否する。例: `GPU_MEM_UTIL=0.88 MST=140 bash scripts/dgemma/serve_vllm.sh`（未指定の既定は 0.80 / 予算指定なし）。 「その他」系の質問文は `--dgemma-catchall-style criteria` を付けると、キャラの完全一致が 27 → 29〜30/31 になり、「その他」のラベル質量の失敗も出なくなった（既定は従来の文面。結果は `experiments/dgemma/followup-12.md` の「「その他」の質問文」）。迷ったときだけ読み増す方式（任意）は、`--dgemma-adaptive-threshold 0.87 --dgemma-adaptive-max 3` を `classify` / `evaluate` に付けて有効にする（`run_eval.sh <label> <seed> <modes> <note> --dgemma-adaptive-threshold 0.87 --dgemma-adaptive-max 3` のように、5番目以降の引数は evaluate にそのまま渡る）。#12 の測定は、vLLM を `GPU_MEM_UTIL=0.88 MST=140` で起動した状態で行った。結果は [`experiments/dgemma/followup-12.md`](experiments/dgemma/followup-12.md)。

```bash
# WSL 内(Python 3.12)。評価用 venv を作る
python3.12 -m venv ~/dgemma-eval-venv && ~/dgemma-eval-venv/bin/pip install -e .
export VENV_EVAL=~/dgemma-eval-venv REPO_DIR=$(pwd)
export VENV_VLLM=<vLLM の venv> MODEL_DIR=<モデルのディレクトリ> GPU_UUID=<3090 の UUID>
# 別ターミナルで vLLM、dgemma-server の順に起動
bash scripts/dgemma/serve_vllm.sh
bash scripts/dgemma/wait_vllm.sh && bash scripts/dgemma/serve_dgemma.sh
# 単一ケースの確認(probe 相当)
~/dgemma-eval-venv/bin/python -m classifier_demo classify dataset/images/M01.png --mode dgemma_choice --model dgemma --base-url http://127.0.0.1:8000/v1 --dgemma-url http://127.0.0.1:8012 --dgemma-yn-style yn
# 全件評価(vLLM 再起動直後に seed 0(choice と json)、続けて seed 1・2(choice のみ))
bash scripts/dgemma/run_all.sh
```

`run_eval.sh <label> <seed> <modes> <note>` が1回分の evaluate で、中身は `--dgemma-url http://127.0.0.1:8012 --dgemma-samples 1 --dgemma-seed N --dgemma-yn-style yn --modes dgemma_choice[,dgemma_json] --warmup 0 --image-format jpeg --max-edge 1024 --dataset-version v1.0.0`（`--prime` なし）です。ログは `LOG_DIR`（既定 `~/dgemma-logs`）、出力は `results/<label>/`。`--warmup 0` なので最初のケースはコールドスタートの時間（画像処理込み）を含みます。seed0 は vLLM の再起動直後（画像のプレフィックスキャッシュが空）で測りました。記録した実行は `doc/experiments/dgemma/`（`seed0` `seed1` `seed2`）です。

方式の中身と記録は [`cli.md`](cli.md) の「DiffusionGemma のモード」。`--warmup 0` なので最初のケースはコールドスタートの時間を含みます（`cases.csv` で確認）。

### OpenJev(llama-server の `/v1/systemone`。追加実験)

上流 llama.cpp の PR #29818 で入った `/v1/systemone` を、OpenJev の GGUF で使います。**重みは CC BY-NC 4.0(非商用)で、本リポジトリには含めません**。結果は [`experiments/openjev/README.md`](experiments/openjev/README.md)。

1. llama.cpp を b11447(commit `da263e7275dfbaeefcd61504eaa4fd5247540e11`)でビルドする。設定は [`experiments/phase4-runbook.md`](experiments/phase4-runbook.md) と同じ(CUDA 12.8、`CMAKE_CUDA_ARCHITECTURES` 86;89)。この実験は未改造(パッチなし)で動かした。
2. `ggml-org/OpenJev-GGUF` の rev `10840f375658dea7afc5ff4711127bca8218b560` から `OpenJev-Q4_K_M.gguf` と `mmproj-OpenJev-Q8_0.gguf` を `<モデルのディレクトリ>` にダウンロードする(sha256 は `experiments/openjev/README.md`)。
3. 起動(RTX 3090 のみ):

```
<llama-server> -m <モデルのディレクトリ>/OpenJev-Q4_K_M.gguf --mmproj <モデルのディレクトリ>/mmproj-OpenJev-Q8_0.gguf --host 127.0.0.1 --port <port> -ngl 99 -c 16384 -np 16 --kv-unified -sm none -mg 0
```

4. 評価(既存の `dgemma_choice` モードがそのまま使える):

```
.venv\Scripts\python.exe -m classifier_demo evaluate --modes dgemma_choice --dgemma-url http://127.0.0.1:<port> --model openjev --warmup 0 --image-format jpeg --max-edge 1024 --dataset-version v1.0.0 [--dgemma-catchall-style criteria] --output-dir results/openjev_<label>
```

## 実験の比較条件

| 主比較 | 条件 | 測定値 |
|---|---|---|
| 選択式 vs 通常JSON | 同じモデル、画像、分類体系、メタデータ条件 | ケース別正誤、軸別精度、処理時間、失敗・再試行 |
| Qwen vs Gemma | 共通のSFW画像・分類基準 | 精度、応答形式、時間。モデル・量子化・ロード設定も記録 |
| Qwenのllama.cpp未改造 vs 改造 | 同じ上流コミット、GGUF、mmproj、画像、質問 | 時間、判定結果の差、画像切替時の混線 |
| メタデータあり vs なし | 同じ元画像の派生ケース | 生成情報の寄与と、容姿判定との食い違い |

JSONベースラインの主比較は**分類フィールドだけ**を生成します。モデルロード時間とウォームアップは画像当たりの分類時間から分け、推論が失敗したケースも分母に残します。候補間の相対スコアは、実世界の正答確率を意味しません。

## 結果の要約(旧 README から移動)

結果フォルダは `doc/experiments/` の下です(上の一覧の「結果フォルダ」は、そこからの相対名)。

元画像31件の小標本です。数値は方式の挙動を見るための材料で、1〜2件の差は誤差の範囲です。一次資料は [`doc/experiments/report.md`](experiments/report.md) と各フォルダの `summary.md`・`run.json` です。

**測り直し（F1〜F3。本表）**: 記事の本表として、1つのコミット・同じ条件で取り直した F1〜F3（Qwen3.5 9B 改造版・未改造版、Gemma 4 12B 改造版。長辺1024と768）を、`--prime` の不公平（通常JSONにだけ準備が効いていなかった。PR #20で修正）を直したあと、**準備なし**（初めて見る画像。画像の読み込みを判定時間に含む）と**準備あり**（画像を読み込み済み）の2条件で測り直しました（各1回、コミット `07086c0`）。失敗は、1軸ずつ・束ね質問で0件、通常JSONの形式不正がQwenで1件・Gemmaで2件でした（旧測定と同じ）。判定は準備の有無や旧測定とのあいだで数件変わったセルがあります（F1・F2は準備の有無で同一、F3の1軸ずつで最大1軸あたり1件）。

> 旧測定（最終の取り直し。3回）では、`--prime` が選択式のsystem文で準備していたため、通常JSONの判定時間にだけ画像の読み込みが含まれていました。「束ね質問はJSONの約半分」という当初の比較はこの不公平による見かけの差で、公平にすると2〜3割の短縮です（report §1.2・§6）。精度は影響を受けません。

判定時間（Qwen3.5 9B・改造版・長辺1024。「準備なし / 準備あり」）と精度（元画像31件、%。服装・キャラは完全一致。準備なしの値。F1 長辺1024では準備ありでも同じ）:

| 方式 | 判定時間（準備なし / 準備あり） | リクエスト数 | image_type | art_style | color | subject | situation | outfit | character |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 束ね質問（順位付け） | 1188 / 713 ms | 1.00 | 90.3 | 90.3 | 100.0 | 96.8 | 90.3 | 93.5 | 87.1 |
| 束ね質問（Y/N欄） | 1619 / 1278 ms | 1.00 | 96.8 | 87.1 | 100.0 | 96.8 | 90.3 | 83.9 | 100.0 |
| 1軸ずつ（確認オフ） | 1890 / 1285 ms | 7.00 | 93.5 | 77.4 | 100.0 | 96.8 | 83.9 | 90.3 | 90.3 |
| 1軸ずつ（確認オン） | 2671 / 1998 ms | 11.29 | 93.5 | 77.4 | 100.0 | 96.8 | 83.9 | 77.4 | 93.5 |
| 通常JSON | 1624 / 963 ms | 1.06 | 96.8 | 77.4 | 93.5 | 93.5 | 83.9 | 90.3 | 87.1 |

別条件の判定時間（長辺1024、準備なし / 準備あり。束ね rank / 束ね Y/N / 1軸ずつ 確認オフ / 確認オン / JSON、ms）: F2 Qwen未改造 1384 / 1285、1899 / 1864、5163 / 5124、8262 / 8214、1609 / 1546。F3 Gemma改造 1579 / 1203、2223 / 1964、3003 / 2740、3884 / 3499、2087 / 1713。長辺768、Gemmaの精度、他の全条件は [`report.md`](experiments/report.md) §1.2 と [`doc/experiments/reprime/summary.md`](experiments/reprime/summary.md)。旧測定は [`doc/experiments/final/summary.md`](experiments/final/summary.md)。

**小型モデル（S1〜S9。準備なし・準備あり各1回）**: 1軸ずつの選択式は小さくしても単一選択の精度がほとんど落ちません（0.8Bでも平均86.5〜87.1%）。通常JSONは小型ほど形式不正が増えて崩れます（Qwen 2B Q4で9件）。束ね質問は、モデルが回答の形式を守れるかに成否が依存し、Qwen 4Bはほぼ全件が形式不正でした（中身は1軸ずつとほぼ同じ答え）。Y/N欄（16欄）は小型でほぼ崩れます。時間はJSONとの比で、1軸ずつ（確認オフ）が0.70〜1.29倍（同程度）、束ね（順位付け）が0.34〜0.65倍（ただしQwen 4Bは失敗した回答の時間）でした。境目ははっきりせず、1回の測定です（[`report.md`](experiments/report.md) §1.3）。

**旧・経緯の数値（E1b/E2b。taxonomy 0.4.1、確認オン・PNG1024、選択式 / 通常JSON、%。服装・キャラは完全一致）**:

| 軸 | Qwen3.5 9B | Gemma 4 12B |
|---|---|---|
| image_type | 96.8 / 96.8 | 100.0 / 90.3 |
| art_style | 77.4 / 77.4 | 83.9 / 80.6 |
| color | 100.0 / 93.5 | 100.0 / 93.5 |
| subject | 96.8 / 93.5 | 96.8 / 90.3 |
| situation | 80.6 / 83.9 | 80.6 / 67.7 |
| outfit | 74.2 / 90.3 | 71.0 / 83.9 |
| character | 93.5 / 87.1 | 93.5 / 93.5 |

E1b/E2b・E3/E4の処理時間（Qwen、選択式1画像あたり。E4以外はPNG1024、確認オン）: LM Studio 9.3秒（E1b）→ llama-server 未改造 8.7秒（E3）→ 改造 3.1秒（E4）。判定はE3とE4で全件一致。

**主な観察**:
- 束ね質問（順位付け）は通常JSONより速い（F1 長辺1024で、準備なし1.19秒 vs 1.62秒、準備あり0.71秒 vs 0.96秒。全12条件で14〜33%短縮）。未改造サーバー（F2）でも束ね質問はJSONより速い（準備なし1.38秒 vs 1.61秒）が、1軸ずつは未改造だと5.16秒と遅い（改造版1.89秒）。
- 1軸ずつ（確認オフ）は通常JSONより遅い（1.14〜3.31倍）。1リクエストは速くても回数が積み重なる。
- 束ね質問のY/N欄は、キャラが全条件（F1〜F3、長辺1024・768）で31/31。服装は83.9〜93.5%。
- Gemmaは1リクエストの固定費がQwenより大きく（約350ms vs 約130ms、サーバーログ。SWAの復元とみられるが未検証）、1軸ずつがQwenより遅い（長辺1024、準備なし3.00秒 vs 1.89秒、準備あり2.74秒 vs 1.29秒）。キャッシュ復元の重さはモデルの構造に依存します。
- 長辺768は、準備なしでは全方式を速くします（F1: 束ね（順位付け）1188→1008ms、JSON 1624→1340ms、1軸ずつ（確認オフ）1890→1637ms）。準備ありでは長辺の差が小さく、F1では768のほうが遅い方式もありました（1回測定のため原因は切り分けていません）。精度は多くの軸で同じでした。
- 単一選択の軸では、選択式は通常JSONと同等以上でした。characterは taxonomy 0.4.1 の `none` の説明文で、選択式が両モデルとも93.5%に上がりました（E1b/E2b。選択肢の文言設計の影響であり、方式の限界ではない）。
- outfit は、確認オン（1軸ずつ）で `school_uniform` の誤検出が残ります（近い候補にも「はい」が付きやすい）。確認オフでは90.3%（F1）、ただし複数写る画像のキャラは順位付けだと取りこぼします。
- 軸の並列送信は、この組み合わせでは得になりませんでした（E9b/b2/e）。
- 出力形式の崩れやすさ（E10・E10b、追加課題）: 白色ノイズ100枚では通常JSONは6モデルとも失敗0でしたが、束ね質問は小型モデルで大きく崩れました（例: Qwen 2B Q8 は 100/100）。実画像31枚では、通常JSONの形式不正（S3 9、S5 10、S7 7、S9 6、F1 1件）が `json_schema`（制約付きデコード）で全モデル0になり、単一項目の正答率も選択式の水準に上がりました（S3 63.2→89.7%）。
- 含意: 形式を文法で縛れば、通常JSONと選択式の形式の崩れやすさの差は、この小標本では埋まります。選択式に残る固有の価値は、候補ごとの相対スコア（閾値・不確かさ・Y/Nによる複数選択）と、形式を守れるモデルでの束ね質問の速さです。`json_schema` は値の集合（enum）も縛るので、構文だけの効果ではありません。ノイズは実画像とかけ離れ、実データは31枚・各1回です。
- 同じ回の中でのキャッシュの使われ方（準備の有無・並列）で、際どいケースの予測が揺れました（report §4.9）。旧の最終の取り直し（3回）では、サーバーを再起動して同じ条件で測れば判定は完全に再現しました。測り直し（各1回）では、準備の有無や旧測定とのあいだで数件の違いがありました。
- メタデータの読み取りは全35ケースで正解（LoRA名・トリガーワード・形式）。メタデータ除去コピーでも画素からのキャラ判定は変わりませんでした。

**既知の失敗**:
- 選択式の`outfit`で、Alisaの制服(`office_wear`)に`school_uniform`が誤って追加されるケースが残る（作者の判断で taxonomy は変えずに報告）。
- `reasoning_effort: "none"` を指定しないと、thinkingが先に出て logprobs が取れない。
- 通常JSON方式では、単一選択の軸をリストで返す形式不正が起こることがある。
- E2（Gemma）で、LM Studioサーバー側のChannel Errorによりyes/no確認が2件失敗した（失敗として分母に残した。E2bでは再発せず）。

結果をグラフで見るダッシュボード: <https://yumeno.github.io/vlm-decision-classifier/>(GitHub Pages)。`site/index.html` を直接ブラウザで開いてもローカルで見られます。データは `doc/experiments/reprime/` の測定(各条件1回)です。

詳細は [`doc/experiments/report.md`](experiments/report.md) §1.2〜§1.4（本表・小型モデル・形式の崩れやすさ）、§4〜§6（経緯・限界）を参照。

## 判定構造・モデルと実行環境

### データセット

専用データセット v1.0.0(元画像31枚 + メタデータ除去コピー4件 = 35ケース、全件SFW)。生成経緯・正解の分布・権利確認は [`dataset/DATASET_CARD.md`](../dataset/DATASET_CARD.md) を参照。

### 判定構造

軸ごとに短い選択式質問を送り、回答ラベルの`logprobs`から候補間の相対スコアを計算します（単一選択の軸は1回の質問。複数選択の軸は既定で相対スコアに閾値を当てて採用し、`--confirm` なら候補ごとのyes/no確認を追加で送る。束ね質問は全軸を1リクエストにまとめる）。候補間の相対スコアは、列挙した候補内で再正規化した値であり、実世界の正答確率ではありません。

### モデルと実行環境

モデル重み、視覚プロジェクタ、LoRA重みはこのリポジトリに同梱しません。使用したGGUFの配布元・リビジョン・量子化・SHA256、対応するmmproj、LM Studio/llama-serverの版、GPUオフロード、画像縮小設定は [`doc/experiments/report.md`](experiments/report.md) §2に固定して記録しています。自家製アプリの製作中に無検閲派生モデルで測った値は、このデモの結果として扱いません。

Qwen3.5の連続質問では、llama.cppが同じ画像を再エンコードする問題があります（上流 [issue #26994](https://github.com/ggml-org/llama.cpp/issues/26994)）。そのキャッシュ挙動を変える改造（`doc/patches/llamacpp-mtmd-checkpoint.patch`）は**速度比較用の任意条件**で、判定結果は変わりません（report §4.5）。初回の分類実行には不要で、未改造環境に戻すには、パッチを当てずに同じコミットからビルドした `llama-server` を使います。新規セットアップの llama.cpp は b11447(commit `da263e7275dfbaeefcd61504eaa4fd5247540e11`)を推奨します(パッチはそのまま当たり、引き続き必要。5枚の確認で回答は同一、時間は約1〜4%短い。[`experiments/openjev/README.md`](experiments/openjev/README.md) §7)。これまでの報告結果はすべて `f95b0d95394d5e311ba8228689972843178c5e28` で測ったものです。

## 開発の経緯

1. [`doc/requirements.md`](requirements.md) に沿って、公開する自作キャラの外見基準と画像生成条件を確定。
2. シナリオ表と正解付与規則を作り、専用データセット(v1.0.0)を生成・検収して版を固定。
3. QwenとGemmaで、画像入力・`logprobs`・回答書式が成立するか接続試験を実施。
4. メタデータ抽出、分類体系、選択式判定、通常JSON分類、共通の評価器を実装。
5. 固定データセットで全方式を評価し、Qwenのllama.cpp改造比較を実施（E1〜E5）。
6. ホットロード・確認のオン・オフ・軸の並列（E9）、束ね質問と送信画像（E7）、束ね質問の複数選択（E7e）を追加。デモUIを作成。
7. 記事の本表として、最終の取り直し（F1〜F3、36回）と小型モデルの比較（S1〜S9）を実施。
8. `--prime` が通常JSONだけに効いていなかった不公平に気づき（デモの動作から）、PR #20 で方式ごとの準備に修正。F1〜F3とS1〜S9を準備なし・準備ありの2条件で測り直した（`doc/experiments/reprime/`）。

詳細な順序と退出条件は [`doc/implementation-experiment-plan.md`](implementation-experiment-plan.md) に記載しています。E6(説明文付きJSON)と E8(メタデータ補助の対照実験)は実施していません。
