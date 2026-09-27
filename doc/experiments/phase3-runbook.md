# Phase 3 実行手順(E1/E1-J: Qwen、E2/E2-J: Gemma、LM Studio)

- 対象: データセット v1.0.0(`dataset/manifest.jsonl`、SHA256 `941faf84…`)、taxonomy 0.4.0(SHA256 `ec7b2a2f…`)
- 1回の `evaluate` で選択式(choice)と通常JSON(json)を両方実行する。モードの順はケースごとに反転する(順序効果の抑制)。これで E1 と E1-J(Gemma は E2 と E2-J)が同じモデル・同じ画素・同じ taxonomy で対応比較になる。
- 実行環境の記録: `doc/experiments/runtime/{qwen,gemma}-lmstudio.json` を `--runtime-info` で `run.json` に写す。`gpu_used` はロード後に記入する。
- **GPU を使う操作(モデルのロード)の前に、必ず作者に確認する**(CLAUDE.md)。

## 手順(モデルごとに繰り返す。コマンドはリポジトリのルートで実行)

1. **状態確認(GPU を使わない)**
   - `lms ps` でロード中のモデルを確認する。別プロジェクトのモデルがあれば、作者に確認してからアンロードする(`lms unload <id>`)。
2. **ロード(作者の確認後)**
   - Qwen: `lms load qwen3.5-9b --identifier qwen3.5-9b -y`(Gemma: `lms load gemma-4-12b-it --identifier gemma-4-12b-it -y`)
   - `lms ps` でコンテキスト長・並列数を、`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` と `nvidia-smi` で使っている GPU を確認し、runtime JSON の `gpu_used` に記入する。
3. **probe**: `.venv/Scripts/python.exe -m classifier_demo probe --model qwen3.5-9b`
   - logprobs あり・thinking なし・reasoning_effort が落ちていないことを確認する。
4. **1件の試打**(両モード)
   - `.venv/Scripts/python.exe -m classifier_demo classify dataset/images/M01.png --model qwen3.5-9b --mode choice --output results/trial/qwen_M01_choice.json`
   - 同じく `--mode json`。エラーがないこと、7軸すべてに結果が出ることを確認する。
5. **全件評価**
   ```
   .venv/Scripts/python.exe -m classifier_demo evaluate --manifest dataset/manifest.jsonl --model qwen3.5-9b --modes choice,json --warmup 1 --dataset-version v1.0.0 --runtime-info doc/experiments/runtime/qwen-lmstudio.json --runtime-label qwen-lmstudio --output-dir results/E1_qwen-lmstudio
   ```
   - Gemma は `--model gemma-4-12b-it`、`--runtime-info doc/experiments/runtime/gemma-lmstudio.json`、`--runtime-label gemma-lmstudio`、`--output-dir results/E2_gemma-lmstudio`。
6. **アンロード**: `lms unload qwen3.5-9b`(Gemma も同様)。作者に「使い終わった」と報告する。
7. **記録**
   - `results/`(Git 管理外)から `run.json`・`cases.csv`・`summary.md` を `doc/experiments/E1_qwen-lmstudio/` にコピーする(個人パスがないことを確認する)。ケース別 JSON は `results/` に残す。
   - `doc/worklog.md` に、実行日時、コミット、所要時間、気づいた失敗・異常を追記する。
   - 数値の解釈は、E1・E2 の両方がそろってから `doc/experiments/report.md` にまとめる(小標本のため、ケース別の結果を主資料にする)。

## 事前に分かっていること(probe、2026-09-24)

- 両モデルとも `reasoning_effort: "none"` が無いと thinking が先に出て、logprobs が取れない。クライアントは常に付けて送る(拒否されたら外して再送し、その事実を記録する)。
- 選択式は、軸ごとに画像を送り直す(Qwen の試し分類で1軸あたり約0.9〜1.3秒)。キャッシュ改造(E3/E4)は、この再エンコードに効く見込み。
