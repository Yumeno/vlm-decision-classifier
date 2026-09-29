# 最終の取り直し(F1〜F3)の手順

記事の本表を、1つのコミット・同じ条件で取り直す(作者判断、2026-09-29)。これまでの E1〜E9・E7 は試行錯誤の経緯として残す。

## 条件

- 共通: 最新のコード、taxonomy 0.4.1、データセット v1.0.0、JPEG(quality 90)、`--prime`、`--warmup 1`、`--axis-concurrency 1`、RTX 3090 のみ(`-sm none -mg 0`)、llama-server の起動引数は phase4-runbook と同じ(`-ngl 99 -c 8192 -np 4 --kv-unified`)。回ごとにサーバーを再起動し、起動前に 3090 の使用量を記録する。
- 条件:
  - F1: Qwen3.5 9B、改造版 llama-server(`doc/experiments/runtime/final-qwen-patched.json`)
  - F2: Qwen3.5 9B、未改造の llama-server(`final-qwen-vanilla.json`)
  - F3: Gemma 4 12B、改造版 llama-server(`final-gemma-patched.json`。F1 と同じ実行ファイルで、違いはモデルだけ)
- 長辺: 1024 と 768
- 回:
  - a: `--modes choice,json,bundled --rank-threshold 0.5 --bundled-multi rank`(1軸ずつは確認オフ、束ね質問は順位付け)
  - b: `--modes choice,bundled --confirm --bundled-multi yn`(1軸ずつは確認オン、束ね質問は Y/N 欄)
- 繰り返し: 3回。順番は「繰り返し → 条件 → 長辺 → 回」(時間とともに条件が偏らないように、繰り返しを一番外側に)。
- 合計 36 回。出力は `results/final/<条件>_e<長辺>_<回>_r<繰り返し>/`(管理外)、公開用は `doc/experiments/final/` にコピーする。

## 実行前の確認で分かったこと

- Gemma を llama-server で束ね質問(順位付け)にかけると、M01 で `Question 1: A` の形で答え、7軸目を飛ばした(キャラの軸が bundled_format_error)。条件をそろえるため、測定の途中でプロンプトは直さない。
- 対策(作者判断、2026-09-29): 束ね質問の rank にも yn と同じ回答の雛形(`1: ?` 〜 `7: ?`)を付けた。M01・G03・G05 で Gemma・Qwen とも7軸すべてに答え、形式不正0件を確認してから本測定に入った。雛形なしの版で1回分だけ流した結果は `results/final_aborted_no_template/`(管理外)に退避し、本表には使わない。
