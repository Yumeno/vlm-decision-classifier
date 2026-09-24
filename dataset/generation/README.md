# dataset/generation/

データセット画像の生成スクリプトと生成記録。仕様の正は `doc/dataset-plan.md`(§2.7、§4、§5)。

## ファイル

- `prompts.yaml` — 生成枠(slot)の固定プロンプト表。**内容は生成前に固定済み。変更しない。**
- `comfy_anima_workflow.json` — ComfyUI用ワークフロー(HuggingFaceの Anima 公式サンプル画像に埋め込まれたAPI形式グラフをそのまま保存したもの)。`scripts/generate_comfy.py` が読み込んで改変する。
- `{slot_id}_a{attempt}.json` — 生成1回ごとのパラメータ・応答の記録(`scripts/gen_common.py` の `write_params` が書く)。
- `log.csv` — 生成試行のログ(1行=1回の生成)。列: `slot_id, attempt, seed, tool, file, sha256, generated_at, result, reason`。
  `result` の種類: `generated`(正常に生成できた) / `error`(送信後の失敗。タイムアウト・取得失敗・保存失敗など) / `setup_error`(Forgeで返ってきた画像のModel hashが一致しない。この時点でスクリプト全体を中断する)。
  **検収結果(採用/除外)は作者が後からこの行に手で書き加える**(`accepted` / `rejected` など)。
- `{slot_id}_a{attempt}.submitted.json` / `{slot_id}_a{attempt}.prompt_id.txt` — 生成リクエストを**送信する直前**に書く送信済みマーカー(後者はComfyUIのprompt_id)。

## スクリプト(`scripts/`)

- `scripts/gen_common.py` — prompts.yaml読み込み、プロンプト組み立て、seed計算、log.csv/paramsJSON書き出しなどの共通処理。単体では実行しない。
- `scripts/generate_forge.py` — Forge Neo(Automatic1111互換API)で `tool: forge` の枠を生成する。
  ```
  python scripts/generate_forge.py --url http://127.0.0.1:7870 --attempt 1
  python scripts/generate_forge.py --url http://127.0.0.1:7870 --attempt 1 --slots A01,A02
  ```
  生成前にロード中チェックポイントが `anima-base-v1.0` であることを確認する(違えば中断。切り替えは作者がForge Neo側で手動で行う)。
- `scripts/generate_comfy.py` — ComfyUIで `tool: comfyui` の枠を生成する。
  ```
  python scripts/generate_comfy.py --url http://127.0.0.1:8188 --attempt 1
  python scripts/generate_comfy.py --url http://127.0.0.1:8188 --attempt 1 --slots A03,A04
  ```
- `scripts/strip_metadata.py` — PNGのテキストチャンク(生成メタデータ)を除去したコピーを作る(`*-strip` 派生ケース用)。
  ```
  python scripts/strip_metadata.py dataset/images/A01.png dataset/images/A01-strip.png
  ```

## 運用ルール(dataset-plan.md §2.7)

- **1枠1枚**。同じ設定で複数枚作って選ぶことはしない。
- 検収で落ちた場合だけ、attempt を1つ進めて作り直す(1枠あたり最大2回、`attempt=1,2`)。作り直しの理由は `log.csv` に残す。2回とも落ちたらその枠は欠数として公表する。
- 生成スクリプトは**同じ slot+attempt の画像/params/送信済みマーカーが既にあれば上書きを拒否してエラーにする**(後選び防止)。
- **送信済みマーカーがある枠は再実行できない**。失敗時はComfyUIのoutputフォルダ等から手で回収するか、attempt 2 に進む。
- 生成画像は `dataset/staging/`(Git管理外)に置く。検収前・権利未確認の画像をコミットしない。
- 検収後、採用画像を `dataset/images/` に移し、manifest (`dataset/manifest.jsonl`) を書くのは別work(`doc/dataset-plan.md` §5)。
- 外部生成(G06〜G08, N01, N02, M03, C02の元画像)はこのスクリプト群の対象外。依頼文・サービス名・利用日は `dataset/generation/{slot_id}_a1.json` に手で記録する。
