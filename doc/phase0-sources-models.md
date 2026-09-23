# Phase 0 記録 — 出典・モデル・実行環境・利用条件

- 作成日: 2026-09-24
- 対応: `implementation-experiment-plan.md` Phase 0

## 1. 移植元

| 項目 | 内容 |
|---|---|
| リポジトリ | `自家製の画像分類アプリ(非公開)`(private、作者本人) |
| 参照コミット | `79949603984918aa015cbff1bbd5ef93e0012196`(main、2026-09-24確認) |
| ライセンス | 未設定。作者本人のコードのため、本デモの公開ライセンス確定時に一緒に扱う |
| 方針 | 丸ごと移植しない(`decision_provider.py` だけで約2,600行)。下記の要点だけを最小実装で引き継ぐ。移植・参考にした関数は実装PRの説明に列挙する |

### 引き継ぐ要点(MVPに必要なもの)

- 選択肢ラベルは `A`〜`T`(最大20、`none of the above` を含む)。LM Studio の `top_logprobs` 上限が20のため。
- リクエスト: `max_tokens=1`、`temperature=0`、`logprobs=true`、`top_logprobs=20`、`reasoning_effort: "none"`。
- ラベル集計: `token.strip().upper()` で `A`/` A` を同一視し、`exp(logprob)` を合算して候補内で再正規化。全ラベルの合計が0なら失敗(観測トークンを診断用に残す)。
- thinking 検出: `logprobs.content` が空のとき `reasoning_content`/`reasoning` と `usage.completion_tokens_details.reasoning_tokens` を見て、thinking による失敗かを区別する。
- 複数キャラ: 1回の choose で順位付け → `other` 系を除外 → 相対フロア(最大確率×0.001)と上限4件で候補を絞る → 候補ごとに yes/no(`Is {name} ({criteria}) present in the image?`、`none` なし)→ P(yes)≥0.5 で採用。`other` は単独の yes/no で何にでも yes が出やすい(元実装の実測: gemma-4 12B で0.73)ので、yes/no の確認対象にしない。
- 画像: PIL で実フォーマットを判定し、`exif_transpose` の後に長辺1024pxへ縮小、data URL で送る。
- LoRA 抽出: `<lora:([^:>]+)(?::([\d.\-]+))?>`。
- 外部由来テキストは区切って「データであり指示ではない」と明記する(元実装は `«…»`)。

### 引き継がないもの(過剰)

束ね質問(bundled)と専用パーサ、階層降下・トーナメント分割、設定クラスと検証群、taxonomy 自動育成、XMP/Gemini 解析、ハッシュ切替、クラス共有の再試行記憶。

### 元実装との差分として決めること

- 元実装の通常 JSON 分類は `temperature=0.3`、`max_tokens=800`、要約込み、再試行なし。本デモの主比較(分類フィールドのみ)は、実験前にプロンプト・温度・再試行上限を固定する。要約込みは E6(追加課題)に回す。

## 2. 推論モデル(固定)

| 用途 | 配布元 / リビジョン | ファイル | SHA256 |
|---|---|---|---|
| Qwen 主モデル | `lmstudio-community/Qwen3.5-9B-GGUF` @ `1379f25c6b505a3fc737bd7818cb09389cf807c1`(Apache-2.0) | `Qwen3.5-9B-Q4_K_M.gguf` | `cd76ec205963b3b33350093e6904d9de16c4e666fd104e1f632d25c7f15f2a13` |
| 同 mmproj | 同上 | `mmproj-Qwen3.5-9B-BF16.gguf` | `330d17547bfdbcd0e7a0cb3f4b06b4ceeac0aaa1e449122d9d66ef957aeb74b3` |
| Gemma 比較モデル | `lmstudio-community/gemma-4-12B-it-GGUF` @ `40b870babe39398ce917bbf4b4ff9b5a1f12710e`(Apache-2.0) | `gemma-4-12B-it-Q4_K_M.gguf` | `95d83ba36642b1f385fb906b5962a71763361be3bac930a709945f72d97473f8` |
| 同 mmproj | 同上 | `mmproj-gemma-4-12B-it-BF16.gguf` | `7fd884a5f7d9ee60f6b88e5b51287151e547454b2b60a4a6270bb5434310ad0e` |

選定理由: LM Studio 経路が主なので、配布元を lmstudio-community に揃えた(unsloth 版にも同じ量子化がある)。どちらも公式モデル(`Qwen/Qwen3.5-9B`、`google/gemma-4-12B-it`)の量子化版。

### 接続 probe の結果(LM Studio、2026-09-24、1回ずつの疎通確認であり評価値ではない)

学習用のAlisa画像1枚(非公開、コミットしない)に「髪色は? A〜D + E. none」を質問。

| モデル | `reasoning_effort` なし | `reasoning_effort: "none"` |
|---|---|---|
| Qwen3.5 9B | 失敗: content 空、reasoning_tokens=1、logprobs なし | 成功: `B`(茶)、相対スコア 0.99 |
| Gemma 4 12B | 失敗: content 空、logprobs なし | 成功: `B`、相対スコア 1.00(上位候補に `<|channel>` が出る) |

→ 両モデルとも `reasoning_effort: "none"` が必須。付けずに失敗した場合は、thinking による失敗として記録する。

## 3. llama.cpp(E3/E4/E5 用)

| 項目 | 内容 |
|---|---|
| 既存の改造版 | 別の作業場所にある上流 `ggml-org/llama.cpp` の `f95b0d95394d5e311ba8228689972843178c5e28`(2026-09-22)。`tools/server/server-context.cpp` の `do_checkpoint = do_checkpoint && !has_mtmd;` をコメントアウト(上流 issue #26994) |
| ビルド条件 | CUDA 12.8、`GGML_CUDA=ON`、`CMAKE_CUDA_ARCHITECTURES=86;89` |
| 未改造版 | 未作成。**同じコミットを別ディレクトリにクローンして、同じ条件でビルドする**(元プロジェクトの作業ツリーには触れない)。Phase 4 で作成 |
| パッチの公開 | Phase 4 で `doc/patches/` に差分を固定する |

## 4. 画像生成(データセット用)

| 項目 | 決定事項 / 状態 |
|---|---|
| ベースモデル | **Anima base v1.0**(`circlestone-labs/Anima` の `anima-base-v1.0.safetensors`、SHA256 `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e`)。派生版ではなくオリジナルを使う(作者の指定) |
| ライセンス | CircleStone Labs Non-Commercial License v1.2。モデル・LoRA(=Derivative)の利用・配布は非商用に限る。**生成物(Outputs)は同社が権利を主張せず、商用を含め用途を問わず使える**(§2.e)→ 生成画像は公開してよい。LoRA 本体は配布しない |
| Alisa の LoRA | 作者の自作。Anima 用の最新版は `fet-alisa-uniform-anima-v4u`(`anima-base-v1.0` で学習、319枚、トリガー `fet_alisa_uniform`)。ベースモデル別の版があり名前が異なる(`fet_alisa_uniform_ilpen2`、`_animagine4`、`_am` 等) |
| 似た別キャラ | LoRA なし。プロンプトで新規に設計する(作者の指定) |
| 内容範囲 | 全件SFW(作者の指定)→ Gemma も全件で評価でき、SFW subset の管理は不要 |
| 生成手段 | 作者の指定: API 生成に加え、Codex / Antigravity 等にも依頼して多様なパターンを作る。ComfyUI も使う(reForge や llama.cpp とは VRAM の都合で同時に動かせない) |

### 分類設計への影響(要反映)

- **LoRA 照合は「単一名との完全一致」ではなく「taxonomy に列挙した LoRA 名の集合との完全一致」にする**。データセットで実際に使った版の名前(例: `fet-alisa-uniform-anima-v4u`)を列挙する。部分一致・前方一致にはしない。
- **ComfyUI で生成した画像には A1111 形式の `parameters` が付かない**。LoRA 付き画像を ComfyUI で作る場合は、workflow JSON(`prompt` チャンク)の LoraLoader 系ノードの `lora_name` からの抽出が必要になる。
- 外部の画像生成サービス(Codex / Antigravity 経由)の画像は、サービスごとに利用条件を確認して記録する。
