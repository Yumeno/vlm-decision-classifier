# VLM Decision Classifier

ローカルの視覚言語モデル（VLM）を使い、生成画像を**選択式の質問**で分類する実験用リポジトリです。分類軸ごとの選択肢と判定基準をモデルへ渡し、回答ラベルの `logprobs` から候補間の相対スコアを得ます。画像の生成メタデータから得たLoRA指定は、画素からの判定とは別の根拠として表示します。

**分類コアの最小実装（Phase 2a）まで完了しています。画像データセット・実験結果はまだ入っていません。** 下記の「実行方法（開発中）」で単体分類の動かし方を確認できます。まず [`doc/requirements.md`](doc/requirements.md)、[`doc/basic-design.md`](doc/basic-design.md)、[`doc/implementation-experiment-plan.md`](doc/implementation-experiment-plan.md) を参照してください。

## 何を実演するか

1. `image_type`、`art_style`、`subject`、`character` などの独立した軸で画像を分類します。複数人の画像ではキャラクターを複数選べるようにします。
2. 自作キャラクター「Alisa」を、画像上の容姿と、PNG生成情報に記録されたAlisa用LoRA名（例: `fet-alisa-uniform-anima-v4u`）の二経路で調べます。LoRA名の記録は**生成時の指定の証拠**であり、実際にLoRAが効いたことや、そのキャラクターが画面に写っていることの証明ではありません。
3. 同じ専用画像データセットで、通常の**分類JSONを生成させる方式**と、回答ラベルの確率分布を読む**選択式方式**の正答率、所要時間、形式不正、再試行を比較します。
4. 通常版 **Qwen3.5 9B GGUF** と **Gemma 4 12B** の対応GGUFを評価します。GemmaはSFW画像での実用性を中心に確認します。Qwenではさらに、同じllama.cppコミットからビルドした未改造版と画像キャッシュ改造版を比較します。

実験用画像は、このリポジトリのために新規生成します。元の画像管理プロジェクトの画像や既存の評価画像は含めません。公開可能な画像、正解ラベル、生成条件、権利確認記録をセットにした固定データセットを準備します。

## 着想と技術上の位置付け

発想の起点はJevと、[Google Gemmaが紹介した「DiffusionGemma as Jev」の投稿](https://x.com/googlegemma/status/2101069861598482817)です。紹介された[vLLMのPR #57250](https://github.com/vllm-project/vllm/pull/57250)は、DiffusionGemmaの出力キャンバスに回答スロットを設け、選択肢の分布を読み出す仕組みを示しています。

このデモはJevやそのPRの実装を移植するものではありません。既存の非公開画像管理プロジェクト `自家製の画像分類アプリ(非公開)` で試している方式から分類部分を切り出し、Qwen/Gemmaの**自己回帰VLM**へ画像と選択肢を送り、chat completionsの `logprobs` を使う独立した実演にします。DiffusionGemmaのキャンバス読み出しと、ここでの先頭回答トークンの読み出しは仕組みが異なります。「logprobsによる分類そのものが新発明」とは主張しません。

## 計画している構成

| パス | 役割 | 現在 |
|---|---|---|
| `README.md` | 概要・開発の入口 | あり |
| `AGENTS.md`、`CLAUDE.md` | AIコーディングエージェント向けの作業規約 | あり |
| `doc/requirements.md` | 要件、対象範囲、公開条件 | あり |
| `doc/basic-design.md` | 分類・データ・アダプタ設計 | あり |
| `doc/implementation-experiment-plan.md` | 実装順と実験行列 | あり |
| `doc/phase0-sources-models.md` | 移植元の要点、使用モデルとSHA256、probe結果、画像生成の条件 | あり |
| `doc/dataset-plan.md` | 正解付与規則、31枠の生成計画表、manifest仕様、検収と正解付与の経緯 | あり |
| `doc/worklog.md` | 作業記録（新しい順） | あり |
| `classifier_demo/` | メタデータ抽出、選択式判定、JSONベースライン、評価器、CLI（`probe`/`classify`/`check-manifest`/`evaluate`） | あり（分類コア・評価器） |
| `taxonomy/default.yaml` | 分類体系 0.4.1（7軸）と自作キャラの定義 | あり |
| `dataset/` | 評価用データセット v1.0.0(`images/` 35ケース、`manifest.jsonl`、`DATASET_CARD.md`、正解の一次記録 `labels/`、生成記録 `generation/`) | あり |
| `scripts/` | データセットの生成（Forge/ComfyUI）、メタデータ除去、データセット組み立て、`benchmark_cache.py`（Phase 4 / E5: 画像キャッシュ改造の再現用ベンチマーク） | あり |
| `tests/` | pytest（pooling・taxonomy検証・メタデータ照合・JSON解析・pipeline・評価器・生成スクリプト） | あり |
| `doc/patches/` | llama.cpp画像キャッシュ改造の固定差分(`llamacpp-mtmd-checkpoint.patch`、上流 `f95b0d9` に当てる1行) | あり |

設計文書中のディレクトリ案は実装時の指針です。コードを追加した時点で、この表と起動方法を実態に合わせて更新してください。

## 実行方法（開発中）

Python 3.11以上。この開発機では `python`（3.14）/ `python3`（3.10）ではなく `py -3.12` で venv を作ります。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

推論サーバー（LM Studio / llama-server の OpenAI互換 chat completions）はユーザーが別途起動しておきます。このツールはサーバーの自動起動・モデルの自動ダウンロードを行いません。

```powershell
# 接続・vision・logprobs の疎通確認（実画像1枚で回答ラベルとlogprobsを試す）
.venv\Scripts\python.exe -m classifier_demo probe --base-url http://127.0.0.1:1234/v1 --model <モデルID>

# 1枚を分類（選択式）。結果JSONは --output 省略時は標準出力
.venv\Scripts\python.exe -m classifier_demo classify path\to\image.png --model <モデルID> --output results\demo.json

# 通常JSON分類ベースラインとの比較
.venv\Scripts\python.exe -m classifier_demo classify path\to\image.png --model <モデルID> --mode json

# manifestの検証(リポジトリルートから実行する。画像パスはカレントディレクトリ基準で解決する)
.venv\Scripts\python.exe -m classifier_demo check-manifest --manifest dataset\manifest.jsonl

# データセット全件の評価(check-manifestを内部で先に実行し、エラーがあれば中断する)
.venv\Scripts\python.exe -m classifier_demo evaluate --manifest dataset\manifest.jsonl --model <モデルID> --output-dir results\<名前>

# Phase 4 / E5: 画像キャッシュ改造の再現用ベンチマーク(未改造版/改造版のllama-serverでそれぞれ実行して比較する)
.venv\Scripts\python.exe scripts\benchmark_cache.py --model <モデルID> --label vanilla --output results\cache\vanilla.json
```

`evaluate` は選択式(`choice`)と通常JSON(`json`)を既定で両方実行し(`--modes choice,json`)、ケースごとに交互の順で実行して順序効果を抑える。出力先(`--output-dir`)には `run.json`（実行条件・除外ケース）、`cases.csv`（ケース別採点）、`summary.md`（集計）、`cases/<case_id>.<mode>.json`（生の分類結果）を書き出す。`rights_confirmed` が true でないケースは評価から除外され、`run.json` の `excluded_cases` に理由とともに記録される。`--runtime-info path\to\runtime.json` で、モデル/mmprojのSHA256・サーバー種別やcommit・パッチ有無・起動引数・GPUオフロードなど実行環境を記した任意のJSONファイルを渡すと、中身をそのまま（ファイル名とSHA256も添えて）`run.json` に記録する。

テスト実行（ネットワーク・実サーバー不要、偽バックエンドのみ使用）:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## 実験の比較条件

| 主比較 | 条件 | 測定値 |
|---|---|---|
| 選択式 vs 通常JSON | 同じモデル、画像、分類体系、メタデータ条件 | ケース別正誤、軸別精度、処理時間、失敗・再試行 |
| Qwen vs Gemma | 共通のSFW画像・分類基準 | 精度、応答形式、時間。モデル・量子化・ロード設定も記録 |
| Qwenのllama.cpp未改造 vs 改造 | 同じ上流コミット、GGUF、mmproj、画像、質問 | 時間、判定結果の差、画像切替時の混線 |
| メタデータあり vs なし | 同じ元画像の派生ケース | 生成情報の寄与と、容姿判定との食い違い |

JSONベースラインの主比較は**分類フィールドだけ**を生成します。要約文も書かせる条件は出力トークン数が増えるため別に測ります。モデルロード時間とウォームアップは画像当たりの分類時間から分け、推論が失敗したケースも分母に残します。候補間の相対スコアは、実世界の正答確率を意味しません。

## モデルと実行環境

モデル重み、視覚プロジェクタ、LoRA重みはこのリポジトリに同梱しません。選定するGGUFの配布元・リビジョン・量子化・SHA256、対応するmmproj、LM Studioまたはllama-serverの版、GPUオフロード、画像縮小設定は実験時に固定し、結果とともに公開します。通常版Qwen3.5 9Bでの数値はこれから測定します。元プロジェクトに記録された無検閲派生モデルでの測定値を、このデモの結果として扱いません。

Qwen3.5の連続質問では、llama.cppが同じ画像を再エンコードする問題と、そのキャッシュ挙動を変える改造が元プロジェクトの技術記録(非公開リポジトリ)にあります。この改造は**速度比較用の任意条件**です。初回の分類実行には不要です。公開する改造手順は、上流コミット・差分・ビルド条件・結果整合の確認が済んでから追加します。

## 開発を始める順番

1. [`doc/requirements.md`](doc/requirements.md) に沿って、公開する自作キャラの外見基準と画像生成条件を確定します。
2. 画像を生成する前に、シナリオ表と正解付与規則を作ります。その後、専用データセットを生成・検収して版を固定します。
3. QwenとGemmaの候補GGUFで、画像入力・`logprobs`・回答書式が成立するか小さな接続試験をします。
4. メタデータ抽出、分類体系、選択式判定、通常JSON分類、共通の評価器を実装します。
5. 固定データセットで全方式を評価し、Qwenのllama.cpp改造比較と公開用READMEの実行手順を完成させます。

詳細な順序と退出条件は [`doc/implementation-experiment-plan.md`](doc/implementation-experiment-plan.md) に記載しています。現時点で動作するインストール手順や実測結果はありません。

## 公開前の確認

- 公開画像と使用した生成モデル・LoRAについて、権利と配布条件を確認する。
- 正解ラベルをモデル出力を見る前に付け、元画像とメタデータ除去コピーを独立画像として水増ししない。
- 秘密情報、個人パス、元プロジェクトの画像やDB、モデル重みが履歴を含めて混入していないことを確認する。
- 実際のモデル・量子化・環境、画像別の結果、失敗例、使用したコミットをREADMEと実験レポートで示す。
- ライセンスは公開前に確定し、コードと画像データの扱いをそれぞれ明記する。
- 公開に切り替えた直後に、privateの無料プランでは使えないGitHub設定を有効化する（下記）。

### 公開時のGitHub設定

設定済み: Wiki・Projects無効、squash/merge commitのみ許可、merge後のブランチ自動削除、Dependabotアラートとセキュリティ更新、topics。

公開への切替後に実行する（privateの無料プランではAPIが403を返す）:

```powershell
gh repo edit Yumeno/vlm-decision-classifier --visibility public --accept-visibility-change-consequences
gh api -X PATCH repos/Yumeno/vlm-decision-classifier -f "security_and_analysis[secret_scanning][status]=enabled" -f "security_and_analysis[secret_scanning_push_protection][status]=enabled"
```

続けて、デフォルトブランチのruleset（削除禁止・force push禁止・PR必須、承認数0）を作成する。`gh api -X POST repos/Yumeno/vlm-decision-classifier/rulesets` に次のJSONを渡す:

```json
{"name":"protect-main","target":"branch","enforcement":"active",
 "conditions":{"ref_name":{"include":["~DEFAULT_BRANCH"],"exclude":[]}},
 "rules":[{"type":"deletion"},{"type":"non_fast_forward"},
  {"type":"pull_request","parameters":{"required_approving_review_count":0,"dismiss_stale_reviews_on_push":false,"require_code_owner_review":false,"require_last_push_approval":false,"required_review_thread_resolution":false}}]}
```

## 関連資料

- [Qwen3.5 9B（公式モデル）](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Gemma 4 12B it（公式モデル）](https://huggingface.co/google/gemma-4-12B-it)
- [vLLM PR #57250 — DiffusionGemmaの構造化読み出し](https://github.com/vllm-project/vllm/pull/57250)
- [llama.cppの画像キャッシュに関するissue](https://github.com/ggml-org/llama.cpp/issues/26994)
