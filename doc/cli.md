# CLI オプション一覧

`python -m classifier_demo <サブコマンド>` の全オプションです(`--help` でも確認できます)。使い方の流れは [`README.md`](../README.md) を参照してください。サブコマンドは `probe` / `classify` / `check-manifest` / `evaluate` / `serve` / `systemone` の6つです。以降のコマンドは、リポジトリのルートで実行します(`check-manifest` と `evaluate` は、画像パスをカレントディレクトリ基準で解決します)。

既定値は `classifier_demo/__main__.py` と `scripts/benchmark_cache.py` の実装のとおりです。

**既定が変わったもの**: (1) 複数選択軸の候補ごとのyes/no確認は**既定オフ**（`--confirm` でオン）。(2) 送信画像は**既定でJPEG**（quality 90）。E1〜E5・E9・E7a/b はPNG・長辺1024で測りました（E7c/d はJPEG）。E1〜E5、E9a/b/b2/e は確認オン、E9c と E7a/b は確認オフです。そのため **E1〜E5・E9・E7a/b の再現には `--image-format png` が必要で、E1〜E5・E9a/b/b2/e ではさらに `--confirm` が必要**です（[`reproduce.md`](reproduce.md)）。


## 共通（`probe` / `classify` / `evaluate`）

| オプション | 既定値 | 意味 |
|---|---|---|
| `--base-url` | `http://127.0.0.1:1234/v1` | OpenAI互換サーバーのURL（LM Studioの既定ポート） |
| `--model` | （必須） | モデルID |

## `probe`

`--base-url`、`--model` のみ。

## `classify <image>`

| オプション | 既定値 | 意味 |
|---|---|---|
| `image`（位置引数） | （必須） | 分類する画像のパス |
| `--mode` | `choice` | `choice`（1軸ずつの選択式）/ `json`（通常JSON）/ `bundled`（束ね質問）/ `json_schema`（通常JSONと同じ質問を制約付きデコードのJSONスキーマで生成。llama-server で測定。E10）/ `dgemma_choice`・`dgemma_json`（DiffusionGemma。下記） |
| `--dgemma-url` | なし | `dgemma_choice` で必須。vLLM の example structured server のURL（例 `http://127.0.0.1:8011`。`/v1/systemone` を使う）。`--base-url` は vLLM 本体（例 `http://127.0.0.1:8000/v1`） |
| `--dgemma-samples` | `auto` | `dgemma_choice` の samples。`auto`=サーバー既定（標準）、整数=ノイズ draw の固定回数（例 `1`） |
| `--taxonomy` | `taxonomy/default.yaml` | 分類体系のYAML |
| `--max-edge` | `1024` | モデルへ送る画像の長辺（px） |
| `--image-format` | `jpeg` | `jpeg`（quality 90）/ `png`。E1〜E9の再現には `png` |
| `--confirm` | オフ | 複数選択軸で上位候補ごとにyes/noを確認する（リクエストが増える）。E1〜E4等はこの方式 |
| `--rank-threshold` | `0.5` | `--confirm` なしのとき、複数選択軸で採用する相対スコアの閾値 |
| `--bundled-multi` | `rank` | 束ね質問での複数選択軸の扱い。`rank`=相対スコアと閾値、`yn`=候補ごとのYes/No欄（E7e） |
| `--output` | （標準出力） | 結果JSONの書き出し先（親フォルダがなければ作る） |

## `check-manifest`

| オプション | 既定値 | 意味 |
|---|---|---|
| `--manifest` | `dataset/manifest.jsonl` | 検証するmanifest |
| `--taxonomy` | `taxonomy/default.yaml` | 正解ラベルの照合に使う分類体系 |

## `evaluate`

`--base-url`、`--model`（必須）に加えて:

| オプション | 既定値 | 意味 |
|---|---|---|
| `--manifest` | `dataset/manifest.jsonl` | 評価するmanifest |
| `--modes` | `choice,json` | カンマ区切りで `choice` / `json` / `bundled` / `json_schema`（制約付きデコードのJSON。E10）/ `dgemma_choice` / `dgemma_json`（重複不可）。2モードはケースごとに順序を交互に、3モード以上は回転して実行 |
| `--taxonomy` | `taxonomy/default.yaml` | 分類体系 |
| `--max-edge` | `1024` | 送信画像の長辺 |
| `--image-format` | `jpeg` | `jpeg` / `png` |
| `--warmup` | `1` | 評価前のウォームアップ回数（分類時間に含めず別記録） |
| `--runtime-label` | なし | 実行環境のラベル（`run.json`に記録） |
| `--runtime-info` | なし | 実行環境を記したJSON（モデル/mmprojのSHA256、サーバーのcommit、パッチ、起動引数、GPUなど）。ファイル名とSHA256を添えて `run.json` にそのまま記録。例は `doc/experiments/runtime/*.json` |
| `--dataset-version` | なし | データセット版（例 `v1.0.0`）。`DATASET_CARD.md` と照合し `run.json` に記録 |
| `--note` | なし | 任意のメモ（`run.json`に記録） |
| `--output-dir` | （必須） | 出力先フォルダ |
| `--prime` | オフ | 各ケースの判定前に画像だけの準備リクエストを1回送る（ホットロード）。`prime_ms` を別記録 |
| `--axis-concurrency` | `1` | `choice` で軸ごとの質問を同時に送る数（1=逐次）。サーバーの対応スロット数が前提。`--prime` と併用（N≥2）すると準備をN本同時に送る |
| `--confirm` | オフ | `classify` と同じ |
| `--rank-threshold` | `0.5` | `classify` と同じ |
| `--bundled-multi` | `rank` | `classify` と同じ |
| `--dgemma-url` / `--dgemma-samples` | なし / `auto` | `classify` と同じ |

### DiffusionGemma のモード（`dgemma_choice` / `dgemma_json`、issue #4）

vLLM で動かした DiffusionGemma（`--base-url` が vLLM 本体、`--dgemma-url` が example サーバー）用。どちらも**ケースごとに全軸を1リクエスト**で聞く。モード名に `dgemma_` を含むと、全リクエストに `chat_template_kwargs: {"enable_thinking": false}` を足し、`temperature` は送らない（vLLM が拡散モデルへの指定を拒否するため。`run.json` の `dgemma.dropped_params` に記録。JSONは temperature 0 でなくサーバー既定で動く）。

| モード | 送り先 | 内容 |
|---|---|---|
| `dgemma_choice` | `--dgemma-url` の `POST /v1/systemone` | 質問IDは `q0`, `q1`, … の連番（11問以上だとサーバーが「ID ラベル」を空白区切りで書き、長いIDではラベルが1トークンにならず 422 になるため）。単一選択軸は `choice` 質問（選択肢名=`name`、説明=`criteria`、`allow_none` の軸は `none of the above` を足す）。複数選択軸は選択肢ごとの `noul` 質問（`Is <name> (<criteria>) present in the image?`）で、P(yes) が 0.5 以上を採用（束ね質問の `yn` と同じ規則。`none` は聞かず、1つも無ければ空）。値は相対スコア / P(yes)。サーバーが返した `diagnostics.samples.n`・`timing.reads` をケース別JSONの `dgemma` に記録 |
| `dgemma_json` | `--base-url` の `/chat/completions` | 通常JSON（`json` と同じ質問文・検証、制約なし）。vLLM は拡散モデルへの `json_schema` を拒否するので制約付きは使わない |

- `--prime` を付けても `dgemma_choice` には準備を送らない（example サーバーが読み出しを自前で行うため）。`run.json` の `dgemma.prime_skipped_modes` に記録。`dgemma_json` には他のJSONと同じ準備を送る。
- `--warmup` の準備リクエストは logprobs 付きの選択式質問を vLLM に送るため、DiffusionGemma では `--warmup 0` にする。
- vLLM・example サーバーの commit は `--runtime-info`（例 `doc/experiments/runtime/dgemma-vllm.json`）に書く。

## `serve`

| オプション | 既定値 | 意味 |
|---|---|---|
| `--port` | `8765` | デモUIの待ち受けポート（`127.0.0.1` 固定。使用中ならエラー） |
| `--taxonomy` | `taxonomy/default.yaml` | 分類体系 |

## Jev 風の呼び出し(System One 形式、実験的)

TypeSafe の Jev SDK(`typesafe-sdk`)と同じ呼び出しの形で、手元の VLM を呼べる最小クライアント(`classifier_demo/systemone.py`)。SDK には依存せず、名前とフィールドを写しただけで、TypeSafe 社とは無関係。

```python
from classifier_demo.systemone import SystemOneClient

client = SystemOneClient(base_url="http://127.0.0.1:1234/v1", model="<モデル名>")
r = client.system_one(
    state="顧客からの問い合わせ文...",
    questions={
        "dept": {"type": "choice", "instructions": "担当部署は?", "criteria": {"returns": "返品", "billing": "請求"}},
        "urgent": {"type": "noul", "instructions": "緊急か?"},
        "tone": {"type": "score", "instructions": "怒りの強さは?", "criteria": ["平静", "不満", "激怒"]},
    },
    images=["photo.png"],  # 本実装の拡張(Jev 本体は画像を受け付けない)
)
r.choices["dept"].choice, r.choices["dept"].probabilities, r.nouls["urgent"].noul, r.scores["tone"].score
```

CLI(手動確認用): `python -m classifier_demo systemone --base-url ... --model ... --questions q.json [--state TEXT | --state-file F] [--image PATH ...] [--prime]`。`q.json` は `{質問名: {type, instructions, criteria}}`。

`systemone` のオプション:

| オプション | 既定値 | 意味 |
|---|---|---|
| `--base-url` | `http://127.0.0.1:1234/v1` | サーバーURL |
| `--model` | （必須） | モデルID |
| `--questions` | （必須） | 質問のJSON（`{質問名: {type, instructions, criteria}}`） |
| `--state` / `--state-file` | なし | 状態テキスト（直接指定 / ファイルから） |
| `--image` | なし | 画像のパスまたはdata URL（複数指定可） |
| `--max-edge` | `1024` | 送信前に縮小する長辺（px） |
| `--image-format` | `jpeg` | `jpeg` / `png` |
| `--prime` | オフ | 先頭だけのリクエストを先に1回送る |


| Jev の質問 | 本実装 | 返すもの |
|---|---|---|
| `choice` | ラベル `A`〜`Z`,`a`〜`z` を順に割り当て、1質問1リクエスト(`max_tokens=1`, `top_logprobs=20`) | `choice`、`probabilities` |
| `noul` | yes/no(`A`/`B`)の2択 | `noul`(yes の相対スコア) |
| `score`(2〜10段階。`criteria` は順序付きリスト) | n 段階ならラベル `0`〜`n-1` | `score`(0〜n-1 の目盛りでの、相対スコアで重み付けした期待値)、`legend`、`probabilities` |

質問は SDK の形の素の dict で渡す(SDK の `Choice`/`Noul`/`Score` ヘルパークラスは受け付けない)。選択肢名・説明・instructions の改行や制御文字は空白にして1行にする。先頭(system → 画像 → state)を全質問で共有し、質問は互いに見えない(束ねない)。`prime=True` で先頭だけのリクエストを先に送れる(時間は `usage.prime_elapsed_ms` に別記録)。

限界:

- `probabilities` / `noul` / `score` は候補内で割り直した**相対スコア**で、較正されていない(正答確率ではない)。較正(正解付きデータで温度などを合わせる)は提供しない。
- 全候補を観測できるのは20候補以内(`top_logprobs` が上位20件。`A` と ` A` のような表記ゆれも枠を使う)。
- choice は最大52ラベル(Jev は255)。超過は明示エラー。ラベルトークンが出ない・thinking が先に出る・logprobs 欠損も `SystemOneError`(任意の選択肢には強制しない)。
- `confidence` は返さない(公式文書で算出式を確認できないため。`probabilities` から計算する)。
- `images` は本実装の拡張で、TypeSafe の Jev 本体は画像を受け付けない。形式は openjev の `images`(`data:image/...;base64,` URL、または `{"content_type", "base64"}`)と同じで、加えてファイルパスと bytes も受ける。8枚まで・1枚5MiBまで・jpeg/png/webp/gif のみ(違反は明示エラー)。openjev はサーバー側でリサイズしないが、本実装は既存実験と揃えるため送信前に `prepare_image`(長辺 `--max-edge`、既定 jpeg q90)を通す。
- 複数選択は Jev にない。本リポジトリの `classify` / 確認(yes/no)を使う。

## `scripts/benchmark_cache.py`（E5: 画像キャッシュ改造の再現用ベンチマーク）

未改造版・改造版のllama-serverを別々に起動して、それぞれで実行します（サーバーの起動・切り替えは利用者が行います）。

| オプション | 既定値 | 意味 |
|---|---|---|
| `--base-url` | `http://127.0.0.1:1234/v1` | サーバーURL |
| `--model` | （必須） | モデルID |
| `--images IMAGE_A IMAGE_B` | `dataset/images/M01.png dataset/images/G05.png` | 1枚目がA、2枚目がB |
| `--repeats` | `3` | 繰り返し回数 |
| `--taxonomy` | `taxonomy/default.yaml` | 分類体系 |
| `--max-edge` | `1024` | 送信画像の長辺 |
| `--image-format` | `jpeg` | `jpeg` / `png`。E5の再現には `png` |
| `--label` | なし | 実行の識別ラベル（例 `vanilla` / `patched`） |
| `--output` | （必須） | 出力JSONのパス |


## データセットの評価(`check-manifest` / `evaluate`)

正解ラベル付きのデータセット(manifest)があるときに、精度と時間を集計します。書式は [`dataset-plan.md`](dataset-plan.md) §4 と [`../dataset/DATASET_CARD.md`](../dataset/DATASET_CARD.md)、実物の [`../dataset/manifest.jsonl`](../dataset/manifest.jsonl) を参照してください(正解ラベルは、モデルの出力を見る前に付けます)。

1. **manifestの検証**（画像の存在・SHA256・正解ラベルの整合）:

   ```powershell
   .venv\Scripts\python.exe -m classifier_demo check-manifest --manifest dataset\manifest.jsonl
   ```

   最後に `OK` が出ます。

2. **データセット全件の評価**（内部で `check-manifest` を先に実行し、エラーがあれば中断）。これは選択式と通常JSONの対応比較で、確認オフ・JPEGの既定のままの最小の形です。

   ```powershell
   .venv\Scripts\python.exe -m classifier_demo evaluate --manifest dataset\manifest.jsonl --model <モデルID> --output-dir results\my_first_run
   ```

`--output-dir` には次のファイルが書き出されます。

| ファイル | 内容 |
|---|---|
| `run.json` | 実行条件（モデル、モード、画像の形式・長辺、確認の有無、manifest・taxonomyのSHA256、ツールのコミット、OS・Pythonなど）、除外ケース（`excluded_cases`）。`--runtime-info` を渡すとその中身も記録 |
| `cases.csv` | ケース別の予測と採点、時間、リクエスト数 |
| `summary.md` | 軸別の精度、候補別のTP/FP/FN、メタデータの一致率、処理時間、失敗の一覧 |
| `cases/<case_id>.<mode>.json` | 生の分類結果 |

- `rights_confirmed` が true でないケースは評価から除外され、理由とともに `run.json` に記録されます。
- 推論失敗・形式不正のケースも評価の分母に残ります。1件の失敗で全件は止まりません。
- 評価前に、ウォームアップ1回（`--warmup 1`）を別記録で行います（分類時間には含めません）。
- `results/` はGit管理外です。公開するときは個人のパスを除いて `doc/experiments/` に写します。
