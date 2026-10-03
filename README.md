# VLM Decision Classifier

ローカルの視覚言語モデル(VLM)に画像と**選択肢**を渡し、モデルが最初に返す回答ラベル(`A`、`B` …)の確率から、候補ごとの**相対スコア**を読み取って画像を分類するツールです。

- 画像はこのPCの推論サーバー(LM Studio または llama-server)にしか送りません。クラウドには送りません。
- 「この画像はどれ?」を文章で書かせるのではなく、選択肢の記号を1つ答えさせるので、形式が崩れにくく、候補ごとのスコアも取れます。
- Windows 11 で開発・確認しています(Linux も動く想定ですが未検証)。
- 結果の要約: Qwen3.5 9B / Gemma 4 12B では、全項目を1回で聞く「束ね質問」が通常JSONより14〜33%速く、小型モデルでは1項目ずつの選択式が形式不正0件で頑健でした(元画像31枚、各条件1回の小標本)。
- 解説記事(note): <https://note.com/yumenojmd/n/nefe57d7a951b> / 結果ダッシュボード: <https://yumeno.github.io/vlm-decision-classifier/>

## 何ができるか

- **1項目ずつの選択式**: 項目(画風、色、被写体など)ごとに選択肢を提示し、回答ラベルのスコアを読む。
- **束ね質問**: 全項目を1回のリクエストで答えさせる(速い。モデルによっては形式が崩れる)。
- **複数回答可の項目**(服装・キャラなど): 候補ごとに yes/no を確認して採用できる(`--confirm`、束ね質問では Y/N 欄)。
- **通常JSON / 型を縛ったJSON**: 比較用に、JSONを生成させる方式(`json`)と、llama-server の制約付きデコードを使う方式(`json_schema`)も選べる。
- **デモ画面**: ブラウザで画像をドロップして、スコアが伸びる様子を見られる。2方式の並べて比較もできる。
- **Jev 風の呼び出し `system_one`**: TypeSafe の Jev SDK と同じ形(`choice` / `noul` / `score`)で、手元のVLMやテキストだけの判定に使える(実験的)。

## 必要なもの

| 項目 | 内容 |
|---|---|
| OS | Windows 11 で確認。Linux / macOS は未検証 |
| Python | 3.11 以上(依存は `pillow`・`pyyaml` のみ) |
| 推論サーバー | LM Studio、または llama-server(llama.cpp)。OpenAI互換の chat completions に接続する。**このツールはサーバーの起動・モデルのダウンロードをしない** |
| GPU / VRAM | 記録があるのは RTX 3090 24GB。Qwen3.5 9B(Q4_K_M、コンテキスト8192、並列4)を llama-server で載せて約6.8GB、Gemma 4 12B(Q4_K_M)は LM Studio の `lms ps` で7.56GB。小型モデル(Qwen3.5 4B/2B/0.8B、Gemma 4 E4B/E2B)のVRAM使用量は記録していません。8GB未満のGPUでは確認していません |

## 導入

### 1. このツールを入れる

Windows(PowerShell):

```powershell
git clone https://github.com/Yumeno/vlm-decision-classifier.git
cd vlm-decision-classifier
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

Linux / macOS(bash):

```bash
git clone https://github.com/Yumeno/vlm-decision-classifier.git
cd vlm-decision-classifier
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

`py -3.12` は開発機の都合です。3.11 以上の Python があればそれで構いません。以降のコマンドは、リポジトリのルートで実行します(Linux / macOS は `.venv\Scripts\python.exe` を `.venv/bin/python`、パス区切りの `\` を `/` に読み替え)。テストを回すなら `pip install -e ".[dev]"` のあと `.venv\Scripts\python.exe -m pytest -q`(サーバー不要)。

### 2. モデルを用意する

モデル重み・視覚プロジェクタ(mmproj)は同梱していません。Hugging Face から取得します(GGUF と mmproj の両方が必要)。

| モデル | Hugging Face リポジトリ | GGUF | mmproj |
|---|---|---|---|
| Qwen3.5 9B | `lmstudio-community/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-Q4_K_M.gguf` | `mmproj-Qwen3.5-9B-BF16.gguf` |
| Gemma 4 12B | `lmstudio-community/gemma-4-12B-it-GGUF` | `gemma-4-12B-it-Q4_K_M.gguf` | `mmproj-gemma-4-12B-it-BF16.gguf` |

実験的に、拡散言語モデル DiffusionGemma(4bit AWQ)を WSL2 の vLLM で動かす方式(画像1枚・全質問を1回の読み出し、`dgemma_choice`)も試しています。GGUF ではなく別の手順で、結果と条件は [`doc/experiments/dgemma/README.md`](doc/experiments/dgemma/README.md)、手順は [`doc/reproduce.md`](doc/reproduce.md) にあります。

小型モデルは `lmstudio-community` の `Qwen3.5-4B/2B/0.8B-GGUF`、`gemma-4-E4B/E2B-it-GGUF`(ファイル名は [`doc/experiments/runtime/small-S*.json`](doc/experiments/runtime/))。

ダウンロードしたファイルは、[`doc/experiments/runtime/*.json`](doc/experiments/runtime/) に記録した SHA256 と照合してから使ってください(実験で使ったファイルと同じかの確認にもなります)。

### 3. サーバーを起動する

**LM Studio**: モデルを取得してロードします。先に `lms ps` で、ほかのモデルがロード中でないか確認してください(押し出しやVRAM不足の原因になります)。実験では次のコマンドで、コンテキスト8192・並列4になっていました。

```powershell
lms load qwen3.5-9b --identifier qwen3.5-9b -y
```

**モデルID** は `--identifier` の値で、`lms ps` または `GET http://127.0.0.1:1234/v1/models` で確認できます。以降の `--model` に渡します。LM Studio の接続先の既定は `http://127.0.0.1:1234/v1` です。

**llama-server**(`--base-url http://127.0.0.1:1235/v1` のように場所を渡す)。実験で使った起動コマンド(`doc/experiments/runtime/qwen-llamaserver-patched.json` の記録):

```
llama-server -m Qwen3.5-9B-Q4_K_M.gguf --mmproj mmproj-Qwen3.5-9B-BF16.gguf --alias qwen3.5-9b -ngl 99 -c 8192 -np 4 --kv-unified -sm none -mg 0 --port 1235
```

- `--alias` が `--model` に渡すモデルIDになります。
- `--kv-unified` を付けないと、`-c 8192 -np 4` がスロットごとに 2048 に分割されます。
- `-sm none -mg 0` は複数GPUのうち1枚に載せる指定で、番号はCUDAの並び順です(`nvidia-smi` の番号と違うことがあります。`llama-server --list-devices` で確認)。GPUが1枚なら不要です。

### 4. (任意)llama.cpp の画像キャッシュ改造パッチ

Qwen3.5 では、llama.cpp が連続する質問で同じ画像を再エンコードしてしまう問題があります([上流 issue #26994](https://github.com/ggml-org/llama.cpp/issues/26994))。[`doc/patches/llamacpp-mtmd-checkpoint.patch`](doc/patches/llamacpp-mtmd-checkpoint.patch)(上流 `f95b0d9` の `tools/server/server-context.cpp` の1行を無効にするだけ)を当てて `llama-server` をビルドすると、同じ画像への2回目以降のリクエストが速くなります。**判定結果は変わりません。**

- 効くのは、リクエストを何回も送る使い方(1項目ずつの選択式、確認つき)です。Qwen3.5 9B での1項目ずつ(確認オフ)は、未改造版5.16秒 → 改造版1.89秒(初めて見る画像)、5.12秒 → 1.29秒(画像読み込み済み)。束ね質問や通常JSON(リクエスト1回)への効果は小さめです。
- 適用とビルドは、上流の `f95b0d9` を取得して `git apply doc/patches/llamacpp-mtmd-checkpoint.patch` し、CUDA でビルドします。手順とコマンドは [`doc/experiments/phase4-runbook.md`](doc/experiments/phase4-runbook.md) §1 にあります。
- 改造なしでも動きます。使う必要はありません。

### 5. 疎通確認

`reasoning_effort: "none"` は、`probe` / `classify` / `serve` などが既定でリクエストに入れます(外すと Qwen/Gemma とも thinking が先に出て、1トークン目の回答ラベルの確率が取れません)。サーバーが拒否して400を返したときは、外して再送し、その事実を結果に記録します。

```powershell
.venv\Scripts\python.exe -m classifier_demo probe --model qwen3.5-9b
```

`answer:` と `relative_scores:` が出て、`logprobs present: yes`、`thinking detected: no`、`reasoning_effort dropped: False` なら準備完了です。llama-server なら `--base-url http://127.0.0.1:1235/v1` を付けます。

## デモ画面で試す

```powershell
.venv\Scripts\python.exe -m classifier_demo serve --port 8765
```

1. 先に、LM Studio / llama-server でモデルをロードしておく(`serve` はサーバーもモデルも起動しません)。
2. ブラウザで `http://127.0.0.1:8765/` を開く。待ち受けはこのPCだけ(`127.0.0.1` 固定)です。使用中のポートを指定するとエラーで終了するので、`--port` を変えてください。
3. 右上の「設定」で、サーバーURLとモデルIDを入れる。
4. 画像をドロップ(またはクリックして選択)し、「判定する」を押す。

画面の読み方:

- 各結果列は、上から「経過時間・リクエスト数」「判定結果の表(項目ごとに1行。UI上の表記は「軸」)」「出力の中身」の順に並びます。
- **青い帯(スコア帯)** は、その項目の候補ごとの相対スコアです(候補内で合計1に割り直した値で、正答確率ではありません)。複数回答可の項目で「候補ごとに確認する」がオフのときは、採用の閾値の位置に縦線が出ます。
- **黄土色の棒** は、候補ごとの yes/no 確認での P(yes) です(縦線が 0.5)。
- **JSON の列** では、通常JSONの生テキストとラベルが出ます。
- 比較モードでは、左右のラベルが食い違う項目に「不一致」の印が付きます。
- 失敗(ラベルが出ない、thinking が先に出る、確率が取れない、通信エラー、JSON形式不正)は、例外で止めずにその場に理由を表示します。別方式へのフォールバックはしません。

![束ね質問と通常JSONの比較](doc/images/demo-bundled-vs-json.png)

*束ね質問(順位付け、0.8秒・1リクエスト)と通常JSON(1.7秒)の比較。M01(Alisa と second original の2人)で、画像を先に読み込み済み。キャラの行に「不一致」の印: 順位付けは相対スコアを候補どうしで取り合うため other original の1つだけを選び、JSON は2人とも挙げた。2人を拾うには「束ね質問の複数選択」を「候補ごとに Y/N」にする。*

![1項目ずつのスコア帯とY/N](doc/images/demo-choice-axes.png)

*1項目ずつ・候補ごとに確認(yes/no)オンの服装とキャラ。青い帯が順位付けの相対スコア、黄土色の棒が候補ごとの P(yes)(縦線が 0.5)。確認では alisa と second original の両方が採用された。*

## 設定画面の説明

設定はブラウザに保存され、次回開いたときに復元されます。「初期値に戻す」で初期値に戻せます。UI上は、項目のことを「軸」と表記しています。

![設定パネル](doc/images/demo-settings.png)

| UIの項目 | 既定 | 何が変わるか・使いどころ |
|---|---|---|
| 比較 | 比較なし | 「選択式 と 通常JSON」: 同じ画像を選択式→通常JSONの順に実行し、左右に並べる(同時実行はしない)。方式の違いを見るとき。「サーバー1 と サーバー2」: 2つのサーバーURLに選択式を順に流して時間を比べる(未改造/改造の llama-server など) |
| 選択式の聞き方 | 1軸ずつ | 「1軸ずつ」は項目ごとに1リクエスト。形式が崩れにくい。「束ね質問」は全項目を1リクエストで聞く。速いが、小型モデルでは形式が崩れることがある |
| 束ね質問の複数選択 | 相対スコアで判定 | 束ね質問のときだけ有効。「相対スコアで判定」は候補どうしでスコアを取り合うため、2つ以上写る画像を取りこぼす。複数写る画像(服装・キャラ)を拾いたいときは「候補ごとに Y/N」(CLIの `--bundled-multi yn`) |
| サーバーURL 1・2 / モデルID | LM Studio の URL | 接続先とモデルID。URL 2 は比較で「サーバー1 と サーバー2」を選んだときだけ使う。接続先はこのPC上のサーバーだけ(ループバック以外は拒否) |
| メタデータ証拠を表示 | オフ | PNG の生成メタデータ(LoRA名・トリガーワード)からのキャラ一致と、画素からのキャラ判定を並べて表示する。両者は統合せず別々に並べる。LoRA名の記録は「生成時の指定」の証拠で、LoRAが効いたことや、そのキャラが写っていることの証明ではない |
| 先に画像を読み込む(ホットロード) | オフ | 判定前に、画像だけの準備リクエストを1回送ってサーバーのキャッシュに載せる。準備した後の判定時間(「読み込み済み」)が測れる。準備の時間は画像欄の下に小さく出て、列の経過時間には含まない。比較では、選択式と JSON は先頭が違い、サーバー1と2は別サーバーなので、各列が自分の準備を送る。キャッシュ改造版のサーバーで効く |
| 軸を並列に送る | オフ | 1項目ずつの質問を4並列で送る。**llama-server では得にならなかった**(E9)ので、通常は使わない |
| 候補ごとに確認する(yes/no) | オフ | オンにすると、複数回答可の項目で順位付けのあとに候補ごとの yes/no 確認(P(yes))を行う。リクエストが増える。オフなら、スコア帯に閾値の縦線が出て、閾値を超えた候補を採用する |
| 採用の閾値(confirmオフ時) | 0.5 | 複数回答可の項目で、相対スコアがいくつ以上の候補を採用するか(`--rank-threshold` と同じ)。下げると拾いやすく、上げると厳しくなる |
| 画像の長辺 | 1024 | モデルへ送る画像の長辺(px)。768 に下げると、初めて見る画像の判定時間が短くなる |
| 送信形式 | JPEG | JPEG(quality 90)または PNG。既定のJPEGで十分。PNGは過去の実験の再現用 |

## 分類体系(YAML)の書き方

「何を、どんな選択肢で判定するか」は、YAML(分類体系、taxonomy)で決まります。既定は [`taxonomy/default.yaml`](taxonomy/default.yaml)(生成画像向けの7項目。`image_type`・`art_style`・`color`・`subject`・`situation`・`outfit`・`character`)です。自分の項目で使うには、これを元に書き換えて `--taxonomy` で渡します(`classify` / `serve` / `evaluate` / `check-manifest`)。デモ画面の設定には、分類体系の切り替えはありません(`serve --taxonomy` で起動時に指定)。

```yaml
version: "1.0"            # 版。結果に記録される(SHA256も記録)
axes:                     # 判定する項目(ここでは「軸」と呼ぶ)
  - id: color              # 項目のID
    question: "How is color used in this image?"   # モデルに見せる質問文
    multi: false            # true なら複数選択(複数の候補を採用できる)
    allow_none: false       # true なら「どれでもない(none)」を選べる
    choices:
      - {id: full_color, name: full color, criteria: "an image in full color"}
      - {id: other, name: other, criteria: "none of the above"}
  - id: character
    question: "Which character appears in this image? Ignore tiny background figures."
    multi: true
    allow_none: true
    none_criteria: "no character appears in the image"   # noneの説明文(allow_none時)
    choices:
      - id: alisa
        name: Alisa
        criteria: "girl with straight brown bob-cut hair ..."
        lora_names: [fet-alisa-uniform-anima-v4u]       # 生成メタデータのLoRA名と規定正規化後の完全一致で照合する(任意)
        trigger_words: [fet_alisa_uniform]               # 任意
      - id: other_original
        name: other character
        criteria: "any person or humanoid character who is neither Alisa nor the second original character"
        catch_all: true    # 「その他」の受け皿(任意)
```

- **項目**: `id`・`question`・`choices` が必須。`multi`(既定false)、`allow_none`(既定false)、`none_criteria` は任意。
- **選択肢**: `id`・`name`・`criteria`(判定基準の文)が必須。`catch_all`、`lora_names`・`trigger_words` は任意。`lora_names`・`trigger_words` はモデルに渡さず、PNG の生成メタデータ(A1111 / ComfyUI 形式)との照合だけに使います(LoRA名は規定正規化後の完全一致のみ)。画素の判定と突き合わせた `combined_evidence` は、項目のIDが `character` のときに作られます。メタデータを使わないなら不要です。
- **単一選択(`multi: false`)** は、選択肢の中から1つを選ぶ項目。**複数選択(`multi: true`)** は、閾値または yes/no 確認で複数を採用する項目です。
- **質問文・選択肢の文言に、正解(LoRA名など)を漏らさない**でください。
- **選択肢の数**: 1項目あたり `none` を含めて最大52(ラベルは `A`〜`Z`・`a`〜`z` を自動で割り当て)。超過はエラーです(黙って切り捨てません)。各ラベルが使用モデルで1トークンになり、1トークン目が互いに異なる必要があります(Qwen/Gemma で確認)。26以下は大文字小文字を区別しない照合、27以上は区別する照合です。
- **全候補のスコアを観測できるのは20候補以内**です(サーバーが返す `top_logprobs` が上位20件のため。21以上ある項目では、上位20に現れない候補の相対スコアは0として扱います)。選択肢が多いときは、項目を分けてください。
- 分類体系を変えたら、`version` を上げ、結果を見てから文言を調整しないでください(精度を比べる実験に使う場合)。

## `system_one` から呼ぶ

TypeSafe の Jev SDK(`typesafe-sdk`)と同じ呼び出しの形で、手元の VLM を呼ぶ最小クライアント(`classifier_demo/systemone.py`、実験的)です。SDK には依存せず、名前とフィールドを写しただけで、TypeSafe 社とは無関係です。

```python
from classifier_demo.systemone import SystemOneClient

client = SystemOneClient(base_url="http://127.0.0.1:1234/v1", model="<モデルID>")
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

CLI(手動確認用): `python -m classifier_demo systemone --base-url ... --model ... --questions q.json [--state TEXT | --state-file F] [--image PATH ...] [--prime]`。`q.json` は `{質問名: {type, instructions, criteria}}` です。

| 質問の型 | 返すもの |
|---|---|
| `choice` | 選択肢ごとの相対スコア `probabilities` と、最大の `choice` |
| `noul` | yes/no の2択。yes の相対スコア |
| `score`(2〜10段階。`criteria` は順序付きリスト) | 相対スコアで重み付けした期待値 `score`(0〜n-1)、`legend`、`probabilities` |

- **`images`**: openjev の `images` と同じ形式(`data:image/...;base64,` URL、または `{"content_type", "base64"}`)。加えてファイルパスと bytes も受けます。8枚まで・1枚5MiBまで・jpeg/png/webp/gif のみ(違反は明示エラー)。送信前に長辺 `--max-edge`(既定1024)・JPEG q90 へ縮小します。
- **`prime=True`**: 先頭(system → 画像 → state)だけのリクエストを先に送ってキャッシュに載せる。時間は `usage.prime_elapsed_ms` に別記録されます。先頭は全質問で共有し、質問は互いに見えません(束ねません)。
- **限界**: スコアは候補内で割り直した相対スコアで較正されていません。全候補を観測できるのは20候補以内、`choice` は最大52ラベルです。`confidence` は返しません。複数選択は Jev にないので、次の `classify` を使います。詳細は [`doc/cli.md`](doc/cli.md)。

## CLI で処理する

### 1枚を分類する(`classify`)

`system_one` にない機能(分類体系の全項目を1枚に対して判定、複数回答可の項目の採用)は `classify` を使います。

```powershell
.venv\Scripts\python.exe -m classifier_demo classify dataset\images\M01.png --model qwen3.5-9b --confirm --output results\demo.json
```

- `--mode`: `choice`(1項目ずつ。既定)/ `bundled`(束ね質問)/ `json`(通常JSON)/ `json_schema`(JSONを制約付きデコードで生成。llama-server で測定)/ `dgemma_choice`・`dgemma_json`(DiffusionGemma、vLLM。[`doc/cli.md`](doc/cli.md) と [`doc/reproduce.md`](doc/reproduce.md))。
- 複数回答可の項目(服装・キャラ)の採用: `--confirm` で上位候補ごとに yes/no を確認する。付けなければ相対スコアを閾値(`--rank-threshold`、既定0.5)で採用する。束ね質問では `--bundled-multi yn` で候補ごとの Y/N 欄にできる。
- 出力: 画面に、項目ごとの採用タグ、エラー、判定時間、リクエスト数が出ます。`--output` を指定した先(省略時は画面)には、次を含む結果JSONが出ます。`vision_tags`(項目ごとの採用タグ)、`axis_decisions`(候補ごとの相対スコア・yes/no 確認・失敗の有無)、`metadata_evidence`(PNG の生成メタデータ)、`combined_evidence`、`timing_ms`、`request_count`、`errors`。
- 全オプションは [`doc/cli.md`](doc/cli.md)。

### たくさんの画像をまとめて処理する

フォルダ単位のバッチ処理は、まだ組み込んでいません([issue #9](https://github.com/Yumeno/vlm-decision-classifier/issues/9))。当面は、`classify` をファイルごとに呼ぶループで代用できます(PowerShell。1枚ごとに `results\batch\<ファイル名>.json` を書き出し、失敗した画像があっても続行します)。

```powershell
foreach ($f in Get-ChildItem path\to\images\* -Include *.png,*.jpg,*.jpeg,*.webp -File) {
    .venv\Scripts\python.exe -m classifier_demo classify $f.FullName --model qwen3.5-9b --confirm --output "results\batch\$($f.BaseName).json"
}
```

結果の採用タグだけを一覧にするなら:

```powershell
foreach ($j in Get-ChildItem results\batch\*.json) {
    $r = Get-Content $j.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    "$($r.image.name)`t$($r.vision_tags | ConvertTo-Json -Compress)"
}
```

モデルのロードは1回で済みますが、`classify` は毎回Pythonを起動します。正解ラベル付きのデータセットがあれば、`evaluate`(精度・時間の集計つき)も使えます([`doc/cli.md`](doc/cli.md))。

## モデルの選び方とおすすめの設定

根拠は [`doc/experiments/reprime/summary.md`](doc/experiments/reprime/summary.md)(F1〜F3・S1〜S9。各条件1回、元画像31枚)と [`doc/experiments/e10/summary.md`](doc/experiments/e10/summary.md)(E10・E10b)です。小標本なので、数件の差や数十msの差から強い結論は出せません。

| 目的 | おすすめ | 根拠 |
|---|---|---|
| 9B / 12B で速く | **束ね質問**(順位付け、`--mode bundled`) | 通常JSONより14〜33%短い(F1〜F3の全12条件)。Qwen3.5 9B 改造版・長辺1024で、初めて見る画像 1.19秒 対 JSON 1.62秒、読み込み済み 0.71秒 対 0.96秒。失敗0件 |
| 9B / 12B で複数回答可の項目(服装・キャラ)も拾う | 束ね質問 + **Y/N 欄**(`--bundled-multi yn`) | キャラは全条件で31/31、服装 83.9〜93.5%。時間は JSON と同程度か遅い(1.00〜1.33倍)。順位付けは候補どうしで取り合うので、複数写る画像を取りこぼす |
| 小型モデルで安定して | **1項目ずつ**(`--mode choice`) | Qwen3.5 4B/2B/0.8B、Gemma 4 E4B/E2B の全モデルで形式不正0件。単一選択5項目の平均は 85.8〜91.0%(0.8B でも 86.5〜87.1%)。ただし0.8Bは服装・キャラの完全一致が低い(0.8B Q4: 服装48.4%・キャラ58.1%) |
| 画像1枚の全質問を1回で読む(実験的、DiffusionGemma) | **`dgemma_choice`**(WSL2 の vLLM + 自前の判定サーバー。推奨設定は下の注意点) | 同じモデルの通常JSON(`dgemma_json`)と同じ31枚で比べて、単一5項目 142〜143/155(JSON 136)、平均 約0.47秒(JSON 1.19秒)。キャラ 29〜30/31(「その他」の書き方 `criteria` のとき)。服装は通常JSONより低め(20〜22/31) |
| 小型モデルで JSON を使いたい | **`json_schema`**(制約付きデコード、llama-server) | 通常JSONは小型ほど形式不正が増える(S1〜S9で31枚中2〜10件)。`json_schema` は E10b で測った5モデル(2B Q4、0.8B Q4、E4B、E2B Q4、9B)すべてで0件になり、単一5項目の平均は 83.9〜91.0%。時間は通常JSONと同程度。**構文だけでなく値の集合も縛るので、精度の改善には両方の効果が含まれる** |

注意点:

- **束ね質問が崩れるモデルがある**。rank(順位付け)で形式不正が多かったのは Qwen3.5 4B(Q4: 31枚中30件、Q8: 28件)、Gemma 4 E4B(12件)、Qwen3.5 2B Q8(6件)。Y/N 欄はさらに崩れやすく、Qwen 4B、2B、Gemma E4B/E2B で10件以上の崩れがありました。使う前に、自分のモデルで少数の画像を試してください。ノイズ画像100枚の E10 でも、束ね質問は小型モデルで崩れ(Qwen 2B Q8 は100/100)、1項目ずつは全モデル0/100でした。
- **パッチの効果**: 1項目ずつや確認つきのように、リクエストが多い使い方で効きます(上記「導入」の4)。未改造の llama-server では、1項目ずつが遅く(Qwen 9B で 5.16秒)、束ね質問でも通常JSONより速いのは変わりません(1.38秒 対 1.61秒)。
- **画像の長辺**: 768 にすると、初めて見る画像(準備なし)では全方式が短くなります(F1: 束ね 1188→1008ms、JSON 1624→1340ms、1項目ずつ(確認オフ) 1890→1637ms)。精度は多くの項目で同じでした。準備ありでは差が小さく、768のほうが遅い方式もありました(1回測定で原因は切り分けていません)。
- **Gemma 4 12B** は1リクエストの固定費が Qwen 9B より大きく、1項目ずつが遅い(長辺1024、準備なし 3.00秒 対 1.89秒)。
- **DiffusionGemma(実験的)**: RTX 3090(24GB)1枚で、4bit AWQ 版を vLLM で動かしています(0.88 の設定で VRAM 約22GB)。推奨は、vLLM を `GPU_MEM_UTIL=0.88 MST=140`(画像トークン 140)で起動し、評価や分類に `--dgemma-yn-style yn --dgemma-catchall-style criteria --dgemma-adaptive-threshold 0.87 --dgemma-adaptive-max 3`(迷ったときだけ読み増す)を付けることです。並列に投げてもほぼ速くなりません。同じ画像への再質問は、キャッシュが効いて速くなります。同じ入力でも読むたびに値がわずかに揺れます(エンジン側の非決定性。出どころは未調査)。詳細は [`doc/experiments/dgemma/README.md`](doc/experiments/dgemma/README.md) と [`followup-12.md`](doc/experiments/dgemma/followup-12.md)、手順は [`doc/reproduce.md`](doc/reproduce.md)。
- **確認(yes/no)は万能ではない**: 1項目ずつ+確認オンの outfit は、近い候補にも「はい」が付きやすく、確認オフ(90.3%)より低い(77.4%、F1 長辺1024)一方、character は少し上がります(93.5% 対 90.3%)。

## 限界と注意

- **スコアは相対スコア**です。列挙した候補内で合計1に割り直した値で、正答確率ではなく、較正もしていません([issue #8](https://github.com/Yumeno/vlm-decision-classifier/issues/8))。「スコア0.9だから90%正しい」とは読めません。
- **全候補を観測できるのは20候補以内**です(`top_logprobs` が上位20件)。1項目あたりの最大は52選択肢です(超過はエラー)。
- **小標本**です。評価に使ったのは元画像31枚(メタデータ除去コピーを加えて35ケース)で、各条件は1回の測定です。少数の結果から強い結論を出さないでください。
- **測定は1台のPC**(RTX 3090 の外付けGPU)だけです。ほかのGPUで同じ速度は出ません。
- **束ね質問は形式が崩れやすい**モデルがあります(上記)。失敗は失敗として記録し、別方式へ黙ってフォールバックしません。
- **速度の前提**: 既存のVLMを使う方式では、画像の最初の読み込みと、リクエストごとの画像の扱い(送り直しとキャッシュからの復元)が必ず乗ります。テキストだけの Jev ほどは速くなりません([`doc/experiments/report.md`](doc/experiments/report.md) §1.1)。
- 画像内・メタデータ内のテキストは命令として扱いません。
- 本デモの目的は、選択式判定の**技術検証**です。精度を追い込むことや、メタデータ補助による精度向上は目的にしていません。
- このリポジトリは TypeSafe 社とは無関係です(`system_one` は同社の Jev SDK の呼び出しの形を写しただけです)。

### 困ったとき

| 症状 | 原因と対処 |
|---|---|
| `probe` や `classify` が HTTP 400 で失敗する | LM Studio にそのモデルがロードされていない、またはモデルIDの誤り。`lms ps` で確認し、`--base-url` と `--model` を合わせる |
| `thinking_before_answer` | thinking が先に出て、回答ラベルの確率が取れない。`reasoning_effort: "none"` が効いていない(サーバーが拒否して外された場合は `probe` の `reasoning_effort dropped` が `True`)。サーバー/モデルのテンプレートを確認する |
| `no_label_tokens` | 回答に選択肢のラベルが出ていない。強制せず、失敗として記録される。`probe` でモデルの対応を確認する |
| `json_format_error` | 通常JSONの形式不正(単一選択の項目をリストで返すなど)。失敗として分母に残る。`json_schema` で避けられる |
| `bundled_format_error` | 束ね質問の回答が形式を守れなかった(小型モデルで多い)。1項目ずつに切り替える |
| LM Studio で連続実行が500/Channel Errorになる | クライアントを直す前に、LM Studio のサーバーログを確認する |
| デモ画面が起動しない / 別アプリの画面が出る | ポートが使用中。`--port` で別の番号を指定し、ブラウザのURLと合わせる |
| llama-server で `-mg` が意図と違うGPUを指す | `-mg` はCUDAの並び順で、`nvidia-smi` の番号と違うことがある。`llama-server --list-devices` で確認する |
| llama-server の停止 | イメージ名で止めると別の llama-server も止まる。PID を指定する(Windows は `taskkill /F /PID <PID>`) |
| 速度が思ったより出ない | ほかのプロセスがGPUを使っていないか、サーバーが画像キャッシュ改造版か(未改造だと1項目ずつが遅い)、準備(`--prime`)の有無を確認する |

## ライセンス

- コード・文書・データセットのテキスト(manifest・正解ラベル・生成記録など): [MIT](LICENSE)
- データセットの画像(`dataset/images/`): 著作権者 資材部の懲りない面々(http://jmd.ickx.jp/fet)。Alisa を含む画像は [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)、それ以外は [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。対象の画像の一覧は [`dataset/DATASET_CARD.md`](dataset/DATASET_CARD.md)。
- モデルの重み・LoRA は同梱していません。各モデルのライセンスに従ってください。

## クレジット

- 発想の起点は Jev と、[Google Gemma が紹介した「DiffusionGemma as Jev」の投稿](https://x.com/googlegemma/status/2101069861598482817)です。紹介された [vLLM の PR #57250](https://github.com/vllm-project/vllm/pull/57250) は、DiffusionGemma の出力キャンバスに回答スロットを設け、選択肢の分布を読み出す仕組みを示しています。このデモは Jev やその PR の実装を移植するものではなく、Qwen/Gemma の**自己回帰VLM**へ画像と選択肢を送り、chat completions の `logprobs`(先頭の回答トークン)を読む独立した実演です。「logprobs による分類そのものが新発明」とは主張しません。
- 評価用の画像は、このリポジトリのために新規生成したものです(元の画像管理プロジェクトの画像や既存の評価画像は含みません)。著作権者は上記のとおり。
- llama.cpp の画像キャッシュ改造パッチは、上流 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) の1行を無効にするものです。

## 関連資料

- [この実験の解説記事(note)](https://note.com/yumenojmd/n/nefe57d7a951b)
- [結果ダッシュボード](https://yumeno.github.io/vlm-decision-classifier/)(`site/index.html` を直接ブラウザで開いてもローカルで見られます。データは `doc/experiments/reprime/` の測定)
- [Qwen3.5 9B(公式モデル)](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Gemma 4 12B it(公式モデル)](https://huggingface.co/google/gemma-4-12B-it)
- [vLLM PR #57250 — DiffusionGemmaの構造化読み出し](https://github.com/vllm-project/vllm/pull/57250)
- [llama.cppの画像キャッシュに関するissue](https://github.com/ggml-org/llama.cpp/issues/26994)

## ほかの文書

- [`doc/reproduce.md`](doc/reproduce.md): 実験の再現手順(E1〜E10、F1〜F3、S1〜S9)、結果の表、開発の経緯。
- [`doc/cli.md`](doc/cli.md): CLI のオプション一覧、データセット全件の評価。
- [`doc/experiments/report.md`](doc/experiments/report.md): 実験レポート(条件・結果・限界)。
- [`doc/maintenance.md`](doc/maintenance.md): 保守メモ(ダッシュボードの更新手順、リポジトリ設定)。
- [`dataset/DATASET_CARD.md`](dataset/DATASET_CARD.md): 評価用データセット v1.0.0(元画像31枚 + メタデータ除去コピー4件 = 35ケース、全件SFW)。
- 設計文書: [`doc/requirements.md`](doc/requirements.md)・[`doc/basic-design.md`](doc/basic-design.md)・[`doc/implementation-experiment-plan.md`](doc/implementation-experiment-plan.md)、作業記録 [`doc/worklog.md`](doc/worklog.md)。

## 構成

| パス | 役割 |
|---|---|
| `classifier_demo/` | コア(メタデータ抽出、選択式判定、JSONベースライン、束ね質問、評価器)、CLI、デモ画面のサーバー(`server.py`、`web/index.html`)、Jev 風クライアント(`systemone.py`)、DiffusionGemma 用の方式と判定サーバー(`dgemma.py`、`dgemma_server.py`) |
| `taxonomy/default.yaml` | 分類体系 0.4.1(7項目)と自作キャラの定義 |
| `dataset/` | 評価用データセット v1.0.0(画像、`manifest.jsonl`、`DATASET_CARD.md`、正解・生成記録) |
| `scripts/` | データセット生成・組み立て、実験の実行ループ・集計(`run_final_matrix.sh` など。[`doc/reproduce.md`](doc/reproduce.md)) |
| `doc/` | 設計文書、実験記録(`experiments/`)、パッチ(`patches/`) |
| `site/` | 結果ダッシュボード(GitHub Pages) |
| `tests/` | pytest |
| `AGENTS.md`、`CLAUDE.md` | AIコーディングエージェント向けの作業規約 |
