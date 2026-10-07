# EmbeddingGemma 2 のゼロショット分類の試し(llama-server --embeddings、/v1/embeddings)。
# 事前に固定した設計(流す前に決めたもの。31枚を見て変えない):
#   - 選択肢の文: "task: classification | query: <name>: <criteria>"(分類体系の名前と説明文をそのまま)
#   - 画像: 長辺1024 の JPEG をそのまま(前置きなし)
#   - 判定: 単一選択の項目は、画像と各選択肢の文のコサイン類似度の argmax
#   - 複数選択(服装・キャラ)は、この試しでは類似度の並びを見るだけ(判定規則は未定のため採点しない)
import argparse, base64, json, sys, time, urllib.request
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from classifier_demo.image import prepare_image
from classifier_demo.taxonomy import load

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:1235/v1/embeddings")
URL = ap.parse_args().url
tax = load(str(R / "taxonomy/default.yaml"))
cases = [json.loads(l) for l in open(R / "dataset/manifest.jsonl", encoding="utf-8")]
cases = [c for c in cases if not c["derived_from"]]


def embed(items):
    body = {"input": [{"content": it} for it in items]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.perf_counter_ns()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return [d["embedding"] for d in out["data"]], (time.perf_counter_ns() - t) / 1e6


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))  # 応答は L2 正規化済み


def text(name, crit):
    return [{"type": "text", "text": f"task: classification | query: {name}: {crit}"}]


# 選択肢の文の埋め込み(先に1回だけ)
opt_vecs = {}
t_text = 0.0
for ax in tax.axes:
    vecs, ms = embed([text(c.name, c.criteria) for c in ax.choices])
    t_text += ms
    opt_vecs[ax.id] = dict(zip([c.id for c in ax.choices], vecs))
print(f"選択肢の文の埋め込み: {sum(len(v) for v in opt_vecs.values())} 本、合計 {t_text:.0f} ms")

single = [a for a in tax.axes if not a.multi]
correct = {a.id: 0 for a in single}
img_ms = []
multi_rank = []
for c in cases:
    img, mime = prepare_image(str(R / c["image_path"]), 1024, "jpeg")[:2]
    url = f"data:{mime};base64," + base64.b64encode(img).decode()
    (v,), ms = embed([[{"type": "image_url", "image_url": {"url": url}}]])
    img_ms.append(ms)
    exp = c["expected"]
    for ax in single:
        sims = {cid: cos(v, ov) for cid, ov in opt_vecs[ax.id].items()}
        correct[ax.id] += max(sims, key=sims.get) == exp[ax.id]
    for ax in [a for a in tax.axes if a.multi]:
        sims = sorted(((round(cos(v, ov), 3), cid) for cid, ov in opt_vecs[ax.id].items()), reverse=True)
        multi_rank.append((c["case_id"], ax.id, exp[ax.id], sims[:3]))
n = len(cases)
print("単一選択の正答:", {k: f"{v}/{n}" for k, v in correct.items()}, "合計", sum(correct.values()), f"/{5 * n}")
print(f"画像1枚の埋め込み: 平均 {sum(img_ms) / n:.0f} ms(最初の1枚 {img_ms[0]:.0f} ms)")
print("複数選択の類似度の上位3(参考、採点しない):")
for row in multi_rank[:16]:
    print("  ", row)
