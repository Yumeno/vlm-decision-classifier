"""C02(メタデータと画像の矛盾・人工ケース)用: 人物なしの風景画像に、Alisa の LoRA を指定した
A1111 形式の生成情報を人工的に書き込む。画素は変えない(書き込み後に画素一致を検証する)。
評価用に作った改変メタデータであり、実際の生成記録ではない(dataset-plan §3 C02)。"""
from PIL import Image, PngImagePlugin

SRC = "dataset/staging/C02_a1.png"
DST = "dataset/staging/C02_a1_injected.png"
PARAMETERS = (
    "masterpiece, best quality, score_7, safe, <lora:fet-alisa-uniform-anima-v4u:1> fet_alisa_uniform, "
    "1girl, brown hair, straight hair, bob cut, bangs, blue eyes, white collared shirt, long sleeves, "
    "pink vest, red neck ribbon, dark pink pencil skirt, lakeside, morning\n"
    "Negative prompt: worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, "
    "chromatic aberration, nsfw\n"
    "Steps: 36, Sampler: ER SDE, Schedule type: Simple, CFG scale: 4.5, Seed: 20260952, Size: 1254x1254, "
    "Model: anima-base-v1.0, Lora hashes: \"fet-alisa-uniform-anima-v4u: 8c33eb667a34\", "
    "Note: ARTIFICIAL METADATA written for evaluation (vlm-decision-classifier C02)"
)

src = Image.open(SRC)
info = PngImagePlugin.PngInfo()
info.add_text("parameters", PARAMETERS)
src.save(DST, pnginfo=info)
out = Image.open(DST)
assert out.tobytes() == src.tobytes() and out.mode == src.mode and out.size == src.size, "pixel mismatch"
assert out.info.get("parameters") == PARAMETERS
print("ok", DST)
