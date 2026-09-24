"""モデル送信用の画像準備。"""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageOps


def prepare_image(path: str, max_edge: int = 1024) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """画像を読み込み、EXIF回転補正・RGB化・縮小(長辺 max_edge、拡大はしない)して
    PNGバイト列にエンコードする。

    Returns: (png_bytes, mime, original_size, sent_size)
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    original_size = img.size
    img = img.convert("RGB")

    width, height = img.size
    long_edge = max(width, height)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    return png_bytes, "image/png", original_size, img.size


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
