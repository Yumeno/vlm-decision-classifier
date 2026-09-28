"""モデル送信用の画像準備。"""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageOps


JPEG_QUALITY = 90


def prepare_image(
    path: str,
    max_edge: int = 1024,
    image_format: str = "jpeg",
    jpeg_quality: int = JPEG_QUALITY,
) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """画像を読み込み、EXIF回転補正・RGB化・縮小(長辺 max_edge、拡大はしない)して
    image_format("jpeg" または "png")でエンコードする。

    Returns: (image_bytes, mime, original_size, sent_size)
    """
    if image_format not in ("jpeg", "png"):
        raise ValueError(f"unknown image_format: {image_format}")
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
    if image_format == "jpeg":
        img.save(buf, format="JPEG", quality=jpeg_quality)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return buf.getvalue(), mime, original_size, img.size


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
