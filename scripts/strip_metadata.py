"""PNGのテキストチャンク(生成メタデータ)を全て除去したコピーを作る。

使い方:
    python scripts/strip_metadata.py SRC DST

*-strip 派生ケース(A01-strip 等)用。画素は変えずメタデータだけ落とす。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def strip_metadata(src: Path, dst: Path) -> Path:
    with Image.open(src) as im:
        im.load()
        mode = im.mode
        size = im.size
        palette = im.getpalette() if mode == "P" else None
        pixel_bytes = im.tobytes()

    clean = Image.frombytes(mode, size, pixel_bytes)
    if palette is not None:
        clean.putpalette(palette)

    dst.parent.mkdir(parents=True, exist_ok=True)
    # pnginfo を渡さないので、生成時にPIL/サーバーが書き込んだtEXt/iTXtチャンクは一切引き継がれない
    clean.save(dst, format="PNG")

    with Image.open(dst) as check:
        check.load()
        if check.tobytes() != pixel_bytes:
            raise ValueError(f"メタデータ除去後に画素が一致しません: {src} -> {dst}")

    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="PNGのメタデータ(テキストチャンク)を除去する")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()

    if not args.src.exists():
        print(f"[エラー] 入力ファイルがありません: {args.src}", file=sys.stderr)
        sys.exit(1)

    strip_metadata(args.src, args.dst)
    print(f"-> {args.dst}")


if __name__ == "__main__":
    main()
