from PIL import Image

from classifier_demo import pipeline
from classifier_demo.image import JPEG_QUALITY, prepare_image
from tests.fakes import FakeBackend, make_no_logprobs_response
from tests.test_pipeline import _test_taxonomy


def _make_png(path, size=(200, 100)):
    Image.new("RGB", size, (200, 100, 100)).save(path)


def test_prepare_image_default_is_jpeg(tmp_path):
    p = tmp_path / "a.png"
    _make_png(p)
    data, mime, _, _ = prepare_image(str(p))
    assert mime == "image/jpeg"
    assert data[:2] == b"\xff\xd8"


def test_prepare_image_png(tmp_path):
    p = tmp_path / "a.png"
    _make_png(p)
    data, mime, _, _ = prepare_image(str(p), image_format="png")
    assert mime == "image/png"
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_prepare_image_resizes_long_edge_for_both_formats(tmp_path):
    p = tmp_path / "a.png"
    _make_png(p, (200, 100))
    for fmt in ("jpeg", "png"):
        _, _, original, sent = prepare_image(str(p), max_edge=100, image_format=fmt)
        assert original == (200, 100)
        assert sent == (100, 50)


def test_pipeline_settings_record_image_format(tmp_path):
    p = tmp_path / "a.png"
    _make_png(p)
    tax = _test_taxonomy()
    for fmt, quality in (("jpeg", JPEG_QUALITY), ("png", None)):
        backend = FakeBackend([make_no_logprobs_response()] * 8)
        result = pipeline.classify(str(p), tax, backend, mode="choice", max_edge=64, image_format=fmt)
        assert result["settings"]["image_format"] == fmt
        assert result["settings"]["jpeg_quality"] == quality
