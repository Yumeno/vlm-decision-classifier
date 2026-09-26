from pathlib import Path

import pytest

from classifier_demo import taxonomy


def _write(tmp_path: Path, content: str) -> str:
    path = tmp_path / "t.yaml"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_default_taxonomy_ok():
    tax = taxonomy.load("taxonomy/default.yaml")
    assert tax.version == "0.4.1"
    axis_ids = [a.id for a in tax.axes]
    assert axis_ids == [
        "image_type",
        "art_style",
        "color",
        "subject",
        "situation",
        "outfit",
        "character",
    ]
    assert tax.sha256  # non-empty hash


def test_default_taxonomy_multi_axes_have_none_criteria():
    tax = taxonomy.load("taxonomy/default.yaml")
    assert tax.axis("outfit").none_criteria == "no character appears in the image"
    assert tax.axis("character").none_criteria == "no character appears in the image"


def test_none_criteria_defaults_to_none_when_absent(tmp_path):
    content = """
version: "0.1.0"
axes:
  - id: a
    question: q
    multi: false
    allow_none: false
    choices:
      - {id: x, name: x, criteria: c}
"""
    path = _write(tmp_path, content)
    tax = taxonomy.load(path)
    assert tax.axis("a").none_criteria is None


def test_none_criteria_loaded_when_present(tmp_path):
    content = """
version: "0.1.0"
axes:
  - id: a
    question: q
    multi: true
    allow_none: true
    none_criteria: "no character appears in the image"
    choices:
      - {id: x, name: x, criteria: c}
"""
    path = _write(tmp_path, content)
    tax = taxonomy.load(path)
    assert tax.axis("a").none_criteria == "no character appears in the image"


def test_duplicate_axis_id_raises(tmp_path):
    content = """
version: "0.1.0"
axes:
  - id: a
    question: q
    multi: false
    allow_none: false
    choices:
      - {id: x, name: x, criteria: c}
  - id: a
    question: q2
    multi: false
    allow_none: false
    choices:
      - {id: y, name: y, criteria: c}
"""
    path = _write(tmp_path, content)
    with pytest.raises(ValueError, match="duplicate axis id"):
        taxonomy.load(path)


def test_duplicate_choice_id_raises(tmp_path):
    content = """
version: "0.1.0"
axes:
  - id: a
    question: q
    multi: false
    allow_none: false
    choices:
      - {id: x, name: x, criteria: c}
      - {id: x, name: x2, criteria: c2}
"""
    path = _write(tmp_path, content)
    with pytest.raises(ValueError, match="duplicate choice id"):
        taxonomy.load(path)


def test_too_many_choices_raises(tmp_path):
    choices = "\n".join(f"      - {{id: c{i}, name: c{i}, criteria: c}}" for i in range(20))
    content = f"""
version: "0.1.0"
axes:
  - id: a
    question: q
    multi: false
    allow_none: false
    choices:
{choices}
"""
    path = _write(tmp_path, content)
    with pytest.raises(ValueError, match="max is 19"):
        taxonomy.load(path)
