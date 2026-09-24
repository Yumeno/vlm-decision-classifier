from pathlib import Path

import pytest

from classifier_demo import taxonomy


def _write(tmp_path: Path, content: str) -> str:
    path = tmp_path / "t.yaml"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_default_taxonomy_ok():
    tax = taxonomy.load("taxonomy/default.yaml")
    assert tax.version == "0.2.0"
    axis_ids = [a.id for a in tax.axes]
    assert axis_ids == ["image_type", "art_style", "subject", "character"]
    assert tax.sha256  # non-empty hash


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
