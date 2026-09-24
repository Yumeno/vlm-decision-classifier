import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, PngImagePlugin

from classifier_demo import evaluate
from tests.fakes import FakeBackend, make_logprobs_response, make_text_response

import urllib.error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_plain_png(path: Path) -> None:
    Image.new("RGB", (16, 16), (10, 20, 30)).save(path)


def _make_alisa_lora_png(path: Path) -> None:
    img = Image.new("RGB", (16, 16), (200, 100, 100))
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "parameters",
        "masterpiece, <lora:fet-alisa-uniform-anima-v4u:0.8>, 1girl\nSteps: 20",
    )
    img.save(path, pnginfo=info)


def _minimal_case(case_id: str, image_path: str, image_sha256: str, **overrides) -> dict:
    case = {
        "case_id": case_id,
        "source_image_id": case_id,
        "derived_from": None,
        "image_path": image_path,
        "image_sha256": image_sha256,
        "split": "test",
        "scenario": "general",
        "expected": {"image_type": "illustration", "art_style": "anime_2d", "subject": "person", "character": []},
        "expected_metadata": {
            "format": "none",
            "loras": [],
            "trigger_words": [],
            "characters": [],
            "artificial": False,
        },
        "generation": {"tool": "test", "seed": 1},
        "rights": {"terms": "test-license", "rights_confirmed": True},
        "review": {"status": "accepted", "note": None},
    }
    case.update(overrides)
    return case


def _write_manifest(path: Path, cases: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")


# ---------------------------------------------------------------------------
# check_manifest: 異常系
# ---------------------------------------------------------------------------


def test_check_manifest_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    case = _minimal_case("A01", "does_not_exist.png", "0" * 64)
    _write_manifest(Path("manifest.jsonl"), [case])

    result = evaluate.check_manifest("manifest.jsonl")
    assert any("does not exist" in e for e in result.errors)


def test_check_manifest_sha_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    case = _minimal_case("A01", "img.png", "0" * 64)  # わざと違うsha256
    _write_manifest(Path("manifest.jsonl"), [case])

    result = evaluate.check_manifest("manifest.jsonl")
    assert any("image_sha256 mismatch" in e for e in result.errors)


def test_check_manifest_duplicate_case_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    sha = _sha256(img)
    case1 = _minimal_case("A01", "img.png", sha)
    case2 = _minimal_case("A01", "img.png", sha)
    _write_manifest(Path("manifest.jsonl"), [case1, case2])

    result = evaluate.check_manifest("manifest.jsonl")
    assert any("duplicate case_id" in e for e in result.errors)


def test_check_manifest_bad_derived_from(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    sha = _sha256(img)

    # 存在しないcase_idを参照
    case_unknown = _minimal_case("A01-strip", "img.png", sha, derived_from="does-not-exist")
    _write_manifest(Path("manifest.jsonl"), [case_unknown])
    result = evaluate.check_manifest("manifest.jsonl")
    assert any("unknown case_id" in e for e in result.errors)

    # source_image_id が食い違う
    case_origin = _minimal_case("A01", "img.png", sha, source_image_id="A01")
    case_bad_source = _minimal_case(
        "A01-strip", "img.png", sha, derived_from="A01", source_image_id="OTHER"
    )
    _write_manifest(Path("manifest2.jsonl"), [case_origin, case_bad_source])
    result2 = evaluate.check_manifest("manifest2.jsonl")
    assert any("different source_image_id" in e for e in result2.errors)


def test_check_manifest_ok_counts_and_rights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    sha = _sha256(img)

    origin = _minimal_case("A01", "img.png", sha, scenario="alisa_lora")
    strip = _minimal_case(
        "A01-strip", "img.png", sha, derived_from="A01", source_image_id="A01", scenario="alisa_lora"
    )
    other = _minimal_case("S01", "img.png", sha, scenario="similar")
    unconfirmed = _minimal_case("BR01", "img.png", sha, scenario="general")
    unconfirmed["rights"] = {"terms": "test-license", "rights_confirmed": False}

    _write_manifest(Path("manifest.jsonl"), [origin, strip, other, unconfirmed])

    result = evaluate.check_manifest("manifest.jsonl")
    assert result.errors == []
    assert result.scenario_counts == {"alisa_lora": 2, "similar": 1, "general": 1}
    assert result.source_count == 3  # A01, S01, BR01 (derived_from が null)
    assert result.derived_count == 1  # A01-strip
    assert result.rights_false_count == 1


# ---------------------------------------------------------------------------
# run_evaluate: 合成データセットでのエンドツーエンド
# ---------------------------------------------------------------------------


def _build_synthetic_dataset(tmp_path: Path):
    """A01(alisa)・A01-strip(派生, メタデータなし)・S01(second_original)・
    G01(人物なし)・BR01(rights未確認, 除外) の5ケースを組み立てる。
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    a01_path = images_dir / "A01.png"
    _make_alisa_lora_png(a01_path)
    a01_strip_path = images_dir / "A01-strip.png"
    _make_plain_png(a01_strip_path)  # 画素は違ってもテストの都合上は別ファイルでよい(メタデータ有無の対比が目的)
    s01_path = images_dir / "S01.png"
    _make_plain_png(s01_path)
    g01_path = images_dir / "G01.png"
    _make_plain_png(g01_path)
    br01_path = images_dir / "BR01.png"
    _make_plain_png(br01_path)

    a01 = _minimal_case(
        "A01",
        "images/A01.png",
        _sha256(a01_path),
        scenario="alisa_lora",
        expected={"image_type": "illustration", "art_style": "anime_2d", "subject": "person", "character": ["alisa"]},
        expected_metadata={
            "format": "a1111",
            "loras": [{"name": "fet-alisa-uniform-anima-v4u", "weight": 0.8}],
            "trigger_words": [],
            "characters": ["alisa"],
            "artificial": False,
        },
    )
    a01_strip = _minimal_case(
        "A01-strip",
        "images/A01-strip.png",
        _sha256(a01_strip_path),
        derived_from="A01",
        source_image_id="A01",
        scenario="alisa_lora",
        expected={"image_type": "illustration", "art_style": "anime_2d", "subject": "person", "character": ["alisa"]},
        expected_metadata={
            "format": "none",
            "loras": [],
            "trigger_words": [],
            "characters": [],
            "artificial": False,
        },
    )
    s01 = _minimal_case(
        "S01",
        "images/S01.png",
        _sha256(s01_path),
        scenario="similar",
        expected={
            "image_type": "illustration",
            "art_style": "anime_2d",
            "subject": "person",
            "character": ["second_original"],
        },
    )
    g01 = _minimal_case(
        "G01",
        "images/G01.png",
        _sha256(g01_path),
        scenario="general",
        expected={"image_type": "illustration", "art_style": "anime_2d", "subject": "landscape", "character": []},
    )
    br01 = _minimal_case("BR01", "images/BR01.png", _sha256(br01_path), scenario="general")
    br01["rights"] = {"terms": "test-license", "rights_confirmed": False}

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [a01, a01_strip, s01, g01, br01])
    return manifest_path


def _build_response_queue(modes: list[str]) -> list:
    """評価順(A01, A01-strip, S01, G01)・交互モード順に対応する応答列を組み立てる。"""
    alisa_ranking = make_logprobs_response({"A": 0.97, "B": 0.000001, "C": 0.01, "D": 0.02})
    alisa_confirm_yes = make_logprobs_response({"A": 0.95, "B": 0.05})
    second_ranking = make_logprobs_response({"B": 0.97, "C": 0.01, "D": 0.02})
    second_confirm_yes = make_logprobs_response({"A": 0.95, "B": 0.05})
    none_ranking = make_logprobs_response({"C": 0.05, "D": 0.95})
    confirm_no = make_logprobs_response({"A": 0.1, "B": 0.9})

    it_illustration = make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02})
    as_anime = make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02})
    subj_person = make_logprobs_response({"A": 0.8, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.03, "F": 0.02})
    subj_landscape = make_logprobs_response({"A": 0.05, "B": 0.8, "C": 0.05, "D": 0.05, "E": 0.03, "F": 0.02})

    json_alisa = make_text_response(
        json.dumps({"image_type": "illustration", "art_style": "anime_2d", "subject": "person", "character": ["alisa"]})
    )
    json_second = make_text_response(
        json.dumps(
            {"image_type": "illustration", "art_style": "anime_2d", "subject": "person", "character": ["second_original"]}
        )
    )
    json_bad = make_text_response("not valid json at all")

    responses_by_case_mode = {
        ("A01", "choice"): [it_illustration, as_anime, subj_person, alisa_ranking, alisa_confirm_yes],
        ("A01", "json"): [json_alisa],
        ("A01-strip", "choice"): [it_illustration, as_anime, subj_person, alisa_ranking, alisa_confirm_yes],
        ("A01-strip", "json"): [json_alisa],
        ("S01", "choice"): [
            urllib.error.URLError("connection refused"),  # image_type axis を失敗させる
            as_anime,
            subj_person,
            second_ranking,
            second_confirm_yes,
        ],
        ("S01", "json"): [json_second],
        ("G01", "choice"): [it_illustration, as_anime, subj_landscape, none_ranking, confirm_no, confirm_no],
        ("G01", "json"): [json_bad, json_bad, json_bad],  # 形式不正が再試行込みで失敗し続ける
    }

    order_for = {}
    case_ids = ["A01", "A01-strip", "S01", "G01"]
    for i, case_id in enumerate(case_ids):
        order_for[case_id] = list(modes) if i % 2 == 0 else list(reversed(modes))

    queue: list = []
    for case_id in case_ids:
        for mode in order_for[case_id]:
            queue.extend(responses_by_case_mode[(case_id, mode)])
    return queue


def test_run_evaluate_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest_path = _build_synthetic_dataset(tmp_path)

    modes = ["choice", "json"]
    backend = FakeBackend(_build_response_queue(modes))

    taxonomy_path = str(Path(__file__).resolve().parent.parent / "taxonomy" / "default.yaml")
    output_dir = str(tmp_path / "results")

    exit_code = evaluate.run_evaluate(
        manifest_path=str(manifest_path),
        taxonomy_path=taxonomy_path,
        backend=backend,
        modes=modes,
        max_edge=64,
        warmup=0,
        runtime_label="test-runtime",
        note="test note",
        output_dir=output_dir,
    )
    assert exit_code == 0

    # --- run.json ---
    run_json_path = Path(output_dir) / "run.json"
    run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
    assert run_data["case_counts"]["total"] == 5
    assert run_data["case_counts"]["evaluated"] == 4
    assert run_data["case_counts"]["excluded"] == 1
    assert run_data["excluded_cases"] == [{"case_id": "BR01", "reason": "rights_not_confirmed"}]
    assert run_data["modes"] == modes
    assert run_data["runtime_label"] == "test-runtime"
    assert run_data["note"] == "test note"
    assert "sha256" in run_data["manifest"]
    assert run_data["taxonomy"]["version"]
    assert run_data["taxonomy"]["sha256"]

    # 絶対パス(tmp_pathの実体)が run.json のどこにも含まれない
    dumped = json.dumps(run_data)
    assert str(tmp_path) not in dumped
    assert str(tmp_path).replace("\\", "/") not in dumped

    # --- ケース別結果ファイル ---
    cases_dir = Path(output_dir) / "cases"
    expected_files = {
        "A01.choice.json", "A01.json.json",
        "A01-strip.choice.json", "A01-strip.json.json",
        "S01.choice.json", "S01.json.json",
        "G01.choice.json", "G01.json.json",
    }
    assert {p.name for p in cases_dir.iterdir()} == expected_files

    # --- cases.csv ---
    csv_path = Path(output_dir) / "cases.csv"
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8  # 4ケース x 2モード

    by_case_mode = {(r["case_id"], r["mode"]): r for r in rows}

    # モード順の交互: A01(偶数番目)=choice->json, A01-strip(奇数番目)=json->choice
    assert by_case_mode[("A01", "choice")]["order_index"] == "0"
    assert by_case_mode[("A01", "json")]["order_index"] == "1"
    assert by_case_mode[("A01-strip", "json")]["order_index"] == "0"
    assert by_case_mode[("A01-strip", "choice")]["order_index"] == "1"
    assert by_case_mode[("S01", "choice")]["order_index"] == "0"
    assert by_case_mode[("G01", "json")]["order_index"] == "0"

    # 失敗(image_type request_error)しても分母に残り、不正解として記録される
    s01_choice = by_case_mode[("S01", "choice")]
    assert s01_choice["pred_image_type"] == ""
    assert s01_choice["ok_image_type"] == "False"
    assert "request_error" in s01_choice["error_types"]
    # 他の軸は失敗の影響を受けず続行している
    assert s01_choice["ok_art_style"] == "True"
    assert s01_choice["pred_character"] == "second_original"
    assert s01_choice["character_exact_match"] == "True"

    # A01 vs A01-strip: 画像判定は同じでもメタデータ検出は異なる
    assert by_case_mode[("A01", "choice")]["detected_meta_format"] == "a1111"
    assert by_case_mode[("A01", "choice")]["meta_chars_ok"] == "True"
    assert by_case_mode[("A01-strip", "choice")]["detected_meta_format"] == "none"
    assert by_case_mode[("A01-strip", "choice")]["meta_chars_ok"] == "True"  # 両方とも空集合

    # G01: choiceは空集合同士で完全一致、jsonは形式不正で失敗 -> 期待が空集合でも不正解
    g01_choice = by_case_mode[("G01", "choice")]
    assert g01_choice["expected_character"] == ""
    assert g01_choice["pred_character"] == ""
    assert g01_choice["character_exact_match"] == "True"
    assert g01_choice["char_axis_failed"] == "False"

    g01_json = by_case_mode[("G01", "json")]
    assert g01_json["char_axis_failed"] == "True"
    assert g01_json["character_exact_match"] == "False"
    assert g01_json["json_attempts"] == "3"

    # --- summary.md ---
    summary_path = Path(output_dir) / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "N=4" in summary_text
    assert "キャラクター" in summary_text
    assert "Latency" in summary_text
    assert "点推定を強い結論として扱わないこと" in summary_text
