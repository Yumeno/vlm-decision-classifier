import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from classifier_demo import evaluate
from classifier_demo.__main__ import build_parser
from classifier_demo.taxonomy import load as load_taxonomy
from tests.fakes import FakeBackend, make_logprobs_response, make_text_response

import urllib.error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_test_taxonomy(tmp_path: Path) -> str:
    """default.yaml から独立した、evaluate テスト専用の小さな taxonomy。

    軸構成(単一選択3軸: image_type・art_style・subject + 複数選択2軸: outfit・character)を
    固定し、FakeBackend の応答列(リクエスト順・件数)をテスト側で完全に制御できるようにする。
    outfit は非catch_all候補を1つ(sailor_uniform)だけにしてあるので、ランキングの値に関わらず
    常に「ランキング1回+確認1回」の2リクエストになる(候補フロアの境界に左右されない)。
    character 軸は元の default.yaml と同じ構成(alisa の LoRA名・トリガーワードあり)。
    """
    content = """\
version: "test-0.1"
axes:
  - id: image_type
    question: "What kind of image is this?"
    multi: false
    allow_none: false
    choices:
      - {id: illustration, name: illustration, criteria: "illustration"}
      - {id: comic, name: comic, criteria: "comic"}
      - {id: ui, name: user interface, criteria: "ui"}
      - {id: other, name: other, criteria: "other"}
  - id: art_style
    question: "What is the art style of this image?"
    multi: false
    allow_none: false
    choices:
      - {id: anime_2d, name: anime style, criteria: "anime"}
      - {id: painterly_2d, name: painterly, criteria: "painterly"}
      - {id: pixel_art, name: pixel art, criteria: "pixel"}
      - {id: other, name: other, criteria: "other"}
  - id: subject
    question: "What is the main subject of this image?"
    multi: false
    allow_none: false
    choices:
      - {id: person, name: person, criteria: "person"}
      - {id: landscape, name: landscape, criteria: "landscape"}
      - {id: mecha_vehicle, name: mecha or vehicle, criteria: "mecha"}
      - {id: object, name: object, criteria: "object"}
      - {id: creature, name: creature, criteria: "creature"}
      - {id: other, name: other, criteria: "other"}
  - id: outfit
    question: "What outfit is worn by the characters in this image?"
    multi: true
    allow_none: true
    choices:
      - {id: sailor_uniform, name: sailor uniform, criteria: "sailor uniform"}
      - {id: other, name: other clothing, criteria: "other clothing", catch_all: true}
  - id: character
    question: "Which character appears in this image?"
    multi: true
    allow_none: true
    choices:
      - id: alisa
        name: Alisa
        criteria: "alisa"
        lora_names: [fet-alisa-uniform-anima-v4u]
        trigger_words: [fet_alisa_uniform]
      - id: second_original
        name: Second original character
        criteria: "second"
      - id: other_original
        name: other character
        criteria: "other"
        catch_all: true
"""
    path = tmp_path / "test_taxonomy.yaml"
    path.write_text(content, encoding="utf-8")
    return str(path)


def _test_taxonomy(tmp_path: Path):
    """`_write_test_taxonomy` と同じ内容を Taxonomy オブジェクトとして返す
    (check_manifest を taxonomy_path 経由でなく直接呼ぶテスト用)。"""
    return load_taxonomy(_write_test_taxonomy(tmp_path))


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
        "expected": {
            "image_type": "illustration",
            "art_style": "anime_2d",
            "subject": "person",
            "outfit": [],
            "character": [],
        },
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

    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert any("does not exist" in e for e in result.errors)


def test_check_manifest_sha_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    case = _minimal_case("A01", "img.png", "0" * 64)  # わざと違うsha256
    _write_manifest(Path("manifest.jsonl"), [case])

    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert any("image_sha256 mismatch" in e for e in result.errors)


def test_check_manifest_duplicate_case_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    sha = _sha256(img)
    case1 = _minimal_case("A01", "img.png", sha)
    case2 = _minimal_case("A01", "img.png", sha)
    _write_manifest(Path("manifest.jsonl"), [case1, case2])

    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert any("duplicate case_id" in e for e in result.errors)


def test_check_manifest_bad_derived_from(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    sha = _sha256(img)

    # 存在しないcase_idを参照
    case_unknown = _minimal_case("A01-strip", "img.png", sha, derived_from="does-not-exist")
    _write_manifest(Path("manifest.jsonl"), [case_unknown])
    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert any("unknown case_id" in e for e in result.errors)

    # source_image_id が食い違う
    case_origin = _minimal_case("A01", "img.png", sha, source_image_id="A01")
    case_bad_source = _minimal_case(
        "A01-strip", "img.png", sha, derived_from="A01", source_image_id="OTHER"
    )
    _write_manifest(Path("manifest2.jsonl"), [case_origin, case_bad_source])
    result2 = evaluate.check_manifest("manifest2.jsonl", _test_taxonomy(tmp_path))
    assert any("different source_image_id" in e for e in result2.errors)


def test_check_manifest_nested_derivation_is_error(tmp_path, monkeypatch):
    # 派生の派生(derived_from が指す先自体も derived_from を持つ)は禁止する
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    sha = _sha256(img)

    origin = _minimal_case("A01", "img.png", sha, source_image_id="A01")
    strip = _minimal_case("A01-strip", "img.png", sha, derived_from="A01", source_image_id="A01")
    nested = _minimal_case(
        "A01-strip-strip", "img.png", sha, derived_from="A01-strip", source_image_id="A01"
    )
    _write_manifest(Path("manifest.jsonl"), [origin, strip, nested])

    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert any("itself derived" in e for e in result.errors)


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

    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert result.errors == []
    assert result.scenario_counts == {"alisa_lora": 2, "similar": 1, "general": 1}
    assert result.source_count == 3  # A01, S01, BR01 (derived_from が null)
    assert result.derived_count == 1  # A01-strip
    assert result.rights_false_count == 1


def test_check_manifest_expected_missing_axis_field(tmp_path, monkeypatch):
    # taxonomy の軸(例: outfit)が expected に無ければエラーにする(ハードコードでなく
    # taxonomy 由来で全軸をチェックしていることの確認)。
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    case = _minimal_case("A01", "img.png", _sha256(img))
    del case["expected"]["outfit"]
    _write_manifest(Path("manifest.jsonl"), [case])

    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert any("expected missing field(s) ['outfit']" in e for e in result.errors)


def test_check_manifest_expected_unknown_choice_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    sha = _sha256(img)

    # 単一選択軸(image_type)に taxonomy に無い choice id
    case_single = _minimal_case("A01", "img.png", sha)
    case_single["expected"]["image_type"] = "not_a_real_choice"
    _write_manifest(Path("manifest.jsonl"), [case_single])
    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert any("expected.image_type has unknown choice id" in e for e in result.errors)

    # 複数選択軸(character)の要素に taxonomy に無い choice id
    case_multi = _minimal_case("A02", "img.png", sha)
    case_multi["expected"]["character"] = ["alisa", "not_a_real_character"]
    _write_manifest(Path("manifest2.jsonl"), [case_multi])
    result2 = evaluate.check_manifest("manifest2.jsonl", _test_taxonomy(tmp_path))
    assert any("expected.character has unknown choice id(s)" in e for e in result2.errors)


def test_check_manifest_expected_multi_axis_non_string_element_is_type_error_not_crash(tmp_path, monkeypatch):
    # list の要素自体が文字列でない(例: ネストしたlist)場合、`in known_ids`(set)で
    # TypeError を送出せず、型エラーとして報告する。未知IDの要素があれば、それは別に報告する。
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    case = _minimal_case("A01", "img.png", _sha256(img))
    case["expected"]["character"] = ["alisa", ["nested", "list"], "not_a_real_character"]
    _write_manifest(Path("manifest.jsonl"), [case])

    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert any(
        "expected.character has non-string element(s)" in e and "['nested', 'list']" in e
        for e in result.errors
    )
    assert any("expected.character has unknown choice id(s)" in e and "not_a_real_character" in e for e in result.errors)


def test_check_manifest_expected_multi_axis_must_be_a_list(tmp_path, monkeypatch):
    # 複数選択軸(character)が list でなければエラーにする(文字列を渡す典型ミスを検出)。
    monkeypatch.chdir(tmp_path)
    img = Path("img.png")
    _make_plain_png(img)
    case = _minimal_case("A01", "img.png", _sha256(img))
    case["expected"]["character"] = "alisa"  # list ではなく文字列(誤り)
    _write_manifest(Path("manifest.jsonl"), [case])

    result = evaluate.check_manifest("manifest.jsonl", _test_taxonomy(tmp_path))
    assert any("expected.character must be a list (multi axis)" in e for e in result.errors)


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
        expected={
            "image_type": "illustration",
            "art_style": "anime_2d",
            "subject": "person",
            "outfit": ["sailor_uniform"],
            "character": ["alisa"],
        },
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
        expected={
            "image_type": "illustration",
            "art_style": "anime_2d",
            "subject": "person",
            "outfit": ["sailor_uniform"],
            "character": ["alisa"],
        },
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
            "outfit": ["sailor_uniform"],
            "character": ["second_original"],
        },
    )
    g01 = _minimal_case(
        "G01",
        "images/G01.png",
        _sha256(g01_path),
        scenario="general",
        expected={
            "image_type": "illustration",
            "art_style": "anime_2d",
            "subject": "landscape",
            "outfit": [],
            "character": [],
        },
    )
    br01 = _minimal_case("BR01", "images/BR01.png", _sha256(br01_path), scenario="general")
    br01["rights"] = {"terms": "test-license", "rights_confirmed": False}

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [a01, a01_strip, s01, g01, br01])
    return manifest_path


def _build_response_queue(modes: list[str]) -> list:
    """評価順(A01, A01-strip, S01, G01)・交互モード順に対応する応答列を組み立てる。

    choice モードの軸呼び出し順は taxonomy の並び: image_type, art_style, subject,
    outfit, character。outfit は非catch_all候補が sailor_uniform 1つだけなので、
    ランキングの値に関わらず常に「ランキング1回+確認1回」の2リクエストになる。
    """
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

    # outfit: 候補は sailor_uniform 1つだけなので、ランキングの値によらず常に1候補になる。
    outfit_ranking_worn = make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.05})
    outfit_ranking_none = make_logprobs_response({"A": 0.03, "B": 0.02, "C": 0.95})
    outfit_confirm_yes = make_logprobs_response({"A": 0.9, "B": 0.1})
    outfit_confirm_no = make_logprobs_response({"A": 0.1, "B": 0.9})

    json_alisa = make_text_response(
        json.dumps(
            {
                "image_type": "illustration",
                "art_style": "anime_2d",
                "subject": "person",
                "outfit": ["sailor_uniform"],
                "character": ["alisa"],
            }
        )
    )
    json_second = make_text_response(
        json.dumps(
            {
                "image_type": "illustration",
                "art_style": "anime_2d",
                "subject": "person",
                "outfit": ["sailor_uniform"],
                "character": ["second_original"],
            }
        )
    )
    json_bad = make_text_response("not valid json at all")

    responses_by_case_mode = {
        ("A01", "choice"): [
            it_illustration, as_anime, subj_person,
            outfit_ranking_worn, outfit_confirm_yes,
            alisa_ranking, alisa_confirm_yes,
        ],
        ("A01", "json"): [json_alisa],
        ("A01-strip", "choice"): [
            it_illustration, as_anime, subj_person,
            outfit_ranking_worn, outfit_confirm_yes,
            alisa_ranking, alisa_confirm_yes,
        ],
        ("A01-strip", "json"): [json_alisa],
        ("S01", "choice"): [
            urllib.error.URLError("connection refused"),  # image_type axis を失敗させる
            as_anime,
            subj_person,
            outfit_ranking_worn, outfit_confirm_yes,
            second_ranking, second_confirm_yes,
        ],
        ("S01", "json"): [json_second],
        ("G01", "choice"): [
            it_illustration, as_anime, subj_landscape,
            outfit_ranking_none, outfit_confirm_no,
            none_ranking, confirm_no, confirm_no,
        ],
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

    taxonomy_path = _write_test_taxonomy(tmp_path)
    output_dir = str(tmp_path / "results")

    runtime_info_path = tmp_path / "runtime.json"
    runtime_info_content = {
        "model_sha256": "deadbeef",
        "mmproj_sha256": "cafef00d",
        "server": "llama.cpp",
        "server_commit": "abc1234",
        "patched": False,
        "gpu": "RTX 3090 24GB",
    }
    runtime_info_path.write_text(json.dumps(runtime_info_content), encoding="utf-8")

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
        runtime_info_path=str(runtime_info_path),
        dataset_version="dataset-v1.0.0",
    )
    assert exit_code == 0

    # --- run.json ---
    run_json_path = Path(output_dir) / "run.json"
    run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
    assert run_data["dataset_version"] == "dataset-v1.0.0"
    assert run_data["case_counts"]["total"] == 5
    assert run_data["case_counts"]["evaluated"] == 4
    assert run_data["case_counts"]["excluded"] == 1
    assert run_data["excluded_cases"] == [{"case_id": "BR01", "reason": "rights_not_confirmed"}]
    # manifest全体(BR01含む)と実際に評価したケース(BR01除く)のシナリオ別件数を分けて記録する
    assert run_data["case_counts"]["manifest_by_scenario"] == {"alisa_lora": 2, "similar": 1, "general": 2}
    assert run_data["case_counts"]["evaluated_by_scenario"] == {"alisa_lora": 2, "similar": 1, "general": 1}
    assert run_data["modes"] == modes
    assert run_data["runtime_label"] == "test-runtime"
    assert run_data["note"] == "test note"
    assert "sha256" in run_data["manifest"]
    assert run_data["taxonomy"]["version"]
    assert run_data["taxonomy"]["sha256"]

    # --- runtime_info: JSONの中身をそのまま保存し、ファイル名・SHA256を別途記録する ---
    assert run_data["runtime_info"] == runtime_info_content
    assert run_data["runtime_info_file"]["name"] == "runtime.json"
    assert run_data["runtime_info_file"]["sha256"] == evaluate.sha256_file(str(runtime_info_path))

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
    assert s01_choice["exact_character"] == "True"
    assert s01_choice["pred_outfit"] == "sailor_uniform"
    assert s01_choice["exact_outfit"] == "True"

    # A01 vs A01-strip: 画像判定は同じでもメタデータ検出は異なる
    a01_choice = by_case_mode[("A01", "choice")]
    assert a01_choice["detected_meta_format"] == "a1111"
    assert a01_choice["meta_chars_ok"] == "True"
    assert a01_choice["detected_meta_loras"] == "fet-alisa-uniform-anima-v4u:0.8"
    assert a01_choice["meta_loras_ok"] == "True"
    a01_strip_choice = by_case_mode[("A01-strip", "choice")]
    assert a01_strip_choice["detected_meta_format"] == "none"
    assert a01_strip_choice["meta_chars_ok"] == "True"  # 両方とも空集合
    assert a01_strip_choice["detected_meta_loras"] == ""
    assert a01_strip_choice["meta_loras_ok"] == "True"  # 両方とも空集合

    # G01: choiceは空集合同士で完全一致、jsonは形式不正で失敗 -> 期待が空集合でも不正解
    g01_choice = by_case_mode[("G01", "choice")]
    assert g01_choice["expected_character"] == ""
    assert g01_choice["pred_character"] == ""
    assert g01_choice["exact_character"] == "True"
    assert g01_choice["character_axis_failed"] == "False"
    assert g01_choice["expected_outfit"] == ""
    assert g01_choice["pred_outfit"] == ""
    assert g01_choice["exact_outfit"] == "True"
    assert g01_choice["outfit_axis_failed"] == "False"

    g01_json = by_case_mode[("G01", "json")]
    assert g01_json["character_axis_failed"] == "True"
    assert g01_json["exact_character"] == "False"
    assert g01_json["outfit_axis_failed"] == "True"
    assert g01_json["exact_outfit"] == "False"
    assert g01_json["json_attempts"] == "3"

    # --- summary.md ---
    summary_path = Path(output_dir) / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "N=4" in summary_text  # 全体件数(元画像3件+派生1件)
    assert "点推定を強い結論として扱わないこと" in summary_text

    # 画像判定の集計(単一軸・複数選択軸・シナリオ別・Latency・ペア比較)は
    # 元画像3件(A01-stripを除く)だけを分母にする
    assert "元画像 N=3 件" in summary_text
    assert summary_text.count("元画像 N=3 件") >= 5  # 単一軸/outfit/character/シナリオ別/Latency の各見出し

    # メタデータは全4ケース(派生を含む)を分母にする
    assert "集計元モード: choice、元画像・派生を含む全ケース" in summary_text
    assert "loras 完全一致率: 100.0% (4/4)" in summary_text
    assert "trigger_words 完全一致率: 100.0% (4/4)" in summary_text

    # 派生ケース節: A01-strip と元ケースA01のcharacter予測・正誤が同じかを一覧にする
    assert "派生ケース(メタデータ除去コピー)" in summary_text
    assert "| A01-strip | A01 | choice | alisa | alisa | 一致 | 正 | 正 | 一致 |" in summary_text
    assert "| A01-strip | A01 | json | alisa | alisa | 一致 | 正 | 正 | 一致 |" in summary_text

    # 画像判定の主集計(元画像分母)には全ケース分母の参考値も併記する
    assert "／ 全ケース" in summary_text  # 単一軸・完全一致率・シナリオ別のセル併記
    assert "TP(元)" in summary_text and "TP(全)" in summary_text  # 候補別TP/FP/FNは列を2つにする
    assert "- 元画像:" in summary_text and "- 全ケース:" in summary_text  # Latency
    assert "### 元画像(choice vs json" in summary_text
    assert "### 全ケース(choice vs json" in summary_text

    # 複数選択軸(outfit・character)ごとにセクションが出る(taxonomy由来で汎用化されている)
    assert "## outfit(集合" in summary_text
    assert "## character(集合" in summary_text

    # macro Precision/Recall/F1(micro の隣、TP/FP/FN列は対象外)
    assert "| macro | - | - | - | - | - | - |" in summary_text
    assert "macro は候補別値" in summary_text

    # artificial は採点対象外である旨の注記
    assert "`artificial` は画像から検出できない記録上の事実なので採点対象外" in summary_text


def test_run_evaluate_without_runtime_info_records_null(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "images").mkdir()
    img_path = Path("images/A01.png")
    _make_plain_png(img_path)
    case = _minimal_case("A01", "images/A01.png", _sha256(img_path))
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [case])

    # choice: image_type/art_style/subject + outfit(ランキング+確認1回、候補1つのみ) +
    # character ranking(none優勢だがfloor境界でalisa/second_originalの両方が候補になり、
    # それぞれ確認(no)が入る) = 3 + 2 + 1 + 2 = 8リクエスト
    responses = [
        make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),
        make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),
        make_logprobs_response({"A": 0.8, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.03, "F": 0.02}),
        make_logprobs_response({"A": 0.03, "B": 0.02, "C": 0.95}),  # outfit ranking(noneを優勢に)
        make_logprobs_response({"A": 0.1, "B": 0.9}),  # outfit confirm -> no
        make_logprobs_response({"C": 0.05, "D": 0.95}),
        make_logprobs_response({"A": 0.1, "B": 0.9}),
        make_logprobs_response({"A": 0.1, "B": 0.9}),
    ]
    backend = FakeBackend(responses)
    taxonomy_path = _write_test_taxonomy(tmp_path)
    output_dir = str(tmp_path / "results")

    exit_code = evaluate.run_evaluate(
        manifest_path=str(manifest_path),
        taxonomy_path=taxonomy_path,
        backend=backend,
        modes=["choice"],
        max_edge=64,
        warmup=0,
        runtime_label=None,
        note=None,
        output_dir=output_dir,
    )
    assert exit_code == 0

    run_data = json.loads((Path(output_dir) / "run.json").read_text(encoding="utf-8"))
    assert run_data["runtime_info"] is None
    assert run_data["runtime_info_file"] is None
    assert run_data["dataset_version"] is None  # --dataset-version 未指定時


def test_run_evaluate_aborts_before_any_request_on_bad_runtime_info(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "images").mkdir()
    img_path = Path("images/A01.png")
    _make_plain_png(img_path)
    case = _minimal_case("A01", "images/A01.png", _sha256(img_path))
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [case])

    taxonomy_path = _write_test_taxonomy(tmp_path)

    # 存在しないruntime-infoファイル -> チェックにも評価にも進まない
    backend = FakeBackend([])
    output_dir_missing = str(tmp_path / "results-missing")
    exit_code = evaluate.run_evaluate(
        manifest_path=str(manifest_path),
        taxonomy_path=taxonomy_path,
        backend=backend,
        modes=["choice"],
        max_edge=64,
        warmup=1,  # warmupより前にruntime-infoを読むことを確認するため意図的に1にする
        runtime_label=None,
        note=None,
        output_dir=output_dir_missing,
        runtime_info_path=str(tmp_path / "does_not_exist.json"),
    )
    assert exit_code == 1
    assert backend.request_count == 0  # warmupすら始まっていない
    assert not Path(output_dir_missing).exists()

    # 壊れたJSON -> 同様に評価を始めない
    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("{not valid json", encoding="utf-8")
    output_dir_bad = str(tmp_path / "results-bad")
    exit_code2 = evaluate.run_evaluate(
        manifest_path=str(manifest_path),
        taxonomy_path=taxonomy_path,
        backend=backend,
        modes=["choice"],
        max_edge=64,
        warmup=1,
        runtime_label=None,
        note=None,
        output_dir=output_dir_bad,
        runtime_info_path=str(bad_json_path),
    )
    assert exit_code2 == 1
    assert backend.request_count == 0
    assert not Path(output_dir_bad).exists()


def test_validate_modes_rejects_invalid_and_duplicates():
    with pytest.raises(ValueError):
        evaluate.validate_modes(["choice", "xml"])
    with pytest.raises(ValueError):
        evaluate.validate_modes(["choice", "choice"])
    with pytest.raises(ValueError):
        evaluate.validate_modes([])
    evaluate.validate_modes(["choice"])  # 例外なし
    evaluate.validate_modes(["json", "choice"])  # 例外なし


def test_cli_modes_rejects_invalid_choice_and_duplicates():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--model", "m", "--output-dir", "out", "--modes", "choice,xml"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--model", "m", "--output-dir", "out", "--modes", "choice,choice"])
    args = parser.parse_args(["evaluate", "--model", "m", "--output-dir", "out", "--modes", "json"])
    assert args.modes == ["json"]


def test_run_evaluate_aborts_on_invalid_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "images").mkdir()
    img_path = Path("images/A01.png")
    _make_plain_png(img_path)
    case = _minimal_case("A01", "images/A01.png", _sha256(img_path))
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [case])

    taxonomy_path = _write_test_taxonomy(tmp_path)
    backend = FakeBackend([])
    output_dir = str(tmp_path / "results")

    exit_code = evaluate.run_evaluate(
        manifest_path=str(manifest_path),
        taxonomy_path=taxonomy_path,
        backend=backend,
        modes=["choice", "choice"],  # 重複は直接呼び出しでも拒否する
        max_edge=64,
        warmup=0,
        runtime_label=None,
        note=None,
        output_dir=output_dir,
    )
    assert exit_code == 1
    assert backend.request_count == 0
    assert not Path(output_dir).exists()


def test_run_evaluate_records_warmup_details(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "images").mkdir()
    img_path = Path("images/A01.png")
    _make_plain_png(img_path)
    case = _minimal_case("A01", "images/A01.png", _sha256(img_path))
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [case])

    # ウォームアップ2回(image_type軸への choose 呼び出し) + 本編1回(choice: image_type/art_style/
    # subject + outfit(ランキング+確認1回) + character ranking。none優勢だがfloor境界で両候補とも
    # 確認要、確認2回)= 2 + 3 + 2 + 1 + 2 = 10リクエスト
    responses = [
        make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # warmup 1
        make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # warmup 2
        make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type
        make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # art_style
        make_logprobs_response({"A": 0.8, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.03, "F": 0.02}),  # subject
        make_logprobs_response({"A": 0.03, "B": 0.02, "C": 0.95}),  # outfit ranking
        make_logprobs_response({"A": 0.1, "B": 0.9}),  # outfit confirm -> no
        make_logprobs_response({"C": 0.05, "D": 0.95}),  # character ranking
        make_logprobs_response({"A": 0.1, "B": 0.9}),  # confirm alisa -> no
        make_logprobs_response({"A": 0.1, "B": 0.9}),  # confirm second_original -> no
    ]
    backend = FakeBackend(responses)
    taxonomy_path = _write_test_taxonomy(tmp_path)
    output_dir = str(tmp_path / "results")

    exit_code = evaluate.run_evaluate(
        manifest_path=str(manifest_path),
        taxonomy_path=taxonomy_path,
        backend=backend,
        modes=["choice"],
        max_edge=64,
        warmup=2,
        runtime_label=None,
        note=None,
        output_dir=output_dir,
    )
    assert exit_code == 0

    run_data = json.loads((Path(output_dir) / "run.json").read_text(encoding="utf-8"))
    warmup = run_data["warmup"]
    assert warmup["count"] == 2
    assert warmup["case_id"] == "A01"
    assert warmup["axis_id"] == "image_type"
    assert len(warmup["elapsed_ms"]) == 2
    assert all(isinstance(v, (int, float)) for v in warmup["elapsed_ms"])
    assert warmup["errors"] == 0


def test_run_evaluate_excludes_derived_case_when_source_excluded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "images").mkdir()
    img_path = Path("images/A01.png")
    _make_plain_png(img_path)
    sha = _sha256(img_path)

    origin = _minimal_case("A01", "images/A01.png", sha, scenario="alisa_lora")
    origin["rights"] = {"terms": "test-license", "rights_confirmed": False}
    # A01-strip自体は rights_confirmed=true だが、元ケースA01が除外されているので
    # 連動して除外されるべき(派生の派生は check_manifest で禁止済み)。
    strip = _minimal_case(
        "A01-strip", "images/A01.png", sha, derived_from="A01", source_image_id="A01", scenario="alisa_lora"
    )
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [origin, strip])

    backend = FakeBackend([])
    taxonomy_path = _write_test_taxonomy(tmp_path)
    output_dir = str(tmp_path / "results")

    exit_code = evaluate.run_evaluate(
        manifest_path=str(manifest_path),
        taxonomy_path=taxonomy_path,
        backend=backend,
        modes=["choice"],
        max_edge=64,
        warmup=0,
        runtime_label=None,
        note=None,
        output_dir=output_dir,
    )
    assert exit_code == 0
    assert backend.request_count == 0  # 評価対象ケースが0件になっている

    run_data = json.loads((Path(output_dir) / "run.json").read_text(encoding="utf-8"))
    assert run_data["case_counts"]["evaluated"] == 0
    assert run_data["case_counts"]["excluded"] == 2
    excluded = {e["case_id"]: e["reason"] for e in run_data["excluded_cases"]}
    assert excluded == {"A01": "rights_not_confirmed", "A01-strip": "source excluded"}
