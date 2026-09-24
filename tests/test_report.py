import pytest

from classifier_demo import report


def _case(expected_character=None, expected_axes=None):
    return {
        "expected": {
            "image_type": "illustration",
            "art_style": "anime_2d",
            "subject": "person",
            "character": expected_character or [],
            **(expected_axes or {}),
        },
        "expected_metadata": {
            "format": "a1111",
            "loras": [],
            "trigger_words": [],
            "characters": expected_character or [],
            "artificial": False,
        },
    }


def test_nearest_rank_percentile_basic():
    values = [10, 20, 30, 40, 50]
    assert report.nearest_rank_percentile(values, 50) == 30
    assert report.nearest_rank_percentile(values, 90) == 50
    assert report.nearest_rank_percentile(values, 1) == 10
    assert report.nearest_rank_percentile(values, 100) == 50


def test_nearest_rank_percentile_unsorted_input_and_single_value():
    assert report.nearest_rank_percentile([5], 50) == 5
    assert report.nearest_rank_percentile([30, 10, 20], 50) == 20


def test_nearest_rank_percentile_empty_raises():
    with pytest.raises(ValueError):
        report.nearest_rank_percentile([], 50)


def test_score_single_axis_correct_and_wrong():
    case = _case()
    result = {"vision_tags": {"image_type": ["illustration"], "art_style": ["other"]}}
    expected, pred, ok = report.score_single_axis(case, result, "image_type")
    assert (expected, pred, ok) == ("illustration", "illustration", True)
    expected, pred, ok = report.score_single_axis(case, result, "art_style")
    assert (expected, pred, ok) == ("anime_2d", "other", False)


def test_score_single_axis_failed_axis_is_empty_pred_and_wrong():
    case = _case()
    result = {"vision_tags": {}}  # image_type axis failed / missing
    expected, pred, ok = report.score_single_axis(case, result, "image_type")
    assert pred == ""
    assert ok is False


def test_score_character_exact_match():
    case = _case(expected_character=["alisa"])
    result = {"vision_tags": {"character": ["alisa"]}}
    scored = report.score_character(case, result)
    assert scored["exact_match"] is True
    assert scored["tp"] == {"alisa"}
    assert scored["fp"] == set()
    assert scored["fn"] == set()
    assert scored["failed"] is False


def test_score_character_empty_vs_empty_is_exact_match():
    case = _case(expected_character=[])
    result = {"vision_tags": {"character": []}}
    scored = report.score_character(case, result)
    assert scored["exact_match"] is True
    assert scored["tp"] == set()
    assert scored["fp"] == set()
    assert scored["fn"] == set()


def test_score_character_failed_axis_forces_wrong_even_if_expected_empty():
    case = _case(expected_character=[])
    result = {"vision_tags": {}}  # character axis missing => failed
    scored = report.score_character(case, result)
    assert scored["failed"] is True
    assert scored["exact_match"] is False
    # 失敗時は予測集合を空として TP/FP/FN を数える(期待も空なので寄与なし)
    assert scored["tp"] == set()
    assert scored["fp"] == set()
    assert scored["fn"] == set()


def test_score_character_tp_fp_fn_with_mismatch():
    case = _case(expected_character=["alisa", "second_original"])
    result = {"vision_tags": {"character": ["alisa", "other_original"]}}
    scored = report.score_character(case, result)
    assert scored["exact_match"] is False
    assert scored["tp"] == {"alisa"}
    assert scored["fp"] == {"other_original"}
    assert scored["fn"] == {"second_original"}


def test_score_metadata_match_and_mismatch():
    case = _case(expected_character=["alisa"])
    case["expected_metadata"] = {
        "format": "a1111",
        "loras": [],
        "trigger_words": [],
        "characters": ["alisa"],
        "artificial": False,
    }
    result_ok = {
        "metadata_evidence": {
            "format": "a1111",
            "loras": [],
            "prompt_tags": [],
            "matches": [{"kind": "lora_name", "value": "x", "weight": 1.0, "character": "alisa"}],
        }
    }
    scored = report.score_metadata(case, result_ok)
    assert scored["format_ok"] is True
    assert scored["chars_ok"] is True

    result_wrong = {"metadata_evidence": {"format": "none", "loras": [], "prompt_tags": [], "matches": []}}
    scored = report.score_metadata(case, result_wrong)
    assert scored["format_ok"] is False
    assert scored["chars_ok"] is False


def test_score_metadata_failed_evidence_forces_wrong_even_if_expected_matches_by_accident():
    case = _case(expected_character=[])
    case["expected_metadata"] = {
        "format": "none",
        "loras": [],
        "trigger_words": [],
        "characters": [],
        "artificial": False,
    }
    # メタデータ抽出自体が失敗(format: error)。期待値がたまたま none/空集合でも不正解扱いにする。
    result = {"metadata_evidence": {"format": "error", "loras": [], "prompt_tags": [], "matches": []}}
    scored = report.score_metadata(case, result)
    assert scored["failed"] is True
    assert scored["format_ok"] is False
    assert scored["chars_ok"] is False

    result_none_evidence = {"metadata_evidence": None}
    scored = report.score_metadata(case, result_none_evidence)
    assert scored["failed"] is True
    assert scored["format_ok"] is False
    assert scored["chars_ok"] is False


def test_build_case_row_has_all_csv_fieldnames():
    case = {
        "case_id": "A01",
        "source_image_id": "A01",
        "derived_from": None,
        "scenario": "alisa_lora",
        "expected": {"image_type": "illustration", "art_style": "anime_2d", "subject": "person", "character": ["alisa"]},
        "expected_metadata": {
            "format": "a1111",
            "loras": [],
            "trigger_words": [],
            "characters": ["alisa"],
            "artificial": False,
        },
    }
    result = {
        "vision_tags": {
            "image_type": ["illustration"],
            "art_style": ["anime_2d"],
            "subject": ["person"],
            "character": ["alisa"],
        },
        "metadata_evidence": {
            "format": "a1111",
            "loras": [],
            "prompt_tags": [],
            "matches": [{"kind": "lora_name", "value": "x", "weight": 1.0, "character": "alisa"}],
        },
        "timing_ms": {"classification_wall_ms": 12.5},
        "request_count": 4,
        "errors": [],
    }
    row = report.build_case_row(case, "choice", 0, result)
    assert set(row.keys()) == set(report.CSV_FIELDNAMES)
    assert row["ok_image_type"] is True
    assert row["character_exact_match"] is True
    assert row["json_attempts"] == ""  # choice モードでは空
