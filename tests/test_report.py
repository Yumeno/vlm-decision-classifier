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
    # loras/trigger_words とも期待値が空集合で検出も空集合なので一致扱い
    assert scored["loras_ok"] is True
    assert scored["triggers_ok"] is True

    result_wrong = {"metadata_evidence": {"format": "none", "loras": [], "prompt_tags": [], "matches": []}}
    scored = report.score_metadata(case, result_wrong)
    assert scored["format_ok"] is False
    assert scored["chars_ok"] is False


def test_score_metadata_loras_match_with_normalized_name_and_weight():
    case = _case()
    case["expected_metadata"] = {
        "format": "a1111",
        "loras": [{"name": "Fet-Alisa-Uniform-Anima-v4u", "weight": 0.8}],
        "trigger_words": [],
        "characters": [],
        "artificial": False,
    }
    result = {
        "metadata_evidence": {
            "format": "a1111",
            # 検出側は前後空白・大小文字が異なっていても正規化後の名前で一致すればよい
            "loras": [{"name": "  fet-alisa-uniform-anima-v4u  ", "weight": 0.8, "source": "a1111"}],
            "prompt_tags": [],
            "matches": [],
        }
    }
    scored = report.score_metadata(case, result)
    assert scored["loras_ok"] is True


def test_score_metadata_loras_weight_mismatch_is_wrong():
    case = _case()
    case["expected_metadata"] = {
        "format": "a1111",
        "loras": [{"name": "fet-alisa-uniform-anima-v4u", "weight": 1.0}],
        "trigger_words": [],
        "characters": [],
        "artificial": False,
    }
    result = {
        "metadata_evidence": {
            "format": "a1111",
            "loras": [{"name": "fet-alisa-uniform-anima-v4u", "weight": 0.8, "source": "a1111"}],
            "prompt_tags": [],
            "matches": [],
        }
    }
    scored = report.score_metadata(case, result)
    assert scored["loras_ok"] is False


def test_score_metadata_loras_none_weight_matches_none_weight():
    # ComfyUI由来で strength_model が取得できない場合 weight は None。期待値も None なら一致。
    case = _case()
    case["expected_metadata"] = {
        "format": "comfyui",
        "loras": [{"name": "fet-alisa-uniform-anima-v4u", "weight": None}],
        "trigger_words": [],
        "characters": [],
        "artificial": False,
    }
    result = {
        "metadata_evidence": {
            "format": "comfyui",
            "loras": [{"name": "fet-alisa-uniform-anima-v4u", "weight": None, "source": "comfyui"}],
            "prompt_tags": [],
            "matches": [],
        }
    }
    scored = report.score_metadata(case, result)
    assert scored["loras_ok"] is True


def test_score_metadata_triggers_match_and_mismatch():
    case = _case()
    case["expected_metadata"] = {
        "format": "a1111",
        "loras": [],
        "trigger_words": ["fet_alisa_uniform"],
        "characters": [],
        "artificial": False,
    }
    result_match = {
        "metadata_evidence": {
            "format": "a1111",
            "loras": [],
            "prompt_tags": [],
            "matches": [{"kind": "prompt_trigger", "value": "fet_alisa_uniform", "character": "alisa"}],
        }
    }
    assert report.score_metadata(case, result_match)["triggers_ok"] is True

    result_missing = {
        "metadata_evidence": {"format": "a1111", "loras": [], "prompt_tags": [], "matches": []}
    }
    assert report.score_metadata(case, result_missing)["triggers_ok"] is False


def test_format_loras():
    assert report.format_loras([]) == ""
    assert (
        report.format_loras([{"name": "b", "weight": 1.0}, {"name": "a", "weight": None}])
        == "a:None;b:1.0"
    )


def test_score_metadata_loras_duplicate_counts_are_preserved():
    # 同じLoRA指定が2回記録されているケース。set化すると1件に潰れて誤って一致してしまうため、
    # 多重集合(Counter)で比較し、件数のずれを不一致として検出する。
    case = _case()
    case["expected_metadata"] = {
        "format": "a1111",
        "loras": [
            {"name": "fet-alisa-uniform-anima-v4u", "weight": 1.0},
            {"name": "fet-alisa-uniform-anima-v4u", "weight": 1.0},
        ],
        "trigger_words": [],
        "characters": [],
        "artificial": False,
    }

    result_missing_dup = {
        "metadata_evidence": {
            "format": "a1111",
            "loras": [{"name": "fet-alisa-uniform-anima-v4u", "weight": 1.0, "source": "a1111"}],
            "prompt_tags": [],
            "matches": [],
        }
    }
    assert report.score_metadata(case, result_missing_dup)["loras_ok"] is False

    result_matching_dup = {
        "metadata_evidence": {
            "format": "a1111",
            "loras": [
                {"name": "fet-alisa-uniform-anima-v4u", "weight": 1.0, "source": "a1111"},
                {"name": "fet-alisa-uniform-anima-v4u", "weight": 1.0, "source": "a1111"},
            ],
            "prompt_tags": [],
            "matches": [],
        }
    }
    assert report.score_metadata(case, result_matching_dup)["loras_ok"] is True


def test_pred_label_shows_failed_distinctly_from_genuine_empty():
    ok_empty = report.score_character(_case(expected_character=[]), {"vision_tags": {"character": []}})
    assert report._pred_label(ok_empty) == "(空)"

    failed = report.score_character(_case(expected_character=[]), {"vision_tags": {}})
    assert report._pred_label(failed) == "FAILED"

    has_chars = report.score_character(_case(expected_character=["alisa"]), {"vision_tags": {"character": ["alisa"]}})
    assert report._pred_label(has_chars) == "alisa"


def test_same_pred_label_distinguishes_both_failed_from_genuine_match():
    ok_empty_1 = report.score_character(_case(expected_character=[]), {"vision_tags": {"character": []}})
    ok_empty_2 = report.score_character(_case(expected_character=[]), {"vision_tags": {"character": []}})
    assert report._same_pred_label(ok_empty_1, ok_empty_2) == "一致"

    failed_1 = report.score_character(_case(expected_character=[]), {"vision_tags": {}})
    failed_2 = report.score_character(_case(expected_character=[]), {"vision_tags": {}})
    # 両方失敗している場合、予測集合は偶然どちらも空集合になるが、
    # 本物の一致とは区別して「同一(失敗)」と表示する。
    assert report._same_pred_label(failed_1, failed_2) == "同一(失敗)"

    # 片方だけ失敗している場合は不一致扱い
    assert report._same_pred_label(ok_empty_1, failed_1) == "不一致"

    has_alisa = report.score_character(_case(expected_character=["alisa"]), {"vision_tags": {"character": ["alisa"]}})
    has_second = report.score_character(
        _case(expected_character=["second_original"]), {"vision_tags": {"character": ["second_original"]}}
    )
    assert report._same_pred_label(has_alisa, has_second) == "不一致"


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
    assert scored["loras_ok"] is False
    assert scored["triggers_ok"] is False

    result_none_evidence = {"metadata_evidence": None}
    scored = report.score_metadata(case, result_none_evidence)
    assert scored["failed"] is True
    assert scored["format_ok"] is False
    assert scored["chars_ok"] is False
    assert scored["loras_ok"] is False
    assert scored["triggers_ok"] is False


def test_build_case_row_has_all_csv_fieldnames():
    case = {
        "case_id": "A01",
        "source_image_id": "A01",
        "derived_from": None,
        "scenario": "alisa_lora",
        "expected": {"image_type": "illustration", "art_style": "anime_2d", "subject": "person", "character": ["alisa"]},
        "expected_metadata": {
            "format": "a1111",
            "loras": [{"name": "fet-alisa-uniform-anima-v4u", "weight": 0.8}],
            "trigger_words": ["fet_alisa_uniform"],
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
            "loras": [{"name": "fet-alisa-uniform-anima-v4u", "weight": 0.8, "source": "a1111"}],
            "prompt_tags": [],
            "matches": [
                {"kind": "lora_name", "value": "fet-alisa-uniform-anima-v4u", "weight": 0.8, "character": "alisa"},
                {"kind": "prompt_trigger", "value": "fet_alisa_uniform", "character": "alisa"},
            ],
        },
        "timing_ms": {"classification_wall_ms": 12.5},
        "request_count": 4,
        "errors": [],
    }
    row = report.build_case_row(case, "choice", 0, result)
    assert set(row.keys()) == set(report.CSV_FIELDNAMES)
    assert row["ok_image_type"] is True
    assert row["character_exact_match"] is True
    assert row["meta_loras_ok"] is True
    assert row["meta_triggers_ok"] is True
    assert row["expected_meta_artificial"] is False
    assert row["expected_meta_loras"] == "fet-alisa-uniform-anima-v4u:0.8"
    assert row["detected_meta_loras"] == "fet-alisa-uniform-anima-v4u:0.8"
    assert row["json_attempts"] == ""  # choice モードでは空
