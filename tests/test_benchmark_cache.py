import urllib.error

import benchmark_cache

from classifier_demo.taxonomy import Axis, Choice, Taxonomy
from tests.fakes import FakeBackend, make_logprobs_response, make_no_logprobs_response


def _axis(axis_id: str) -> Axis:
    return Axis(
        id=axis_id,
        question="q",
        multi=False,
        allow_none=False,
        choices=[
            Choice(id="x", name="x", criteria="c"),
            Choice(id="y", name="y", criteria="c"),
        ],
    )


def _two_axis_taxonomy() -> Taxonomy:
    return Taxonomy(version="test", axes=[_axis("axis1"), _axis("axis2")], sha256="deadbeef")


def _one_axis_taxonomy() -> Taxonomy:
    return Taxonomy(version="test", axes=[_axis("axis1")], sha256="deadbeef")


def test_send_ranking_records_success():
    backend = FakeBackend([make_logprobs_response({"A": 0.9, "B": 0.1})])
    axis = _axis("axis1")
    record = benchmark_cache._send_ranking(backend, b"img", "image/png", axis)
    assert record["error"] is None
    assert record["top_label"] == "x"
    assert record["relative_scores"]["x"] > record["relative_scores"]["y"]
    assert record["elapsed_ms"] == 1.0


def test_send_ranking_records_decision_error_without_raising():
    backend = FakeBackend([make_no_logprobs_response()])
    axis = _axis("axis1")
    record = benchmark_cache._send_ranking(backend, b"img", "image/png", axis)
    assert record["error"] is not None
    assert "no_logprobs" in record["error"]
    assert record["top_label"] is None
    assert record["elapsed_ms"] is None


def test_send_ranking_records_request_error_without_raising():
    backend = FakeBackend([urllib.error.URLError("connection refused")])
    axis = _axis("axis1")
    record = benchmark_cache._send_ranking(backend, b"img", "image/png", axis)
    assert record["error"] is not None
    assert "URLError" in record["error"]


def test_run_test1_separates_first_axis_from_later_axes_and_continues_on_failure():
    tax = _two_axis_taxonomy()
    # repeat1: reset, axis1 ok, axis2 fails / repeat2: reset, axis1 ok, axis2 ok
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.4, "B": 0.6}),  # repeat1 reset (image B, axis1)
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # repeat1 axis1 (image A)
            make_no_logprobs_response(),  # repeat1 axis2 -> fails
            make_logprobs_response({"A": 0.3, "B": 0.7}),  # repeat2 reset (image B, axis1)
            make_logprobs_response({"A": 0.8, "B": 0.2}),  # repeat2 axis1 (image A)
            make_logprobs_response({"A": 0.7, "B": 0.3}),  # repeat2 axis2 (image A)
        ]
    )

    result = benchmark_cache.run_test1(backend, b"img_a", "image/png", b"img_b", "image/png", tax, repeats=2)

    assert len(result["requests"]) == 4
    summary = result["summary"]
    assert summary["first_axis_count"] == 2
    assert summary["later_axes_count"] == 1
    assert summary["first_axis_elapsed_ms_median"] == 1.0
    assert summary["later_axes_elapsed_ms_median"] == 1.0
    assert summary["excluded_repeats"] == 0
    assert summary["failures"] == 1


def test_run_test1_reset_failure_marks_repeat_and_excludes_it_from_summary():
    tax = _one_axis_taxonomy()
    backend = FakeBackend(
        [
            make_no_logprobs_response(),  # repeat1 reset (image B) -> fails
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # repeat1 axis1 (image A) -> 送信は止めない
            make_logprobs_response({"A": 0.4, "B": 0.6}),  # repeat2 reset (image B) -> 成功
            make_logprobs_response({"A": 0.8, "B": 0.2}),  # repeat2 axis1 (image A)
        ]
    )

    result = benchmark_cache.run_test1(backend, b"img_a", "image/png", b"img_b", "image/png", tax, repeats=2)

    assert len(result["requests"]) == 2
    repeat1_req, repeat2_req = result["requests"]
    assert repeat1_req["reset_failed"] is True
    assert repeat1_req["error"] is None  # Aリクエスト自体は送って成功している
    assert repeat2_req["reset_failed"] is False

    summary = result["summary"]
    assert summary["excluded_repeats"] == 1
    # reset失敗したrepeat1のAリクエストは中央値・件数から除外され、repeat2の1件のみが残る
    assert summary["first_axis_count"] == 1
    assert summary["first_axis_elapsed_ms_median"] == 1.0
    assert summary["failures"] == 0


def test_run_test1_records_one_reset_per_repeat_excluded_from_summary():
    tax = _two_axis_taxonomy()
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.1, "B": 0.9}),  # repeat1 reset (image B, axis1) -> top y
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # repeat1 axis1 (image A)
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # repeat1 axis2 (image A)
            make_logprobs_response({"A": 0.2, "B": 0.8}),  # repeat2 reset (image B, axis1) -> top y
            make_logprobs_response({"A": 0.8, "B": 0.2}),  # repeat2 axis1 (image A)
            make_logprobs_response({"A": 0.7, "B": 0.3}),  # repeat2 axis2 (image A)
        ]
    )

    result = benchmark_cache.run_test1(backend, b"img_a", "image/png", b"img_b", "image/png", tax, repeats=2)

    resets = result["resets"]
    assert len(resets) == 2
    assert [r["repeat"] for r in resets] == [1, 2]
    assert all(r["axis"] == "axis1" for r in resets)
    # 画像Bへのリセットなので top_label は画像Aの1軸目とは独立に y になる
    assert all(r["top_label"] == "y" for r in resets)

    # reset は要約(件数・中央値)の計算対象に含めない(通常のA軸リクエストのみ4件)
    summary = result["summary"]
    assert summary["first_axis_count"] == 2
    assert summary["later_axes_count"] == 2
    assert summary["excluded_repeats"] == 0
    assert len(result["requests"]) == 4


def test_run_test2_detects_label_mismatch_and_score_diff():
    tax = _one_axis_taxonomy()
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # A1 -> x
            make_logprobs_response({"A": 0.5, "B": 0.5}),  # B (unrelated)
            make_logprobs_response({"A": 0.2, "B": 0.8}),  # A2 -> y (画像混線を模した不一致)
        ]
    )

    result = benchmark_cache.run_test2(backend, b"img_a", "image/png", b"img_b", "image/png", tax, repeats=1)

    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["top_label_match"] is False
    assert group["max_abs_diff"] > 0.5

    summary = result["summary"]
    assert summary["label_mismatch_count"] == 1
    assert summary["max_abs_diff_overall"] == group["max_abs_diff"]
    assert summary["invalid_groups"] == 0
    assert summary["failures"] == 0


def test_run_test2_matching_labels_report_no_mismatch():
    tax = _one_axis_taxonomy()
    responses = [make_logprobs_response({"A": 0.9, "B": 0.1})] * 3  # A1/B/A2 とも同じ分布
    backend = FakeBackend(responses)

    result = benchmark_cache.run_test2(backend, b"img_a", "image/png", b"img_b", "image/png", tax, repeats=1)

    group = result["groups"][0]
    assert group["top_label_match"] is True
    assert group["max_abs_diff"] == 0.0
    assert result["summary"]["label_mismatch_count"] == 0


def test_run_test2_records_failure_and_continues():
    tax = _one_axis_taxonomy()
    backend = FakeBackend(
        [
            make_no_logprobs_response(),  # A1 fails
            make_logprobs_response({"A": 0.5, "B": 0.5}),  # B
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # A2
        ]
    )

    result = benchmark_cache.run_test2(backend, b"img_a", "image/png", b"img_b", "image/png", tax, repeats=1)

    group = result["groups"][0]
    assert group["requests"]["a1"]["error"] is not None
    # 片方が失敗している場合、一致判定・差分は計算しない(Noneのまま)
    assert group["top_label_match"] is None
    assert group["max_abs_diff"] is None
    assert result["summary"]["failures"] == 1


def test_run_test2_b_failure_marks_group_invalid_and_excludes_from_summary():
    tax = _one_axis_taxonomy()
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # A1 -> x
            make_no_logprobs_response(),  # B -> fails(画像が実際にBへ切り替わった保証がない)
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # A2 -> x(A1と一致するが無効扱い)
        ]
    )

    result = benchmark_cache.run_test2(backend, b"img_a", "image/png", b"img_b", "image/png", tax, repeats=1)

    group = result["groups"][0]
    assert group["b_failed"] is True
    assert group["top_label_match"] is None
    assert group["max_abs_diff"] is None

    summary = result["summary"]
    assert summary["invalid_groups"] == 1
    assert summary["label_mismatch_count"] == 0
    assert summary["max_abs_diff_overall"] is None
    assert summary["failures"] == 1
