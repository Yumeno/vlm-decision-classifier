import urllib.error
from pathlib import Path

from PIL import Image, PngImagePlugin

from classifier_demo import pipeline
from classifier_demo.taxonomy import Axis, Choice, Taxonomy
from tests.fakes import (
    FakeBackend,
    KeyedFakeBackend,
    make_logprobs_response,
    make_no_logprobs_response,
)


def _test_taxonomy() -> Taxonomy:
    """default taxonomy から独立した、pipeline テスト専用の小さな taxonomy。
    軸構成(単一選択3軸 + 複数選択1軸)は既存テストの FakeBackend レスポンス列
    (ラベル数・リクエスト順)に合わせてある。"""
    image_type = Axis(
        id="image_type",
        question="q",
        multi=False,
        allow_none=False,
        choices=[
            Choice(id="illustration", name="illustration", criteria="c"),
            Choice(id="comic", name="comic", criteria="c"),
            Choice(id="ui", name="user interface", criteria="c"),
            Choice(id="other", name="other", criteria="c"),
        ],
    )
    art_style = Axis(
        id="art_style",
        question="q",
        multi=False,
        allow_none=False,
        choices=[
            Choice(id="anime_2d", name="anime style", criteria="c"),
            Choice(id="painterly_2d", name="painterly", criteria="c"),
            Choice(id="pixel_art", name="pixel art", criteria="c"),
            Choice(id="other", name="other", criteria="c"),
        ],
    )
    subject = Axis(
        id="subject",
        question="q",
        multi=False,
        allow_none=False,
        choices=[
            Choice(id="person", name="person", criteria="c"),
            Choice(id="landscape", name="landscape", criteria="c"),
            Choice(id="mecha_vehicle", name="mecha or vehicle", criteria="c"),
            Choice(id="object", name="object", criteria="c"),
            Choice(id="creature", name="creature", criteria="c"),
            Choice(id="other", name="other", criteria="c"),
        ],
    )
    character = Axis(
        id="character",
        question="Which character appears in this image?",
        multi=True,
        allow_none=True,
        choices=[
            Choice(
                id="alisa",
                name="Alisa",
                criteria="c",
                lora_names=["fet-alisa-uniform-anima-v4u"],
                trigger_words=["fet_alisa_uniform"],
            ),
            Choice(id="second_original", name="second", criteria="c"),
            Choice(id="other_original", name="other", criteria="c", catch_all=True),
        ],
    )
    return Taxonomy(version="test", axes=[image_type, art_style, subject, character], sha256="deadbeef")


def _make_image_with_alisa_lora(path: Path) -> None:
    img = Image.new("RGB", (16, 16), (200, 100, 100))
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "parameters",
        "masterpiece, <lora:fet-alisa-uniform-anima-v4u:0.8>, 1girl\nSteps: 20",
    )
    img.save(path, pnginfo=info)


def test_classify_choice_mode_records_axis_failure_and_succeeds_others(tmp_path):
    image_path = tmp_path / "img.png"
    _make_image_with_alisa_lora(image_path)

    tax = _test_taxonomy()

    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type -> illustration
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style -> anime_2d
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject -> person
            make_no_logprobs_response(),  # character axis fails
        ]
    )

    result = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64)

    assert result["mode"] == "choice"
    assert result["vision_tags"]["image_type"] == ["illustration"]
    assert result["vision_tags"]["art_style"] == ["anime_2d"]
    assert result["vision_tags"]["subject"] == ["person"]
    assert "character" not in result["vision_tags"]

    assert len(result["errors"]) == 1
    assert result["errors"][0]["axis"] == "character"
    assert result["errors"][0]["type"] == "no_logprobs"

    # メタデータで alisa の LoRA が一致しているが、character軸が失敗しているので unknown
    assert result["combined_evidence"] == {"alisa": "unknown"}
    assert result["metadata_evidence"]["format"] == "a1111"

    # 画像の絶対パスを結果に含めない
    assert result["image"]["name"] == "img.png"
    assert "\\" not in result["image"]["name"] and "/" not in result["image"]["name"]

    assert result["timing_ms"]["classification_wall_ms"] > 0
    assert result["request_count"] == 4
    assert "axis_decisions" in result
    assert "image_type" in result["axis_decisions"]


def test_classify_records_request_error_and_continues_other_axes(tmp_path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(image_path)
    tax = _test_taxonomy()

    backend = FakeBackend(
        [
            urllib.error.URLError("connection refused"),  # image_type -> 通信失敗
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style -> anime_2d
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject -> person
            make_logprobs_response(
                {"A": 0.6, "B": 0.0001, "C": 0.1999, "D": 0.2}
            ),  # character ranking -> alisa top, second_original below floor
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # alisa yes/no -> yes
        ]
    )

    result = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64)

    assert "image_type" not in result["vision_tags"]
    assert result["vision_tags"]["art_style"] == ["anime_2d"]
    assert any(e["axis"] == "image_type" and e["type"] == "request_error" for e in result["errors"])


def test_classify_image_error_is_recorded_without_crashing(tmp_path):
    missing_path = tmp_path / "does_not_exist.png"
    tax = _test_taxonomy()
    backend = FakeBackend([])

    result = pipeline.classify(str(missing_path), tax, backend, mode="choice", max_edge=64)

    assert result["vision_tags"] == {}
    assert result["metadata_evidence"] is None
    assert any(e["type"] == "image_error" for e in result["errors"])
    assert result["timing_ms"]["classification_wall_ms"] > 0
    assert result["request_count"] == 0


def test_classify_character_confirmation_failure_excludes_from_vision_tags(tmp_path):
    # character軸の確認(yes/no)が1件失敗した場合、その軸は失敗扱いとなり、
    # vision_tags に character を入れない(catch_all へもフォールバックしない)。
    image_path = tmp_path / "img.png"
    _make_image_with_alisa_lora(image_path)
    tax = _test_taxonomy()

    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject
            make_logprobs_response(
                {"A": 0.6, "B": 0.0001, "C": 0.1999, "D": 0.2}
            ),  # character ranking -> alisa candidate only
            urllib.error.URLError("connection refused"),  # alisa yes/no -> 通信失敗
        ]
    )

    result = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64)

    assert "character" not in result["vision_tags"]
    assert result["axis_decisions"]["character"]["failed"] is True
    assert result["axis_decisions"]["character"]["tags"] == []
    # メタデータで alisa の LoRA が一致しているが、character軸が失敗しているので unknown
    assert result["combined_evidence"] == {"alisa": "unknown"}
    assert any(
        e["axis"] == "character" and e["type"] == "candidate_confirmation_error" for e in result["errors"]
    )


def test_classify_on_progress_receives_metadata_then_axis_events_in_order(tmp_path):
    # デモUIサーバー用の拡張: on_progress を渡しても戻り値は変わらず、
    # メタデータイベント1回 → 軸ごとに1回(taxonomy.axesの順)呼ばれる。
    image_path = tmp_path / "img.png"
    _make_image_with_alisa_lora(image_path)
    tax = _test_taxonomy()

    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject
            make_no_logprobs_response(),  # character axis fails
        ]
    )

    events: list[dict] = []
    result = pipeline.classify(
        str(image_path), tax, backend, mode="choice", max_edge=64, on_progress=events.append
    )

    assert events[0]["type"] == "metadata"
    assert events[0]["metadata_evidence"]["format"] == "a1111"

    axis_events = events[1:]
    assert [e["axis_id"] for e in axis_events] == ["image_type", "art_style", "subject", "character"]
    assert axis_events[0]["error"] is None
    assert axis_events[0]["selected"] == "illustration"
    assert axis_events[-1]["error"]["type"] == "no_logprobs"

    # on_progress を渡しても戻り値そのものは変わらない
    result_without_callback = pipeline.classify(
        str(image_path),
        tax,
        FakeBackend(
            [
                make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),
                make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),
                make_logprobs_response(
                    {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
                ),
                make_no_logprobs_response(),
            ]
        ),
        mode="choice",
        max_edge=64,
    )
    # classification_wall_ms は実行のたびに変わるので、それ以外が一致することを見る
    result["timing_ms"]["classification_wall_ms"] = None
    result_without_callback["timing_ms"]["classification_wall_ms"] = None
    assert result == result_without_callback


def test_classify_metadata_error_is_recorded_and_classification_continues(tmp_path, monkeypatch):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(image_path)
    tax = _test_taxonomy()

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "extract_evidence", _raise)

    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject
            make_logprobs_response(
                {"A": 0.6, "B": 0.0001, "C": 0.1999, "D": 0.2}
            ),  # character ranking -> alisa candidate only
            make_logprobs_response({"A": 0.1, "B": 0.9}),  # alisa yes/no -> no
        ]
    )

    result = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64)

    assert result["metadata_evidence"] == {"format": "error", "loras": [], "prompt_tags": [], "matches": []}
    assert any(e["type"] == "metadata_error" for e in result["errors"])
    # メタデータ抽出が失敗しても分類自体は続行される
    assert result["vision_tags"]["image_type"] == ["illustration"]


# ---------------------------------------------------------------------------
# E9: axis_concurrency(軸の並列送信)
# ---------------------------------------------------------------------------


def _concurrency_taxonomy() -> Taxonomy:
    """軸ごとに一意な目印文字列を質問文・候補criteriaに埋め込んだ、並列テスト専用の
    小さな taxonomy(単一選択3軸 + 複数選択1軸)。KeyedFakeBackend がプロンプト中の
    目印でリクエストを振り分けるので、軸の到着順(並列実行では保証されない)に依らず
    正しい応答を返せる。"""
    image_type = Axis(
        id="image_type",
        question="AXIS_IMAGE_TYPE marker",
        multi=False,
        allow_none=False,
        choices=[Choice(id="illustration", name="illustration", criteria="c"), Choice(id="comic", name="comic", criteria="c")],
    )
    art_style = Axis(
        id="art_style",
        question="AXIS_ART_STYLE marker",
        multi=False,
        allow_none=False,
        choices=[Choice(id="anime_2d", name="anime", criteria="c"), Choice(id="painterly_2d", name="painterly", criteria="c")],
    )
    subject = Axis(
        id="subject",
        question="AXIS_SUBJECT marker",
        multi=False,
        allow_none=False,
        choices=[Choice(id="person", name="person", criteria="c"), Choice(id="landscape", name="landscape", criteria="c")],
    )
    character = Axis(
        id="character",
        question="AXIS_CHARACTER marker",
        multi=True,
        allow_none=True,
        choices=[
            Choice(id="alisa", name="Alisa", criteria="ALISA_CRITERIA marker"),
            Choice(id="other_original", name="Other", criteria="OTHER_CRITERIA marker", catch_all=True),
        ],
    )
    return Taxonomy(version="test", axes=[image_type, art_style, subject, character], sha256="deadbeef")


def _concurrency_backend() -> KeyedFakeBackend:
    return KeyedFakeBackend(
        {
            "AXIS_IMAGE_TYPE": make_logprobs_response({"A": 0.9, "B": 0.1}),  # -> illustration
            "AXIS_ART_STYLE": make_logprobs_response({"A": 0.2, "B": 0.8}),  # -> painterly_2d
            "AXIS_SUBJECT": make_logprobs_response({"A": 0.7, "B": 0.3}),  # -> person
            "AXIS_CHARACTER": make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.05}),  # ranking -> alisa candidate
            "ALISA_CRITERIA": make_logprobs_response({"A": 0.95, "B": 0.05}),  # confirm alisa -> yes
        }
    )


def test_classify_axis_concurrency_matches_sequential_content_and_order(tmp_path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path)
    tax = _concurrency_taxonomy()

    # 同じ(消費されない)偽バックエンドを両方の実行で使い回せる: KeyedFakeBackendは
    # プロンプトの目印で応答を選ぶだけで、呼び出し順に依存しない。
    backend = _concurrency_backend()

    result_sequential = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64, axis_concurrency=1)
    result_parallel = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64, axis_concurrency=4)

    # classification_wall_ms は実行のたびに変わるので、それ以外の内容が完全に一致することを見る
    result_sequential["timing_ms"]["classification_wall_ms"] = None
    result_parallel["timing_ms"]["classification_wall_ms"] = None
    assert result_sequential == result_parallel

    # 戻り値の辞書の並び順も taxonomy.axes の順のまま(on_progress は完了順でよいが、
    # マージ結果は axis_concurrency に依らず決定的)。
    expected_order = ["image_type", "art_style", "subject", "character"]
    assert list(result_parallel["axis_decisions"].keys()) == expected_order
    assert list(result_parallel["vision_tags"].keys()) == expected_order
    assert list(result_parallel["timing_ms"].keys())[1:] == expected_order  # classification_wall_ms の次から

    assert result_parallel["vision_tags"]["image_type"] == ["illustration"]
    assert result_parallel["vision_tags"]["art_style"] == ["painterly_2d"]
    assert result_parallel["vision_tags"]["subject"] == ["person"]
    assert result_parallel["vision_tags"]["character"] == ["alisa"]


def test_classify_axis_concurrency_request_count_is_correct(tmp_path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path)
    tax = _concurrency_taxonomy()
    backend = _concurrency_backend()

    result = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64, axis_concurrency=4)

    # image_type(1) + art_style(1) + subject(1) + character(ranking 1 + confirm alisa 1) = 5
    assert result["request_count"] == 5
    assert backend.request_count == 5
    assert result["errors"] == []


def test_classify_axis_concurrency_on_progress_thread_safe_and_complete(tmp_path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path)
    tax = _concurrency_taxonomy()
    backend = _concurrency_backend()

    events: list[dict] = []
    lock_probe = {"concurrent_calls": 0, "max_concurrent": 0}

    def on_progress(event: dict) -> None:
        # on_progress 自体がロックで直列化されていることを検証する
        # (呼び出し中に他スレッドが割り込んで同時に events.append しないか)。
        lock_probe["concurrent_calls"] += 1
        lock_probe["max_concurrent"] = max(lock_probe["max_concurrent"], lock_probe["concurrent_calls"])
        events.append(event)
        lock_probe["concurrent_calls"] -= 1

    pipeline.classify(
        str(image_path), tax, backend, mode="choice", max_edge=64, axis_concurrency=4, on_progress=on_progress
    )

    # metadata 1回 + axis 4回(image_type/art_style/subject/character)
    assert len(events) == 5
    assert events[0]["type"] == "metadata"
    axis_events = events[1:]
    assert {e["axis_id"] for e in axis_events} == {"image_type", "art_style", "subject", "character"}
    assert lock_probe["max_concurrent"] == 1
