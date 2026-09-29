import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import aggregate_final as af  # noqa: E402

COLS = [
    "case_id", "source_image_id", "derived_from", "scenario", "mode", "order_index",
    "expected_color", "pred_color", "ok_color",
    "expected_outfit", "pred_outfit", "exact_outfit",
    "classification_wall_ms", "prime_ms", "request_count", "error_types",
]


def make_run(root: Path, name: str, wall: list[float], colors: list[str], errors=("", "")):
    d = root / name
    d.mkdir()
    meta = {
        "tool_commit": {"commit": "abc", "dirty": False},
        "model": "m", "modes": ["choice"], "confirm": False, "bundled_multi": "rank",
        "runtime_info": {"server": {"image_cache_patch": True}},
    }
    (d / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with open(d / "cases.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for i, (ms, col) in enumerate(zip(wall, colors)):
            w.writerow({
                "case_id": f"C{i}", "source_image_id": f"C{i}", "derived_from": "", "scenario": "s",
                "mode": "choice", "order_index": i, "expected_color": "red", "pred_color": col,
                "ok_color": col == "red", "expected_outfit": "", "pred_outfit": "", "exact_outfit": True,
                "classification_wall_ms": ms, "prime_ms": 100, "request_count": 2, "error_types": errors[i],
            })
        # 派生ケース(集計から除く)
        w.writerow({"case_id": "C0-strip", "source_image_id": "C0", "derived_from": "C0", "scenario": "s",
                    "mode": "choice", "order_index": 9, "expected_color": "red", "pred_color": "blue",
                    "ok_color": False, "expected_outfit": "", "pred_outfit": "", "exact_outfit": True,
                    "classification_wall_ms": 99999, "prime_ms": 100, "request_count": 2, "error_types": ""})


def test_mean_sd_and_fluctuation(tmp_path):
    make_run(tmp_path, "F1_e1024_a_r1", [1000, 2000], ["red", "red"])
    make_run(tmp_path, "F1_e1024_a_r2", [3000, 4000], ["red", "blue"], errors=("", "logprobs_missing"))
    (tmp_path / "F1_e1024_a_r3").mkdir()  # 実行中(run.json なし)は無視
    runs = af.load_runs(tmp_path)
    assert len(runs) == 2
    md = af.build_summary(runs)
    # 回ごとの平均は 1500 と 3500 -> 平均 2500、標準偏差 1414
    assert "2500 ± 1414 (2)" in md
    # 正答率は 100% と 50% -> 75.0 ± 35.4
    assert "75.0 ± 35.4 (2)" in md
    # C1 の color だけが揺れた(派生は除外)
    assert "1 / 4" in md
    assert "C1/color: red | blue" in md
    assert "logprobs_missing=1" in md


def test_single_repetition_has_no_sd(tmp_path):
    make_run(tmp_path, "F2_e768_a_r1", [1000, 3000], ["red", "red"])
    md = af.build_summary(af.load_runs(tmp_path))
    assert "2000 ± n/a (1)" in md
