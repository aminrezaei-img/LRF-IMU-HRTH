import csv

from lrf_imu.evaluation.harth_sanity import _write_report


def test_flow_report_accepts_optional_real_class_fields(tmp_path):
    _write_report(
        tmp_path,
        "Flow",
        {"schema_version": "test", "dataset": "test", "held_out_subject": "S006"},
        [
            {"class_id": 0, "class_name": "walking_slow"},
            {
                "class_id": 1,
                "class_name": "walking_moderate",
                "real_windows": 3,
                "real_rms": 0.5,
            },
        ],
        "flow_class_metrics.csv",
    )

    with (tmp_path / "flow_class_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["real_windows"] == "3"
