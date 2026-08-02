
from pathlib import Path
import hashlib
import json
import math
import statistics

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def test_fixture_manifest_and_checksum_scope():
    manifest = load(FIXTURE_ROOT / "fixture_manifest.json")
    listed = set(manifest["files"])
    actual = {
        p.relative_to(FIXTURE_ROOT).as_posix()
        for p in FIXTURE_ROOT.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS"
    }
    assert listed == actual
    assert "SHA256SUMS" not in listed
    assert manifest["integrity"]["checksum_excludes_itself"] is True
    assert "CRLF normalized to LF" in manifest["integrity"]["checksum_line_endings"]
    assert manifest["safety"] == {
        "participant_derived_data": False,
        "checkpoint_payloads": False,
        "license_safe": True,
        "deterministic": True,
    }
    checksum_lines = (FIXTURE_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    entries = {}
    for line in checksum_lines:
        digest, rel = line.split("  ", 1)
        entries[rel] = digest
    assert set(entries) == listed
    assert list(entries) == sorted(listed)
    for rel, expected in entries.items():
        raw = (FIXTURE_ROOT / rel).read_bytes()
        canonical = raw.replace(b"\r\n", b"\n")
        actual_digest = hashlib.sha256(canonical).hexdigest().upper()
        assert actual_digest == expected

def test_fixture_text_has_no_personal_paths_or_payload_extensions():
    forbidden_markers = (
        "C" + ":/",
        "D" + ":/",
        "C" + ":" + "\\",
        "D" + ":" + "\\",
        "A" + "minR",
        ".pt",
        ".pth",
        ".ckpt",
        ".npz",
        ".npy",
    )
    for path in FIXTURE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in forbidden_markers), path
        assert "participant_id" not in text.lower()
        assert "raw participant data" not in text.lower()

def test_raw_logs_are_deterministic_120_column_synthetic_records():
    allowed = {1, 3, 4, 33}
    expected_labels = {
        1: [1, 1, 99, 99, 1, 1] + [3] * 6 + [4] * 6 + [33] * 6,
        2: [1] * 6 + [3] * 6 + [4] * 6 + [33] * 6,
        3: [1] * 6 + [3] * 6 + [4] * 6 + [33] * 6,
        5: [1] * 6 + [3] * 6 + [4] * 6 + [33] * 6,
    }
    logs = sorted((FIXTURE_ROOT / "raw").glob("subject*_ideal.log"))
    assert [p.stem[7:9] for p in logs] == ["01", "02", "03", "05"]
    for path in logs:
        subject = int(path.stem[7:9])
        rows = path.read_text(encoding="utf-8").splitlines()
        assert len(rows) == 24
        for row_index, line in enumerate(rows):
            fields = line.split("\t")
            assert len(fields) == 120
            assert int(fields[119]) == expected_labels[subject][row_index]
            assert fields[119] in {"1", "3", "4", "33", "99"}
            for col, value in enumerate(fields):
                if col == 119:
                    continue
                actual = float(value)
                if 80 <= col <= 85:
                    expected = subject * 1000 + row_index * 10 + col - 80
                else:
                    expected = -(subject * 100 + col + 1)
                assert actual == expected
        assert set(expected_labels[subject]) - allowed == ({99} if subject == 1 else set())

def _compact_counts(labels, allowed, window=4, stride=2):
    filtered = [label for label in labels if label in allowed]
    runs = []
    start = 0
    while start < len(filtered):
        stop = start + 1
        while stop < len(filtered) and filtered[stop] == filtered[start]:
            stop += 1
        runs.append(filtered[start:stop])
        start = stop
    return sum(max(0, (len(run) - window) // stride + 1) for run in runs), runs

def test_filter_before_runs_and_safe_16_7_8_recipe():
    case = load(FIXTURE_ROOT / "preprocessing_cases.json")
    assert case["gap_bridge_case"]["run_detection_order"] == ["filter", "bridge_short_gaps", "detect_runs"]
    assert case["gap_bridge_case"]["filtered_row_indices"] == [0, 1, 4, 5]
    assert case["gap_bridge_case"]["selected_acc_x_after_bridge"] == [1000, 1010, 1040, 1050]
    assert case["gap_bridge_case"]["compact_window_starts"] == [0]
    labels_by_subject = {
        1: [1, 1, 99, 99, 1, 1] + [3] * 6 + [4] * 6 + [33] * 6,
        2: [1] * 6 + [3] * 6 + [4] * 6 + [33] * 6,
        3: [1] * 6 + [3] * 6 + [4] * 6 + [33] * 6,
        5: [1] * 6 + [3] * 6 + [4] * 6 + [33] * 6,
    }
    counts = {}
    for subject, labels in labels_by_subject.items():
        counts[f"{subject:02d}"], runs = _compact_counts(labels, {1, 3, 4, 33})
        assert all(len(run) >= 4 for run in runs)
    assert counts == {"01": 7, "02": 8, "03": 8, "05": 8}
    split = load(FIXTURE_ROOT / "loso_split_cases.json")
    assert split["safe_split"]["window_counts"] == {"train": 16, "validation": 7, "held_out_test": 8}
    assert split["safe_split"]["shapes"] == {"train": [16, 6, 4], "validation": [7, 6, 4], "held_out_test": [8, 6, 4]}
    assert sum(counts.values()) == 31
    assert split["validation_fraction_separation"] == {"vae_subject_fraction": 0.15, "cnn_window_fraction": 0.20}

def test_channel_standardization_duplicate_and_shapes():
    channels = load(FIXTURE_ROOT / "channel_selection.json")
    assert channels["observed_six_channel_columns"] == [80, 81, 82, 83, 84, 85]
    assert channels["intended_three_channel_columns"] == [80, 81, 82]
    assert channels["three_channel_is_separate_schema"] is True
    assert channels["allow_inference_time_channel_drop"] is False
    std = load(FIXTURE_ROOT / "standardization_cases.json")
    assert std["axes"] == [0, 2]
    assert std["ddof"] == 0
    for base, mean, scale in zip(std["channel_bases"], std["expected_train_means"], std["expected_train_stds"]):
        values = [base - 1, base + 1, base - 1, base + 1]
        assert statistics.fmean(values) == mean
        assert math.sqrt(statistics.fmean([(v - mean) ** 2 for v in values])) == scale
    assert std["constant_channel_std_after_floor"] == 1e-8
    duplicate = load(FIXTURE_ROOT / "duplicate_audit_cases.json")
    assert duplicate["required_extension"] == "train_held_out_test"
    assert {case["case_id"] for case in duplicate["cases"]} == {"clean", "train_val_duplicate", "train_test_only_duplicate"}
    assert all(len(value) == 40 for value in duplicate["expected_hashes"].values())

def test_vae_and_flow_contract_probes():
    vae = load(FIXTURE_ROOT / "vae_probe.json")
    assert vae["compact"]["input_shape"] == [2, 6, 16]
    assert vae["compact"]["reconstruction_shape"] == [2, 6, 16]
    assert vae["production"]["latent_shape"] == [1, 48, 40]
    assert vae["three_channel"]["input_shape"] == [1, 3, 160]
    assert "no_checkpoint_payload" in vae["determinism_assertions"]
    flow = load(FIXTURE_ROOT / "flow_probe.json")
    forward = flow["forward_interpolation"]
    z0, z1, t = forward["z0"][0], forward["z1"][0], forward["t"]
    expected = [[(1 - t) * z0[row][col] + t * z1[row][col] for col in range(2)] for row in range(2)]
    assert expected == forward["expected_zt"][0]
    assert [[z1[row][col] - z0[row][col] for col in range(2)] for row in range(2)] == forward["expected_target_velocity"][0]
    assert sum((1 - 4) ** 2 for _ in range(4)) / 4 == forward["expected_mse"]
    reverse = flow["reverse_euler"]
    current = reverse["initial"][0]
    for _ in range(reverse["steps"]):
        current = [[current[row][col] + reverse["dt"] * reverse["velocity"][0][row][col] for col in range(2)] for row in range(2)]
    assert current == reverse["after_final_step"][0]
    assert flow["production_profile"]["paper_steps"] == 10
    assert flow["production_profile"]["website_euler_steps"] == 100
    assert flow["production_profile"]["website_native_samples"] == 500

def test_website_overlap_trajectory_and_metrics():
    overlap = load(FIXTURE_ROOT / "website_overlap_add.json")
    left, right = overlap["windows"]
    start = len(left) - overlap["overlap"]
    crossfaded = [
        left[start + index] * (1 - weight) + right[index] * weight
        for index, weight in enumerate(overlap["crossfade_weights"])
    ]
    reconstructed = left[:start] + crossfaded + right[overlap["overlap"]:]
    assert reconstructed == overlap["expected_reconstruction"]
    trajectory = load(FIXTURE_ROOT / "website_trajectory.json")
    assert trajectory["native_signal"]["sampling_hz"] == 50
    assert trajectory["stored_trajectory"]["signals_shape"] == [3, 6, 6]
    assert trajectory["stored_trajectory"]["stored_sample_indices"] == [0, 2, 4]
    assert trajectory["stored_trajectory"]["base_seed"] == 42
    assert trajectory["stored_trajectory"]["subject_id"] == 2
    assert trajectory["stored_trajectory"]["activity_class_id"] == 1
    assert trajectory["stored_trajectory"]["computed_seed"] == 2142
    assert trajectory["stored_trajectory"]["seed_formula"] == "base_seed + subject_id * 1000 + activity_class_id * 100"
    assert trajectory["production_profile"]["stored_steps"] == 100
    assert trajectory["production_profile"]["state_count"] == 51
    reference = load(FIXTURE_ROOT / "evaluation_reference.json")
    labels = reference["labels"]
    assert labels == [0, 1, 2, 3]
    assert reference["label_space"] == "encoded"
    assert reference["raw_code_to_encoded"] == {"1": 0, "3": 1, "4": 2, "33": 3}
    y_true = reference["y_true"]
    predictions = reference["y_pred"]
    assert predictions == reference["predictions"]
    f1_values = []
    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(y_true, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(y_true, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(y_true, predictions))
        precision = tp / (tp + fp) if tp + fp else reference["zero_division"]
        recall = tp / (tp + fn) if tp + fn else reference["zero_division"]
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    assert f1_values == pytest.approx([reference["per_class_f1"][str(label)] for label in labels])
    assert statistics.fmean(f1_values) == pytest.approx(reference["macro_f1"])
    assert reference["confusion_matrix"] == [[2,0,0,0],[0,1,0,1],[0,0,1,1],[0,0,0,2]]
    assert statistics.stdev(reference["fold_macro_f1"]) == pytest.approx(reference["fold_sample_std_ddof1"])
    assert statistics.pstdev(reference["fold_macro_f1"]) == pytest.approx(reference["fold_population_std_ddof0"])

def test_csv_fixture_is_human_auditable():
    rows = (FIXTURE_ROOT / "metadata_summary.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "subject,raw_rows,compact_windows,selected_channels,window_samples,stride_samples"
    assert len(rows) == 5
    assert rows[1:] == [
        "01,24,7,6,4,2",
        "02,24,8,6,4,2",
        "03,24,8,6,4,2",
        "05,24,8,6,4,2",
    ]
