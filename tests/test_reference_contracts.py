
from pathlib import Path
import json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def test_existing_milestone_2a_contracts_remain_pinned():
    names = [
        "source_inventory.json",
        "data_preprocessing_contract.json",
        "vae_contract.json",
        "rectified_flow_contract.json",
        "evaluation_contract.json",
        "runtime_contract.json",
    ]
    contracts = {name: load(REPOSITORY_ROOT / "contracts" / name) for name in names}
    assert all(contract["schema_version"] == "2A.1" for contract in contracts.values())
    preprocessing = contracts["data_preprocessing_contract.json"]["contract"]
    assert preprocessing["dataset"]["sampling_rate_hz"] == 50
    assert preprocessing["dataset"]["log_format"]["expected_columns"] == 120
    assert preprocessing["columns"]["right_thigh_zero_based"] == [80, 81, 82, 83, 84, 85]
    assert preprocessing["columns"]["label_zero_based"] == 119
    assert preprocessing["windowing"]["main_window_samples"] == 160
    assert preprocessing["windowing"]["main_hop_samples"] == 40
    assert preprocessing["normalization"]["fit_axes"] == [0, 2]
    assert preprocessing["normalization"]["standard_deviation_ddof"] == 0
    assert preprocessing["split_interface"]["observed_return_count"] == 8
    assert preprocessing["split_interface"]["standard_alias_expected_return_count"] == 7
    vae = contracts["vae_contract.json"]["contract"]
    assert vae["geometry"]["latent_mean_shape"] == "[B,48,40]"
    assert vae["channels"]["three_channel_policy"].startswith("separate model")
    flow = contracts["rectified_flow_contract.json"]["contract"]
    assert flow["latent"]["shape"] == "[B,48,40]"
    assert flow["paper_sampling"]["steps"] == 10
    assert flow["paper_sampling"]["seed"] == 42
    evaluation = contracts["evaluation_contract.json"]["contract"]
    assert evaluation["metrics"]["primary"] == "macro-F1"
    assert evaluation["metrics"]["zero_division"] == 0
    assert evaluation["metrics"]["standard_deviation_ddof"] == 1
    assert evaluation["split"]["vae_safe_subject_validation_fraction"] == 0.15
    assert evaluation["split"]["cnn_internal_validation_fraction"] == 0.20

def test_configuration_decision_matrix_preserves_qualified_evidence():
    contract = load(REPOSITORY_ROOT / "contracts" / "configuration_decision_matrix.json")
    assert contract["schema_version"] == "2B.configuration-evidence.1"
    assert contract["exact_paper_reproduction"] is False
    assert contract["authority_pins"]["public_repository"]["commit"] == "8445c9cff1b93e92844b94394258635b5d25fd54"
    assert contract["authority_pins"]["audit_repository"]["commit"] == "f38bebce36c4f21d857dc084ac8d06759c2c012d"
    items = {item["decision_id"]: item for item in contract["items"]}
    assert len(items) == 16
    assert items["accepted_source_candidate_identity"]["decision_status"] == "HOLD"
    assert items["vae_batch_size"]["selected_public_compatibility_value"] == "256"
    assert items["rectified_flow_sampling"]["decision_status"] == "RESOLVED"
    assert items["channel_lineage_and_augmentation_ablation"]["decision_status"] == "HOLD"
    assert contract["compatibility_profile"]["data"]["window_samples"] == 160
    assert contract["compatibility_profile"]["vae"]["latent_shape"] == ["B", 48, 40]
    assert contract["compatibility_profile"]["flow"]["base_width"] == 256
    assert contract["compatibility_profile"]["flow"]["sampler"]["paper_steps"] == 10
    assert contract["compatibility_profile"]["vae"]["loss_weights"] == {"l1": 0.1, "l2": 0.5}
    assert contract["compatibility_profile"]["vae"]["augmentation"] == {"jitter": 0.008, "scale": 0.04, "time_mask": 0.05}
    assert contract["compatibility_profile"]["flow"]["sampler"]["website_euler_steps"] == 100
    assert contract["compatibility_profile"]["flow"]["sampler"]["website_native_samples"] == 500
    assert contract["hash_hygiene"]["recomputed_sha256"] == "9F1210C6034695061E83A648F2941CA9F6E0E8A057F547FE320D3F24D967F3EE"
    assert contract["hash_hygiene"]["recomputed_sha256"] != contract["hash_hygiene"]["copied_report_sha256"]
    assert "transcription" in contract["hash_hygiene"]["discrepancy"]

def test_three_channel_lineage_is_blocked_without_silent_projection():
    lineage = load(REPOSITORY_ROOT / "contracts" / "three_channel_lineage.json")
    assert lineage["outcome"] == "PUBLIC_RECONSTRUCTION_REQUIRED"
    assert lineage["exact_paper_reproduction"] is False
    assert lineage["observed_parser"]["selected_columns"] == [80, 81, 82, 83, 84, 85]
    assert lineage["intended_three_channel_route"]["selected_columns"] == [80, 81, 82]
    assert lineage["intended_three_channel_route"]["inference_time_drop_allowed"] is False
    assert lineage["channel_contract"]["six_channel_input_shape"] == ["B", 6, 160]
    assert lineage["channel_contract"]["three_channel_input_shape"] == ["B", 3, 160]
    assert lineage["release_decision"]["requires_public_reconstruction"] is True

def test_parity_contract_captures_safe_fixture_boundary():
    parity = load(REPOSITORY_ROOT / "contracts" / "parity_fixture_contract.json")
    assert parity["license_safe"] is True
    assert parity["participant_derived_data"] is False
    assert parity["checkpoint_payloads"] is False
    assert parity["raw_recipe"]["columns"] == 120
    assert parity["raw_recipe"]["selected_columns"] == [80,81,82,83,84,85]
    assert parity["expected_safe_split"]["subject_window_counts"] == {"01":7, "02":8, "03":8, "05":8}
    assert parity["expected_safe_split"]["train"] == 16
    assert parity["expected_safe_split"]["validation"] == 7
    assert parity["expected_safe_split"]["held_out_test"] == 8
    assert parity["split_separation"] == {"vae_subject_validation_fraction": 0.15, "cnn_window_validation_fraction": 0.20}
    assert parity["sampling"]["paper_steps"] == 10
    assert parity["sampling"]["website_euler_steps"] == 100
    assert parity["sampling"]["keep_protocols_separate"] is True

def test_reference_snapshots_have_reconciled_counts_and_holds():
    snapshots = load(REPOSITORY_ROOT / "contracts" / "reference_behavior_snapshots.json")
    assert snapshots["snapshot_summary"] == {"total": 19, "pass": 14, "hold": 5}
    entries = snapshots["snapshots"]
    assert len(entries) == 19
    assert sum(entry["status"] == "PASS" for entry in entries) == 14
    assert sum(entry["status"] == "HOLD" for entry in entries) == 5
    required_holds = {
        "preprocessing.proposed_3ch_projection_source_lineage",
        "vae.safe_split_eight_item_contract_and_legacy_alias_gate",
        "holds.vae_schedule_conflict",
        "holds.flow_base_width_conflict",
        "holds.fixture_report_website_source_hash_transcription",
    }
    assert set(snapshots["unresolved_holds"]) == required_holds
    website = next(entry for entry in entries if entry["snapshot_id"] == "holds.fixture_report_website_source_hash_transcription")
    assert website["source"]["sha256"] == "9F1210C6034695061E83A648F2941CA9F6E0E8A057F547FE320D3F24D967F3EE"
    assert website["source"]["reported_sha256"] != website["source"]["sha256"]
    assert snapshots["integrity_policy"]["participant_data"] is False
    assert snapshots["integrity_policy"]["checkpoint_payloads"] is False

def test_public_contract_text_is_sanitized_and_json_is_stable():
    paths = [
        REPOSITORY_ROOT / "contracts" / "configuration_decision_matrix.json",
        REPOSITORY_ROOT / "contracts" / "three_channel_lineage.json",
        REPOSITORY_ROOT / "contracts" / "parity_fixture_contract.json",
        REPOSITORY_ROOT / "contracts" / "reference_behavior_snapshots.json",
    ]
    markers = ("C" + ":/", "D" + ":/", "C" + ":" + "\\", "D" + ":" + "\\", "A" + "minR")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in markers), path
        assert "participant data" not in text.lower()
        assert json.loads(text)

def test_immutable_source_paths_use_known_allowlist_without_machine_paths():
    known = {
        "manuscript/Manuscript_02_06_2026.tex",
        "VAE/Run_VAE_Pretraings.ps1",
        "VAE/VAE_logic.py",
        "models/unet_1d.py",
        "LRF/rectified_flow.py",
        "1_train_flow.ps1",
        "TSTR.py",
        "8_tstr_classification_figure_org_PATCHED_v3.py",
        "8_export_website_trajectories.py",
        "sanitized investigation evidence",
    }
    configuration = load(REPOSITORY_ROOT / "contracts" / "configuration_decision_matrix.json")
    lineage = load(REPOSITORY_ROOT / "contracts" / "three_channel_lineage.json")
    snapshots = load(REPOSITORY_ROOT / "contracts" / "reference_behavior_snapshots.json")
    declared = [
        ref["path"]
        for item in configuration["items"]
        for ref in item["evidence_refs"]
        if ref.get("authority") == "immutable-source"
    ]
    declared.append(configuration["hash_hygiene"]["website_exporter_source_relative_path"])
    declared.extend(
        [
            lineage["observed_parser"]["source_relative_path"],
            lineage["intended_three_channel_route"]["source_relative_path"],
        ]
    )
    declared.extend(
        snapshot["source"]["path"]
        for snapshot in snapshots["snapshots"]
        if snapshot["source"].get("path") != "sanitized investigation evidence"
    )
    assert declared
    assert set(declared) <= known
    assert not any(path.startswith("scripts/") for path in declared)
    assert "scripts/" not in json.dumps(configuration)
    assert "scripts/" not in json.dumps(lineage)


def test_evaluation_label_order_concords_across_contract_and_fixture():
    evaluation_contract = load(REPOSITORY_ROOT / "contracts" / "evaluation_contract.json")
    fixture = load(
        REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic" / "evaluation_reference.json"
    )
    expected_labels = [0, 1, 2, 3]
    expected_mapping = {"1": 0, "3": 1, "4": 2, "33": 3}
    assert evaluation_contract["contract"]["metrics"]["primary_labels"] == expected_labels
    assert evaluation_contract["contract"]["dataset"]["labels"] == {
        "0": "walking",
        "1": "running",
        "2": "jump_up",
        "3": "cycling",
    }
    assert fixture["labels"] == expected_labels
    assert fixture["raw_code_to_encoded"] == expected_mapping
