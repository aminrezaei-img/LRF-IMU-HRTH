"""Focused contract tests for the paper-specific activity and sensor schema."""

from pathlib import Path
import json
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lrf_imu.data.activities import (  # noqa: E402
    ACTIVITY_CODE_TO_LABEL,
    ACTIVITY_LABEL_TO_NAME,
    ActivityMappingError,
    ENCODED_LABELS,
    PAPER_ACTIVITY_CODES,
    PAPER_ACTIVITY_NAMES,
    encode_activity_codes,
    validate_activity_mapping,
)
from lrf_imu.data.schema import (  # noqa: E402
    LABEL_COLUMN_INDEX,
    RAW_COLUMN_COUNT,
    RIGHT_THIGH_SIX_CHANNEL_INDICES,
    RIGHT_THIGH_THREE_CHANNEL_INDICES,
    SIX_CHANNEL_NAMES,
    THREE_CHANNEL_NAMES,
    SchemaValidationError,
    SensorMode,
    get_sensor_schema,
    validate_channel_mapping,
    validate_raw_column_count,
)


def test_paper_activity_mapping_keeps_raw_and_encoded_namespaces_explicit() -> None:
    assert PAPER_ACTIVITY_CODES == (1, 3, 4, 33)
    assert ENCODED_LABELS == (0, 1, 2, 3)
    assert PAPER_ACTIVITY_NAMES == ("walking", "running", "jump_up", "cycling")
    assert dict(ACTIVITY_CODE_TO_LABEL) == {1: 0, 3: 1, 4: 2, 33: 3}
    assert dict(ACTIVITY_LABEL_TO_NAME) == {
        0: "walking",
        1: "running",
        2: "jump_up",
        3: "cycling",
    }
    assert encode_activity_codes((1, 33, 3, 4)) == (0, 3, 1, 2)
    assert 33 not in ENCODED_LABELS


def test_mapping_validation_is_strict_by_default() -> None:
    assert dict(validate_activity_mapping({1: 0, 3: 1, 4: 2, 33: 3})) == dict(
        ACTIVITY_CODE_TO_LABEL
    )

    with pytest.raises(ActivityMappingError, match="missing raw code"):
        validate_activity_mapping({1: 0, 3: 1, 4: 2})
    with pytest.raises(ActivityMappingError, match="unsupported raw activity code"):
        validate_activity_mapping({1: 0, 3: 1, 4: 2, 33: 3, 2: 4})
    with pytest.raises(ActivityMappingError, match="unique"):
        validate_activity_mapping({1: 0, 3: 0, 4: 2, 33: 3})
    with pytest.raises(ActivityMappingError, match="canonical"):
        validate_activity_mapping({1: 1, 3: 0, 4: 2, 33: 3})


def test_nonstrict_mapping_is_still_limited_to_the_four_class_vocabulary() -> None:
    assert dict(validate_activity_mapping({1: 0}, strict=False)) == {1: 0}
    with pytest.raises(ActivityMappingError, match="unsupported raw activity code"):
        validate_activity_mapping({2: 0}, strict=False)


def test_sensor_schema_matches_raw_layout_and_channel_order() -> None:
    six = get_sensor_schema("six_channel")
    assert six.mode is SensorMode.SIX_CHANNEL
    assert six.channel_indices == (80, 81, 82, 83, 84, 85)
    assert six.channel_indices == RIGHT_THIGH_SIX_CHANNEL_INDICES
    assert six.channel_names == ("ax", "ay", "az", "gx", "gy", "gz")
    assert six.channel_names == SIX_CHANNEL_NAMES
    assert six.channel_count == 6
    assert six.raw_column_count == RAW_COLUMN_COUNT == 120
    assert six.label_column_index == LABEL_COLUMN_INDEX == 119
    assert six.training_mode == "separate_model"
    assert six.inference_policy == "declared_channel_set_only"
    assert six.allow_inference_time_channel_drop is False
    assert six.reconstructed_three_channel is False


def test_three_channel_schema_is_explicit_reconstruction_and_separate_training() -> None:
    three = get_sensor_schema("three_channel")
    assert three.mode is SensorMode.THREE_CHANNEL
    assert three.channel_indices == RIGHT_THIGH_THREE_CHANNEL_INDICES == (80, 81, 82)
    assert three.channel_names == THREE_CHANNEL_NAMES == ("ax", "ay", "az")
    assert three.channel_count == 3
    assert three.reconstructed_three_channel is True
    assert three.training_mode == "separate_model"
    assert three.allow_inference_time_channel_drop is False
    assert three.inference_policy == "declared_channel_set_only"


@pytest.mark.parametrize("mode", ["", "four_channel", "six", "all_activities", 6, None])
def test_invalid_sensor_modes_are_rejected(mode) -> None:
    with pytest.raises(SchemaValidationError):
        get_sensor_schema(mode)


def test_documented_accelerometer_mode_alias_requires_non_strict_parsing_for_legacy_aliases() -> None:
    assert get_sensor_schema("accelerometer_only").is_three_channel
    assert get_sensor_schema("3ch", strict=False).is_three_channel
    with pytest.raises(SchemaValidationError):
        get_sensor_schema("3ch")


def test_channel_mapping_requires_exact_indices_and_order() -> None:
    assert validate_channel_mapping(
        SensorMode.SIX_CHANNEL,
        (80, 81, 82, 83, 84, 85),
        ("ax", "ay", "az", "gx", "gy", "gz"),
    ) == (RIGHT_THIGH_SIX_CHANNEL_INDICES, SIX_CHANNEL_NAMES)
    with pytest.raises(SchemaValidationError, match="requires channel indices"):
        validate_channel_mapping("six_channel", (80, 81, 82, 84, 83, 85))
    with pytest.raises(SchemaValidationError, match="requires channel names"):
        validate_channel_mapping(
            "three_channel", (80, 81, 82), ("gx", "gy", "gz")
        )
    with pytest.raises(SchemaValidationError, match="duplicates"):
        validate_channel_mapping("three_channel", (80, 80, 82))


def test_raw_column_validation_rejects_malformed_layouts() -> None:
    assert validate_raw_column_count(120) == 120
    with pytest.raises(SchemaValidationError, match="expected 120"):
        validate_raw_column_count(119)
    with pytest.raises(SchemaValidationError, match="integer"):
        validate_raw_column_count("120")


def test_channel_selection_fixture_agrees_with_public_schema() -> None:
    fixture = json.loads(
        (
            REPOSITORY_ROOT
            / "tests"
            / "fixtures"
            / "synthetic"
            / "channel_selection.json"
        ).read_text(encoding="utf-8")
    )
    assert tuple(fixture["observed_six_channel_columns"]) == RIGHT_THIGH_SIX_CHANNEL_INDICES
    assert tuple(fixture["intended_three_channel_columns"]) == RIGHT_THIGH_THREE_CHANNEL_INDICES
    assert tuple(fixture["observed_six_channel_names"]) == (
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    )
    assert tuple(fixture["intended_three_channel_names"]) == (
        "acc_x",
        "acc_y",
        "acc_z",
    )
    assert fixture["allow_inference_time_channel_drop"] is False
    assert fixture["three_channel_is_separate_schema"] is True
