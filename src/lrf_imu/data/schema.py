"""Validated REALDISP raw-column and sensor-channel schemas.

The release exposes two explicit input schemas.  The six-channel schema is
the observed right-thigh IMU path.  The three-channel schema is a public
reconstruction using the accelerometer columns only; it is a separate
downstream training schema and is never an inference-time projection of a
six-channel input or checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Any, Optional, Sequence, Tuple, Union


class SchemaValidationError(ValueError):
    """Raised when a raw layout, sensor mode, or channel mapping is invalid."""


class SensorMode(str, Enum):
    """Explicit public sensor modes."""

    SIX_CHANNEL = "six_channel"
    THREE_CHANNEL = "three_channel"


RAW_COLUMN_COUNT = 120
REQUIRED_RAW_COLUMNS = RAW_COLUMN_COUNT
LABEL_COLUMN_INDEX = 119

RIGHT_THIGH_SIX_CHANNEL_INDICES: Tuple[int, ...] = (80, 81, 82, 83, 84, 85)
RIGHT_THIGH_THREE_CHANNEL_INDICES: Tuple[int, ...] = (80, 81, 82)

# Names and order follow the public compatibility configuration and the
# scientific contract: accelerometers first, then gyroscopes.
SIX_CHANNEL_NAMES: Tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")
THREE_CHANNEL_NAMES: Tuple[str, ...] = ("ax", "ay", "az")

# Descriptive aliases for callers that prefer the sensor placement in the
# constant name.  They reference the same immutable tuples.
RIGHT_THIGH_6CH_INDICES = RIGHT_THIGH_SIX_CHANNEL_INDICES
RIGHT_THIGH_3CH_INDICES = RIGHT_THIGH_THREE_CHANNEL_INDICES
CHANNEL_NAMES_6CH = SIX_CHANNEL_NAMES
CHANNEL_NAMES_3CH = THREE_CHANNEL_NAMES
ACCELEROMETER_ONLY_INDICES = RIGHT_THIGH_THREE_CHANNEL_INDICES

_SCHEMA_INDICES = {
    SensorMode.SIX_CHANNEL: RIGHT_THIGH_SIX_CHANNEL_INDICES,
    SensorMode.THREE_CHANNEL: RIGHT_THIGH_THREE_CHANNEL_INDICES,
}
_SCHEMA_NAMES = {
    SensorMode.SIX_CHANNEL: SIX_CHANNEL_NAMES,
    SensorMode.THREE_CHANNEL: THREE_CHANNEL_NAMES,
}


def _normalise_indices(indices: Sequence[Any], field_name: str) -> Tuple[int, ...]:
    if isinstance(indices, (str, bytes)):
        raise SchemaValidationError("{} must be a sequence of integer indices".format(field_name))
    try:
        result = tuple(indices)
    except TypeError as exc:
        raise SchemaValidationError(
            "{} must be a sequence of integer indices".format(field_name)
        ) from exc
    normalised = []
    for index in result:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise SchemaValidationError(
                "{} must contain only integer indices".format(field_name)
            )
        normalised.append(int(index))
    if len(set(normalised)) != len(normalised):
        raise SchemaValidationError("{} must not contain duplicates".format(field_name))
    return tuple(normalised)


def _normalise_names(names: Sequence[Any], field_name: str) -> Tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise SchemaValidationError("{} must be a sequence of channel names".format(field_name))
    try:
        result = tuple(names)
    except TypeError as exc:
        raise SchemaValidationError(
            "{} must be a sequence of channel names".format(field_name)
        ) from exc
    normalised = []
    for name in result:
        if not isinstance(name, str) or not name.strip():
            raise SchemaValidationError(
                "{} must contain non-empty string names".format(field_name)
            )
        normalised.append(name.strip())
    if len(set(normalised)) != len(normalised):
        raise SchemaValidationError("{} must not contain duplicates".format(field_name))
    return tuple(normalised)


def validate_raw_column_count(column_count: Any) -> int:
    """Require the exact 120-column raw REALDISP layout."""

    if isinstance(column_count, bool) or not isinstance(column_count, Integral):
        raise SchemaValidationError("raw column count must be an integer")
    count = int(column_count)
    if count != RAW_COLUMN_COUNT:
        raise SchemaValidationError(
            "expected {} raw columns, got {}".format(RAW_COLUMN_COUNT, count)
        )
    return count


def normalize_sensor_mode(
    mode: Union[SensorMode, str], *, strict: bool = True
) -> SensorMode:
    """Normalize an explicit sensor mode and reject unsupported variants.

    Strict mode accepts the two canonical values and the documented
    ``accelerometer_only`` configuration spelling.  Relaxed mode additionally
    accepts short, unambiguous aliases for callers migrating old configuration
    files.  It never accepts an arbitrary channel count or an inference-time
    drop mode.
    """

    if isinstance(mode, SensorMode):
        return mode
    if not isinstance(mode, str):
        raise SchemaValidationError(
            "sensor mode must be 'six_channel' or 'three_channel'"
        )

    value = mode.strip().lower() if not strict else mode
    canonical = {item.value: item for item in SensorMode}
    if value in canonical:
        return canonical[value]
    if value == "accelerometer_only":
        return SensorMode.THREE_CHANNEL
    if not strict:
        aliases = {
            "3ch": SensorMode.THREE_CHANNEL,
            "3_channel": SensorMode.THREE_CHANNEL,
            "6ch": SensorMode.SIX_CHANNEL,
            "full_imu": SensorMode.SIX_CHANNEL,
        }
        if value in aliases:
            return aliases[value]
    accepted = "six_channel, three_channel"
    raise SchemaValidationError(
        "unsupported sensor mode {!r}; expected {}".format(mode, accepted)
    )


# Common spelling for callers that use validation terminology.
validate_sensor_mode = normalize_sensor_mode


def validate_channel_mapping(
    mode: Union[SensorMode, str],
    channel_indices: Sequence[Any],
    channel_names: Optional[Sequence[Any]] = None,
    *,
    strict: bool = True,
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    """Validate the exact indices and ordering required by a sensor mode.

    ``strict`` applies to mode parsing; the public column contract itself is
    always exact.  This prevents a custom six-channel ordering from being
    mistaken for the characterized right-thigh schema.
    """

    sensor_mode = normalize_sensor_mode(mode, strict=strict)
    indices = _normalise_indices(channel_indices, "channel_indices")
    expected_indices = _SCHEMA_INDICES[sensor_mode]
    if indices != expected_indices:
        raise SchemaValidationError(
            "{} requires channel indices {}, got {}".format(
                sensor_mode.value, expected_indices, indices
            )
        )

    expected_names = _SCHEMA_NAMES[sensor_mode]
    if channel_names is None:
        names = expected_names
    else:
        names = _normalise_names(channel_names, "channel_names")
        if names != expected_names:
            raise SchemaValidationError(
                "{} requires channel names in order {}, got {}".format(
                    sensor_mode.value, expected_names, names
                )
            )
    if len(indices) != len(names):
        raise SchemaValidationError("channel indices and names must have equal length")
    if any(index < 0 or index >= RAW_COLUMN_COUNT for index in indices):
        raise SchemaValidationError(
            "channel indices must be within the {} raw columns".format(
                RAW_COLUMN_COUNT
            )
        )
    if LABEL_COLUMN_INDEX in indices:
        raise SchemaValidationError("label column must not be selected as a signal channel")
    return indices, names


@dataclass(frozen=True)
class SensorSchema:
    """Immutable raw-column schema for one separately trained input mode."""

    mode: SensorMode
    channel_indices: Tuple[int, ...]
    channel_names: Tuple[str, ...]
    raw_column_count: int = RAW_COLUMN_COUNT
    label_column_index: int = LABEL_COLUMN_INDEX
    training_mode: str = "separate_model"
    inference_policy: str = "declared_channel_set_only"
    allow_inference_time_channel_drop: bool = False
    reconstructed_three_channel: bool = False
    exact_paper_reproduction: bool = False

    def __post_init__(self) -> None:
        mode = normalize_sensor_mode(self.mode, strict=True)
        indices, names = validate_channel_mapping(
            mode, self.channel_indices, self.channel_names, strict=True
        )
        validate_raw_column_count(self.raw_column_count)
        if self.label_column_index != LABEL_COLUMN_INDEX:
            raise SchemaValidationError(
                "label column index must be {}".format(LABEL_COLUMN_INDEX)
            )
        if self.training_mode != "separate_model":
            raise SchemaValidationError(
                "sensor modes require training_mode='separate_model'"
            )
        if self.inference_policy != "declared_channel_set_only":
            raise SchemaValidationError(
                "inference_policy must be 'declared_channel_set_only'"
            )
        if self.allow_inference_time_channel_drop:
            raise SchemaValidationError(
                "inference-time channel dropping is not part of the public schema"
            )
        expected_reconstructed = mode is SensorMode.THREE_CHANNEL
        if self.reconstructed_three_channel != expected_reconstructed:
            raise SchemaValidationError(
                "three-channel mode must be marked as the explicit public reconstruction"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "channel_indices", indices)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "raw_column_count", RAW_COLUMN_COUNT)
        object.__setattr__(self, "label_column_index", LABEL_COLUMN_INDEX)

    @property
    def channel_count(self) -> int:
        return len(self.channel_indices)

    @property
    def is_three_channel(self) -> bool:
        return self.mode is SensorMode.THREE_CHANNEL

    @property
    def is_six_channel(self) -> bool:
        return self.mode is SensorMode.SIX_CHANNEL

    @classmethod
    def from_mode(
        cls, mode: Union[SensorMode, str], *, strict: bool = True
    ) -> "SensorSchema":
        sensor_mode = normalize_sensor_mode(mode, strict=strict)
        return _SCHEMA_BY_MODE[sensor_mode]

    # A readable alias for data-preparation callers.
    for_mode = from_mode

    def validate_raw_layout(self, column_count: Any) -> int:
        """Validate a file's raw column count against this schema."""

        return validate_raw_column_count(column_count)


_SCHEMA_BY_MODE = MappingProxyType(
    {
        SensorMode.SIX_CHANNEL: SensorSchema(
            mode=SensorMode.SIX_CHANNEL,
            channel_indices=RIGHT_THIGH_SIX_CHANNEL_INDICES,
            channel_names=SIX_CHANNEL_NAMES,
            reconstructed_three_channel=False,
        ),
        SensorMode.THREE_CHANNEL: SensorSchema(
            mode=SensorMode.THREE_CHANNEL,
            channel_indices=RIGHT_THIGH_THREE_CHANNEL_INDICES,
            channel_names=THREE_CHANNEL_NAMES,
            reconstructed_three_channel=True,
        ),
    }
)

SIX_CHANNEL_SCHEMA = _SCHEMA_BY_MODE[SensorMode.SIX_CHANNEL]
THREE_CHANNEL_SCHEMA = _SCHEMA_BY_MODE[SensorMode.THREE_CHANNEL]


def sensor_schema_for_mode(
    mode: Union[SensorMode, str], *, strict: bool = True
) -> SensorSchema:
    """Return the immutable schema for an explicit public sensor mode."""

    return SensorSchema.from_mode(mode, strict=strict)


# Another concise alias used in orchestration code.
get_sensor_schema = sensor_schema_for_mode


__all__ = [
    "ACCELEROMETER_ONLY_INDICES",
    "CHANNEL_NAMES_3CH",
    "CHANNEL_NAMES_6CH",
    "LABEL_COLUMN_INDEX",
    "RAW_COLUMN_COUNT",
    "REQUIRED_RAW_COLUMNS",
    "RIGHT_THIGH_3CH_INDICES",
    "RIGHT_THIGH_6CH_INDICES",
    "RIGHT_THIGH_SIX_CHANNEL_INDICES",
    "RIGHT_THIGH_THREE_CHANNEL_INDICES",
    "SIX_CHANNEL_NAMES",
    "SIX_CHANNEL_SCHEMA",
    "THREE_CHANNEL_NAMES",
    "THREE_CHANNEL_SCHEMA",
    "SchemaValidationError",
    "SensorMode",
    "SensorSchema",
    "get_sensor_schema",
    "normalize_sensor_mode",
    "sensor_schema_for_mode",
    "validate_channel_mapping",
    "validate_raw_column_count",
    "validate_sensor_mode",
]
