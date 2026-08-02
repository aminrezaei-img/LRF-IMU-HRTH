"""The paper-specific activity vocabulary used by the public data layer.

REALDISP contains more activities than the four-class path characterized for
this release.  The constants in this module intentionally describe only that
four-class path.  A raw activity code is the value read from column 119 of a
log; an encoded label is the contiguous value consumed by downstream models.
They are different namespaces and must not be conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple


class ActivityMappingError(ValueError):
    """Raised when a raw-code to encoded-label mapping is not valid."""


@dataclass(frozen=True)
class ActivitySpec:
    """One activity in both the raw and encoded label namespaces."""

    raw_code: int
    encoded_label: int
    name: str

    def __post_init__(self) -> None:
        if isinstance(self.raw_code, bool) or not isinstance(self.raw_code, Integral):
            raise TypeError("raw_code must be an integer")
        if isinstance(self.encoded_label, bool) or not isinstance(
            self.encoded_label, Integral
        ):
            raise TypeError("encoded_label must be an integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("name must be a non-empty string")
        object.__setattr__(self, "raw_code", int(self.raw_code))
        object.__setattr__(self, "encoded_label", int(self.encoded_label))
        object.__setattr__(self, "name", self.name.strip())


# This is deliberately a small paper-specific vocabulary, not a declaration
# that all REALDISP activities are supported by the public release.
PAPER_ACTIVITY_SPECS: Tuple[ActivitySpec, ...] = (
    ActivitySpec(raw_code=1, encoded_label=0, name="walking"),
    ActivitySpec(raw_code=3, encoded_label=1, name="running"),
    ActivitySpec(raw_code=4, encoded_label=2, name="jump_up"),
    ActivitySpec(raw_code=33, encoded_label=3, name="cycling"),
)

PAPER_ACTIVITY_CODES: Tuple[int, ...] = tuple(
    spec.raw_code for spec in PAPER_ACTIVITY_SPECS
)
RAW_ACTIVITY_CODES = PAPER_ACTIVITY_CODES

ENCODED_ACTIVITY_LABELS: Tuple[int, ...] = tuple(
    spec.encoded_label for spec in PAPER_ACTIVITY_SPECS
)
ENCODED_LABELS = ENCODED_ACTIVITY_LABELS

PAPER_ACTIVITY_NAMES: Tuple[str, ...] = tuple(
    spec.name for spec in PAPER_ACTIVITY_SPECS
)
ACTIVITY_NAMES = PAPER_ACTIVITY_NAMES

_RAW_TO_ENCODED = {
    spec.raw_code: spec.encoded_label for spec in PAPER_ACTIVITY_SPECS
}
_ENCODED_TO_RAW = {
    spec.encoded_label: spec.raw_code for spec in PAPER_ACTIVITY_SPECS
}
_ENCODED_TO_NAME = {spec.encoded_label: spec.name for spec in PAPER_ACTIVITY_SPECS}
_RAW_TO_NAME = {spec.raw_code: spec.name for spec in PAPER_ACTIVITY_SPECS}

# Mapping proxies keep the vocabulary immutable while preserving normal
# Mapping semantics for callers and serializers.
RAW_CODE_TO_ENCODED_LABEL: Mapping[int, int] = MappingProxyType(_RAW_TO_ENCODED)
ENCODED_LABEL_TO_RAW_CODE: Mapping[int, int] = MappingProxyType(_ENCODED_TO_RAW)
ENCODED_LABEL_TO_NAME: Mapping[int, str] = MappingProxyType(_ENCODED_TO_NAME)
RAW_CODE_TO_NAME: Mapping[int, str] = MappingProxyType(_RAW_TO_NAME)

# Short aliases used by data-preparation callers.  The longer names above are
# preferred because they make the raw-vs-encoded distinction explicit.
ACTIVITY_CODE_TO_LABEL = RAW_CODE_TO_ENCODED_LABEL
ACTIVITY_LABEL_TO_CODE = ENCODED_LABEL_TO_RAW_CODE
ACTIVITY_LABEL_TO_NAME = ENCODED_LABEL_TO_NAME


def _normalise_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ActivityMappingError("{} must contain integers".format(field_name))
    return int(value)


def validate_activity_mapping(
    mapping: Optional[Mapping[Any, Any]] = None, *, strict: bool = True
) -> Mapping[int, int]:
    """Validate and freeze a raw-code to encoded-label mapping.

    In strict mode the mapping must be exactly the characterized mapping for
    raw codes ``1, 3, 4, 33`` and encoded labels ``0..3``.  Non-strict mode is
    useful for a deliberately partial vocabulary, but it still rejects
    unknown REALDISP codes, duplicate encoded labels, and labels outside the
    four encoded labels.  Neither mode expands support to the other REALDISP
    activities.
    """

    if mapping is None:
        candidate: Mapping[Any, Any] = RAW_CODE_TO_ENCODED_LABEL
    elif not isinstance(mapping, Mapping):
        raise ActivityMappingError("activity mapping must be a mapping")
    else:
        candidate = mapping

    normalised = {
        _normalise_integer(raw_code, "raw activity codes"): _normalise_integer(
            encoded_label, "encoded activity labels"
        )
        for raw_code, encoded_label in candidate.items()
    }

    expected_codes = set(PAPER_ACTIVITY_CODES)
    unknown_codes = set(normalised).difference(expected_codes)
    if unknown_codes:
        raise ActivityMappingError(
            "unsupported raw activity code(s): {}".format(
                sorted(unknown_codes)
            )
        )
    if not normalised:
        raise ActivityMappingError("activity mapping must not be empty")

    labels = tuple(normalised.values())
    if len(set(labels)) != len(labels):
        raise ActivityMappingError("encoded activity labels must be unique")
    unknown_labels = set(labels).difference(ENCODED_ACTIVITY_LABELS)
    if unknown_labels:
        raise ActivityMappingError(
            "unsupported encoded activity label(s): {}".format(
                sorted(unknown_labels)
            )
        )

    if strict:
        missing_codes = expected_codes.difference(normalised)
        if missing_codes:
            raise ActivityMappingError(
                "strict activity mapping is missing raw code(s): {}".format(
                    sorted(missing_codes)
                )
            )
        if normalised != dict(RAW_CODE_TO_ENCODED_LABEL):
            raise ActivityMappingError(
                "strict activity mapping must equal the canonical raw-to-encoded mapping"
            )
        if set(labels) != set(ENCODED_ACTIVITY_LABELS):
            raise ActivityMappingError(
                "strict activity mapping must use encoded labels 0, 1, 2, and 3"
            )

    return MappingProxyType(normalised)


def raw_code_to_encoded_label(
    raw_code: Any,
    mapping: Optional[Mapping[Any, Any]] = None,
    *,
    strict: bool = True,
) -> int:
    """Translate one raw log code to its encoded model label."""

    code = _normalise_integer(raw_code, "raw activity code")
    validated = validate_activity_mapping(mapping, strict=strict)
    try:
        return validated[code]
    except KeyError as exc:
        raise ActivityMappingError(
            "raw activity code {} is not present in the selected mapping".format(code)
        ) from exc


def encoded_label_to_raw_code(encoded_label: Any) -> int:
    """Translate one encoded model label back to its raw log code."""

    label = _normalise_integer(encoded_label, "encoded activity label")
    try:
        return ENCODED_LABEL_TO_RAW_CODE[label]
    except KeyError as exc:
        raise ActivityMappingError(
            "encoded activity label {} is not one of 0, 1, 2, or 3".format(label)
        ) from exc


def activity_name_for_raw_code(raw_code: Any) -> str:
    """Return the paper activity name for a raw log code."""

    code = _normalise_integer(raw_code, "raw activity code")
    try:
        return RAW_CODE_TO_NAME[code]
    except KeyError as exc:
        raise ActivityMappingError(
            "raw activity code {} is outside the four-class public vocabulary".format(
                code
            )
        ) from exc


def activity_name_for_encoded_label(encoded_label: Any) -> str:
    """Return the paper activity name for an encoded model label."""

    label = _normalise_integer(encoded_label, "encoded activity label")
    try:
        return ENCODED_LABEL_TO_NAME[label]
    except KeyError as exc:
        raise ActivityMappingError(
            "encoded activity label {} is not one of 0, 1, 2, or 3".format(label)
        ) from exc


def encode_activity_codes(
    raw_codes: Iterable[Any],
    mapping: Optional[Mapping[Any, Any]] = None,
    *,
    strict: bool = True,
) -> Tuple[int, ...]:
    """Encode an iterable of raw log codes without changing its order."""

    if isinstance(raw_codes, (str, bytes)):
        raise TypeError("raw_codes must be an iterable of integer codes")
    validated = validate_activity_mapping(mapping, strict=strict)
    encoded = []
    for raw_code in raw_codes:
        code = _normalise_integer(raw_code, "raw activity code")
        try:
            encoded.append(validated[code])
        except KeyError as exc:
            raise ActivityMappingError(
                "raw activity code {} is not present in the selected mapping".format(
                    code
                )
            ) from exc
    return tuple(encoded)


def activity_spec_for_raw_code(raw_code: Any) -> ActivitySpec:
    """Return the immutable specification for one raw activity code."""

    code = _normalise_integer(raw_code, "raw activity code")
    for spec in PAPER_ACTIVITY_SPECS:
        if spec.raw_code == code:
            return spec
    raise ActivityMappingError(
        "raw activity code {} is outside the four-class public vocabulary".format(code)
    )


__all__ = [
    "ACTIVITY_CODE_TO_LABEL",
    "ACTIVITY_LABEL_TO_CODE",
    "ACTIVITY_LABEL_TO_NAME",
    "ACTIVITY_NAMES",
    "ActivityMappingError",
    "ActivitySpec",
    "ENCODED_ACTIVITY_LABELS",
    "ENCODED_LABELS",
    "ENCODED_LABEL_TO_NAME",
    "ENCODED_LABEL_TO_RAW_CODE",
    "PAPER_ACTIVITY_CODES",
    "PAPER_ACTIVITY_NAMES",
    "PAPER_ACTIVITY_SPECS",
    "RAW_ACTIVITY_CODES",
    "RAW_CODE_TO_ENCODED_LABEL",
    "RAW_CODE_TO_NAME",
    "activity_name_for_encoded_label",
    "activity_name_for_raw_code",
    "activity_spec_for_raw_code",
    "encode_activity_codes",
    "encoded_label_to_raw_code",
    "raw_code_to_encoded_label",
    "validate_activity_mapping",
]
