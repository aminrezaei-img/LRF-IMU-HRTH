"""Portable filesystem paths used by the release configuration.

The release does not assume a particular workstation layout.  Relative roots
are resolved against an explicit ``base_dir`` (or the caller's current working
directory when no base is supplied), while absolute roots are preserved.
No directories are created by this module.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Union


PathLike = Union[str, Path]


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")
DEFAULT_RESULTS_ROOT = Path("results")


def _base_path(base_dir: Optional[PathLike]) -> Path:
    """Return an absolute base path without requiring it to exist."""

    if base_dir is None:
        return Path.cwd()
    return Path(base_dir).expanduser().resolve()


def _resolve_root(value: PathLike, base_dir: Path) -> Path:
    """Resolve one configured root against ``base_dir`` when it is relative."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """The four roots shared by data preparation and experiment outputs."""

    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    results_root: Path = DEFAULT_RESULTS_ROOT

    def resolved(self, base_dir: Optional[PathLike] = None) -> "ProjectPaths":
        """Return roots with relative values resolved against ``base_dir``."""

        base = _base_path(base_dir)
        return ProjectPaths(
            data_root=_resolve_root(self.data_root, base),
            output_root=_resolve_root(self.output_root, base),
            checkpoint_root=_resolve_root(self.checkpoint_root, base),
            results_root=_resolve_root(self.results_root, base),
        )

    def with_overrides(self, **roots: Optional[PathLike]) -> "ProjectPaths":
        """Return a copy with any explicitly supplied roots replaced."""

        valid = {
            "data_root",
            "output_root",
            "checkpoint_root",
            "results_root",
        }
        unknown = set(roots).difference(valid)
        if unknown:
            raise ValueError("Unknown path root(s): {}".format(", ".join(sorted(unknown))))

        values = {
            key: value
            for key, value in roots.items()
            if value is not None
        }
        return replace(self, **values)

    def data(self, *parts: PathLike) -> Path:
        return self.data_root.joinpath(*parts)

    def output(self, *parts: PathLike) -> Path:
        return self.output_root.joinpath(*parts)

    def checkpoint(self, *parts: PathLike) -> Path:
        return self.checkpoint_root.joinpath(*parts)

    def result(self, *parts: PathLike) -> Path:
        return self.results_root.joinpath(*parts)

    def fold(self, subject: int) -> "FoldPaths":
        """Return conventional subject-scoped paths without creating them."""

        return FoldPaths(subject=int(subject), roots=self)

    def as_dict(self) -> Dict[str, str]:
        return {
            "data_root": str(self.data_root),
            "output_root": str(self.output_root),
            "checkpoint_root": str(self.checkpoint_root),
            "results_root": str(self.results_root),
        }


@dataclass(frozen=True)
class FoldPaths:
    """Subject-scoped view over :class:`ProjectPaths`."""

    subject: int
    roots: ProjectPaths

    @property
    def subject_name(self) -> str:
        return "subject_{:02d}".format(self.subject)

    @property
    def data_root(self) -> Path:
        return self.roots.data(self.subject_name)

    @property
    def output_root(self) -> Path:
        return self.roots.output(self.subject_name)

    @property
    def checkpoint_root(self) -> Path:
        return self.roots.checkpoint(self.subject_name)

    @property
    def results_root(self) -> Path:
        return self.roots.result(self.subject_name)

    def data(self, *parts: PathLike) -> Path:
        return self.data_root.joinpath(*parts)

    def output(self, *parts: PathLike) -> Path:
        return self.output_root.joinpath(*parts)

    def checkpoint(self, *parts: PathLike) -> Path:
        return self.checkpoint_root.joinpath(*parts)

    def result(self, *parts: PathLike) -> Path:
        return self.results_root.joinpath(*parts)


def paths_from_mapping(
    mapping: Dict[str, object],
    base_dir: Optional[PathLike] = None,
) -> ProjectPaths:
    """Build resolved paths from a config mapping.

    The mapping deliberately uses the four public top-level names so that a
    caller can replace roots with a small CLI-style override dictionary.
    """

    paths = ProjectPaths(
        data_root=Path(mapping.get("data_root", DEFAULT_DATA_ROOT)),
        output_root=Path(mapping.get("output_root", DEFAULT_OUTPUT_ROOT)),
        checkpoint_root=Path(mapping.get("checkpoint_root", DEFAULT_CHECKPOINT_ROOT)),
        results_root=Path(mapping.get("results_root", DEFAULT_RESULTS_ROOT)),
    )
    return paths.resolved(base_dir)


def fold_paths(paths: ProjectPaths, subject: int) -> FoldPaths:
    """Convenience function for callers that already have resolved roots."""

    return paths.fold(subject)
