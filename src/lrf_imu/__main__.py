"""Run ``python -m lrf_imu``."""

from .cli import main


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(main())
