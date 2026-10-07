"""Parse repository Python sources without creating bytecode artifacts."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "tests").rglob("*.py")
    )
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"syntax_check=PASS files={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
