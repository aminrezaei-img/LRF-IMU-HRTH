#!/usr/bin/env bash
set -euo pipefail

# Bounded release validation; no training or data generation is performed.

usage() {
  cat <<'EOF'
Usage: validate_release.sh [--python PATH]

Run tests, compilation, diff hygiene, targeted Ruff, and public CLI help checks.
EOF
}

python_cmd="${PYTHON:-python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --python) python_cmd="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_cmd" -m compileall -q src
git diff --check
"$python_cmd" -m ruff check \
  src/lrf_imu/integration/dayforge.py \
  src/lrf_imu/integration/physical_state.py \
  src/lrf_imu/integration/dayforge_audit.py \
  src/lrf_imu/integration/__init__.py \
  src/lrf_imu/cli.py \
  tests/test_dayforge_handoff.py \
  tests/test_dayforge_fusion.py
"$python_cmd" -m pytest -p no:cacheprovider -q

for command in \
  prepare-harth-data train-harth-vae train-harth-flow generate-harth \
  evaluate-harth-vae evaluate-harth-flow map-dayforge-physical-states \
  synthesize-dayforge; do
  "$python_cmd" -m lrf_imu "$command" --help >/dev/null
done

echo "release validation passed"
