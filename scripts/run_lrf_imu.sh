#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the canonical command: python -m lrf_imu generate-harth.

usage() {
  cat <<'EOF'
Usage: run_lrf_imu.sh --vae-checkpoint PATH --flow-checkpoint PATH --class NAME [options]

Generate one metadata-only HARTH window with the existing VAE and Flow.

Options:
  --class NAME                 Canonical HARTH class name or ID
  --seed N                     Random seed (default: 42)
  --device cpu|cuda            Execution device (default: cpu)
  --python PATH                Python launcher (default: $PYTHON or python)
  --output PATH                Capture canonical JSON output at PATH
  --expected-vae-sha256 HASH   Verify VAE SHA-256 before execution
  --expected-flow-sha256 HASH  Verify Flow SHA-256 before execution
  --dry-run                    Validate inputs and print the canonical command
  -h, --help                   Show this help
EOF
}

python_cmd="${PYTHON:-python}"
vae_checkpoint=""
flow_checkpoint=""
activity=""
seed=42
device="cpu"
output=""
expected_vae=""
expected_flow=""
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --python) python_cmd="$2"; shift 2 ;;
    --vae-checkpoint) vae_checkpoint="$2"; shift 2 ;;
    --flow-checkpoint) flow_checkpoint="$2"; shift 2 ;;
    --class|--activity) activity="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --expected-vae-sha256) expected_vae="$2"; shift 2 ;;
    --expected-flow-sha256) expected_flow="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$vae_checkpoint" || -z "$flow_checkpoint" || -z "$activity" ]]; then
  echo "--vae-checkpoint, --flow-checkpoint, and --class are required." >&2
  usage >&2
  exit 2
fi
[[ -f "$vae_checkpoint" ]] || { echo "VAE checkpoint not found: $vae_checkpoint" >&2; exit 2; }
[[ -f "$flow_checkpoint" ]] || { echo "Flow checkpoint not found: $flow_checkpoint" >&2; exit 2; }

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

sha256() {
  "$python_cmd" -c 'import hashlib, sys; h=hashlib.sha256(); f=open(sys.argv[1], "rb"); [h.update(b) for b in iter(lambda: f.read(1048576), b"")]; f.close(); print(h.hexdigest().upper())' "$1"
}

verify_hash() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256 "$path")"
  [[ "$actual" == "${expected^^}" ]] || {
    echo "SHA-256 mismatch for $path: expected ${expected^^}, got $actual" >&2
    exit 3
  }
}

[[ -z "$expected_vae" ]] || verify_hash "$vae_checkpoint" "$expected_vae"
[[ -z "$expected_flow" ]] || verify_hash "$flow_checkpoint" "$expected_flow"

"$python_cmd" --version >/dev/null
if [[ "$dry_run" == true ]]; then
  echo "dry_run=true"
  printf 'canonical=python -m lrf_imu generate-harth --flow-checkpoint %q --vae-checkpoint %q --activity %q --seed %q --device %q\n' \
    "$flow_checkpoint" "$vae_checkpoint" "$activity" "$seed" "$device"
  exit 0
fi

if [[ "$device" == "cuda" ]]; then
  "$python_cmd" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in the selected Python environment"'
fi

args=( -m lrf_imu generate-harth --flow-checkpoint "$flow_checkpoint" --vae-checkpoint "$vae_checkpoint" --activity "$activity" --seed "$seed" --device "$device" )
if [[ -n "$output" ]]; then
  mkdir -p "$(dirname -- "$output")"
  "$python_cmd" "${args[@]}" > "$output"
  cat "$output"
else
  "$python_cmd" "${args[@]}"
fi
