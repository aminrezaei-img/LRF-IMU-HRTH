#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the canonical commands: python -m lrf_imu map-dayforge-physical-states
# and, only when explicitly requested, python -m lrf_imu synthesize-dayforge.

usage() {
  cat <<'EOF'
Usage: run_paper3_dayforge.sh --dayforge-root PATH --mapping-output PATH [options]

Run the read-only Module B mapping. Add --run-synthesis for one explicitly
selected person-day through the existing Module C command.

Options:
  --dayforge-root PATH         Validated semantic DayForge root
  --derived-root PATH          Optional in-bed handoff root
  --mapping-config PATH        Mapping YAML (default: configs/paper/dayforge_harth_mapping.yaml)
  --mapping-output PATH        Mapping output directory
  --run-synthesis              Invoke Module C after mapping
  --vae-checkpoint PATH        Required with --run-synthesis
  --flow-checkpoint PATH       Required with --run-synthesis
  --normalization PATH         Required with --run-synthesis
  --output PATH                Module C output directory
  --persona ID                 One person for Module C
  --date YYYY-MM-DD            One day for Module C
  --seed N                     Seed (default: 42)
  --device cpu|cuda            Device (default: cpu)
  --python PATH                Python launcher (default: $PYTHON or python)
  -h, --help                   Show this help
EOF
}

python_cmd="${PYTHON:-python}"
dayforge_root=""
derived_root=""
mapping_config="configs/paper/dayforge_harth_mapping.yaml"
mapping_output=""
run_synthesis=false
vae_checkpoint=""
flow_checkpoint=""
normalization=""
output=""
persona=""
date=""
seed=42
device="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --python) python_cmd="$2"; shift 2 ;;
    --dayforge-root) dayforge_root="$2"; shift 2 ;;
    --derived-root) derived_root="$2"; shift 2 ;;
    --mapping-config) mapping_config="$2"; shift 2 ;;
    --mapping-output) mapping_output="$2"; shift 2 ;;
    --run-synthesis) run_synthesis=true; shift ;;
    --vae-checkpoint) vae_checkpoint="$2"; shift 2 ;;
    --flow-checkpoint) flow_checkpoint="$2"; shift 2 ;;
    --normalization) normalization="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --persona) persona="$2"; shift 2 ;;
    --date) date="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$dayforge_root" || -z "$mapping_output" ]]; then
  echo "--dayforge-root and --mapping-output are required." >&2
  usage >&2
  exit 2
fi
[[ -d "$dayforge_root" ]] || { echo "DayForge root not found: $dayforge_root" >&2; exit 2; }
[[ -z "$derived_root" || -d "$derived_root" ]] || { echo "Derived root not found: $derived_root" >&2; exit 2; }
[[ -f "$mapping_config" ]] || { echo "Mapping config not found: $mapping_config" >&2; exit 2; }

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

map_args=( -m lrf_imu map-dayforge-physical-states --dayforge-root "$dayforge_root" --config "$mapping_config" --output-dir "$mapping_output" )
[[ -z "$derived_root" ]] || map_args+=( --derived-root "$derived_root" )
"$python_cmd" "${map_args[@]}"

if [[ "$run_synthesis" == true ]]; then
  [[ -n "$vae_checkpoint" && -f "$vae_checkpoint" ]] || { echo "--vae-checkpoint is required for synthesis" >&2; exit 2; }
  [[ -n "$flow_checkpoint" && -f "$flow_checkpoint" ]] || { echo "--flow-checkpoint is required for synthesis" >&2; exit 2; }
  [[ -n "$normalization" && -f "$normalization" ]] || { echo "--normalization is required for synthesis" >&2; exit 2; }
  [[ -n "$output" && -n "$persona" && -n "$date" ]] || { echo "--output, --persona, and --date are required for synthesis" >&2; exit 2; }
  "$python_cmd" -m lrf_imu synthesize-dayforge \
    --dayforge-root "$dayforge_root" \
    --mapping-root "$mapping_output" \
    --vae-checkpoint "$vae_checkpoint" \
    --flow-checkpoint "$flow_checkpoint" \
    --normalization-metadata "$normalization" \
    --output-dir "$output" \
    --persona "$persona" \
    --date "$date" \
    --seed "$seed" \
    --device "$device"
fi
