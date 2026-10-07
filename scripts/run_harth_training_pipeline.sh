#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_harth_training_pipeline.sh --data-root PATH --output-root PATH [options]

Run prepare-harth-data -> train-harth-vae -> train-harth-flow through the
canonical python -m lrf_imu interface.

Options:
  --config PATH              Frozen HARTH configuration
  --held-out-subject ID      LOSO subject (default: harth:S006)
  --seed N                   Global seed (default: 42)
  --vae-epochs N             Optional VAE epoch override
  --flow-epochs N            Optional Flow epoch override
  --max-train-batches N      Optional bounded smoke override
  --max-val-batches N        Optional bounded smoke override
  --python PATH              Python launcher
  --dry-run                  Print commands without writing or training
  -h, --help                 Show this help
EOF
}

python_cmd="${PYTHON:-python}"
data_root=""
output_root=""
config="configs/paper/harth_10class_160_40.yaml"
held_out="harth:S006"
seed=42
vae_epochs=""
flow_epochs=""
max_train_batches=""
max_val_batches=""
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --python) python_cmd="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --held-out-subject) held_out="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --vae-epochs) vae_epochs="$2"; shift 2 ;;
    --flow-epochs) flow_epochs="$2"; shift 2 ;;
    --max-train-batches) max_train_batches="$2"; shift 2 ;;
    --max-val-batches) max_val_batches="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$data_root" || -z "$output_root" ]]; then
  echo "--data-root and --output-root are required." >&2
  usage >&2
  exit 2
fi
[[ -d "$data_root" ]] || { echo "Data root not found: $data_root" >&2; exit 2; }

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$config" != /* ]]; then config="$repo_root/$config"; fi
[[ -f "$config" ]] || { echo "Configuration not found: $config" >&2; exit 2; }
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

vae_output="$output_root/vae"
flow_output="$output_root/flow"
vae_checkpoint="$vae_output/vae_s3_z48.pt"
flow_checkpoint="$flow_output/flow_unet_best.pt"

prepare=( -m lrf_imu prepare-harth-data --data-root "$data_root" --composition harth_walking_speed --held-out-subject "$held_out" --window-length 160 --hop-length 40 --seed "$seed" )
vae=( -m lrf_imu train-harth-vae --data-root "$data_root" --composition harth_walking_speed --held-out-subject "$held_out" --config "$config" --output-dir "$vae_output" --seed "$seed" )
flow=( -m lrf_imu train-harth-flow --data-root "$data_root" --composition harth_walking_speed --held-out-subject "$held_out" --config "$config" --vae-checkpoint "$vae_checkpoint" --output-dir "$flow_output" --seed "$seed" )
[[ -z "$vae_epochs" ]] || vae+=( --epochs "$vae_epochs" )
[[ -z "$flow_epochs" ]] || flow+=( --epochs "$flow_epochs" )
[[ -z "$max_train_batches" ]] || { vae+=( --max-train-batches "$max_train_batches" ); flow+=( --max-train-batches "$max_train_batches" ); }
[[ -z "$max_val_batches" ]] || { vae+=( --max-val-batches "$max_val_batches" ); flow+=( --max-val-batches "$max_val_batches" ); }

if [[ "$dry_run" == true ]]; then
  echo "dry_run=true"
  printf 'prepare='; printf '%q ' "$python_cmd" "${prepare[@]}"; printf '\n'
  printf 'vae='; printf '%q ' "$python_cmd" "${vae[@]}"; printf '\n'
  printf 'flow='; printf '%q ' "$python_cmd" "${flow[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$output_root"
"$python_cmd" "${prepare[@]}"
"$python_cmd" "${vae[@]}"
[[ -f "$vae_checkpoint" ]] || { echo "Missing expected VAE checkpoint: $vae_checkpoint" >&2; exit 3; }
"$python_cmd" "${flow[@]}"
[[ -f "$flow_checkpoint" ]] || { echo "Missing expected Flow checkpoint: $flow_checkpoint" >&2; exit 3; }
echo "training_pipeline=complete"
echo "vae_checkpoint=$vae_checkpoint"
echo "flow_checkpoint=$flow_checkpoint"
