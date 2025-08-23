#!/usr/bin/env bash
# Run training and/or testing for every experiments/**/config.json as individual Slurm jobs.
# Usage: ./scripts/run_all_train_test_slurm.sh [train|test|both] [ROOT_DIR] [--account ...] [--partition ...] [-- ...extra args]

set -uo pipefail

: "${LOG_DIR:=./logs}"
MAX_JOBS=64

mkdir -p "$LOG_DIR"

MODE="${1:-both}"
shift || true

ROOT_DIR="experiments"
PASS_ARGS=()
ACCOUNT="${SLURM_ACCOUNT:-}"
PARTITION="${SLURM_PARTITION:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --) shift; PASS_ARGS=("$@"); break ;;
    *)  ROOT_DIR="$1"; shift ;;
  esac
done

declare -a SBATCH_OPTS=(
  [0]="--output=$LOG_DIR/slurm_%x_%j.out"
  [1]="--error=$LOG_DIR/slurm_%x_%j.err"
)
if [[ -n "$ACCOUNT" ]]; then
  SBATCH_OPTS+=(--account="$ACCOUNT")
fi
if [[ -n "$PARTITION" ]]; then
  SBATCH_OPTS+=(--partition="$PARTITION")
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: '$ROOT_DIR' is not a directory." >&2
  exit 1
fi

CONFIGS=()
while IFS= read -r -d $'\0' file; do
  CONFIGS+=("$file")
done < <(find "$ROOT_DIR" -type f -name 'config.json' -print0)
NUM_CONFIGS=${#CONFIGS[@]}

if (( NUM_CONFIGS == 0 )); then
  echo "No config.json files found in $ROOT_DIR"
  exit 0
fi

SLURM_SCRIPT="./scripts/slurm/train_test.sbatch.sh"
echo "Using Slurm sbatch script $SLURM_SCRIPT"

# Function to limit concurrent jobs
# squeue -h -t pending,running -r
function wait_for_jobs {
  local max_jobs=$1
  while (( $(squeue -h -t pending,running -r | wc -l) >= max_jobs )); do
    sleep 3m
  done
}

for CFG in "${CONFIGS[@]}"; do
  wait_for_jobs "$MAX_JOBS"
  sbatch "${SBATCH_OPTS[@]}" "$SLURM_SCRIPT" "$CFG" "$LOG_DIR" "$MODE" "${PASS_ARGS[@]}"
done

echo "Submitted $NUM_CONFIGS configs as individual Slurm jobs (max $MAX_JOBS concurrent submissions)."
echo "Logs will be in $LOG_DIR and slurm output/error in logs/."
