#!/usr/bin/env bash
# Run training for every experiments/**/config.json found as Slurm jobs.
# Each config runs in parallel, up to 64 concurrent jobs. Logging always enabled.

set -uo pipefail

mkdir -p tmp

# -------- Options via env vars --------
: "${LOG_DIR:=./logs}"             # logs are always written here
MAX_JOBS=64                        # max concurrent jobs

mkdir -p "$LOG_DIR"

# -------- Parse args --------
ROOT_DIR="experiments"
PASS_ARGS=()
#SBATCH_OPTS=()
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

# -------- Build sbatch options --------
declare -a SBATCH_OPTS=(
  [0]="--output=$LOG_DIR/slurm_%x_%A_%a.out"
  [1]="--error=$LOG_DIR/slurm_%x_%A_%a.err"
)
if [[ -n "$ACCOUNT" ]]; then
  SBATCH_OPTS+=(--account="$ACCOUNT")
fi
if [[ -n "$PARTITION" ]]; then
  SBATCH_OPTS+=(--partition="$PARTITION")
fi

# -------- Setup --------
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: '$ROOT_DIR' is not a directory." >&2
  exit 1
fi

# -------- Find configs --------
CONFIGS=()
while IFS= read -r -d $'\0' file; do
  CONFIGS+=("$file")
done < <(find "$ROOT_DIR" -type f -name 'config.json' -print0)
NUM_CONFIGS=${#CONFIGS[@]}

if (( NUM_CONFIGS == 0 )); then
  echo "No config.json files found in $ROOT_DIR"
  exit 0
fi

# -------- Write configs to a file for the job array --------
CONFIG_LIST_FILE=$(mktemp ./tmp/configs.XXXXXX.txt)
for cfg in "${CONFIGS[@]}"; do
  echo "$cfg" >> "$CONFIG_LIST_FILE"
done
echo "Temporary configs list saved to $CONFIG_LIST_FILE "

# -------- Slurm job script --------
SLURM_SCRIPT="./scripts/slurm/train.sbatch.sh"
echo "Using Slurm sbatch script $SLURM_SCRIPT"

# -------- Submit job array --------
set -x
# For compatibility with Bash < 4.4
sbatch "${SBATCH_OPTS[@]}" --array=0-$(($NUM_CONFIGS-1))%$MAX_JOBS "$SLURM_SCRIPT" "$CONFIG_LIST_FILE" "$LOG_DIR" "${PASS_ARGS[@]+${doublequote}PASS_ARGS[@]${doublequote}}"
set +x

echo "Submitted $NUM_CONFIGS configs as a Slurm job array (max $MAX_JOBS concurrent)."
echo "Logs will be in $LOG_DIR and slurm output/error in logs/."
