#!/usr/bin/env bash
#SBATCH --job-name=test_cqf
##SBATCH --output=./logs/slurm_%x_%A_%a.out
##SBATCH --error=./logs/slurm_%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=23:00:00

CONFIG_LIST_FILE="$1"
LOG_DIR="$2"
shift 2
PASS_ARGS=("$@")

CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CONFIG_LIST_FILE")
if [[ -z "$CFG" ]]; then
  echo "No config for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 1
fi

safe_name="${CFG//\//__}"
safe_name="${safe_name// /_}"
log_file="$LOG_DIR/${safe_name%.json}.testing.log"

echo "Running config: $CFG"
srun micromamba run -n cqf python src/infrastructure/testing.py --path_to_experiment_file "$CFG" "${PASS_ARGS[@]}" >"$log_file" 2>&1
