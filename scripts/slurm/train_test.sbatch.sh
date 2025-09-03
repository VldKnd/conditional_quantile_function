#!/usr/bin/env bash
#SBATCH --job-name=train_test_cqf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=23:00:00

CFG="$1"
LOG_DIR="$2"
MODE="${3:-both}"
shift 3
PASS_ARGS=("$@")

if [[ -z "$CFG" ]]; then
  echo "No config file provided"
  exit 1
fi

# Replace all slashes and spaces with a single underscore (no trimming)
safe_name="${CFG//[\/ ]/_}"

if [[ "$MODE" == "train" || "$MODE" == "both" ]]; then
  log_file="$LOG_DIR/${safe_name%.json}.training.log"
  echo "Training config: $CFG"
  srun micromamba run -n cqf python src/infrastructure/training.py --path_to_experiment_file "$CFG" "${PASS_ARGS[@]}" >"$log_file" 2>&1
fi

if [[ "$MODE" == "test" || "$MODE" == "both" ]]; then
  log_file="$LOG_DIR/${safe_name%.json}.testing.log"
  echo "Testing config: $CFG"
  srun micromamba run -n cqf python src/infrastructure/testing.py --path_to_experiment_file "$CFG" "${PASS_ARGS[@]}" >"$log_file" 2>&1
fi
