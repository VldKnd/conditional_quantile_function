datasets=("rf1" "rf2" "scm1d" "scm20d" "sgemm" "bio" "blog")
seeds=(0 1 2 3 4 5 6 7 8 9)
MAX_JOBS=64
LOG_DIR="./logs"

declare -a SBATCH_OPTS=(
    [0]="--account=$SLURM_ACCOUNT"
    [1]="--partition=$SLURM_PARTITION"
)

# Function to limit concurrent jobs
# squeue -h -t pending,running -r
function wait_for_jobs {
  local max_jobs=$1
  while (( $(squeue -h -t pending,running -r | wc -l) >= max_jobs )); do
    sleep 2m
  done
}

for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        wait_for_jobs "$MAX_JOBS"
        SBATCH_OPTS[2]="--job-name=conf_${dataset}_${seed}"
        SBATCH_OPTS[3]="--output=$LOG_DIR/conformal_${dataset}_${seed}.log"
        SBATCH_OPTS[4]="--error=$LOG_DIR/conformal_${dataset}_${seed}.err"
        sbatch "${SBATCH_OPTS[@]}" ./scripts/slurm/conformal.sbatch.sh "$dataset" $seed --all
        sleep 3
    done
done
