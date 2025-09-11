datasets=("rf1" "rf2" "scm1d" "scm20d" "sgemm" "bio" "blog")
seeds=(0 1 2 3 4)
MAX_JOBS=64

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
        sbatch ./scripts/slurm/conformal.sbatch.sh $dataset $seed --all
    done
done
