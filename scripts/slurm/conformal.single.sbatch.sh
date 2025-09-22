#!/bin/bash
#SBATCH --job-name=conf_all 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=12
#SBATCH --time=0-01:00:00 
#SBATCH --mem-per-task=16G

DATSETS=("bio" "blog" "sgemm")
SEEDS=(0 1 2 3 4 5 6 7 8 9)

for dataset in "${DATSETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        #SRUN_OPTS[2]="--job-name=conf_${dataset}_${seed}"
        SRUN_OPTS[0]="--output=$LOG_DIR/conf_single_${dataset}_${seed}.log"
        SRUN_OPTS[1]="--error=$LOG_DIR/conf_single_${dataset}_${seed}.err"
        SRUN_OPTS[2]="-N 1 -n 1 -c 12"
        srun "${SRUN_OPTS[@]}" uv run python -m conformal.experiment --dataset=$dataset --seed=$seed --n_cpus=12 --all &
    done
    wait
    echo Finished $dataset
done
