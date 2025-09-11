#!/usr/bin/env bash
##SBATCH --job-name=train_test_cqf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=23:00:00

dataset="$1"
seed="$2"
shift 2
PASS_ARGS=("$@")


srun micromamba run -n cqf-conformal python -m conformal.experiment --dataset=$dataset --seed=$seed "${PASS_ARGS[@]}"
