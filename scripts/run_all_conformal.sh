datasets=("rf1" "rf2" "scm1d" "scm20d" "sgemm" "bio" "blog")
seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        uv run python -m conformal.experiment --dataset=$dataset --seed=$seed
    done
done
