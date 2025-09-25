datasets=("rf1" "rf2" "scm1d" "scm20d" "sgemm" "bio" "blog")
seeds=(53)

for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        uv run python -m conformal.tune_params --dataset=$dataset --seed=$seed
    done
done
