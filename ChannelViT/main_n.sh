#!/bin/bash

: <<'COMMENT'
This script runs ChannelViT/main_n.py 50 times with different seeds (1 to 50).
For each run, it sets the --seed argument and uses --retrain retrain_second_stage.


for seed in {1..50}
do
    echo "Running with seed $seed"
    python ChannelViT/main_n.py --seed "$seed" --retrain retrain_second_stage
done

COMMENT



for seed in {1..5}
do
    echo "Running with seed $seed"
    python main_n.py --seed "$seed" --retrain retrain_second_stage
done
