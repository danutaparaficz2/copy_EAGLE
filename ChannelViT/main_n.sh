#!/bin/bash

for seed in {1..50}
do
    echo "Running with seed $seed"
    python ChannelViT/main_n.py --seed "$seed" --retrain retrain_second_stage
done