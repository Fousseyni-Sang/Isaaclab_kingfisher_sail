#!/bin/bash

# List of seeds to try
seeds=(15) #250 10 25) #30 44 15 

# Loop over seeds
for seed in "${seeds[@]}"; do
    echo "No_spearman: Launching training with seed $seed..."
    python source/standalone/workflows/rl_games/bipedal_gym.py \
        --seed $seed
done
