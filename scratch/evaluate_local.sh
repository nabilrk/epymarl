#!/bin/bash
source venv/bin/activate

# Find the latest model folder
LATEST_MODEL=$(ls -td results/models/*5x5-4p-2f-custom* | head -1)
echo "Evaluating latest model: $LATEST_MODEL"

xvfb-run -a python3 src/main.py --config=mappo --env-config=gymma with \
    env_args.time_limit=100 \
    env_args.key="lbtransportation:Transportation-5x5-4p-2f-custom-v3" \
    t_max=100 \
    evaluate=True \
    test_nepisode=10 \
    test_greedy=True \
    load_step=10000000 \
    checkpoint_path="$LATEST_MODEL" \
    env_args.record_video=True
