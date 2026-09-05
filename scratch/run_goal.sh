#!/bin/bash
source venv/bin/activate

echo "Updating max_episode_steps to 100..."
sed -i 's/"max_episode_steps": 50/"max_episode_steps": 100/g' /home/nabil/lb-transportation/lbtransportation/__init__.py

LATEST_MODEL=$(ls -td results/models/*10x10-4p-2f-coop* | head -1)
echo "Resuming training from latest model: $LATEST_MODEL"

python3 src/main.py --config=mappo --env-config=gymma with \
    env_args.time_limit=100 \
    env_args.key="lbtransportation:Transportation-10x10-4p-2f-coop-v3" \
    t_max=100000000 \
    save_model=True \
    save_model_interval=2000000 \
    test_nepisode=100 \
    test_interval=2000000 \
    checkpoint_path="$LATEST_MODEL" > training_100m_goal_resumed.log 2>&1

echo "Training complete. Finding latest model..."
FINAL_MODEL=$(ls -td results/models/*10x10-4p-2f-coop* | head -1)
echo "Evaluating model: $FINAL_MODEL"

xvfb-run -a python3 src/main.py --config=mappo --env-config=gymma with \
    env_args.time_limit=100 \
    env_args.key="lbtransportation:Transportation-10x10-4p-2f-coop-v3" \
    t_max=100 \
    evaluate=True \
    test_nepisode=1000 \
    test_greedy=True \
    load_step=100000000 \
    checkpoint_path="$FINAL_MODEL" > eval_100m_goal.log 2>&1

echo "Extracting metrics to CSV..."
python3 -c '
import re, csv

with open("eval_100m_goal.log", "r") as f:
    content = f.read()

ep_len_match = re.search(r"test_ep_length_mean:\s+([\d\.]+)", content)
return_match = re.search(r"test_return_mean:\s+([\d\.\-]+)", content)

ep_len = float(ep_len_match.group(1)) if ep_len_match else 0.0
ret = float(return_match.group(1)) if return_match else 0.0

fruits = (ret + 0.04 * ep_len) / 4.0
percent = (fruits / 2.0) * 100.0

csv_file = "/home/nabil/.gemini/antigravity/brain/6941f617-02f6-48db-9c83-71c1f32fc5d1/evaluation_100m_metrics.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Average Episode Length", ep_len])
    writer.writerow(["Average Return", ret])
    writer.writerow(["Average Fruits Delivered", fruits])
    writer.writerow(["Success Percentage", f"{percent:.2f}%"])

print(f"Metrics saved to {csv_file}")
'
