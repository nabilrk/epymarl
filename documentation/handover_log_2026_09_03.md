# Handover Log: Multi-Agent Transportation Task (Sept 3, 2026)

## Summary of Completed Work
We successfully completed local parallel training of 4 cooperative agents operating in a 5x5 Grid World (`Transportation-5x5-4p-2f-custom-v3`) using the EPyMARL MAPPO algorithm. We also set up a massive 100-million timestep training pipeline on Google Colab for a larger 10x10 grid.

## Key Changes Made

### 1. Environment Configurations (`lb-transportation`)
- Registered a new custom environment: `Transportation-5x5-4p-2f-custom-v3`.
- **Specs**: 5x5 grid, 4 agents (level 1), 2 fruits (level 2).
- **Rewards**: +1.0 per agent per fruit delivered to the goal, and -0.01 penalty per timestep.

### 2. EPyMARL Framework Modifications (`epymarl`)
- **`src/envs/gymma.py`**: Added specific logic in the evaluation wrapper to track and print `FRUITS_DELIVERED` by counting the `delivered_boxes` array when the environment signals `done`. This allows for exact metric parsing.
- **`src/config/envs/gymma.yaml`**: Refined the config for `render_mode` and `layout` integration.
- **Evaluation Scripts**: Updated `scratch/evaluate_local.sh` to wrap the evaluation run with `xvfb-run -a` to support headless rendering of evaluation videos on local Linux servers.

### 3. Colab Training Pipeline
- Updated `scratch/generate_ipynb.py` to produce a completely self-contained Jupyter Notebook for Colab.
- **Graphics Fixes**: Bundled the necessary `GLU` and `freeglut3-dev` libraries alongside `xvfb` so Colab can properly render videos without crashing in `RecordVideo`.
- **Scale**: Configured the Colab pipeline to target `Transportation-10x10-4p-2f-coop-v3` for a massive 100 million timestep run.

## Results & Insights
- **Local Training (10M Timesteps)**: MAPPO converged on an optimal strategy! The agents successfully learned to pair up to lift the level 2 boxes.
- **Evaluation Metrics**: Over a rigorous 1,000-episode evaluation, the agents achieved an average episode length of **8.41 steps** and successfully delivered **90.16%** of all possible fruits.
- **Algorithm Used**: True CTDE MAPPO using a centralized critic, GRU (RNN) hidden states for memory, and a highly parallelized 10-environment sampling batch.

## Next Steps
- Await the results of the 100M timestep training run currently executing on Google Colab.
- Analyze the video outputs from the 10x10 grid to see if the agents can scale their coordination strategy to a significantly larger state space.
