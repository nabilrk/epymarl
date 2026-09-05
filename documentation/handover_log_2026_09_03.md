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

---

## Update: September 5, 2026 (100-Million Timestep Local Evaluation)

**Environment**: `Transportation-10x10-4p-2f-coop-v3`
**Scale**: 100 Million Timesteps (Locally trained via daemon processes over 17+ hours)

### Key Changes
- Updated the base environment configuration in `lb-transportation/__init__.py` to increase `max_episode_steps` from 50 to 100 globally to give agents enough time to navigate the 10x10 grid.
- Engineered `scratch/run_goal.sh` to automate training, model checkpoint recovery, greedy evaluation, and CSV metric extraction.

### Results & Insights (10x10 Grid)
The 100M timestep training run has completed. Over a rigorous 1,000-episode evaluation:
- **Average Episode Length**: 100.0 steps
- **Average Return**: -4.0
- **Success Percentage**: 0.00%

**Conclusion**: The agents completely failed to learn to solve the 10x10 grid. Unlike the 5x5 grid, the state space in a 10x10 grid is 4 times larger. The agents rely on random exploration to stumble upon a reward initially. The statistical probability of two agents randomly deciding to stand next to the same box, both executing the `LOAD` action simultaneously, and then randomly carrying it to the goal is astronomically low. The standard MAPPO exploration simply isn't dense enough.

**Next Steps**: 
To scale to the 10x10 grid, the agents will likely require **Curriculum Learning** (slowly scaling the grid from 5x5 up to 10x10) or dense proxy rewards to bootstrap the initial coordination behavior.
