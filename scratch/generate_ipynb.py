import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Multi-Agent Reinforcement Learning Training in Colab\n",
    "This notebook will clone your private repositories and run the massive 100 Million timestep training on a 10x10 grid with 4 agents and 2 boxes.\n",
    "\n",
    "### Prerequisites:\n",
    "1. On the left sidebar of Colab, click the **🔑 Secrets** icon.\n",
    "2. Add a new secret named `GITHUB_TOKEN` and paste your GitHub Personal Access Token as the value.\n",
    "3. Enable `Notebook access` for the secret."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import userdata\n",
    "import os\n",
    "\n",
    "github_token = userdata.get('GITHUB_TOKEN')\n",
    "github_user = 'nabilrk'\n",
    "\n",
    "# Clone the custom environment and epymarl framework\n",
    "!git clone https://{github_user}:{github_token}@github.com/{github_user}/lb-transportation.git\n",
    "!git clone https://{github_user}:{github_token}@github.com/{github_user}/epymarl.git\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies (this might take a minute or two)\n",
    "!apt-get update\n",
    "!apt-get install -y python-opengl xvfb libglu1-mesa freeglut3-dev\n",
    "!sed -i 's/\"max_episode_steps\": 50/\"max_episode_steps\": 100/g' lb-transportation/lbtransportation/__init__.py\n",
    "!pip install -e lb-transportation/\n",
    "!sed -i 's/PyYAML==5.3.1/PyYAML/g' epymarl/requirements.txt\n",
    "!pip install -r epymarl/requirements.txt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start the 100M timestep training run!\n",
    "# We pipe the output to a log file so it doesn't crash your browser tab with print statements.\n",
    "import os\n",
    "# Change directory to the epymarl folder we just cloned\n",
    "if os.path.exists('epymarl'):\n",
    "    os.chdir('epymarl')\n",
    "\n",
    "!python src/main.py --config=mappo --env-config=gymma with \\\n",
    "    env_args.time_limit=100 \\\n",
    "    env_args.key=\"lbtransportation:Transportation-10x10-4p-2f-coop-v3\" \\\n",
    "    t_max=100000000 \\\n",
    "    save_model=True \\\n",
    "    save_model_interval=2000000 \\\n",
    "    test_nepisode=100 \\\n",
    "    test_interval=2000000 > training.log 2>&1 &\n",
    "\n",
    "print(\"Training started in the background! Run the next cell to monitor progress.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run this cell anytime to see the latest training progress\n",
    "!tail -n 30 training.log\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Once training is complete, generate a video of the evaluation\n",
    "import glob\n",
    "import os\n",
    "\n",
    "# Find the latest saved model\n",
    "models = glob.glob('results/models/*')\n",
    "latest_model = max(models, key=os.path.getctime)\n",
    "\n",
    "!xvfb-run -a python src/main.py --config=mappo --env-config=gymma with \\\n",
    "    env_args.time_limit=100 \\\n",
    "    env_args.key=\"lbtransportation:Transportation-10x10-4p-2f-coop-v3\" \\\n",
    "    t_max=100 \\\n",
    "    evaluate=True \\\n",
    "    test_nepisode=10 \\\n",
    "    test_greedy=True \\\n",
    "    load_step=100000000 \\\n",
    "    checkpoint_path=\"{latest_model}\" \\\n",
    "    env_args.record_video=True\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display the generated video directly in the notebook\n",
    "from IPython.display import HTML\n",
    "from base64 import b64encode\n",
    "\n",
    "video_path = glob.glob('results/replays/*.mp4')[0]\n",
    "mp4 = open(video_path, 'rb').read()\n",
    "data_url = \"data:video/mp4;base64,\" + b64encode(mp4).decode()\n",
    "\n",
    "HTML(f\"\"\"\n",
    "<video width=400 controls>\n",
    "      <source src=\"{data_url}\" type=\"video/mp4\">\n",
    "</video>\n",
    "\"\"\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('/home/nabil/.gemini/antigravity/brain/6941f617-02f6-48db-9c83-71c1f32fc5d1/marl_colab_training.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
