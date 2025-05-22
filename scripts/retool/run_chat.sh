#!/bin/bash
set -eu

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Agent-R1

cd ~/Agent-R1
python -m agent_r1.vllm_infer.chat --config ./scripts/retool/config.py