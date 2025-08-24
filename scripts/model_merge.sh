#!/bin/bash
set -eu

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Agent-R1

BACKEND=fsdp
LOCAL_DIR=/data/Mamba/Project/Agent-R1/ReTool/Qwen2.5-7B-Instruct_GRPO_ReTool_EPOCHS-5_BS-8_KL-0.001_ROLLOUT-8-MAXLEN-8192_20250516/global_step_690/actor
TARGET_DIR=${LOCAL_DIR}_huggingface

cd ~/verl
python scripts/model_merger.py merge \
    --backend $BACKEND \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR