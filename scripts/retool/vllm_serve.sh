#!/bin/bash
set -eu

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Agent-R1

export CUDA_VISIBLE_DEVICES=7
export MODEL_NAME=/data/Mamba/Project/Agent-R1/ReTool/Qwen2.5-7B-Instruct_GRPO_ReTool_EPOCHS-5_BS-8_KL-0.001_ROLLOUT-8-MAXLEN-8192_20250516/global_step_600/actor_huggingface

vllm serve $MODEL_NAME --enable-auto-tool-choice --tool-call-parser hermes --served-model-name agent --port 8000