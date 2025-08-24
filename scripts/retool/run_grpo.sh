#!/bin/bash
set -eu

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Agent-R1

export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
# export all_proxy=socks5://127.0.0.1:7891

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=60000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_USE_V1=1

ADV_ESTIMATOR=GRPO
RL_DATASET=ReTool
MAX_RESPONSE_LENGTH=8192
MICRO_BATCH_SIZE_PER_GPU=4
KL_COEF=0.001
ROLLOUT=8
EPOCHS=5

MODEL_NAME=Tool-Star-Qwen-3B
MODEL_PATH=dongguanting/Tool-Star-Qwen-3B

PROJECT_NAME=Agent-R1_Math-ReTool
DATE_SUFFIX=$(date +"%Y%m%d")
EXPERIMENT_NAME=${MODEL_NAME}_${ADV_ESTIMATOR}_${RL_DATASET}_EPOCHS-${EPOCHS}_BS-${MICRO_BATCH_SIZE_PER_GPU}_KL-${KL_COEF}_ROLLOUT-${ROLLOUT}-MAXLEN-${MAX_RESPONSE_LENGTH}_${DATE_SUFFIX}
OUTPUT_DIR=/data/Mamba/Project/Agent-R1/ReTool/${MODEL_NAME}_${ADV_ESTIMATOR}_${RL_DATASET}_EPOCHS-${EPOCHS}_BS-${MICRO_BATCH_SIZE_PER_GPU}_KL-${KL_COEF}_ROLLOUT-${ROLLOUT}-MAXLEN-${MAX_RESPONSE_LENGTH}_${DATE_SUFFIX}

TRAIN_FILES=/data/Mamba/Project/Agent-R1/ReTool/data/train.parquet
VALID_FILES=/data/Mamba/Project/Agent-R1/ReTool/data/test.parquet

mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}/model_train.log"
if [ -f "$log_file" ]; then
    echo "Overwrite Log: $log_file"
    > "$log_file"
else
    echo "Create Log: $log_file"
    touch "$log_file"
fi

echo "=============================================="
echo "Real-Time Training Log Monitoring"
echo "tail -f ${log_file}"
echo "=============================================="

cd ~/Agent-R1
PYTHONWARNINGS="ignore" python -m agent_r1.src.main_agent \
    algorithm.adv_estimator=${ADV_ESTIMATOR,,} \
    algorithm.use_kl_in_reward=False \
    data.train_files=[$TRAIN_FILES] \
    data.val_files=[$VALID_FILES] \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_response_length_single_turn=8192 \
    data.use_default_tool_template=False \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-06 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n_repeat=$ROLLOUT \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.stop_token_ids=[] \
    'actor_rollout_ref.rollout.stop=["</code>"]' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=1 \
    trainer.total_epochs=$EPOCHS "${@:1}" \
	+trainer.wandb_proxy=http://127.0.0.1:7891 \
    tool.tools=['python'] \
    tool.env=retool \
    tool.max_turns=5 \
    tool.max_tool_response_length=4096 \
    >> "${log_file}" 2>&1
