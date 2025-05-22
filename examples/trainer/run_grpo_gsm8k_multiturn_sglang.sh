# run on 8xH100
# make sure your current working directory is the root of the project

#!/bin/bash
set -eu

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Agent-R1

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=60000
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

ulimit -n 65535

cd ~/verl
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

ADV_ESTIMATOR=GRPO
RL_DATASET=GSM8K
MAX_RESPONSE_LENGTH=1024
BATCH_SIZE=128
MICRO_BATCH_SIZE_PER_GPU=16
KL_COEF=0.001
ROLLOUT=16
EPOCHS=5

TRAIN_FILES=/data/Mamba/Project/Agent-R1/GSM8K/data/train.parquet
VALID_FILES=/data/Mamba/Project/Agent-R1/GSM8K/data/test.parquet

MODEL_NAME=Qwen2.5-3B-Instruct
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct

PROJECT_NAME=Async-RL_GSM8K
DATE_SUFFIX=$(date +"%Y%m%d")
EXPERIMENT_NAME=${MODEL_NAME}_${ADV_ESTIMATOR}_${RL_DATASET}_EPOCHS-${EPOCHS}_BS-${BATCH_SIZE}_MICROBS-${MICRO_BATCH_SIZE_PER_GPU}_KL-${KL_COEF}_ROLLOUT-${ROLLOUT}-MAXLEN-${MAX_RESPONSE_LENGTH}_${DATE_SUFFIX}
OUTPUT_DIR=/data/Mamba/Project/Agent-R1/GSM8K/${MODEL_NAME}_${ADV_ESTIMATOR}_${RL_DATASET}_EPOCHS-${EPOCHS}_BS-${BATCH_SIZE}_MICROBS-${MICRO_BATCH_SIZE_PER_GPU}_KL-${KL_COEF}_ROLLOUT-${ROLLOUT}-MAXLEN-${MAX_RESPONSE_LENGTH}_${DATE_SUFFIX}

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

PYTHONWARNINGS="ignore::FutureWarning" python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$MAX_RESPONSE_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$ROLLOUT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VALID_FILES \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    trainer.total_epochs=$EPOCHS \
    >> "${log_file}" 2>&1
