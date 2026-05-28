#!/usr/bin/env bash
# GRPO RLVR on arithmetic chain data via verl.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${TRAIN_FILE:?TRAIN_FILE is required}"
: "${VAL_FILE:?VAL_FILE is required}"

N_GPUS=${N_GPUS:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
TRAIN_BS=${TRAIN_BS:-256}
MINI_BS=${MINI_BS:-256}
MICRO_BS=${MICRO_BS:-2}
LR=${LR:-1e-6}
KL_COEF=${KL_COEF:-0.001}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-2048}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_STEPS=${TOTAL_STEPS:-150}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(basename "${MODEL_PATH}")-rlvr}
SAVE_DIR=${SAVE_DIR:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}

export VLLM_USE_V1=0
cd "${PROJECT_ROOT}"

exec python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BS}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BS}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BS}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BS}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BS}" \
  custom_reward_function.path="${PROJECT_ROOT}/src/rlvr_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.logger='["console"]'
