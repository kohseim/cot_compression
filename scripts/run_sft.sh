#!/usr/bin/env bash
# Full-FT SFT on arithmetic chain data via LLaMA-Factory.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LF_DIR="${PROJECT_ROOT}/LLaMA-Factory"

: "${DATA_ROOT:?DATA_ROOT is required}"
: "${MODE:?MODE is required}"

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B}
TEMPLATE=${TEMPLATE:-qwen}
OP_MIN=${OP_MIN:-10}
OP_MAX=${OP_MAX:-12}
N_GPUS=${N_GPUS:-1}
PER_DEVICE_BS=${PER_DEVICE_BS:-4}
GRAD_ACC=${GRAD_ACC:-12}
LR=${LR:-2e-5}
EPOCHS=${EPOCHS:-1}
CUTOFF_LEN=${CUTOFF_LEN:-2048}

mode_slug="${MODE//\//_}"
DATASET_NAME="arithmetic_sft_${mode_slug}"
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/saves/arithmetic_sft/${BASE_MODEL//\//-}/${mode_slug}}

python "${LF_DIR}/scripts/prepare_arithmetic_sft.py" \
  --data_root "${DATA_ROOT}" \
  --op_min "${OP_MIN}" --op_max "${OP_MAX}" \
  --modes "${MODE}" \
  --overwrite --register

CONFIG_FILE="$(mktemp -t sft_config.XXXXXX.yaml)"
trap 'rm -f "${CONFIG_FILE}"' EXIT

cat > "${CONFIG_FILE}" <<EOF
model_name_or_path: ${BASE_MODEL}
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: full
template: ${TEMPLATE}
dataset: ${DATASET_NAME}
dataset_dir: ${LF_DIR}/data
cutoff_len: ${CUTOFF_LEN}
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
output_dir: ${OUTPUT_DIR}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: none
per_device_train_batch_size: ${PER_DEVICE_BS}
gradient_accumulation_steps: ${GRAD_ACC}
learning_rate: ${LR}
num_train_epochs: ${EPOCHS}
optim: adamw_torch
weight_decay: 0.1
max_grad_norm: 1.0
lr_scheduler_type: cosine_with_min_lr
lr_scheduler_kwargs: '{"min_lr": 3.0e-6}'
warmup_ratio: 0.05
bf16: true
ddp_timeout: 180000000
EOF

if [[ ${N_GPUS} -gt 1 ]]; then
  export FORCE_TORCHRUN=1
  export NPROC_PER_NODE="${N_GPUS}"
fi

cd "${LF_DIR}"
exec python -m llamafactory.cli train "${CONFIG_FILE}"
